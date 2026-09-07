#!/usr/bin/env python3
"""
judge.py

aggregate_results.py -- turns raw per-model/per-job files 
into one combined file per candidate-per-job 

Takes candidate.json file with a job listing PDF, and sends them to a local Ollama model 
(e.g. deepseek-r1:8b) acting as a "judge" to produce a single, independent final
evaluation and score out of 10.

The job listing PDF is read with the cv-parser pipeline (parsed_cv.parse(...)
.full_text())

Per-summary "score" fields are stripped out before the
prompt is built by default. 
Pass --keep-scores to disable this.

The judge's own output is validated before it is written anywhere: it must
parse as JSON and carry a numeric final_score in [0, 10]. If it doesn't,
nothing is written to the "clean" output path -- the raw response is instead
saved next to it with an ".invalid.json" suffix and the script exits non-zero,
so a driving shell script can flag the failure instead of silently writing a
bad row. CSV output is more forgiving: a validation problem is recorded in the
row itself (in an "errors" column) rather than blocking the write, since a
half-populated CSV row is much lower-risk than a structurally-broken JSON file
representing itself as good data.

Usage:
    python judge.py --summaries combined/secretary/lerato_munche.json \\
        --job job_listings/job_listing_secretary.pdf \\
        --output judged/secretary/lerato_munche.json

    # multiple summary files also still works (e.g. one per grader run,
    # aggregate_results.py not used) -- depends on output of evaluator models
    python judge.py --summaries s1.json s2.json --job job_listing.pdf

    # choose a different local Ollama model / host
    python judge.py --summaries summaries.json --job job.pdf \\
        --model deepseek-r1:8b --host http://localhost:11434

"""

import argparse
import json
import sys
import time
from pathlib import Path
_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR / "common"))
from ollama_utils import call_ollama_json
from results_io import save_json
from document_utils import extract_text_from_pdf

# =====================================================================
# Argument parsing (cmd) 
# =====================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
            description="Judge multiple CV summaries against a job listing using a local Ollama model."
    )
    parser.add_argument(
        "--summaries", nargs="+", required=True,
        help="One or more JSON files, each a summary object or list of summary objects "
                "(typically the single combined file aggregate_results.py produces for one candidate/job).",
    )
    parser.add_argument(
        "--job", required=True,
        help="Path to the job listing PDF.",
    )
    
    parser.add_argument(
        "--job-name", default=None,
        help="Job label to record in output (e.g. for the CSV 'job' column). "
                "Defaults to the job PDF's filename stem.",
    )
    parser.add_argument(
        "--model", default="deepseek-r1:8b",
        help="Ollama model name to use as the judge (default: deepseek-r1:8b).",
    )
    parser.add_argument(
        "--host", default="http://localhost:11434/api/chat",
        help="Ollama host URL (default: http://localhost:11434/api/chat).",
    )
    parser.add_argument(
        "--num-predict", type=int, default=4096,
        help="Max tokens the judge model may generate (default: 4096).", 
    )
    #set higher than indep evaluators, reasoning models (deepseek-r1) spend part of budget on <think>
    # before JSON is ever written. If value is too low, reply can truncate before JSON is complete.
    
    parser.add_argument(
        "--max-retries",type=int,default=3,
        help="Max attempts to get a JSON-parseable reply from the judge model before giving up and writing {}. default=3"
    )

    parser.add_argument(
        "--keep-alive", default="0m",
        help="How long Ollama keeps the judge model loaded after this call (default: '5m'). "
    )

    parser.add_argument(
        "--temperature",type=float,default=0.2
    )

    parser.add_argument(
        "--timeout",type=int,default=3600
    )

    parser.add_argument(
        "--keep-scores", action="store_true",
        help="Include prior reviewers' numeric scores in the prompt (off by default, to reduce anchoring bias).",
    )
    parser.add_argument(
        "--candidate", default=None,
        help="Candidate name to use. Defaults to the 'candidate' field from the first summary.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the final evaluation. "
    )

    return parser.parse_args()

# =====================================================================
# Loading input 
# =====================================================================

def load_summaries(paths):
    """
    Load one or more JSON files, each containing either a single summary
    object or a list of summary objects. 
    
    Returns a flat list of dicts. [{...}, {...}]
    """
    summaries = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            sys.exit(f"error: summary file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                sys.exit(f"error: could not parse {path} as JSON: {e}")

        if isinstance(data, list):
            summaries.extend(data) 
        elif isinstance(data, dict):
            summaries.append(data)
        else:
            sys.exit(f"error: unexpected JSON structure in {path}")

    if not summaries:
        sys.exit("error: no summaries loaded")
    return summaries


# =====================================================================
# Prompt construction 
# =====================================================================

def _as_text(value):
    """
    Render a strengths/weaknesses field as prompt text whether the grader
    wrote it as a single string or a list of bullet strings.
    """
    if isinstance(value, (list, tuple)):
        return "\n".join(f"  - {item}" for item in value if str(item).strip())
    return str(value) if value else ""


def format_summaries_for_prompt(summaries, keep_scores=False):
    """
    Turn the list of summary dicts into a readable block of text for the
    prompt. Scores are dropped by default so the judge isn't anchored by
    earlier models' numbers.
    """
    blocks = []
    for i, s in enumerate(summaries, start=1):
        candidate = s.get("candidate", "Unknown candidate")
        strengths = _as_text(s.get("strengths", ""))
        weaknesses = _as_text(s.get("weaknesses", ""))
        summary_text = s.get("summary", "")

        lines = [f"--- Grader summary #{i} (candidate: {candidate}) ---"]
        if keep_scores and "score" in s:
            lines.append(f"Prior score given: {s['score']}/10")
        lines.append(f"Strengths:\n{strengths}")
        lines.append(f"Weaknesses:\n{weaknesses}")
        lines.append(f"Summary: {summary_text}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_prompt(job_text, summaries_text, candidate_name):
    return f"""You are an impartial hiring judge. You will be given a job listing and \
several independent summaries written by other reviewers about the same candidate's CV. \
Each summary reflects one reviewer's read of the CV; the reviewers may disagree or \
overlap. Some summaries may include a numeric score from that reviewer — ignore any such \
score completely and do not let it influence you. Form your own independent judgement \
based only on the strengths, weaknesses, and descriptive summary text provided.

JOB LISTING:
\"\"\"
{job_text}
\"\"\"

CANDIDATE: {candidate_name}

REVIEWER SUMMARIES:
\"\"\"
{summaries_text}
\"\"\"

Your task:
1. Weigh the strengths and weaknesses raised across all reviewer summaries against the \
actual requirements in the job listing.
2. Note any agreement or disagreement between reviewers, and resolve it using your own \
judgement of relevance to the job listing.
3. Produce ONE final, independent evaluation.

Respond with ONLY a single valid JSON object (no markdown fences, no extra commentary), \
with exactly these keys:
{{
  "candidate": "{candidate_name}",
  "final_score": <integer 0-10>,
  "score_rationale": "<1-3 sentences on why this score>",
  "key_strengths": "<synthesized strengths most relevant to the job listing>",
  "key_weaknesses": "<synthesized weaknesses/gaps most relevant to the job listing>",
  "final_evaluation": "<a few sentences giving your overall independent verdict on fit for the role>"
}}
"""

def call_judge(prompt, model, host, num_predict, keep_alive, 
               temperature=0.2,timeout=3600,
               ):
    """
    Call the judge model via shared ollama_utils helper.

    Go through /api/chat, strips <think> reasoning out of reply before hunting for JSON object.
    Connection-level failure (e.g. ollama not running / model not pulled/ timeout):
        exit with error message. 
    Parse failure (call succeeds but reply is not valid JSON):
        hand back to caller.
    """
    parsed, error = call_ollama_json(
        model, prompt,
        num_predict=num_predict,
        keep_alive=keep_alive,
        host=host,
        temperature=temperature,
        timeout = timeout,
    )

    if error and error.startswith("REQUEST ERROR"):
        sys.exit(f"err: could not get a response from ollama at {host} ({error})."
                 f" Check that `ollama serve` is running, and {model} is pulled.")

    return parsed,error

def call_judge_with_retries(prompt, model, host, num_predict, keep_alive,
                             temperature=0.2, timeout=3600, max_retries=3):
    """
    Call judge model, retry if its reply doesn't parse as a JSON object 
    (PARSE ERROR from ollama_utils). This method is only triggered by a parse faillure. 
    Connection-level failures still exit immediately.

    If every attempt fails parsing (after max_retries), return last attempt's ({},"PARSE EROR")
    """
    attempt = 0
    while True:
        attempt += 1
        parsed, error = call_judge(
            prompt, model, host,
            num_predict=num_predict,
            keep_alive=keep_alive,
            temperature=temperature,
            timeout=timeout,
        )
        if not error:
            return parsed, error
        if attempt >= max_retries:
            print(f"\n{error} on attempt {attempt}/{max_retries} -- giving up after max retries.")
            return parsed, error
        print(f"\n{error} on attempt {attempt}/{max_retries} -- retrying...")
    

# =====================================================================
# Output writing
# =====================================================================

def write_json_output(out_path, result):
    """
    Write the judge's raw output as JSON to out_path, exactly as returned
    by the model (or `{}` if ollama_utils couldn't parse a JSON object out
    of its reply at all)
    """
    out_path = Path(out_path)

    save_json(result,out_path.parent,out_path.name)
    print(f"\nSaved final evaluation to {out_path}")
    
    return True

    
# =====================================================================
# Main 
# =====================================================================

def main():
    
    args = _parse_args()

    summaries = load_summaries(args.summaries)
    job_text = extract_text_from_pdf(args.job)
    job_name = args.job_name or Path(args.job).stem

    candidate_name = args.candidate or summaries[0].get("candidate", "Unknown candidate")
    summaries_text = format_summaries_for_prompt(summaries, keep_scores=args.keep_scores)
    prompt = build_prompt(job_text, summaries_text, candidate_name)
    print(f"prompt:\n{prompt}")

    print(f"Sending {len(summaries)} summary(ies) for '{candidate_name}' ({job_name}) to {args.model} at {args.host} ...")
    start_time = time.time()
    parsed, error = call_judge_with_retries(
        prompt, args.model, args.host,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries
    )


    end_time = time.time()
    print(f"Model finished, time: {end_time - start_time:.4f}")
    
    if error:
        print(f"\nWarning: {error} -- writing model output as-is anyway.")

    print(f"\nFinal score for {candidate_name}: {parsed.get('final_score', '?')}/10")
    print(f"Evaluation: {parsed.get('final_evaluation', '')}")
 
    if not args.output:
        return
 
    write_json_output(args.output, parsed)


if __name__ == "__main__":
    main()
