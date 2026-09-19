"""
judge.py 

The judge of the council.
Several evaluator models (the 'council', Evaluate_CV.py) each review a CV independently; 
this script has a separate model combine their summaries into one final verdict.

Takes one or more reviewer-summary JSON files for a candidate and a job listing PDF, 
and sends them to an Ollama model acting as a "judge" (locally DEFAULT_MODEL, deepseek-r1:8b;
with --cloud, CLOUD_MODEL=nemotron-3-super on Ollama Cloud) to produce a single final evaluation and 
score out of 10.

In batch evaluation (CLI, with run_evaluation.py): aggregate_results.py turns raw per-model/per-job files 
into one combined file per candidate-per-job (prepares data for this judge).

The job listing PDF is read as plain text with document_utils.extract_text_from_pdf (pdfplumber).

Per-summary "score" fields are stripped out before the prompt is built by default. 
Pass --keep-scores to disable this.

The judge's reply must parse as a JSON object: if it doesn't, the call is
retried, up to --max-retries attempts in total (default 3). Parse
failures and empty replies are retried; a request failure (Ollama unreachable, 
HTTP error such as a bad key or rate limit) exits immediately. 
The reply is written unmodified to --output (or only printed when --output is not given).

Checking the score itself is left to the caller: in batch runs,
run_evaluation.py accepts a verdict only if final_score is a number in
[1, 10], and logs any other verdict as a failure to be re-judged.

If the summaries carry an essential-requirements entry (aggregate_results.py
adds one per candidate when given that check's results), it is taken out of
the reviewer summaries and given its own section of the prompt. Pass
--no-essential to leave it out, which reproduces the council-only prompt.

CLI Usage:
    python judge.py --summaries combined/secretary/lerato_munche.json \\
        --job job_listings/job_listing_secretary.pdf \\
        --output judged/secretary/lerato_munche.json

    # several summary files about the SAME candidate, e.g. the evaluators' raw outputs
    # from an Evaluate_CV.py run on a single CV (raw outputs of a multi-CV run hold
    # every candidate, so group them first with aggregate_results.py)
    python judge.py --summaries s1.json s2.json --job job_listing.pdf

    # choose a different local Ollama model / host
    python judge.py --summaries summaries.json --job job.pdf \\
        --model qwen2.5:7b --host http://localhost:11434/api/chat

    # judge on Ollama Cloud with CLOUD_MODEL (needs OLLAMA_API_KEY)
    python judge.py --summaries summaries.json --job job.pdf --cloud

--cloud is for the command line only: judge_candidate (the web app's entry
point) always judges locally.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR / "common"))
from ollama_utils import call_ollama_json, load_env_file
from results_io import save_json
from document_utils import extract_text_from_pdf

# The judge's call settings: this script's command-line defaults, and what the
# web app judges with too (ui/pipeline/deps.py, through judge_candidate) - one
# set of values, so a candidate is judged the same way from either
DEFAULT_MODEL = "deepseek-r1:8b"
DEFAULT_HOST = "http://localhost:11434/api/chat"
# Higher than the independent evaluators' budget: a reasoning model
# (deepseek-r1) spends part of it on <think> before any JSON is written, and
# if the budget is too low, the reply is truncated before the JSON is complete
DEFAULT_NUM_PREDICT = 4096
DEFAULT_MAX_RETRIES = 3
DEFAULT_KEEP_ALIVE = "0m"
DEFAULT_TIMEOUT = 3600

# The judge with --cloud. A model family (NVIDIA) which the 
# cloud council (Evaluate_CV.CLOUD_EVALUATOR_MODELS: Google, OpenAI) does not
# use, so the judge is not grading its own model's summaries
CLOUD_MODEL = "nemotron-3-super"
CLOUD_HOST = "https://ollama.com/api/chat"
CLOUD_API_KEY_ENV = "OLLAMA_API_KEY"
# Its reasoning counts against num_predict too, and on the cloud a larger budget 
# costs no local GPU time, so it gets more room than the local judge
CLOUD_NUM_PREDICT = 8192
# 300s (5 minutes) rather than the local 3600s (same as in Evaluate_CV.py)
CLOUD_TIMEOUT = 300

# =====================================================================
# Argument parsing (cmd) 
# =====================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
            description="Judge multiple CV summaries against a job listing using an Ollama model (local or cloud)."
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
        help="Job label shown in the progress output.",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Ollama model name to use as the judge (default: {DEFAULT_MODEL}, "
             f"or {CLOUD_MODEL} with --cloud).",
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help=f"Judge on Ollama Cloud instead of the local daemon. Needs {CLOUD_API_KEY_ENV} "
             f"set; --host, --num-predict, --keep-alive and --timeout give way to the "
             f"cloud's own settings.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"Ollama host URL (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--num-predict", type=int, default=DEFAULT_NUM_PREDICT,
        help=f"Max tokens the judge model may generate (default: {DEFAULT_NUM_PREDICT}).", 
    )
    
    parser.add_argument(
        "--max-retries",type=int,default=DEFAULT_MAX_RETRIES,
        help=f"Max attempts to get a usable (non-empty, JSON) reply from the judge model before giving up and writing {{}} (default:{DEFAULT_MAX_RETRIES})"
    )

    parser.add_argument(
        "--keep-alive", default=DEFAULT_KEEP_ALIVE,
        help=f"How long Ollama keeps the judge model loaded after this call (default: '{DEFAULT_KEEP_ALIVE}'). "
    )

    parser.add_argument(
        "--timeout",type=int,default=DEFAULT_TIMEOUT,
        help=f"Seconds to wait for the model's reply (default: {DEFAULT_TIMEOUT})."
    )

    parser.add_argument(
        "--keep-scores", action="store_true",
        help="Include prior reviewers' numeric scores in the prompt (off by default, to reduce anchoring bias).",
    )
    parser.add_argument(
        "--no-essential", action="store_true",
        help="Leave the essential-requirements check out of the prompt even when the summaries "
                "include one (reproduces the council-only condition).",
    )
    parser.add_argument(
        "--candidate", default=None,
        help="Candidate name to use. Defaults to the 'candidate' field from the first summary.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the final evaluation (JSON). "
    )

    args = parser.parse_args()
    if args.model is None:
        args.model = CLOUD_MODEL if args.cloud else DEFAULT_MODEL
    return args

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


# Marks the essential-requirements check's entry among the summaries (see
# aggregate_results.py). It is not a reviewer's summary - it has no score,
# strengths or weaknesses - so it is split off before the reviewer summaries
# are formatted, rather than being numbered in among them.
ESSENTIAL_TYPE = "essential_requirements"


def split_essential(entries):
    """
    Separate the essential-requirements check from the reviewer summaries.

    Returns (summaries, essential), where essential is the first entry whose
    "type" is ESSENTIAL_TYPE, or None. All other entries are reviewer summaries.
    """
    summaries = [e for e in entries if e.get("type") != ESSENTIAL_TYPE]
    checks = [e for e in entries if e.get("type") == ESSENTIAL_TYPE]
    if len(checks) > 1:
        print(f"warning: {len(checks)} essential-requirements entries found; using the first.")
    return summaries, (checks[0] if checks else None)


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


def format_essential_for_prompt(essential):
    """
    The essential-requirements check as prompt text, or None when there is
    nothing usable to show the judge: no entry, a check that failed (its
    "error" field is set), or an empty summary. 

    A failed check is left out rather than passed through, so the judge 
    never reads an error message as an assessment of the candidate.
    """
    if not essential or essential.get("error"):
        return None
    summary = str(essential.get("summary") or "").strip()
    return summary or None


def build_prompt(job_text, summaries_text, candidate_name, essential_text=None):
    """
    essential_text: the essential-requirements check's summary (see format_essential_for_prompt). 
    The check gets its own section and task step instead of posing as another reviewer's summary.

    The section says the check read three CV sections - the scoping used by
    Evaluate_CV.py and by the web app's essential stage alike.
    """
    essential_basis = ""
    essential_block = ""
    essential_step = ""
    final_step = 3
    if essential_text:
        essential_basis = ", and on the separate essential requirements check"
        essential_block = f"""ESSENTIAL REQUIREMENTS CHECK:
A separate check read only the candidate's experience, education and certifications \
sections, and summarised how well they meet the job listing's essential requirements. \
It gives no score, and it saw less of the CV than the reviewers did, so a requirement \
it reports as unclear may still be evidenced in the reviewer summaries.
\"\"\"
{essential_text}
\"\"\"

"""
        essential_step = """3. Use the essential requirements check to decide whether the job \
listing's essential requirements are met. Where it and the reviewers disagree, resolve it \
using the evidence each one describes.
"""
        final_step = 4

    return f"""You are an impartial hiring judge. You will be given a job listing and \
several independent summaries written by other reviewers about the same candidate's CV. \
Each summary reflects one reviewer's read of the CV; the reviewers may disagree or \
overlap. Some summaries may include a numeric score from that reviewer — ignore any such \
score completely and do not let it influence you. Form your own independent judgement \
based only on the strengths, weaknesses, and descriptive summary text provided{essential_basis}.

JOB LISTING:
\"\"\"
{job_text}
\"\"\"

REVIEWER SUMMARIES:
\"\"\"
{summaries_text}
\"\"\"

{essential_block}Your task:
1. Weigh the strengths and weaknesses raised across all reviewer summaries against the \
actual requirements in the job listing.
2. Note any agreement or disagreement between reviewers, and resolve it using your own \
judgement of relevance to the job listing.
{essential_step}{final_step}. Produce ONE final, independent evaluation.

Respond with ONLY a single valid JSON object (no markdown fences, no extra commentary), \
with exactly these keys:
{{
  "candidate": "{candidate_name}",
  "final_score": <integer 1-10>,
  "score_rationale": "<1-3 sentences on why this score>",
  "key_strengths": "<synthesized strengths most relevant to the job listing>",
  "key_weaknesses": "<synthesized weaknesses/gaps most relevant to the job listing>",
  "final_evaluation": "<a few sentences giving your overall independent verdict on fit for the role>"
}}
"""


def build_judge_prompt(job_text, summaries, candidate_name, essential=None, keep_scores=False):
    """
    The whole prompt for one candidate, from the reviewer summaries and the
    essential-requirements check's entry (None for the council-only prompt).

    called by main() and judge_candidate() (command line and web app do not drift apart). 
    A failed or empty check is left out (format_essential_for_prompt).
    """
    summaries_text = format_summaries_for_prompt(summaries, keep_scores=keep_scores)
    return build_prompt(job_text, summaries_text, candidate_name,
                        format_essential_for_prompt(essential))


def cloud_call_options(api_key, stats):
    """
    The call_judge arguments that send the judge to Ollama Cloud: 
    the host, budget and timeout in place of the local ones, plus the key.
    """
    return {"host": CLOUD_HOST, 
            "num_predict": CLOUD_NUM_PREDICT, 
            "keep_alive": None,
            "timeout": CLOUD_TIMEOUT,
            "headers": {"Authorization": "Bearer " + api_key}, 
            "stats": stats}


def call_judge(prompt, model, host, num_predict, keep_alive,timeout=3600, headers=None, stats=None):
    """
    Call the judge model via shared ollama_utils helper. 
    
    `headers` and `stats` are for a cloud call (see cloud_call_options) and are left out of a local one.

    Go through /api/chat (on default, set by --host), strips <think> reasoning out of reply before hunting for JSON object.
    Connection-level failure (e.g. ollama not running/model not pulled/timeout/HTTP error):
        exit with error message. 
    Parse failure (call succeeds but reply is not valid JSON) or empty reply:
        hand back to caller.
    """
    cloud = {"headers": headers, "stats": stats} if headers else {}
    parsed, error = call_ollama_json(
        model, prompt,
        num_predict=num_predict,
        keep_alive=keep_alive,
        host=host,
        timeout = timeout,
        **cloud,
    )

    if error and error.startswith("REQUEST ERROR"):
        if headers: # a cloud call
            sys.exit(f"err: could not get a response from Ollama Cloud at {host} ({error})."
                     f" Check {CLOUD_API_KEY_ENV}, the plan's usage, and that {model} is offered.")
        sys.exit(f"err: could not get a response from ollama at {host} ({error})."
                 f" Check that `ollama serve` is running, and {model} is pulled.")

    return parsed,error

def call_judge_with_retries(prompt, model, host, num_predict, keep_alive, timeout=3600, max_retries=3,
                            **cloud):
    """
    Call judge model, retry if its reply doesn't parse as a JSON object 
    (PARSE ERROR from ollama_utils) or the reply is empty. 
    Connection-level failures still exit immediately.

    If every attempt fails (after max_retries), return last attempt's ({},"<error>")

    cloud: headers and stats for a cloud call, passed to each attempt.
    """
    attempt = 0
    while True:
        attempt += 1
        parsed, error = call_judge(
            prompt, model, host,
            num_predict=num_predict,
            keep_alive=keep_alive,
            timeout=timeout,
            **cloud,
        )
        if not error:
            return parsed, error
        if attempt >= max_retries:
            print(f"\n{error} on attempt {attempt}/{max_retries} -- giving up after max retries.")
            return parsed, error
        print(f"\n{error} on attempt {attempt}/{max_retries} -- retrying...")
    

def judge_candidate(job_text, summaries, candidate_name, essential=None, *,
                    model=DEFAULT_MODEL, host=DEFAULT_HOST,
                    num_predict=DEFAULT_NUM_PREDICT, keep_alive=DEFAULT_KEEP_ALIVE,
                    timeout=DEFAULT_TIMEOUT,
                    max_retries=DEFAULT_MAX_RETRIES, keep_scores=False):
    """
    Judge one candidate with the prompt and call settings main() uses by
    default - for callers other than command line (web app).

    summaries: reviewer summaries only
    essential: the essential-requirements check's result, None for the council-only prompt

    Returns (parsed, error) (same as call_judge_with_retries). 
    Never exits the process: call_judge's sys.exit when Ollama cannot be
    reached comes back as an error (SystemExit on a web app's worker thread ends the run silently rather than reporting)
    """
    prompt = build_judge_prompt(job_text, summaries, candidate_name, essential, keep_scores)
    try:
        return call_judge_with_retries(
            prompt, model, host,
            num_predict=num_predict,
            keep_alive=keep_alive,
            timeout=timeout,
            max_retries=max_retries,
        )
    except SystemExit as exc:
        return {}, str(exc)


# =====================================================================
# Output writing
# =====================================================================

def write_json_output(out_path, result):
    """
    Write the judge's raw output as JSON to out_path, exactly as returned
    by the model (or `{}` if every attempt failed)
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

    summaries, essential = split_essential(load_summaries(args.summaries))
    if not summaries:
        sys.exit("error: no reviewer summaries loaded (only an essential-requirements check)")
    job_text = extract_text_from_pdf(args.job)
    job_name = args.job_name or Path(args.job).stem

    candidate_name = args.candidate or summaries[0].get("candidate", "Unknown candidate")
    if args.no_essential:
        essential = None
    essential_text = format_essential_for_prompt(essential)
    if args.no_essential:
        print("Essential-requirements check: left out (--no-essential)")
    elif essential is None:
        print("Essential-requirements check: none in the summaries")
    elif essential_text is None:
        reason = essential.get("error") or "empty summary"
        print(f"Essential-requirements check: not usable ({reason}) -- left out of the prompt")
    else:
        print("Essential-requirements check: included")
    prompt = build_judge_prompt(job_text, summaries, candidate_name, essential,
                                keep_scores=args.keep_scores)

    settings = {"host": args.host, "num_predict": args.num_predict,
                "keep_alive": args.keep_alive, "timeout": args.timeout}
    stats = {}
    if args.cloud:
        api_key = os.environ.get(CLOUD_API_KEY_ENV, "").strip()
        if not api_key:
            sys.exit(f"error: --cloud needs an Ollama API key: set {CLOUD_API_KEY_ENV} in your terminal or add {CLOUD_API_KEY_ENV}=<key> "
                     f"to .env in the repo root (create one at https://ollama.com/settings/keys)")
        settings = cloud_call_options(api_key, stats)
    host = settings.pop("host")

    print(f"Sending {len(summaries)} summary(ies) for '{candidate_name}' ({job_name}) to {args.model} at {host} ...")
    start_time = time.time()
    parsed, error = call_judge_with_retries(
        prompt, args.model, host,
        max_retries=args.max_retries,
        **settings,
    )


    end_time = time.time()
    print(f"Model finished, time: {end_time - start_time:.4f}")
    if args.cloud and stats:
        # The last attempt's tokens only - earlier attempts are not counted.
        print(f"Cloud tokens: {stats.get('prompt_eval_count')} in, {stats.get('eval_count')} out")
    
    if error:
        print(f"\nWarning: {error} -- result is {{}}.")

    print(f"\nFinal score for {candidate_name}: {parsed.get('final_score', '?')}/10")
    print(f"Evaluation: {parsed.get('final_evaluation', '')}")
 
    if not args.output:
        return
 
    write_json_output(args.output, parsed)


if __name__ == "__main__":
    load_env_file()
    main()
