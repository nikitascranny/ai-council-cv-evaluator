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
        --parser-dir cv-parser/pipeline \\
        --output judged/secretary/lerato_munche.json

    # multiple summary files also still works (e.g. one per grader run,
    # if you don't want to use aggregate_results.py)
    python judge.py --summaries s1.json s2.json --job job_listing.pdf --parser-dir cv-parser/pipeline

    # choose a different local Ollama model / host
    python judge.py --summaries summaries.json --job job.pdf \\
        --parser-dir cv-parser/pipeline --model deepseek-r1:8b --host http://localhost:11434

    # append to a shared results CSV instead of writing one JSON per run
    python judge.py --summaries summaries.json --job job.pdf \\
        --parser-dir cv-parser/pipeline --output-format csv --output judged/all_results.csv

Requirements:
    pip install requests pymupdf
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests


# Loading input =====================================================================

def load_summaries(paths):
    """
    Load one or more JSON files, each containing either a single summary
    object or a list of summary objects. Returns a flat list of dicts.
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


def import_parsed_cv(parser_dir):
    """
    Import the cv-parser pipeline's parsed_cv module from parser_dir.
    Used for the job listing PDF.
    """
    parser_dir = Path(parser_dir)
    if not parser_dir.is_dir():
        sys.exit(
            f"error: --parser-dir {parser_dir} not found. Point it at the "
            f"cv-parser 'pipeline' directory (contains parsed_cv.py)."
        )
    if str(parser_dir) not in sys.path:
        sys.path.insert(0, str(parser_dir))
    try:
        import parsed_cv  # noqa: PLC0415 -- deliberately lazy, path-dependent import
    except ImportError as e:
        sys.exit(
            f"error: could not import parsed_cv from {parser_dir}: {e}\n"
            f"Is pymupdf installed? (pip install pymupdf)"
        )
    return parsed_cv


def extract_job_text(pdf_path, parsed_cv):
    """
    Extract job listing text using the cv-parser pipeline's parser.
    """
    path = Path(pdf_path)
    if not path.exists():
        sys.exit(f"error: job listing PDF not found: {path}")

    cv = parsed_cv.parse(path)
    for warning in cv.warnings:
        print(f"  WARN {path.name}: {warning}", file=sys.stderr)

    full_text = cv.full_text()
    if not full_text:
        sys.exit(
            f"error: parsed_cv extracted no text from {path} "
            f"(it may be a scanned image without OCR text -- see cv.warnings above)"
        )
    return full_text


# Prompt construction =================================================================
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


# Calling the local Ollama model ============================================

def call_ollama(prompt, model, host, temperature=0.2):
    """
    Call a local Ollama model's /api/generate endpoint (non-streaming).
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(url, json=payload, timeout=3600)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"error: could not connect to Ollama at {host}. "
            f"Is `ollama serve` running, and is '{model}' pulled "
            f"(`ollama pull {model}`)?"
        )
    except requests.exceptions.RequestException as e:
        sys.exit(f"error: Ollama request failed: {e}")

    data = resp.json()
    return data.get("response", "")


def parse_judge_output(raw_text):
    """
    Attempt to parse the model's response as JSON. 
    Models sometimes wrap JSON in markdown fences or add stray text, so fall back to extracting
    the first {...} block if a direct parse fails.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None

# Output validation ============================================================

REQUIRED_TEXT_FIELDS = ("score_rationale", "key_strengths", "key_weaknesses", "final_evaluation")


def validate_and_coerce(parsed, candidate_name):
    """
    Check the judge's parsed output against the schema judge.py asked for.

    Returns (clean_dict, errors, blocking). clean_dict is always a best-effort
    dict (never None) so a CSV row can be written even when things are messy.
    blocking is True when the result is unsafe to write as "clean" JSON:
    the model's output wasn't a JSON object at all, or final_score could not
    be read as a number.
    """
    errors = []

    if not isinstance(parsed, dict):
        return (
            {"candidate": candidate_name, "final_score": None, "score_rationale": "",
             "key_strengths": "", "key_weaknesses": "", "final_evaluation": ""},
            ["judge output did not parse as a JSON object"],
            True,
        )

    clean = {"candidate": str(parsed.get("candidate") or candidate_name)}

    raw_score = parsed.get("final_score")
    blocking = False
    try:
        score = int(round(float(raw_score)))
    except (TypeError, ValueError):
        errors.append(f"final_score missing or not numeric: {raw_score!r}")
        score = None
        blocking = True
    else:
        if not (0 <= score <= 10):
            errors.append(f"final_score {score} out of expected range 0-10, clamped")
            score = max(0, min(10, score))
    clean["final_score"] = score

    for key in REQUIRED_TEXT_FIELDS:
        val = parsed.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{key} missing or empty")
            val = val.strip() if isinstance(val, str) else ""
        clean[key] = val.strip() if isinstance(val, str) else val

    return clean, errors, blocking


# Output writing

CSV_FIELDS = [
    "candidate", "job", "final_score", "score_rationale",
    "key_strengths", "key_weaknesses", "final_evaluation", "errors",
]


def write_json_output(out_path, clean, errors, blocking, raw_response):
    """
    Write validated output as JSON. If validation was blocking, refuse to
    write the clean path (so nothing downstream can mistake a broken result
    for a good one) and write the raw response to a sibling *.invalid.json
    file instead, returning a non-zero-worthy failure flag.
    """
    out_path = Path(out_path)
    if blocking:
        invalid_path = out_path.with_suffix(out_path.suffix + ".invalid.json")
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_path.write_text(
            json.dumps({"raw_response": raw_response, "errors": errors}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nERROR: judge output failed validation, nothing written to {out_path}", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(f"Raw response saved to {invalid_path}", file=sys.stderr)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved final evaluation to {out_path}")
    if errors:
        print("Note: written with non-blocking validation warnings:")
        for e in errors:
            print(f"  - {e}")
    return True


def write_csv_output(out_path, clean, errors, job_name):
    """
    Append one row to a shared CSV file, creating it with a header if it
    doesn't exist yet. CSV is the lower-risk fallback: a validation problem
    is recorded in the row's "errors" column rather than blocking the write.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()

    row = {
        "candidate": clean.get("candidate", ""),
        "job": job_name,
        "final_score": clean.get("final_score", ""),
        "score_rationale": clean.get("score_rationale", ""),
        "key_strengths": clean.get("key_strengths", ""),
        "key_weaknesses": clean.get("key_weaknesses", ""),
        "final_evaluation": clean.get("final_evaluation", ""),
        "errors": "; ".join(errors),
    }

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nAppended row for {row['candidate']} to {out_path}")
    if errors:
        print("Note: appended with validation warnings (see 'errors' column):")
        for e in errors:
            print(f"  - {e}")
    return True


# Main =====================================================================

def main():
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
        "--parser-dir", required=True,
        help="Path to the cv-parser 'pipeline' directory (contains parsed_cv.py). "
             "The job listing PDF is always read through this parser.",
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
        "--host", default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434).",
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
        help="Path to save the final evaluation. A .json path writes one validated JSON object; "
             "a .csv path appends one row to a shared CSV file. Required unless --output-format is set explicitly.",
    )
    parser.add_argument(
        "--output-format", choices=["json", "csv"], default=None,
        help="Force the output format instead of inferring it from --output's extension.",
    )
    args = parser.parse_args()

    output_format = args.output_format
    if output_format is None and args.output:
        output_format = "csv" if Path(args.output).suffix.lower() == ".csv" else "json"

    summaries = load_summaries(args.summaries)
    parsed_cv = import_parsed_cv(args.parser_dir)
    job_text = extract_job_text(args.job, parsed_cv)
    job_name = args.job_name or Path(args.job).stem

    candidate_name = args.candidate or summaries[0].get("candidate", "Unknown candidate")
    summaries_text = format_summaries_for_prompt(summaries, keep_scores=args.keep_scores)
    prompt = build_prompt(job_text, summaries_text, candidate_name)

    print(f"Sending {len(summaries)} summary(ies) for '{candidate_name}' ({job_name}) to {args.model} at {args.host} ...")
    start_time = time.time()
    raw_response = call_ollama(prompt, args.model, args.host)
    end_time = time.time()
    print(f"Model finished, time: {end_time - start_time:.4f}")

    parsed = parse_judge_output(raw_response)
    if parsed is None:
        print("\nWarning: could not parse model output as JSON. Raw response below:\n")
        print(raw_response)

    clean, errors, blocking = validate_and_coerce(parsed, candidate_name)
    print(f"\nFinal score for {candidate_name}: {clean.get('final_score', '?')}/10")
    print(f"Evaluation: {clean.get('final_evaluation', '')}")

    if not args.output:
        return

    if output_format == "csv":
        ok = write_csv_output(args.output, clean, errors, job_name)
    else:
        ok = write_json_output(args.output, clean, errors, blocking, raw_response)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
