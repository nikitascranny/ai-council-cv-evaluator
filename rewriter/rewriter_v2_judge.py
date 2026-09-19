"""
rewriter_v2_judge.py

For one candidate's rewrite submission (one candidate x one job listing,
i.e. one folder under data/rewrite_results/<job_title>/<candidate>/), reads
both agents' rewritten CVs plus their metadata.csv, reads the respective job
listing PDF (via document_utils.extract_text_from_pdf - see common/
document_utils.py), and asks a local Ollama model acting as a "judge" for
three things:

  * which rewrite to submit for this job, and why
  * a suggested career roadmap
  * a few alternative career paths outside this job listing, based on the
    candidate's demonstrated skills and strengths

rewriter_v2.py deliberately stops at producing the two rewrites side by side.

Input layout (written by whatever drives rewriter_v2.rewrite_cv() per
candidate per job - not this file):

    data/rewrite_results/<job_title>/<candidate>/
        metadata.csv          - one row per agent (RewriteResult fields,
                                 minus the CV text itself)
        <agent_model>.txt     - that agent's rewritten CV, filename derived
                                 from the model name via results_io.safe_path_part()
                                 (the same rule rewriter_v2.save_results writes it with)

Output, one folder per submission, mirroring the input layout:

    data/rewrite_feedback/<job_title>/<candidate>/
        feedback.json   - the judge's raw response (or {} + a note on
                           failure - kept even then, so a bad run is
                           diagnosable rather than silently missing)

Usage:
    # every submission under data/rewrite_results/ that doesn't already have feedback
    python rewrite_judge.py

    # just one
    python rewrite_judge.py --job-title "Computer Repair Technician" --candidate "ALEX PARK"

    # re-judge everything, even submissions with existing feedback
    python rewrite_judge.py --force

    # a different judge model
    python rewrite_judge.py --model deepseek-r1:8b --host http://localhost:11434/api/chat
"""

import argparse
import ast
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR / "common"))

from ollama_utils import call_ollama_json  # noqa: E402
from results_io import save_json, safe_path_part  # noqa: E402
from document_utils import extract_text_from_pdf  # noqa: E402

# Directory constants
REWRITE_RESULTS_DIR = _ROOT_DIR / "data" / "rewrite_results"
JOB_LISTINGS_DIR = _ROOT_DIR / "data" / "job-listings"
FEEDBACK_DIR = _ROOT_DIR / "data" / "rewrite_feedback"

# Judge settings
DEFAULT_MODEL = "deepseek-r1:8b" # kept distinct from the rewriting agents (phi4, qwen2.5:14b)
DEFAULT_HOST = "http://localhost:11434/api/chat"
DEFAULT_NUM_PREDICT = 4096
DEFAULT_NUM_CTX = 8192
REQUIRED_KEYS = {"candidate", "job_title", "recommended_rewrite", "recommendation_reason",
                 "career_roadmap", "alternative_careers"} #recommended output of the rewriter judge


# =====================================================================
# Loading one submission
# =====================================================================

@dataclass
class RewriteSubmission:
    """
    One candidate's two (or more) agent rewrites for one job listing,
    plus the metadata already computed when they were produced.
    """
    job_title: str
    candidate: str
    job_pdf: Path
    rewrites: list = field(default_factory=list)  # list of dicts: metadata row + "text"


def _parse_list_field(raw: str) -> list:
    """
    raw: a Python repr string e.g. "['alex park west edge']" (list fields in metadata.csv)

    Returns list converted string.
    If unparseable, return empty rather than aborting the whole row.
    """
    try:
        value = ast.literal_eval(raw) if raw else []
        return value if isinstance(value, list) else [] 
    except (ValueError, SyntaxError):
        return []


def _parse_bool_field(raw: str) -> bool:
    return str(raw).strip().lower() == "true"


def load_metadata(csv_path: Path) -> list:
    """
    csv_path: path to metadata.csv

    Returns a list of dicts, one per agent.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["used_fallback"] = _parse_bool_field(row.get("used_fallback", ""))
        row["rejected"] = _parse_bool_field(row.get("rejected", ""))
        row["retried"] = _parse_bool_field(row.get("retried", ""))
        row["reasons"] = _parse_list_field(row.get("reasons", ""))
        row["advisory"] = _parse_list_field(row.get("advisory", ""))
    return rows


def find_job_listing_pdf(job_title: str, job_listings_dir: Path = JOB_LISTINGS_DIR):
    """
    job_title: the Title Case folder name (e.g. "Computer Repair Technician"); 
    job_listings_dir: directory in which all job listing PDFs are stored
    
    job listing PDFs are job_listing_<slug>.pdf with an all-lowercase, underscore-joined slug.
    """
    slug = job_title.strip().lower().replace(" ", "_")
    candidate = job_listings_dir / f"job_listing_{slug}.pdf"
    return candidate if candidate.is_file() else None


def load_submission(job_title: str, candidate: str, submission_dir: Path):
    """
    Load one data/rewrite_results/<job_title>/<candidate>/ folder.

    Returns (RewriteSubmission, None) or (None, "<reason>") in the case of a bad submission 
    (no metadata.csv, no matching job listing, a row's .txt file absent)
    """
    csv_path = submission_dir / "metadata.csv"

    if not csv_path.is_file():
        return None, f"no metadata.csv in {submission_dir}"

    job_pdf = find_job_listing_pdf(job_title)
    if job_pdf is None:
        return None, f"no job listing PDF found for job title {job_title!r} under {JOB_LISTINGS_DIR}"

    rows = load_metadata(csv_path)
    if not rows:
        return None, f"metadata.csv is empty: {csv_path}"

    rewrites = []
    for row in rows:
        agent_model = row.get("agent_model", "")
        text_path = submission_dir / f"{safe_path_part(agent_model)}.txt"
        if not text_path.is_file(): 
            return None, f"metadata.csv lists agent {agent_model!r} but {text_path.name} is missing"
        row["text"] = text_path.read_text(encoding="utf-8")
        rewrites.append(row)

    if len(rewrites) < 2:
        return None, f"only {len(rewrites)} rewrite(s) found for {candidate}/{job_title}, need at least 2 to compare"

    return RewriteSubmission(job_title=job_title, 
                             candidate=candidate, 
                             job_pdf=job_pdf, 
                             rewrites=rewrites), None


def discover_submissions(results_dir: Path = REWRITE_RESULTS_DIR) -> list[tuple]:
    """
    Returns a sorted list of (job_title, candidate, submission_dir) tuples for every 
    data/rewrite_results/<job_title>/<candidate>/ folder
    """
    if not results_dir.is_dir():
        return []
    submissions = []
    for job_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()): # sort for stable, reproducible run order
        for candidate_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
            submissions.append((job_dir.name, candidate_dir.name, candidate_dir))
    return submissions


# =====================================================================
# Prompt construction
# =====================================================================

def _format_rewrite_block(label: str, row: dict) -> str:
    """
    label: used in building the prompt ("REWRITE <label>", e.g. "REWRITE A")
    row: RewriteSubmission.rewrites (one row of metadata.csv + rewrite text)
    """
    flags = (f"rejected={row['rejected']}  "
             f"used_fallback={row['used_fallback']}  "
             f"retried={row['retried']}")
    lines = [
        f"REWRITE {label} - model: {row.get('agent_model', 'unknown')}",
        f"  fact-check: {flags}",
    ]
    if row["reasons"]:
        lines.append(f"  flagged issues: {row['reasons']}")
    if row["advisory"]:
        lines.append(f"  advisory notes: {row['advisory']}") #(usually harmless re-casing, not fabrication)
    try:
        lines.append(f"  length vs original: {float(row['length_ratio']):.2f}x"
                     f"   wording similarity to original: {float(row['similarity']):.2f}")
    except (KeyError, TypeError, ValueError):
        pass
    lines.append('"""')
    lines.append(row["text"])
    lines.append('"""')
    return "\n".join(lines)


def build_prompt(job_text: str, submission: RewriteSubmission) -> str:
    
    labels = ["A", "B", "C", "D"][: len(submission.rewrites)] #add more labels for more rewrites
    blocks = "\n\n".join(
        _format_rewrite_block(label, row) for label, row in zip(labels, submission.rewrites)
    )
    agent_names = [row.get("agent_model", "unknown") for row in submission.rewrites]

    return f"""You are helping a candidate decide which of {len(submission.rewrites)} AI-tailored \
CV rewrites to submit for one specific job application, and what to do next in their career.

JOB LISTING ({submission.job_title}):
\"\"\"
{job_text}
\"\"\"

{len(submission.rewrites)} independent models each rewrote a candidate's CV to target the \
job listing above. Automated fact-checking already ran on each rewrite before you see it \
(whether it invented or dropped a fact, and whether it had to fall back to the candidate's \
original CV text) - trust those results rather than re-deriving them from the text:

{blocks}

Your task:
1. Compare the rewrite(s) against what the job listing actually asks for, factoring in the \
fact-check results above - a rewrite that is still flagged, or fell back to the original text, \
is a real quality signal, not a detail to ignore.
2. Pick exactly ONE rewrite to recommend, naming it by its model exactly as given \
({', '.join(repr(n) for n in agent_names)}).
3. Suggest a concrete career roadmap for this candidate: realistic next steps, skills or \
experience worth building, and the kind of roles to target next, given their background and \
this job listing.
4. Separately, set this job listing aside and suggest 2-3 alternative career paths that fit \
the skills and strengths evident in the candidate's CV, but are meaningfully different from \
"{submission.job_title}".

Respond with ONLY a single valid JSON object (no markdown fences, no extra commentary), with \
exactly these keys:
{{
  "candidate": "{submission.candidate}",
  "job_title": "{submission.job_title}",
  "recommended_rewrite": "<exactly one of {agent_names}>",
  "recommendation_reason": "<3-6 sentences comparing the rewrites against the job listing and \
justifying the pick, referencing the fact-check results where relevant>",
  "career_roadmap": [
    {{"step": "<short name for this step, e.g. 'Apply for the role with Rewrite B'>",
     "action": "<what the candidate should concretely do for this step>"}},
    {{"step": "<next step>", "action": "<what to do>"}}
  ],
  "alternative_careers": ["<a different role or career path, distinct from '{submission.job_title}'>", "<another one>"]
}}

career_roadmap must be an ordered JSON array of step objects like the ones above - not a single \
string - starting with applying to this job and continuing with realistic next steps: skills or \
experience worth building, and the kind of roles to target after this one.

alternative_careers must be a JSON array of 2-3 short strings, each naming one specific role or \
career path (not "{submission.job_title}" itself) that suits the candidate's demonstrated \
skills and strengths.
"""


# =====================================================================
# Calling the judge
# =====================================================================

def _valid(parsed: dict, agent_names: list) -> bool:
    if not REQUIRED_KEYS <= parsed.keys():
        return False
    if parsed.get("recommended_rewrite") not in agent_names:
        return False
    roadmap = parsed.get("career_roadmap")
    if not isinstance(roadmap, list) or not roadmap:
        return False
    if not all(isinstance(step, dict) and step.get("step") and step.get("action")
               for step in roadmap):
        return False
    alt_careers = parsed.get("alternative_careers")
    if not isinstance(alt_careers, list) or not alt_careers:
        return False
    return all(isinstance(career, str) and career.strip() for career in alt_careers)


def call_judge_with_retries(prompt: str, agent_names: list, model: str, host: str,
                             num_predict: int, timeout: int,
                             max_retries: int, num_ctx: int = DEFAULT_NUM_CTX):
    """
    Call the judge model, retrying on a parse failure or a reply that
    doesn't name one of the actual agents (mirrors evaluator/judge.py's
    call_judge_with_retries). Connection-level failures exit immediately 
    (if Ollama is unreachable, every remaining submission would fail the
    same way, so there's nothing to gain from grinding through the batch).
    """
    attempt = 0
    while True:
        attempt += 1
        parsed, error = call_ollama_json(
            model, prompt, num_predict=num_predict, keep_alive="5m", #TODO
            host=host, timeout=timeout, num_ctx=num_ctx,
        )
        if error and error.startswith("REQUEST ERROR"):
            sys.exit(f"err: could not get a response from ollama at {host} ({error}). "
                     f"Check that `ollama serve` is running, and {model} is pulled.")

        if not error and not _valid(parsed, agent_names):
            error = f"reply missing required keys or named a rewrite outside {agent_names}"

        if not error:
            return parsed, None
        if attempt >= max_retries:
            print(f"  {error} on attempt {attempt}/{max_retries} -- giving up after max retries.")
            return parsed, error
        print(f"  {error} on attempt {attempt}/{max_retries} -- retrying...")


# =====================================================================
# Output writing
# =====================================================================

def write_feedback(parsed: dict, submission: RewriteSubmission, outdir: Path):
    submission_out = outdir / submission.job_title / submission.candidate
    save_json(parsed, submission_out, "feedback.json")
    return submission_out


# =====================================================================
# Main
# =====================================================================

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-title", help="Only judge this job title's folder name under data/rewrite_results/ "
                                         "(e.g. \"Computer Repair Technician\"). Requires --candidate.")
    ap.add_argument("--candidate", help="Only judge this candidate (folder name, e.g. \"ALEX PARK\"). "
                                         "Requires --job-title.")
    ap.add_argument("--results-dir", type=Path, default=REWRITE_RESULTS_DIR,
                     help=f"Where the rewrite submissions live (default: {REWRITE_RESULTS_DIR}).")
    ap.add_argument("--job-listings-dir", type=Path, default=JOB_LISTINGS_DIR,
                     help=f"Where job_listing_*.pdf files live (default: {JOB_LISTINGS_DIR}).")
    ap.add_argument("--outdir", type=Path, default=FEEDBACK_DIR,
                     help=f"Where to write feedback.json (default: {FEEDBACK_DIR}).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST}).")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                     help="Context window size in tokens (default: %(default)s). "
                          "Judge prompts (job listing + every rewrite) can exceed "
                          "Ollama's 4096-token default, causing a 400 error.")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--force", action="store_true",
                     help="Re-judge submissions that already have feedback.json (default: skip them).")
    args = ap.parse_args()

    if bool(args.job_title) != bool(args.candidate):
        ap.error("--job-title and --candidate must be given together")
    return args


def main():
    args = _parse_args()

    if args.job_title:
        submission_dir = args.results_dir / args.job_title / args.candidate
        if not submission_dir.is_dir():
            sys.exit(f"error: {submission_dir} does not exist")
        targets = [(args.job_title, args.candidate, submission_dir)]
    else:
        targets = discover_submissions(args.results_dir)
        if not targets:
            sys.exit(f"error: no submissions found under {args.results_dir}")

    n_ok = n_failed = n_skipped = n_missing = 0

    for job_title, candidate, submission_dir in targets:
        out_path = args.outdir / job_title / candidate / "feedback.json"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        print(f"-> {job_title} / {candidate}")

        submission, reason = load_submission(job_title, candidate, submission_dir)
        if submission is None:
            print(f"  WARN skipping: {reason}")
            n_missing += 1
            continue

        job_text = extract_text_from_pdf(submission.job_pdf)
        agent_names = [row.get("agent_model", "unknown") for row in submission.rewrites]
        prompt = build_prompt(job_text, submission)

        start = time.time()
        parsed, error = call_judge_with_retries(
            prompt, agent_names, args.model, args.host,
            num_predict=args.num_predict,
            timeout=args.timeout, max_retries=args.max_retries, num_ctx=args.num_ctx,
        )
        print(f"  done in {time.time() - start:.1f}s")

        if error:
            print(f"  WARN: {error} -- writing model output as-is anyway.")
            n_failed += 1
        else:
            print(f"  recommended: {parsed.get('recommended_rewrite')}")
            n_ok += 1

        written_to = write_feedback(parsed, submission, args.outdir)
        print(f"  wrote {written_to / 'feedback.json'}")

    print("\n=== Done ===")
    print(f"Judged successfully : {n_ok}")
    print(f"Judged with warnings: {n_failed}")
    print(f"Skipped (existing)  : {n_skipped}")
    print(f"Skipped (incomplete): {n_missing}")


if __name__ == "__main__":
    main()
