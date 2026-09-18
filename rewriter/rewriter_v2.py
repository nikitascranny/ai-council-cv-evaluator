"""
rewriter_v2.py

Rewriter Variant 2 (graduate use). Given one candidate's CV and a job
title, runs two independent rewriting agents that each tailor the CV
toward that role, and returns both results for the user to compare.

Each agent is given only the job TITLE, not the full job listing text -
the same design rewriter_v1.py uses for its phrasing-sensitivity probe.
Previously this variant deliberately passed the full listing (see git
history for that version); that's been dropped in favour of title-only
so both variants share the same input shape. The fact guard matters just
as much either way: fact_guard.title_fabricated() still checks the
rewrite hasn't claimed to already hold job_title, and the rest of the
guard (new numbers, new qualifications) is independent of how much job
context the model saw. See fact_guard.py for the shared check.

Guard behaviour if an agent's rewrite is still flagged after one strict
retry: rather than showing the user a rewrite we can't vouch for, or
silently dropping it, that agent's slot falls back to the candidate's
ORIGINAL CV text (result.used_fallback = True). The caller/eventual UI
should surface this clearly rather than let a fallback look like a normal
rewrite - the point is the user always gets two things to compare, and
never an unflagged fabrication.

Two sets of agents: the local pair (phi4, qwen2.5:14b) through the Ollama
daemon on this machine, and a larger cloud pair on Ollama Cloud, chosen with
rewrite_cv(profile="cloud"). A cloud agent that fails - no key, no
connection, a usage limit, a timeout - does not cost the candidate that
slot: its local partner rewrites the CV instead, and the result records
which cloud model it replaced and why (replaced_cloud_model, cloud_error).
That is a different thing from used_fallback, which means the fact guard
rejected a rewrite and the slot holds the ORIGINAL CV.
"""

import os
import sys
import csv
import difflib
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------- CONFIG & PATHS ----------------
# Repo layout, confirmed with the team:
#
#   ai-cv-eval-main/
#   |-- common/               (document_utils.py, ollama_utils.py, results_io.py, fact_guard.py)
#   |-- parser/cv_pdf_parser.py   (the one canonical CV PDF parser)
#   |-- data/cvs/Batch1, data/job-listings/
#   |-- evaluator/...
#   \-- rewriter/rewriter_v2.py   <- this file
#
# rewriter/ is a direct child of the repo root, sibling to common/ and
# parser/ - so ROOT_DIR needs exactly one .parent from this file.
BASE_DIR = Path(__file__).resolve().parent   # .../ai-cv-eval-main/rewriter
ROOT_DIR = BASE_DIR.parent                    # .../ai-cv-eval-main

# document_utils.py, ollama_utils.py and fact_guard.py all live in the
# shared common/ folder now - one sys.path insert covers all three,
# rather than rewriter/ reaching into evaluator/'s own folder.
sys.path.insert(0, str(ROOT_DIR / "common"))

from document_utils import load_cv
from ollama_utils import call_ollama_text
from fact_guard import rejection_reasons, advisory_caps_runs

# Real project layout: data/cvs/Batch1, data/job-listings/
CVS_FOLDER = ROOT_DIR / "data" / "cvs" / "Batch1"
JOB_DESCRIPTIONS_FOLDER = ROOT_DIR / "data" / "job-listings"
REWRITE_RESULTS_FOLDER = ROOT_DIR / "data" / "rewrite_results"

# The two independent rewriting agents. Kept distinct from
# Evaluate_CV.py's EVALUATOR_MODELS on purpose - this file's job is
# rewriting, not scoring, even though it happens to reuse the same models.
AGENT_MODELS = ["phi4","qwen2.5:14b"]
# A full CV rewrite can run long - LERATO MUNCHE's CV alone is ~23,500
# characters of full_text(). 4096 gives real headroom over the 1024 used
# for JSON scoring replies, but a CV meaningfully larger than that in the
# corpus would still risk truncation; if that shows up in testing, that's
# a case for parsed_cv's chunks() rather than pushing this number higher.
REWRITE_NUM_PREDICT = 4096

LOCAL = "local"
CLOUD = "cloud"
PROFILES = (LOCAL, CLOUD)

# Ollama Cloud's agents, each paired with the local agent that takes its slot
# if the cloud call fails. Two families again (OpenAI, Google), for the same
# reason the local pair is phi4 and qwen: two versions worth comparing, not
# one model's second draft. Both are on Ollama's free plan's starter models.
CLOUD_AGENTS = [("gpt-oss:120b", "phi4"), ("gemma4:31b", "qwen2.5:14b")]
CLOUD_HOST = "https://ollama.com/api/chat"
CLOUD_API_KEY_ENV = "OLLAMA_API_KEY"
# gpt-oss always reasons before it answers, and those tokens count against
# num_predict - so the cloud budget is larger than the local one, and its
# reasoning is kept to "low" so it cannot spend the budget thinking.
CLOUD_NUM_PREDICT = 16384
CLOUD_THINK = {"gpt-oss:120b": "low"}
# A cloud call that hangs costs the candidate this long before the local
# fallback starts, so it is minutes rather than the local hour.
CLOUD_TIMEOUT = 300


@dataclass
class RewriteResult:
    """One agent's attempt at rewriting one candidate's CV for one job.

    Shaped for the eventual docx export: candidate/job/agent identify the
    version, rewritten_text is what gets shown, and the rest (rejected,
    retried, reasons, advisory, used_fallback) is what a UI needs to decide
    how to present that version - a clean rewrite, a flagged one, or a
    fallback to the original.
    """
    candidate: str
    job_title: str
    agent_model: str
    original_text: str
    rewritten_text: str
    used_fallback: bool = False
    rejected: bool = False
    retried: bool = False
    reasons: list = field(default_factory=list)
    advisory: list = field(default_factory=list)
    length_ratio: float | None = None
    similarity: float | None = None
    error: str | None = None
    # Where agent_model ran: LOCAL or CLOUD.
    provider: str = LOCAL
    # Set when this local result stands in for a cloud agent that failed.
    replaced_cloud_model: str | None = None
    cloud_error: str | None = None
    # Tokens Ollama Cloud counted across this agent's calls (retry included),
    # which is what its usage is billed on. None for local runs.
    tokens_in: int | None = None
    tokens_out: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_REWRITE_PROMPT = """You are helping a candidate tailor their CV for a job application.

The candidate is applying for the following role: {job_title}

Rewrite the CV below so that it highlights the experience, skills and
achievements most relevant to that role, and reads clearly and
professionally throughout.

Rules, all mandatory:
- Every fact must survive exactly: employers, job titles, dates, numbers,
  tools, qualifications, and every section and every role that appears in
  the original CV.
- Do not omit anything. Every employer, job title, role, date, number,
  tool and qualification in the CV must still be there afterwards.
- Do not add any fact that is not already in the CV. You may reorder,
  re-emphasise and rephrase - you may not invent.
- Do not invent employers, job titles, dates, numbers, tools or
  qualifications, and do not state or imply the candidate already holds
  the job title being applied for unless the CV itself says so.
- Do not upgrade a claim: "assisted with" must not become "led".
- Output the rewritten CV as plain text and nothing else. No preamble, no
  commentary, no markdown fences.

CV:
{cv_text}
"""

_STRICT_SUFFIX = """

A previous attempt at this rewrite INVENTED information that was not in the \
CV, or implied the candidate already holds the job being applied for. That \
is the single thing you must not do. Copy every name, number, date and \
qualification from the CV exactly as written. If you are unsure whether \
something is in the CV, leave it out. Do not state or imply the candidate \
already works in the advertised role unless the CV itself says so.
"""


def build_rewrite_prompt(cv_text: str, job_title: str, *, strict: bool = False) -> str:
    """
    Plain string replacement rather than str.format/f-string: a CV body can
    contain literal curly braces (template artifacts, copy-paste from other
    tools), which .format() would misread as fields.

    job_title IS inserted directly into the prompt now (unlike
    rewriter_v1.build_rewrite_prompt, which deliberately withholds it to
    avoid the model adopting the role - see that function's docstring).
    v2 needs the model to know what it's tailoring for, since there's no
    job description left to infer it from, so this is a real trade-off:
    telling the model the target role makes it more likely to slip into
    writing as if the candidate already holds it. That's exactly the
    failure title_fabricated() (via rejection_reasons(), called from
    _run_agent) exists to catch - a rewrite that does this gets a strict
    retry and, if it still fails, falls back to the original CV rather
    than reaching the caller unflagged.
    """
    prompt = _REWRITE_PROMPT.replace("{job_title}", job_title).replace("{cv_text}", cv_text)
    return prompt + _STRICT_SUFFIX if strict else prompt


def _call(model: str, prompt: str, result: RewriteResult, api_key: str | None):
    """One model call, local or cloud depending on result.provider.

    The local call is exactly the one this module always made. A cloud call
    adds the host, the key, the larger budget and the shorter timeout, and
    adds the tokens it used to the result.
    """
    if result.provider != CLOUD:
        return call_ollama_text(model, prompt, num_predict=REWRITE_NUM_PREDICT)

    stats = {}
    kwargs = {"num_predict": CLOUD_NUM_PREDICT, "host": CLOUD_HOST,
              "headers": {"Authorization": "Bearer " + (api_key or "")},
              "keep_alive": None, "timeout": CLOUD_TIMEOUT, "stats": stats}
    if model in CLOUD_THINK:
        kwargs["think"] = CLOUD_THINK[model]
    text, error = call_ollama_text(model, prompt, **kwargs)
    if stats.get("prompt_eval_count") is not None:
        result.tokens_in = (result.tokens_in or 0) + stats["prompt_eval_count"]
    if stats.get("eval_count") is not None:
        result.tokens_out = (result.tokens_out or 0) + stats["eval_count"]
    return text, error


def _run_agent(cv_text: str, job_title: str, candidate_name: str, model: str,
               provider: str = LOCAL, api_key: str | None = None) -> RewriteResult:
    """
    One agent's full attempt: rewrite, guard, retry once with a strict
    reminder if flagged, and fall back to the original CV text if the
    rewrite is still flagged after that retry. See module docstring for
    why a flagged rewrite is never handed back silently.
    """
    result = RewriteResult(
        candidate=candidate_name, job_title=job_title, agent_model=model,
        original_text=cv_text, rewritten_text=cv_text, provider=provider,
    )

    prompt = build_rewrite_prompt(cv_text, job_title)
    text, error = _call(model, prompt, result, api_key)
    if error:
        result.error = error
        return result  # rewritten_text stays as the original, set above

    reasons = rejection_reasons(cv_text, text, job_title)
    if reasons:
        result.retried = True
        strict_prompt = build_rewrite_prompt(cv_text, job_title, strict=True)
        text, error = _call(model, strict_prompt, result, api_key)
        if error:
            result.error = error
            return result
        reasons = rejection_reasons(cv_text, text, job_title)

    result.reasons = reasons
    result.rejected = bool(reasons)
    result.advisory = advisory_caps_runs(cv_text, text)

    if result.rejected:
        # Still flagged after the strict retry - do not hand the user a
        # rewrite we can't vouch for. Fall back to their original text so
        # they still get two versions to compare; the caller/UI should
        # surface *why* clearly rather than let this look like a normal
        # rewrite that happened not to change much.
        result.used_fallback = True
        result.rewritten_text = cv_text
    else:
        result.rewritten_text = text

    result.length_ratio = (len(result.rewritten_text) / len(cv_text)) if cv_text else None
    result.similarity = difflib.SequenceMatcher(
        None, cv_text.split(), result.rewritten_text.split()
    ).ratio()

    return result


def rewrite_cv(cv_text: str, job_title: str, candidate_name: str = "",
               profile: str = LOCAL) -> list[RewriteResult]:
    """
    Run both independent rewriting agents against one candidate/job pair
    and return both results, for the caller to display side by side.

    profile picks the agents: LOCAL runs AGENT_MODELS on this machine, CLOUD
    runs CLOUD_AGENTS on Ollama Cloud with a local fallback per slot (see
    _rewrite_on_cloud).

    Reconciling or critiquing the two versions (the "judge" step from the
    diagram) is intentionally not built here - this function's whole job
    is producing the two candidate rewrites plus their guard status.
    """
    if profile == CLOUD:
        return _rewrite_on_cloud(cv_text, job_title, candidate_name)
    if profile != LOCAL:
        raise ValueError(f"unknown rewriter profile: {profile!r}")
    return [
        _run_agent(cv_text, job_title, candidate_name, model)
        for model in AGENT_MODELS
    ]


def _rewrite_on_cloud(cv_text: str, job_title: str, candidate_name: str) -> list[RewriteResult]:
    """
    Each cloud agent in turn; any that fails is replaced by its local
    partner, so the candidate still gets two versions.

    Only a failed call counts (result.error) - a cloud rewrite the fact guard
    rejected is a finished result about that model, not an outage, and is
    returned as it is.

    Once one cloud call has failed, the rest of the run stays local rather
    than trying the cloud again: the usual causes (no key, no connection, a
    usage limit) will fail the next call the same way, and a timeout would
    cost the candidate another CLOUD_TIMEOUT before its fallback even starts.
    """
    api_key = os.environ.get(CLOUD_API_KEY_ENV, "").strip()
    cloud_error = None if api_key else f"{CLOUD_API_KEY_ENV} is not set"

    results = []
    for cloud_model, local_model in CLOUD_AGENTS:
        if cloud_error is None:
            result = _run_agent(cv_text, job_title, candidate_name, cloud_model,
                                provider=CLOUD, api_key=api_key)
            if not result.error:
                results.append(result)
                continue
            cloud_error = result.error

        result = _run_agent(cv_text, job_title, candidate_name, local_model)
        result.replaced_cloud_model = cloud_model
        result.cloud_error = cloud_error
        results.append(result)
    return results


def _safe_folder_name(name: str) -> str:
    """
    Job titles and candidate names end up as path components. They come
    from filename-derived strings (see job_title_from_filename in
    run_batch.py and cv_path.stem in main()) so this is mostly a
    belt-and-braces pass - strip anything that isn't safe on disk rather
    than trust the caller never hands us a stray "/" or ":".
    """
    # "." is deliberately kept: it's safe mid-name on every OS we care
    # about (Windows/macOS/Linux) and this sanitizer is also applied to
    # agent_model, where version-numbered tags like "qwen2.5" or
    # "qwen2.5:14b" need their dot to survive - only the trailing-dot
    # case (stripped below) and characters like ":" are actually unsafe.
    keep = "-_(). "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    return cleaned.strip().strip(".") or "unnamed"


def save_results(results: list[RewriteResult]):
    """
    Saves the rewritten CVs in individual text files and stores the metadata 
    for all runs in a single CSV file, structured inside
    data/rewrite_results/<job_title>/<candidate_name>/ - job title first so
    that running the same candidate against several jobs produces separate
    output folders instead of each job's results overwriting the last.
    """
    if not results:
        return None

    # Determine job title / candidate name and create the target directory
    job_title = _safe_folder_name(results[0].job_title)
    candidate_name = _safe_folder_name(results[0].candidate)
    target_dir = REWRITE_RESULTS_FOLDER / job_title / candidate_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save rewritten text per model
    for r in results:
        # agent_model may be a tag like "qwen2.5:7b" - the colon is fine in
        # an Ollama model name, but on NTFS a ":" in a filename is read as
        # an alternate-data-stream separator, not a literal character. That
        # silently splits the write into a 0-byte visible file plus a
        # hidden stream holding the actual text, so sanitize before use.
        safe_model_name = _safe_folder_name(r.agent_model)
        text_file_path = target_dir / f"{safe_model_name}.txt"
        with text_file_path.open("w", encoding="utf-8") as f:
            f.write(r.rewritten_text)

    # Save metadata to a single CSV file
    csv_file_path = target_dir / "metadata.csv"
    
    # We exclude the large text fields to keep the CSV clean and purely metadata-focused
    fieldnames = [
        "candidate", "job_title", "agent_model", "used_fallback",
        "rejected", "retried", "reasons", "advisory",
        "length_ratio", "similarity", "error",
        "provider", "replaced_cloud_model", "cloud_error", "tokens_in", "tokens_out"
    ]
    
    with csv_file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row_dict = r.to_dict()
            # Filter the dictionary to only include the specific fieldnames
            filtered_row = {k: v for k, v in row_dict.items() if k in fieldnames}
            writer.writerow(filtered_row)
            
    return target_dir


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Test Rewriter Var2 against one CV and one job listing.")
    ap.add_argument("--cv", help="CV filename inside data/cvs/Batch1/ (defaults to the first one found)")
    ap.add_argument("--job", help="Job listing filename inside data/job-listings/ (defaults to the first one found)")
    ap.add_argument("--job-title", help="Human-readable job title (defaults to the job PDF's filename, cleaned up)")
    ap.add_argument("--cloud", action="store_true",
                    help=f"Rewrite with the Ollama Cloud agents ({', '.join(c for c, _ in CLOUD_AGENTS)}) "
                         f"instead of the local ones. Needs {CLOUD_API_KEY_ENV} set; a cloud agent "
                         f"that fails is replaced by its local partner.")
    args = ap.parse_args()

    cv_files = sorted(CVS_FOLDER.glob("*.pdf"))
    job_files = sorted(JOB_DESCRIPTIONS_FOLDER.glob("*.pdf"))
    if not cv_files or not job_files:
        print("Need at least one CV in data/cvs/Batch1/ and one job listing in data/job-listings/ to run a test.")
        return

    cv_path = (CVS_FOLDER / args.cv) if args.cv else cv_files[0]
    job_path = (JOB_DESCRIPTIONS_FOLDER / args.job) if args.job else job_files[0]
    job_title = args.job_title or job_path.stem.replace("job_listing_", "").replace("_", " ").title()

    candidate_name = cv_path.stem

    print(f"CV: {cv_path.name}")
    print(f"Job: {job_path.name}  (title: {job_title})\n")

    cv = load_cv(cv_path)
    cv_text = cv.full_text()

    results = rewrite_cv(cv_text, job_title, candidate_name=candidate_name,
                         profile=CLOUD if args.cloud else LOCAL)

    # Save outputs to structured directories
    saved_dir = save_results(results)
    if saved_dir:
        print(f"Results successfully saved to {saved_dir.resolve()}\n")

    for r in results:
        print(f"\n{'=' * 60}")
        print(f" AGENT: {r.agent_model}  ({r.provider})")
        print(f"{'=' * 60}")
        if r.replaced_cloud_model:
            print(f"  [!] ran locally in place of {r.replaced_cloud_model}: {r.cloud_error}")
        if r.tokens_in is not None or r.tokens_out is not None:
            print(f"  cloud tokens: {r.tokens_in} in, {r.tokens_out} out")
        if r.error:
            print(f"  [!] {r.error}")
            continue

        print(f"  rejected: {r.rejected}   retried: {r.retried}   used_fallback: {r.used_fallback}")
        if r.reasons:
            print(f"  reasons: {r.reasons}")
        if r.advisory:
            print(f"  advisory (new capitalised runs): {r.advisory}")
        if r.length_ratio is not None and r.similarity is not None:
            print(f"  length_ratio: {r.length_ratio:.2f}   similarity: {r.similarity:.2f}")
        print()
        print(r.rewritten_text[:1000])


if __name__ == "__main__":
    main()