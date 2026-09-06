"""
rewriter_v2.py

Rewriter Variant 2 (graduate use). Given one candidate's CV and one job
listing, runs two independent rewriting agents that each tailor the CV
toward the job, and returns both results for the user to compare.

Unlike rewriter_v1.py, this variant DELIBERATELY gives each agent the full
job listing, not just the title - the goal here is tailoring, not measuring
phrasing sensitivity. That makes the fact guard more important, not less:
a model that can see the actual requirements has more temptation to quietly
upgrade a claim to match one. See fact_guard.py for the shared check.

What this file does NOT do yet, on purpose:
- No judge agent. The two rewrites are returned as-is for the caller to
  display side by side; reconciling/critiquing them is a separate,
  not-yet-built step.
- No docx export. rewrite_cv() returns plain text plus structured metadata
  (RewriteResult) specifically so that export step can be built later
  without reshaping this data - not built now.

Guard behaviour if an agent's rewrite is still flagged after one strict
retry: rather than showing the user a rewrite we can't vouch for, or
silently dropping it, that agent's slot falls back to the candidate's
ORIGINAL CV text (result.used_fallback = True). The caller/eventual UI
should surface this clearly rather than let a fallback look like a normal
rewrite - the point is the user always gets two things to compare, and
never an unflagged fabrication.
"""

import difflib
from dataclasses import dataclass, field, asdict
from pathlib import Path

from document_utils import extract_text_from_pdf, load_cv
from ollama_utils import call_ollama_text
from fact_guard import rejection_reasons, advisory_caps_runs

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
CVS_FOLDER = BASE_DIR / "cvs"
JOB_DESCRIPTIONS_FOLDER = BASE_DIR / "job descriptions"

# The two independent rewriting agents. Kept distinct from
# Evaluate_CV.py's EVALUATOR_MODELS on purpose - this file's job is
# rewriting, not scoring, even though it happens to reuse the same models.
AGENT_MODELS = ["phi4", "qwen2.5:14b"]

# A full CV rewrite can run long - LERATO MUNCHE's CV alone is ~23,500
# characters of full_text(). 4096 gives real headroom over the 1024 used
# for JSON scoring replies, but a CV meaningfully larger than that in the
# corpus would still risk truncation; if that shows up in testing, that's
# a case for parsed_cv's chunks() rather than pushing this number higher.
REWRITE_NUM_PREDICT = 4096


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

    def to_dict(self) -> dict:
        return asdict(self)


_REWRITE_PROMPT = """You are helping a candidate tailor their CV for a specific job application.

Rewrite the CV below so that it highlights the experience, skills and
achievements most relevant to the job description, and reads clearly and
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

JOB DESCRIPTION:
{job_text}

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


def build_rewrite_prompt(cv_text: str, job_text: str, job_title: str, *, strict: bool = False) -> str:
    """
    Plain string replacement rather than str.format/f-string: a CV body can
    contain literal curly braces (template artifacts, copy-paste from other
    tools), which .format() would misread as fields.

    job_title isn't inserted into the prompt directly - the model reads the
    full job_text and infers the role from that, same as a human tailoring
    a CV would. job_title is still passed through to the guard
    (fact_guard.title_fabricated, via rejection_reasons) as the specific
    string to check the rewrite hasn't claimed to already hold.
    """
    prompt = _REWRITE_PROMPT.replace("{job_text}", job_text).replace("{cv_text}", cv_text)
    return prompt + _STRICT_SUFFIX if strict else prompt


def _run_agent(cv_text: str, job_text: str, job_title: str, candidate_name: str, model: str) -> RewriteResult:
    """
    One agent's full attempt: rewrite, guard, retry once with a strict
    reminder if flagged, and fall back to the original CV text if the
    rewrite is still flagged after that retry. See module docstring for
    why a flagged rewrite is never handed back silently.
    """
    result = RewriteResult(
        candidate=candidate_name, job_title=job_title, agent_model=model,
        original_text=cv_text, rewritten_text=cv_text,
    )

    prompt = build_rewrite_prompt(cv_text, job_text, job_title)
    text, error = call_ollama_text(model, prompt, num_predict=REWRITE_NUM_PREDICT)
    if error:
        result.error = error
        return result  # rewritten_text stays as the original, set above

    reasons = rejection_reasons(cv_text, text, job_title)
    if reasons:
        result.retried = True
        strict_prompt = build_rewrite_prompt(cv_text, job_text, job_title, strict=True)
        text, error = call_ollama_text(model, strict_prompt, num_predict=REWRITE_NUM_PREDICT)
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


def rewrite_cv(cv_text: str, job_text: str, job_title: str, candidate_name: str = "") -> list[RewriteResult]:
    """
    Run both independent rewriting agents against one candidate/job pair
    and return both results, for the caller to display side by side.

    Reconciling or critiquing the two versions (the "judge" step from the
    diagram) is intentionally not built here - this function's whole job
    is producing the two candidate rewrites plus their guard status.
    """
    return [
        _run_agent(cv_text, job_text, job_title, candidate_name, model)
        for model in AGENT_MODELS
    ]


def main():
    """
    Manual test harness: run both agents against a single CV and a single
    job listing pulled from the existing cvs/ and 'job descriptions/'
    folders, and print what each agent produced.

    Not the eventual graduate-facing entry point - this is a quick way to
    exercise the rewrite/guard/retry/fallback logic against real CVs while
    building it, before any UI or docx export sits in front of it.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Test Rewriter Var2 against one CV and one job listing.")
    ap.add_argument("--cv", help="CV filename inside cvs/ (defaults to the first one found)")
    ap.add_argument("--job", help="Job listing filename inside 'job descriptions/' (defaults to the first one found)")
    ap.add_argument("--job-title", help="Human-readable job title (defaults to the job PDF's filename, cleaned up)")
    args = ap.parse_args()

    cv_files = sorted(CVS_FOLDER.glob("*.pdf"))
    job_files = sorted(JOB_DESCRIPTIONS_FOLDER.glob("*.pdf"))
    if not cv_files or not job_files:
        print("Need at least one CV in cvs/ and one job listing in 'job descriptions/' to run a test.")
        return

    cv_path = (CVS_FOLDER / args.cv) if args.cv else cv_files[0]
    job_path = (JOB_DESCRIPTIONS_FOLDER / args.job) if args.job else job_files[0]
    job_title = args.job_title or job_path.stem.replace("job_listing_", "").replace("_", " ").title()

    print(f"CV: {cv_path.name}")
    print(f"Job: {job_path.name}  (title: {job_title})\n")

    cv = load_cv(cv_path)
    cv_text = cv.full_text()
    job_text = extract_text_from_pdf(job_path)

    results = rewrite_cv(cv_text, job_text, job_title, candidate_name=cv_path.stem)

    for r in results:
        print(f"\n{'=' * 60}")
        print(f" AGENT: {r.agent_model}")
        print(f"{'=' * 60}")
        if r.error:
            print(f"  [!] {r.error}")
            continue

        print(f"  rejected: {r.rejected}   retried: {r.retried}   used_fallback: {r.used_fallback}")
        if r.reasons:
            print(f"  reasons: {r.reasons}")
        if r.advisory:
            print(f"  advisory (new capitalised runs): {r.advisory}")
        print(f"  length_ratio: {r.length_ratio:.2f}   similarity: {r.similarity:.2f}")
        print()
        print(r.rewritten_text[:1000])


if __name__ == "__main__":
    main()
