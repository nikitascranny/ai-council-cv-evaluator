"""Candidate mode's seam onto rewriter v2.

v2 is another team member's work: given a CV and a job listing it runs two
independent rewriting agents and returns both attempts, each carrying the
fact guard's verdict. The UI shows both and lets the candidate download the
one they want as a Word document.

This module is the only place the UI names rewriter_v2 or its judge,
rewriter_v2_judge. It resolves the
import lazily and raises NotAvailable when it cannot, so a checkout missing
the rewriter still serves a page that says so plainly rather than a stack
trace - the same contract the previous version of this file had, when v2 did
not exist at all.

Note the shape v2 actually settled on, which is not what this seam originally
guessed at: it returns TWO rewrites as plain text, not one document. It also
takes only the job TITLE, not the listing text - v2 used to take the whole
listing, and dropped it because the models borrowed facts from it. The .docx
is made downstream in ui/docx_export.py, from whichever version the user picks.
"""

import sys
from pathlib import Path

# ui/pipeline/candidate.py -> ui/pipeline -> ui -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent

BACKEND = None
FEEDBACK_BACKEND = None

# rewriter_v2_judge's own CLI defaults. Its module constants cover the model,
# host, token budget and context size; these three exist only in argparse.
_JUDGE_TEMPERATURE = 0.2
_JUDGE_TIMEOUT = 3600
_JUDGE_MAX_RETRIES = 3


class NotAvailable(RuntimeError):
    """Rewriter v2 cannot be reached."""


def _ensure_paths():
    for directory in (ROOT / "rewriter", ROOT / "common", ROOT / "parser", ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def _load_backend():
    """rewriter_v2.rewrite_cv, imported on use rather than at module import.

    Importing at module scope would make ui.pipeline.candidate - and so
    app.py - unimportable whenever rewriter/ is absent or its own imports
    fail, which is exactly the case this module exists to report politely.
    """
    _ensure_paths()
    try:
        import rewriter_v2
    except ImportError as exc:
        raise NotAvailable(
            "rewriter v2 could not be loaded ({}). Candidate mode needs it to "
            "rewrite your CV.".format(exc))
    return rewriter_v2.rewrite_cv


ADDED = "added"
DROPPED = "dropped"
TITLE = "title"


def classify_reasons(reasons) -> list:
    """Which kinds of failure the guard actually found.

    rejection_reasons mixes three distinct verdicts, and they are not
    variations on one theme: a new hard fact means the model invented
    something, a lost number means it left something out, and a borrowed job
    title means it claimed a role the candidate has never held. Telling a
    candidate their rewrite "added information your CV does not contain" when
    the guard's complaint was that it DROPPED half their numbers is not a
    rounding error in the wording - it points them at the wrong problem.

    Truncation makes this the common case rather than the rare one: a CV over
    the rewriter's token budget comes back with its tail missing, and every
    number in that tail is reported as dropped.
    """
    kinds = []
    for reason in reasons or []:
        text = str(reason)
        if text.startswith("number dropped from the CV"):
            kind = DROPPED
        elif text.startswith("job title claimed but not held"):
            kind = TITLE
        else:
            kind = ADDED
        if kind not in kinds:
            kinds.append(kind)
    return kinds


def describe_rejection(reasons) -> str:
    """One plain sentence for why the guard rejected a rewrite."""
    kinds = classify_reasons(reasons)
    parts = []
    if ADDED in kinds:
        parts.append("added information your CV does not contain")
    if DROPPED in kinds:
        parts.append("left out details your CV does contain")
    if TITLE in kinds:
        parts.append("claimed a job title you have not held")
    if not parts:
        return "The check could not confirm the rewrite matched your CV."
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + " and " + parts[-1]
    return "The check found that the rewrite " + body + "."


def probe():
    """Raise NotAvailable unless rewriter v2 can be reached.

    Split out so a page can ask "is this mode usable" without starting a
    rewrite. The route used to answer that question by calling generate()
    with placeholder arguments, which was harmless only while the backend
    did not exist - against the real v2 it would run two 14B models to
    render a form.
    """
    if BACKEND is None:
        _load_backend()


def is_rewrite(result) -> bool:
    """Whether a result holds text the model actually wrote.

    A fallback and a failed agent both carry the candidate's original CV in
    rewritten_text. Downloading or scoring either under the model's name
    would present the candidate's own CV as a rewrite.
    """
    return not result.used_fallback and not getattr(result, "error", None)


def generate(cv_text: str, job_title: str, candidate_name: str = "",
             backend=...):
    """Both agents' attempts at one CV for one job.

    Takes text rather than paths: the caller has already parsed the CV once
    (see stages.parse_stage) and re-reading the PDF here would parse it
    twice for no gain. Returns v2's RewriteResult list untouched - the guard
    flags on each one are what the page and the .docx both key off, so
    flattening them into something smaller here would only lose the
    fallback status the UI is obliged to show.
    """
    backend = (BACKEND or _load_backend()) if backend is ... else backend
    if backend is None:
        raise NotAvailable("rewriter v2 is not wired up.")
    results = backend(cv_text, job_title, candidate_name)
    if not results:
        raise NotAvailable("rewriter v2 returned no rewrites.")
    return list(results)


def _load_feedback_backend():
    """rewriter_v2_judge, as (job_text, rows, job_title, candidate_name) ->
    (parsed, error).

    The judge was written to read rewrites back off disk, from files a
    separate batch run saved. The web app already holds them in memory, so
    this builds its RewriteSubmission directly rather than writing files only
    to have them read straight back.
    """
    _ensure_paths()
    try:
        import rewriter_v2_judge as judge
    except ImportError as exc:
        raise NotAvailable(
            "the rewrite judge could not be loaded ({}).".format(exc))

    def run(job_text, rows, job_title, candidate_name):
        submission = judge.RewriteSubmission(
            job_title=job_title, candidate=candidate_name, job_pdf=None,
            rewrites=rows)
        names = [row["agent_model"] for row in rows]
        try:
            return judge.call_judge_with_retries(
                judge.build_prompt(job_text, submission), names,
                judge.DEFAULT_MODEL, judge.DEFAULT_HOST,
                num_predict=judge.DEFAULT_NUM_PREDICT,
                temperature=_JUDGE_TEMPERATURE, timeout=_JUDGE_TIMEOUT,
                max_retries=_JUDGE_MAX_RETRIES, num_ctx=judge.DEFAULT_NUM_CTX)
        except SystemExit as exc:
            # The judge exits the process when Ollama is unreachable - right
            # for its CLI batch, where every submission would fail the same
            # way. On the web app's worker thread a SystemExit slips past
            # `except Exception`, ends the run with no done event, and leaves
            # the page on "waiting" for good.
            return None, str(exc)

    return run


def feedback(results, job_text, job_title, candidate_name="", backend=...):
    """The rewrite judge on both versions: which to send and why, a career
    roadmap, and alternative careers. Returns (parsed, error), as the judge
    itself does, and never raises NotAvailable - the rewrites are done by the
    time this runs, and a missing judge must not cost the candidate those.

    A failed agent is left out: its text is the original CV with none of the
    guard flags set, so the judge would read it as a clean rewrite that
    changed nothing, and could recommend it. A fallback stays in - its flags
    say what it is, and the judge's prompt tells it to weigh them.
    """
    rows = [{"agent_model": r.agent_model, "text": r.rewritten_text or "",
             "rejected": bool(getattr(r, "rejected", False)),
             "used_fallback": bool(r.used_fallback),
             "retried": bool(getattr(r, "retried", False)),
             "reasons": list(getattr(r, "reasons", None) or []),
             "advisory": list(getattr(r, "advisory", None) or []),
             "length_ratio": getattr(r, "length_ratio", None),
             "similarity": getattr(r, "similarity", None)}
            for r in results if not getattr(r, "error", None)]
    if not rows:
        return None, "no model produced a rewrite to judge"
    try:
        if backend is ...:
            backend = FEEDBACK_BACKEND or _load_feedback_backend()
    except NotAvailable as exc:
        return None, str(exc)
    return backend(job_text, rows, job_title, candidate_name)
