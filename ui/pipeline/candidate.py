"""Candidate mode's seam onto rewriter v2.

v2 is another team member's work: given a CV and a job listing it runs two
independent rewriting agents and returns both attempts, each carrying the
fact guard's verdict. The UI shows both and lets the candidate download the
one they want as a Word document.

This module is the only place the UI names rewriter_v2. It resolves the
import lazily and raises NotAvailable when it cannot, so a checkout missing
the rewriter still serves a page that says so plainly rather than a stack
trace - the same contract the previous version of this file had, when v2 did
not exist at all.

Note the shape v2 actually settled on, which is not what this seam originally
guessed at: it returns TWO rewrites as plain text, not one document, and it
needs the full job listing text rather than just the title - the model infers
the role from the listing the way a person tailoring a CV would. The .docx is
made downstream in ui/docx_export.py, from whichever version the user picks.
"""

import sys
from pathlib import Path

# ui/pipeline/candidate.py -> ui/pipeline -> ui -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent

BACKEND = None


class NotAvailable(RuntimeError):
    """Rewriter v2 cannot be reached."""


def _load_backend():
    """rewriter_v2.rewrite_cv, imported on use rather than at module import.

    Importing at module scope would make ui.pipeline.candidate - and so
    app.py - unimportable whenever rewriter/ is absent or its own imports
    fail, which is exactly the case this module exists to report politely.
    """
    for directory in (ROOT / "rewriter", ROOT / "common", ROOT / "parser", ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
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


def generate(cv_text: str, job_text: str, job_title: str,
             candidate_name: str = "", backend=...):
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
    results = backend(cv_text, job_text, job_title, candidate_name)
    if not results:
        raise NotAvailable("rewriter v2 returned no rewrites.")
    return list(results)
