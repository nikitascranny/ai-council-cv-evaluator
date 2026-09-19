"""Candidate mode's seam onto rewriter v2.

Given a CV and a job listing, rewriter v2 runs two
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

The candidate also picks where the rewrite runs: on this computer, or on
Ollama Cloud's larger models. The judge and the scoring stay local either
way. A cloud agent that fails is replaced by a local one inside v2 itself,
so the page's job is to say so, not to arrange it.
"""

import math
import os
import re
import sys
from pathlib import Path

# ui/pipeline/candidate.py -> ui/pipeline -> ui -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent

BACKEND = None
FEEDBACK_BACKEND = None

# rewriter_v2_judge's own CLI defaults. Its module constants cover the model,
# host, token budget and context size; these three exist only in argparse.
_JUDGE_TIMEOUT = 3600
_JUDGE_MAX_RETRIES = 3

# The judge's context window grows to fit its prompt, up to this. Its prompt
# holds the listing and both rewrites whole, and its own default (8192)
# leaves too little room for a long CV twice over plus a reply - but every
# token of context is memory on this machine, so it only grows when needed.
_JUDGE_MAX_CTX = 16384
# Characters per token, rounded down so the estimate of the prompt's size
# errs high: too small a window drops part of a rewrite, too large costs a
# little memory.
_CHARS_PER_TOKEN = 3.5

# rewriter_v2's profile names and the key its cloud agents read. Restated
# rather than imported because rewriter_v2 is only imported on use (see
# _load_rewriter); tests/unit-tests/test_candidate.py pins them together.
LOCAL = "local"
CLOUD = "cloud"
REWRITERS = (LOCAL, CLOUD)
CLOUD_API_KEY_ENV = "OLLAMA_API_KEY"


class NotAvailable(RuntimeError):
    """Rewriter v2 cannot be reached."""


def _ensure_paths():
    for directory in (ROOT / "rewriter", ROOT / "common", ROOT / "parser", ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def _load_rewriter():
    """The rewriter_v2 module, imported on use rather than at module import.

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
    return rewriter_v2


def _load_backend():
    """rewriter_v2.rewrite_cv - see _load_rewriter."""
    return _load_rewriter().rewrite_cv


def cloud_status():
    """None when the cloud rewriter can be offered, otherwise why not.

    Only checks that a key is set. Whether ollama.com accepts it is found out
    by the rewrite itself, which falls back to local if it does not -
    checking here would put a network call in front of drawing a form.
    """
    if os.environ.get(CLOUD_API_KEY_ENV, "").strip():
        return None
    return ("No Ollama Cloud API key is set up. Add {}=<your key> to the .env "
            "file in the repo root (or set it in the terminal) and restart the "
            "app.".format(CLOUD_API_KEY_ENV))


def rewriter_choices() -> list:
    """What the candidate form offers: each choice, the models it runs, and
    why it cannot be picked (None if it can). The models come from v2 itself
    so the form never names a model the rewrite would not run."""
    rewriter_v2 = _load_rewriter()
    return [
        {"value": LOCAL, "models": list(rewriter_v2.AGENT_MODELS),
         "unavailable": None},
        {"value": CLOUD, "models": [c for c, _ in rewriter_v2.CLOUD_AGENTS],
         "unavailable": cloud_status()},
    ]


_HTTP_STATUS = re.compile(r"\b(\d{3}) (?:Client|Server) Error")


def describe_cloud_error(error) -> str:
    """One plain sentence for why Ollama Cloud failed.

    The raw text is a requests exception message - accurate, and useless to
    a candidate. The status code is what tells the causes apart.
    """
    text = str(error or "")
    if CLOUD_API_KEY_ENV in text and "not set" in text:
        return "No Ollama Cloud API key is set up on this computer."
    match = _HTTP_STATUS.search(text)
    status = int(match.group(1)) if match else None
    if status in (401, 403):
        return ("Ollama Cloud did not accept the API key, or the model is not "
                "included in the plan.")
    if status == 429:
        return ("Ollama Cloud's usage limit was reached, or it was busy with "
                "another request.")
    if status == 404:
        return "Ollama Cloud does not offer that model."
    if status is not None and status >= 500:
        return "Ollama Cloud could not run the model just then."
    if "timed out" in text.lower():
        return "Ollama Cloud took too long to answer."
    if "EMPTY RESPONSE" in text:
        return "Ollama Cloud sent back an empty rewrite."
    if "REQUEST ERROR" in text:
        return "Ollama Cloud could not be reached - check the internet connection."
    return "Ollama Cloud failed."


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
             backend=..., profile: str = LOCAL):
    """Both agents' attempts at one CV for one job.

    Takes text rather than paths: the caller has already parsed the CV once
    (see stages.parse_stage) and re-reading the PDF here would parse it
    twice for no gain. Returns v2's RewriteResult list untouched - the guard
    flags on each one are what the page and the .docx both key off, so
    flattening them into something smaller here would only lose the
    fallback status the UI is obliged to show.

    profile is LOCAL or CLOUD. A local rewrite calls v2 exactly as it always
    has; only a cloud one passes the profile on.
    """
    if profile not in REWRITERS:
        raise ValueError("unknown rewriter: {!r}".format(profile))
    backend = (BACKEND or _load_backend()) if backend is ... else backend
    if backend is None:
        raise NotAvailable("rewriter v2 is not wired up.")
    if profile == LOCAL:
        results = backend(cv_text, job_title, candidate_name)
    else:
        results = backend(cv_text, job_title, candidate_name, profile=profile)
    if not results:
        raise NotAvailable("rewriter v2 returned no rewrites.")
    return list(results)


def judge_num_ctx(prompt: str, num_predict: int, floor: int,
                  ceiling: int = _JUDGE_MAX_CTX) -> int:
    """A context window big enough for the judge's prompt and its reply.

    The window holds both: a prompt that fills it leaves the reply no room,
    and one that overflows it loses part of a rewrite before the judge reads
    it. Never below the judge's own default, so a short prompt runs exactly
    as it did; rounded up to a multiple of 2048 so nearby prompt sizes share
    a window rather than each reloading the model at a slightly different
    one.
    """
    needed = math.ceil(len(prompt) / _CHARS_PER_TOKEN) + num_predict
    needed = math.ceil(needed / 2048) * 2048
    return max(floor, min(needed, ceiling))


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
        prompt = judge.build_prompt(job_text, submission)
        try:
            return judge.call_judge_with_retries(
                prompt, names,
                judge.DEFAULT_MODEL, judge.DEFAULT_HOST,
                num_predict=judge.DEFAULT_NUM_PREDICT,
                timeout=_JUDGE_TIMEOUT,
                max_retries=_JUDGE_MAX_RETRIES,
                num_ctx=judge_num_ctx(prompt, judge.DEFAULT_NUM_PREDICT,
                                      judge.DEFAULT_NUM_CTX))
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
