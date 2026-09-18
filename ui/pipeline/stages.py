"""The pipeline's stages, as generators of progress events.

Each stage yields Events as it works and returns its result, so a caller writes

    rows = yield from council_stage(...)

and gets both the progress and the value. The console app printed instead,
which is why it could only ever be driven from a terminal.

Stages are independent on purpose: a lost model costs its own row, and the
judge still runs on whatever came back.
"""

import sys
import time
from pathlib import Path

from ui.pipeline.events import Event

# Pure text segmentation, no Ollama calls - safe to import directly rather
# than route through Deps, same as ui/facts.py does for fact_guard.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT / "parser") not in sys.path:
    sys.path.insert(0, str(_ROOT / "parser"))

import job_req_parser  # noqa: E402
import txt_cv_parser  # noqa: E402


def parse_stage(cv_path, deps, candidate: str = ""):
    """Our parser. Runs first because everything downstream reads its output."""
    yield Event("parse", "start", {"path": str(cv_path)}, candidate=candidate)
    started = time.time()
    cv = deps.load_cv(cv_path)
    yield Event("parse", "done",
                {"sections": [s.label for s in cv.sections],
                 "chars": len(cv.full_text()),
                 "multi_column_pages": list(cv.multi_column_pages or []),
                 "warnings": list(cv.warnings or [])},
                time.time() - started, candidate)
    return cv


def council_stage(job_text, cv_text, candidate, models, deps):
    """Each evaluator in turn. One model failing must not end the run."""
    rows = []
    for model in models:
        yield Event("council", "start", {"model": model}, candidate=candidate)
        started = time.time()
        try:
            row = deps.evaluate_cv(job_text, cv_text, candidate, model)
        except Exception as exc:
            row = {"candidate": candidate, "score": None, "strengths": [],
                   "weaknesses": [], "summary": "MODEL ERROR: {}".format(exc)}
            yield Event("council", "failed", {"model": model, "error": str(exc)},
                        time.time() - started, candidate)
        else:
            yield Event("council", "done",
                        {"model": model, "score": row.get("score"),
                         "summary": row.get("summary", "")},
                        time.time() - started, candidate)
        row["model"] = model
        rows.append(row)
    return rows


_ESSENTIAL_SECTIONS = ("experience", "education", "certifications")


def _scope_rewrite(text: str) -> str:
    """Section-scope a rewrite's plain text the same way a parsed CV is scoped.

    txt_cv_parser segments on the same heading lexicon cv_pdf_parser uses, but
    its own docstring calls it validated against exactly one sample - a
    heading it misreads on a real rewrite could scope to the wrong section or
    to nothing. Falling back to the whole rewrite in that case costs nothing
    the old behaviour did not already accept, and avoids silently handing the
    essential-requirements model less text than it needs.
    """
    sections = txt_cv_parser.segment_text(text)
    scoped = "\n\n".join(s.text() for s in sections
                         if s.label in _ESSENTIAL_SECTIONS and s.text()).strip()
    return scoped or text


def _scope_job_requirements(job_text: str) -> str:
    """The listing's essential-requirements section alone, not the whole ad.

    job_req_parser looks for a heading in its own vocabulary and stops at the
    next section heading. `or job_text` covers both failure modes the same
    way - a heading it doesn't recognise (returns None) and a recognised
    heading immediately followed by another one (returns ""): either way,
    the essential-requirements model should see the whole listing rather
    than an accidentally empty one.
    """
    section = job_req_parser.extract_requirements_section(job_text)
    return section or job_text


def essential_stage(job_text, cv, candidate, model, deps):
    """Scoped input on both sides, not the whole listing and the whole CV.

    The CV side is scoped to experience/education/certifications, by
    sections_named() for a parsed CV or txt_cv_parser for a rewrite (see
    _scope_rewrite). The listing side is scoped to its essential-requirements
    section by job_req_parser (see _scope_job_requirements). Both fall back
    to their whole, unscoped text when scoping finds nothing.
    """
    yield Event("essential", "start", {"model": model}, candidate=candidate)
    started = time.time()
    if isinstance(cv, str):
        scoped_cv = _scope_rewrite(cv)
    else:
        scoped_cv = cv.sections_named(*_ESSENTIAL_SECTIONS)
    scoped_job = _scope_job_requirements(job_text)
    result = deps.check_essential(scoped_job, scoped_cv, candidate, model=model)
    yield Event("essential", "done", {"summary": result.get("summary", "")},
                time.time() - started, candidate)
    return result


def essential_in_prompt(essential) -> bool:
    """Whether the judge's prompt includes this essential-requirements result.

    The rule is judge.format_essential_for_prompt's - no result, a failed
    check or an empty summary is left out - restated rather than imported,
    because stages reach the team's Ollama-calling modules only through Deps.
    tests/unit-tests/test_stages.py pins the two together.
    """
    if not essential or essential.get("error"):
        return False
    return bool(str(essential.get("summary") or "").strip())


def judge_stage(job_text, rows, candidate, model, deps, essential=None):
    """Aggregate the council's summaries into one independent verdict.

    essential is the essential-requirements result when that check ran, and
    the judge weighs it alongside the council unless it failed. The done
    event says whether it did, since that changes what the score means.
    """
    yield Event("judge", "start", {"model": model}, candidate=candidate)
    started = time.time()
    parsed, error = deps.judge(job_text, rows, candidate, model, essential)
    if error:
        yield Event("judge", "failed", {"error": str(error)},
                    time.time() - started, candidate)
        return {"candidate": candidate, "final_score": None, "error": str(error)}
    yield Event("judge", "done",
                {"final_score": parsed.get("final_score"),
                 "final_evaluation": parsed.get("final_evaluation", ""),
                 "with_essential": essential_in_prompt(essential)},
                time.time() - started, candidate)
    return parsed


# The attributes rewriter_v1.paraphrase records on itself after each call.
# probe() reads last_truncated off paraphrase_fn (getattr(paraphrase_fn,
# "last_truncated", None)) to fill ProbeResult.truncated_rewrite - the signal
# that Ollama silently cut the rewrite prompt. capture_paraphrase must copy
# these onto its own wrapper or that signal is always None for every run
# started from the web UI, which is one of the two measured defects this
# project sets out to catch.
_PARAPHRASE_ATTRS = ("last_truncated", "last_prompt_eval_count", "last_prompt_chars")


def capture_paraphrase(fn):
    """Wrap a paraphrase function so the text it returned can be read back.

    probe() takes paraphrase_fn as a parameter, so this captures the rewrite
    without adding a field to ProbeResult - a full CV in every JSONL row would
    bloat the research output the findings are read from.

    The guard retries on rejection, calling this a second time. Keeping the last
    call is deliberate: that is the text which was actually scored.
    """
    def wrapper(cv_text, job_title, **kwargs):
        result = fn(cv_text, job_title, **kwargs)
        wrapper.text = result
        for attr in _PARAPHRASE_ATTRS:
            setattr(wrapper, attr, getattr(fn, attr, None))
        return result

    wrapper.text = None
    for attr in _PARAPHRASE_ATTRS:
        setattr(wrapper, attr, None)
    return wrapper


def rewrite_stage(cv_text, job_title, scorer, deps, candidate: str = "",
                  paraphrase_fn=None, rewriter_model=None):
    """The phrasing-sensitivity probe, reported as a finding.

    A rejected rewrite is a result about the rewriting model, not an error: it
    means the paraphrase introduced a fact the original did not contain.

    rewriter_model is passed through to probe() explicitly, so the run's
    Settings decide which model paraphrases rather than probe()'s own default.
    """
    yield Event("rewrite", "start", {"job_title": job_title}, candidate=candidate)
    started = time.time()
    extra = {"paraphrase_fn": paraphrase_fn} if paraphrase_fn else {}
    if rewriter_model:
        extra["rewriter_model"] = rewriter_model
    result = deps.probe(cv_text, job_title, scorer, **extra)
    out = result if isinstance(result, dict) else vars(result)

    # probe() returns a ProbeResult dataclass in production, and ProbeResult has
    # no `new_facts` field - the invented facts that make a rejection a finding
    # live in `reasons` (set from rejection_reasons() inside probe()). Read
    # `reasons` first for that reason; `new_facts` is a fallback so dict-shaped
    # fakes in tests keep working.
    detail = {"rejected": bool(out.get("rejected")),
              "delta": out.get("delta"),
              "score_before": out.get("score_before"),
              "score_after": out.get("score_after"),
              "similarity": out.get("similarity"),
              "new_facts": list(out.get("reasons") or out.get("new_facts") or [])}
    yield Event("rewrite", "done", detail, time.time() - started, candidate)
    return out


def rewrite_v2_stage(cv_text, job_title, candidate_name, generate):
    """Rewriter v2's two agents, for candidate mode.

    v2 runs both agents itself and returns when both are done, so unlike
    council_stage this cannot report per-agent progress without reaching
    inside it. One start event and one done event naming both agents is the
    honest granularity - a fake per-agent row that jumped from 0 to 2 would
    tell the user less than nothing.

    Both agents failing is not an exception here: v2 records the failure on
    each result's .error and returns them anyway, and the page shows that
    rather than a blank screen.

    A cloud run whose cloud agents failed says which local models stood in
    for them, so the board does not show a cloud run that quietly went local.
    Left out when nothing was replaced, which is every local run.
    """
    yield Event("rewrite_v2", "start", {"job_title": job_title},
                candidate=candidate_name)
    started = time.time()
    results = generate(cv_text, job_title, candidate_name)
    detail = {"agents": [r.agent_model for r in results],
              "fallbacks": [r.agent_model for r in results if r.used_fallback],
              "errors": [r.error for r in results if getattr(r, "error", None)]}
    replaced = ["{} (for {})".format(r.agent_model, r.replaced_cloud_model)
                for r in results if getattr(r, "replaced_cloud_model", None)]
    if replaced:
        detail["ran_locally_instead"] = replaced
    yield Event("rewrite_v2", "done", detail, time.time() - started,
                candidate_name)
    return results


def feedback_stage(results, job_text, job_title, candidate_name, feedback):
    """rewriter_v2_judge on the finished rewrites: which to send, a career
    roadmap, and other careers the CV suits.

    Reported rather than raised when it fails, on the same principle as a
    lost council model: the rewrites it comments on are already done, and a
    judge that timed out must not cost the candidate those.
    """
    yield Event("feedback", "start", {}, candidate=candidate_name)
    started = time.time()
    try:
        parsed, error = feedback(results, job_text, job_title, candidate_name)
    except Exception as exc:
        parsed, error = None, exc
    if error:
        yield Event("feedback", "failed", {"error": str(error)},
                    time.time() - started, candidate_name)
        return {"error": str(error)}
    yield Event("feedback", "done",
                {"recommended": parsed.get("recommended_rewrite")},
                time.time() - started, candidate_name)
    return parsed
