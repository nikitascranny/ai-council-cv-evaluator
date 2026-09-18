"""Each stage, as an event stream.

These are the console app's stage functions with print() replaced by yield.
The behaviour they must keep: a model that dies costs its own row and not the
run, and a rejected rewrite is a result rather than an error.
"""

import types

import pytest

from ui.pipeline import stages
from ui.pipeline.deps import Deps
from ui.pipeline.stages import (capture_paraphrase, council_stage, essential_stage,
                         judge_stage, parse_stage, rewrite_stage)


def _fake_cv(sections=("experience",), text="cv text"):
    cv = types.SimpleNamespace()
    cv.sections = [types.SimpleNamespace(label=s) for s in sections]
    cv.multi_column_pages = []
    cv.warnings = []
    cv.full_text = lambda: text
    cv.sections_named = lambda *names: "scoped"
    return cv


def _deps(**overrides):
    base = dict(load_cv=lambda p: _fake_cv(), extract_text=lambda p: "job",
                evaluate_cv=lambda *a: {"score": 7, "summary": "ok"},
                check_essential=lambda *a, **k: {"summary": "meets"},
                judge=lambda *a: ({"final_score": 8,
                                   "final_evaluation": "good"}, None),
                probe=lambda *a, **k: {"delta": 2, "score_before": 5,
                                       "score_after": 7},
                make_scorer=lambda m: (lambda t: 0), paraphrase=lambda *a, **k: "")
    base.update(overrides)
    return Deps(**base)


def _drain(gen):
    """Run a stage generator to completion, returning (events, return value)."""
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def test_parse_reports_the_sections_it_found():
    events, cv = _drain(parse_stage("cv.pdf", _deps(load_cv=lambda p: _fake_cv(
        sections=("experience", "education")))))
    done = [e for e in events if e.state == "done"]
    assert done[0].detail["sections"] == ["experience", "education"]
    assert cv.full_text() == "cv text"


def test_parse_surfaces_parser_warnings_without_failing():
    cv = _fake_cv()
    cv.warnings = ["page 2 looked multi-column"]
    events, _ = _drain(parse_stage("cv.pdf", _deps(load_cv=lambda p: cv)))
    done = [e for e in events if e.state == "done"][0]
    assert done.detail["warnings"] == ["page 2 looked multi-column"]


def test_council_emits_a_start_and_done_per_model():
    events, rows = _drain(council_stage("job", "cv", "ALEX",
                                        ["phi4", "qwen"], _deps()))
    assert [(e.state, e.detail.get("model")) for e in events] == [
        ("start", "phi4"), ("done", "phi4"), ("start", "qwen"), ("done", "qwen")]
    assert [r["model"] for r in rows] == ["phi4", "qwen"]


def test_a_failing_model_costs_its_own_row_not_the_run():
    """One model dying mid-demo must not end the evaluation."""
    def flaky(job, cv, candidate, model):
        if model == "phi4":
            raise RuntimeError("out of memory")
        return {"score": 6, "summary": "fine"}

    events, rows = _drain(council_stage("job", "cv", "ALEX", ["phi4", "qwen"],
                                        _deps(evaluate_cv=flaky)))
    failed = [e for e in events if e.state == "failed"]
    assert failed[0].detail["model"] == "phi4"
    assert "out of memory" in failed[0].detail["error"]
    assert len(rows) == 2 and rows[0]["score"] is None and rows[1]["score"] == 6


def test_essential_stage_scopes_the_cv_rather_than_sending_all_of_it():
    """The point of sections_named() is that the check reads three sections."""
    seen = {}

    def check(job_text, scoped, candidate, model):
        seen["scoped"] = scoped
        return {"summary": "meets"}

    _drain(essential_stage("job", _fake_cv(), "ALEX", "qwen",
                           _deps(check_essential=check)))
    assert seen["scoped"] == "scoped"


def test_essential_stage_scopes_the_job_listing_to_its_requirements_section():
    """job_req_parser should hand the essential check just the requirements,
    not the whole listing - the same principle already applied to the CV."""
    seen = {}

    def check(job_text, scoped, candidate, model):
        seen["job_text"] = job_text
        return {"summary": "meets"}

    listing = ("About the role\nWe build great things.\n\n"
               "Essential Requirements\n- Own transport\n- 2 years experience\n\n"
               "Benefits\nMedical aid.")
    _drain(essential_stage(listing, _fake_cv(), "ALEX", "qwen",
                           _deps(check_essential=check)))
    assert "Own transport" in seen["job_text"]
    assert "2 years experience" in seen["job_text"]
    assert "great things" not in seen["job_text"]
    assert "Medical aid" not in seen["job_text"]


def test_essential_stage_sends_the_whole_listing_when_no_requirements_heading_is_found():
    """A listing job_req_parser can't segment must still reach the model
    whole, not as an empty string."""
    seen = {}

    def check(job_text, scoped, candidate, model):
        seen["job_text"] = job_text
        return {"summary": "meets"}

    listing = "Just some prose about the role, with no recognised heading at all."
    _drain(essential_stage(listing, _fake_cv(), "ALEX", "qwen",
                           _deps(check_essential=check)))
    assert seen["job_text"] == listing


def test_a_rewrite_with_no_matching_headings_falls_back_to_the_whole_text():
    """Plain prose that segment_text cannot resolve to any of the three
    wanted sections must still reach the model - as the whole rewrite,
    exactly what ran before txt_cv_parser was wired in."""
    seen = {}

    def check(job_text, text, candidate, model):
        seen["text"] = text
        return {"summary": "meets"}

    _drain(essential_stage("job", "the whole rewrite", "ALEX", "qwen",
                           _deps(check_essential=check)))
    assert seen["text"] == "the whole rewrite"


def test_a_rewrite_with_headings_is_scoped_the_same_as_a_parsed_cv():
    """A rewrite whose headings DO resolve through the lexicon should be
    scoped to those sections, not sent whole - the whole point of wiring
    txt_cv_parser into the essential check."""
    seen = {}
    rewrite = ("Profile\nA short summary paragraph.\n\n"
               "Experience\nFive years as a lab technician.\n\n"
               "Education\nBSc Marine Biology.\n\n"
               "References\nAvailable on request.")

    def check(job_text, text, candidate, model):
        seen["text"] = text
        return {"summary": "meets"}

    _drain(essential_stage("job", rewrite, "ALEX", "qwen",
                           _deps(check_essential=check)))
    assert "lab technician" in seen["text"]
    assert "BSc Marine Biology" in seen["text"]
    assert "short summary paragraph" not in seen["text"]
    assert "Available on request" not in seen["text"]


def test_judge_failure_is_reported_as_an_event_and_a_none_score():
    events, verdict = _drain(judge_stage(
        "job", [], "ALEX", "deepseek", _deps(judge=lambda *a: (None, "timed out"))))
    assert [e for e in events if e.state == "failed"]
    assert verdict["final_score"] is None and "timed out" in verdict["error"]


def test_judge_stage_hands_the_essential_result_to_the_judge():
    seen = {}

    def judge(job_text, rows, candidate, model, essential=None):
        seen["essential"] = essential
        return {"final_score": 7}, None

    events, _ = _drain(judge_stage("job", [], "ALEX", "deepseek", _deps(judge=judge),
                                   essential={"summary": "meets"}))
    assert seen["essential"] == {"summary": "meets"}
    assert events[-1].detail["with_essential"] is True


def test_judge_stage_without_the_check_says_the_judge_had_the_council_alone():
    events, _ = _drain(judge_stage("job", [], "ALEX", "deepseek", _deps()))
    assert events[-1].detail["with_essential"] is False


@pytest.mark.parametrize("essential", [
    None, {}, {"summary": ""}, {"summary": "   "},
    {"summary": "partial", "error": "PARSE ERROR"}, {"summary": "meets"},
], ids=["none", "empty_dict", "empty_summary", "blank_summary", "failed", "usable"])
def test_essential_in_prompt_agrees_with_what_judge_py_puts_in_the_prompt(essential):
    """The rule is restated in stages rather than imported - this is what
    keeps the recorded condition honest if judge.py's rule ever changes."""
    import judge

    expected = judge.format_essential_for_prompt(essential) is not None
    assert stages.essential_in_prompt(essential) is expected


def test_rewrite_rejection_is_a_finding_not_a_failure():
    """An invented fact is a result about the rewriting model."""
    rejected = {"rejected": True, "new_facts": ["PhD"], "delta": None}
    events, out = _drain(rewrite_stage("cv", "Engineer", scorer=None,
                                       deps=_deps(probe=lambda *a, **k: rejected)))
    done = [e for e in events if e.state == "done"][0]
    assert done.detail["rejected"] is True
    assert done.detail["new_facts"] == ["PhD"]
    assert out["rejected"] is True


def test_rewrite_reports_the_delta_when_the_guard_passed():
    events, out = _drain(rewrite_stage("cv", "Engineer", scorer=None, deps=_deps()))
    done = [e for e in events if e.state == "done"][0]
    assert done.detail["delta"] == 2
    assert out["score_before"] == 5


def test_a_real_probe_result_object_still_reports_what_was_invented():
    """probe() returns a ProbeResult dataclass, not a dict, in production.

    ProbeResult has no new_facts field - the invented facts live in `reasons`.
    This pins that rewrite_stage reads reasons off a real object, not just off
    the dict-shaped fakes the other tests use.
    """
    from rewriter_v1 import ProbeResult

    result = ProbeResult(candidate="ALEX", job="job", rewriter_model="m",
                         scorer_model="m", rejected=True,
                         reasons=["invented: PhD"])
    events, out = _drain(rewrite_stage("cv", "Engineer", scorer=None,
                                       deps=_deps(probe=lambda *a, **k: result)))
    done = [e for e in events if e.state == "done"][0]
    assert done.detail["new_facts"] == ["invented: PhD"]
    assert out["rejected"] is True


def test_capture_paraphrase_proxies_last_truncated_from_the_wrapped_function():
    """I2: rewriter_v1.paraphrase records truncation on the function object
    itself, and probe() reads it back off paraphrase_fn. A plain closure
    proxies nothing, so truncated_rewrite was always None for every run
    started from the web UI - this pins that the wrapper now copies it."""
    def fake(cv, job, **kw):
        fake.last_truncated = True
        return "rewritten"
    fake.last_truncated = None

    wrapped = capture_paraphrase(fake)
    wrapped("cv", "job")

    assert wrapped.last_truncated is True


def test_capture_paraphrase_proxies_the_sibling_last_attributes_too():
    def fake(cv, job, **kw):
        fake.last_truncated = False
        fake.last_prompt_eval_count = 4096
        fake.last_prompt_chars = 9001
        return "rewritten"
    fake.last_truncated = None
    fake.last_prompt_eval_count = None
    fake.last_prompt_chars = None

    wrapped = capture_paraphrase(fake)
    wrapped("cv", "job")

    assert wrapped.last_prompt_eval_count == 4096
    assert wrapped.last_prompt_chars == 9001


def test_capture_paraphrase_keeps_the_text_that_was_actually_scored():
    """The guard retries on rejection; the last call is the scored one."""
    calls = iter(["first", "second"])
    wrapped = capture_paraphrase(lambda cv, job, **k: next(calls))
    wrapped("cv", "job")
    wrapped("cv", "job")
    assert wrapped.text == "second"


def test_rewrite_stage_passes_the_rewriter_model_through_to_probe():
    """I1: rewrite_stage once left the paraphrase on probe()'s 14b default,
    because it never passed a model through. This pins that it now does."""
    seen = {}

    def probe(cv, job, scorer, **kw):
        seen.update(kw)
        return {"delta": 0, "score_before": 5, "score_after": 5}

    _drain(rewrite_stage("cv", "Engineer", scorer=None,
                         deps=_deps(probe=probe), rewriter_model="qwen2.5:7b"))
    assert seen["rewriter_model"] == "qwen2.5:7b"


def test_rewrite_stage_omits_rewriter_model_when_not_given():
    """Callers that do not pass rewriter_model must not send probe() an
    explicit None, which would override probe()'s own default."""
    seen = {}

    def probe(cv, job, scorer, **kw):
        seen.update(kw)
        return {"delta": 0, "score_before": 5, "score_after": 5}

    _drain(rewrite_stage("cv", "Engineer", scorer=None, deps=_deps(probe=probe)))
    assert "rewriter_model" not in seen


def test_every_event_carries_the_candidate_for_batch_runs():
    events, _ = _drain(council_stage("job", "cv", "ALEX", ["phi4"], _deps()))
    assert all(e.candidate == "ALEX" for e in events)


# --- candidate mode's rewrite stage ----------------------------------------


def _rw(**kw):
    from types import SimpleNamespace
    base = dict(agent_model="phi4", used_fallback=False, error=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def test_rewrite_v2_stage_reports_both_agents_and_returns_their_results():
    results = [_rw(agent_model="phi4"), _rw(agent_model="qwen2.5:14b")]
    events, got = _drain(stages.rewrite_v2_stage(
        "cv", "Secretary", "Jane", lambda *a: results))
    assert [(e.stage, e.state) for e in events] == [
        ("rewrite_v2", "start"), ("rewrite_v2", "done")]
    assert got == results
    assert events[1].detail["agents"] == ["phi4", "qwen2.5:14b"]


def test_the_done_event_names_which_agents_fell_back():
    """The board is the only place a user sees progress, so a run where the
    guard rejected a rewrite must not look identical to one where it did not."""
    results = [_rw(agent_model="phi4"),
               _rw(agent_model="qwen2.5:14b", used_fallback=True)]
    events, _ = _drain(stages.rewrite_v2_stage(
        "cv", "Secretary", "Jane", lambda *a: results))
    assert events[1].detail["fallbacks"] == ["qwen2.5:14b"]


def test_an_agent_error_is_reported_not_raised():
    """v2 records a failed model on result.error and returns anyway, so both
    slots still reach the page. Raising here would lose the good one too."""
    results = [_rw(), _rw(agent_model="qwen2.5:14b", error="timeout")]
    events, got = _drain(stages.rewrite_v2_stage(
        "cv", "Secretary", "Jane", lambda *a: results))
    assert events[1].detail["errors"] == ["timeout"]
    assert len(got) == 2


def test_feedback_stage_reports_the_recommendation():
    events, got = _drain(stages.feedback_stage(
        [_rw()], "job", "Secretary", "Jane",
        lambda *a: ({"recommended_rewrite": "phi4"}, None)))
    assert [(e.stage, e.state) for e in events] == [
        ("feedback", "start"), ("feedback", "done")]
    assert events[1].detail["recommended"] == "phi4"
    assert got == {"recommended_rewrite": "phi4"}


def test_a_failed_rewrite_judge_costs_the_feedback_not_the_run():
    def raises(*a):
        raise RuntimeError("timed out")

    for feedback in (lambda *a: (None, "timed out"), raises):
        events, got = _drain(stages.feedback_stage(
            [_rw()], "job", "Secretary", "Jane", feedback))
        assert events[-1].state == "failed"
        assert "timed out" in got["error"]
