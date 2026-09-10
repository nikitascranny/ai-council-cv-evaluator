"""Candidate mode's seam onto rewriter v2.

v2 exists now. These pin the contract the UI depends on: two results back,
untouched, and an honest failure when the rewriter cannot be reached.
"""

import pytest

from ui.pipeline import candidate


def _result(**kw):
    from types import SimpleNamespace
    base = dict(candidate="Jane Doe", job_title="Secretary", agent_model="phi4",
                original_text="orig", rewritten_text="new", used_fallback=False,
                rejected=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_backend_gets_the_cv_text_the_job_text_and_the_title():
    seen = {}

    def fake(cv_text, job_text, job_title, candidate_name):
        seen["args"] = (cv_text, job_text, job_title, candidate_name)
        return [_result()]

    candidate.generate("cv body", "job body", "Secretary", "Jane Doe", backend=fake)
    assert seen["args"] == ("cv body", "job body", "Secretary", "Jane Doe")


def test_both_agents_results_come_back_untouched():
    """The page and the .docx both key off the guard flags on each result,
    so this seam must not flatten or summarise them."""
    a, b = _result(agent_model="phi4"), _result(agent_model="qwen2.5:14b",
                                                used_fallback=True)
    got = candidate.generate("cv", "job", "Secretary", backend=lambda *a_: [a, b])
    assert got == [a, b]
    assert got[1].used_fallback is True


def test_no_rewrites_is_an_error_not_a_silent_success():
    with pytest.raises(candidate.NotAvailable):
        candidate.generate("cv", "job", "Secretary", backend=lambda *a: [])


def test_an_unreachable_rewriter_names_v2_so_the_ui_can_say_what_is_missing(monkeypatch):
    def boom():
        raise candidate.NotAvailable("rewriter v2 could not be loaded (no module)")
    monkeypatch.setattr(candidate, "_load_backend", boom)
    with pytest.raises(candidate.NotAvailable, match="rewriter v2"):
        candidate.generate("cv", "job", "Secretary")


def test_the_module_backend_is_used_when_none_is_passed(monkeypatch):
    monkeypatch.setattr(candidate, "BACKEND", lambda *a: [_result()])
    assert len(candidate.generate("cv", "job", "Secretary")) == 1


# --- what the guard's verdict means in plain words --------------------------


def test_a_dropped_number_is_not_reported_as_an_invention():
    """The two are opposites. Telling someone their rewrite "added
    information" when the guard's complaint was that it dropped half their
    numbers points them at the wrong problem entirely - and truncation makes
    dropped the common case, not the rare one."""
    said = candidate.describe_rejection(["number dropped from the CV: 2015"])
    assert "left out" in said
    assert "added" not in said


def test_an_invented_fact_is_reported_as_one():
    said = candidate.describe_rejection(["phd"])
    assert "added information" in said
    assert "left out" not in said


def test_a_borrowed_job_title_is_named_as_its_own_failure():
    said = candidate.describe_rejection(
        ["job title claimed but not held: Web Dev Manager"])
    assert "job title" in said


def test_several_kinds_are_all_reported():
    said = candidate.describe_rejection(
        ["phd", "number dropped from the CV: 4",
         "job title claimed but not held: Web Dev Manager"])
    assert "added information" in said and "left out" in said
    assert "job title" in said


def test_no_reasons_does_not_invent_a_cause():
    """A fallback with an empty reasons list must not assert why."""
    said = candidate.describe_rejection([])
    assert "could not confirm" in said


def test_classification_does_not_repeat_a_kind():
    kinds = candidate.classify_reasons(
        ["number dropped from the CV: 4", "number dropped from the CV: 7"])
    assert kinds == [candidate.DROPPED]
