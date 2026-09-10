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
