"""Candidate mode's seam onto rewriter v2, which does not exist yet.

The UI must say so honestly rather than ship a stub that looks like it worked.
When v2 lands, a teammate sets BACKEND and nothing else moves.
"""

import pytest

from ui.pipeline import candidate


def test_without_a_backend_the_call_raises_not_available():
    with pytest.raises(candidate.NotAvailable):
        candidate.generate("cv.pdf", "Secretary", backend=None)


def test_the_message_names_v2_so_the_ui_can_say_what_is_missing():
    with pytest.raises(candidate.NotAvailable, match="rewriter v2"):
        candidate.generate("cv.pdf", "Secretary", backend=None)


def test_a_backend_is_called_with_the_cv_and_the_job_title():
    seen = {}

    def fake(cv_path, job_title):
        seen["args"] = (cv_path, job_title)
        return "out/improved.docx"

    result = candidate.generate("cv.pdf", "Secretary", backend=fake)
    assert seen["args"] == ("cv.pdf", "Secretary")
    assert result.path == "out/improved.docx"


def test_a_backend_returning_nothing_is_an_error_not_a_silent_success():
    with pytest.raises(candidate.NotAvailable):
        candidate.generate("cv.pdf", "Secretary", backend=lambda c, j: None)


def test_the_module_backend_is_used_when_none_is_passed(monkeypatch):
    monkeypatch.setattr(candidate, "BACKEND", lambda c, j: "out/x.docx")
    assert candidate.generate("cv.pdf", "Secretary").path == "out/x.docx"
