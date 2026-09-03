"""Tests for the HTML comparison report.

The report is the demo artefact: job listing, the CV as the parser read it, and
the rewritten CV side by side, with the council and judge above them. It is
generated from data already in memory, so none of this needs a model.
"""

import json

import pytest

import run_demo


# --- capturing the rewritten text -------------------------------------------

def test_capture_records_what_the_paraphraser_returned():
    """probe() takes paraphrase_fn, so wrapping it captures the scored text
    without changing rewriter_v1 or the shape of its research rows."""
    capture = run_demo.capture_paraphrase(lambda cv, title, **kw: "REWRITTEN")

    out = capture("ORIGINAL", "Some Job")

    assert out == "REWRITTEN"
    assert capture.text == "REWRITTEN"


def test_capture_keeps_the_last_attempt_when_the_guard_forces_a_retry():
    """probe calls paraphrase_fn twice on rejection; the second is what gets
    scored, so that is the one the report must show."""
    attempts = iter(["FIRST", "SECOND"])
    capture = run_demo.capture_paraphrase(lambda cv, title, **kw: next(attempts))

    capture("ORIGINAL", "Job")
    capture("ORIGINAL", "Job", strict=True)

    assert capture.text == "SECOND"


def test_capture_passes_keyword_arguments_through():
    seen = {}
    def fake(cv, title, **kw):
        seen.update(kw)
        return "X"
    run_demo.capture_paraphrase(fake)("CV", "Job", model="m", strict=True)

    assert seen == {"model": "m", "strict": True}


# --- word diff --------------------------------------------------------------

def test_word_diff_marks_added_words_in_the_rewrite():
    _, after = run_demo.word_diff("managed a team", "expertly managed a team")
    assert "expertly" in after
    assert "<ins>" in after


def test_word_diff_marks_removed_words_in_the_original():
    before, _ = run_demo.word_diff("managed a large team", "managed a team")
    assert "<del>" in before
    assert "large" in before


def test_word_diff_leaves_identical_text_unmarked():
    before, after = run_demo.word_diff("same words here", "same words here")
    assert "<ins>" not in after and "<del>" not in before


def test_word_diff_escapes_html_in_the_source_text():
    """A CV containing angle brackets must not inject markup into the report."""
    before, after = run_demo.word_diff("<script>alert(1)</script>", "safe")
    assert "<script>" not in before
    assert "&lt;script&gt;" in before


# --- the report -------------------------------------------------------------

def _payload(**over):
    base = {
        "job": "job_listing_secretary",
        "candidate": "ALEX PARK",
        "job_text": "Must have 3 years experience.",
        "cv_text": "I managed a team.",
        "sections": ["contact", "experience"],
        "council": [{"model": "qwen2.5:14b", "score": 3, "summary": "thin"},
                    {"model": "phi4", "score": 6, "summary": "decent"}],
        "judge": {"final_score": 5, "final_evaluation": "borderline"},
    }
    base.update(over)
    return base


def test_report_contains_the_candidate_job_and_every_council_score():
    html = run_demo.render_report(_payload())
    assert "ALEX PARK" in html
    for fragment in ("qwen2.5:14b", "phi4", ">3<", ">6<", "5"):
        assert fragment in html, fragment


def test_report_shows_the_judge_verdict():
    assert "borderline" in run_demo.render_report(_payload())


def test_report_shows_the_rewrite_panes_when_a_probe_ran():
    html = run_demo.render_report(_payload(
        rewrite={"score_before": 3, "score_after": 4, "delta": 1, "rejected": False},
        rewritten_text="I expertly managed a team."))
    assert "expertly" in html
    assert "Rewritten" in html


def test_report_omits_the_rewrite_panes_when_no_probe_ran():
    html = run_demo.render_report(_payload())
    assert "Rewritten" not in html


def test_report_surfaces_a_guard_rejection_as_a_finding_not_an_error():
    html = run_demo.render_report(_payload(
        rewrite={"score_before": 3, "score_after": None, "delta": None,
                 "rejected": True, "reasons": ["bsc"]},
        rewritten_text="I have a BSc."))
    assert "rejected" in html.lower()
    assert "bsc" in html.lower()


def test_report_escapes_the_job_and_cv_text():
    html = run_demo.render_report(_payload(job_text="<b>bold</b>",
                                           cv_text="a & b"))
    assert "<b>bold</b>" not in html
    assert "&amp;" in html


def test_report_survives_a_run_with_a_failed_council_model():
    """score None must render, not crash the page."""
    html = run_demo.render_report(_payload(
        council=[{"model": "broken", "score": None, "summary": "MODEL ERROR"}]))
    assert "broken" in html


def test_write_report_puts_html_beside_the_json(tmp_path):
    path = run_demo.write_report(tmp_path / "run.json", _payload())
    assert path.name == "run.html"
    assert path.exists()
    assert path.read_text(encoding="utf-8").lstrip().startswith("<!doctype html>")


# --- surfacing a no-op rewrite ----------------------------------------------

def test_report_shows_the_similarity_when_a_probe_ran():
    html = run_demo.render_report(_payload(
        rewrite={"score_before": 3, "score_after": 4, "delta": 1,
                 "rejected": False, "similarity": 0.42},
        rewritten_text="I expertly led a team."))
    assert "42%" in html


def test_report_warns_when_the_rewrite_barely_changed_anything():
    """delta 0 on a 99%-identical rewrite is not evidence about phrasing - it
    means nothing was rephrased, and the page has to say so."""
    html = run_demo.render_report(_payload(
        rewrite={"score_before": 3, "score_after": 3, "delta": 0,
                 "rejected": False, "similarity": 0.991},
        rewritten_text="I managed a team."))
    assert "no-op" in html.lower() or "barely" in html.lower()


def test_report_does_not_warn_on_a_genuine_rewrite():
    html = run_demo.render_report(_payload(
        rewrite={"score_before": 3, "score_after": 3, "delta": 0,
                 "rejected": False, "similarity": 0.30},
        rewritten_text="Entirely different wording throughout."))
    assert "no-op" not in html.lower()


def test_report_handles_an_older_run_with_no_similarity_recorded():
    """Runs saved before similarity existed must still render."""
    html = run_demo.render_report(_payload(
        rewrite={"score_before": 3, "score_after": 4, "delta": 1, "rejected": False},
        rewritten_text="something"))
    assert "Rewritten" in html


def test_report_falls_back_visibly_rather_than_silently(tmp_path):
    """Regression: run_evaluation passed the payload before save_run merged the
    job and candidate into it, so every generated page was headed "? - ?".
    The tests passed because the fixture supplied both keys."""
    payload = {"council": [], "judge": {}}
    html = run_demo.render_report(payload)
    assert "?" in html          # renders, does not raise
    # and the real flow must not rely on that fallback:
    assert "<h1>?</h1>" in html


def test_run_payload_carries_job_and_candidate_for_the_report():
    """What save_run persists and what write_report renders must be one dict."""
    payload = run_demo.new_payload("job_listing_secretary", "ALEX PARK")
    assert payload["job"] == "job_listing_secretary"
    assert payload["candidate"] == "ALEX PARK"
    assert "<h1>ALEX PARK</h1>" in run_demo.render_report(payload)
