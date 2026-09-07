"""One candidate, end to end, with no models involved.

The payload this produces is the same dict the saved JSON and the HTML report
already consume, so the report renderer is untouched by the move to a web UI.
"""

import json
import types

from ui.pipeline.backends import Settings
from ui.pipeline.deps import Deps
from ui.pipeline import evaluation


def _fake_cv(text="cv text"):
    cv = types.SimpleNamespace()
    cv.sections = [types.SimpleNamespace(label="experience")]
    cv.multi_column_pages = []
    cv.warnings = []
    cv.full_text = lambda: text
    cv.sections_named = lambda *n: "scoped"
    return cv


def _fake_probe(cv_text, job_title, scorer, **kwargs):
    """The real probe always calls paraphrase_fn; a fake that does not would
    let run_single pass a test it should fail."""
    if "paraphrase_fn" in kwargs:
        kwargs["paraphrase_fn"](cv_text, job_title)
    return {"delta": 2, "score_before": 5, "score_after": 7, "similarity": 0.8}


def _deps(**overrides):
    base = dict(load_cv=lambda p: _fake_cv(), extract_text=lambda p: "job text",
                evaluate_cv=lambda *a: {"score": 7, "summary": "ok"},
                check_essential=lambda *a, **k: {"summary": "meets"},
                judge=lambda *a: ({"final_score": 8,
                                   "final_evaluation": "good"}, None),
                probe=_fake_probe,
                make_scorer=lambda m: (lambda t: 0),
                paraphrase=lambda cv, job, **k: "rewritten cv")
    base.update(overrides)
    return Deps(**base)


def _drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def test_a_default_run_is_parse_council_judge(tmp_path):
    events, payload = _drain(evaluation.run_single(
        tmp_path / "job_listing_secretary.pdf", tmp_path / "ALEX PARK.pdf",
        Settings(council=["phi4"]), _deps(), out_dir=tmp_path))
    stages = [e.stage for e in events if e.state == "done"]
    assert stages == ["parse", "council", "judge", "run"]
    assert payload["judge"]["final_score"] == 8


def test_optional_stages_run_only_when_enabled(tmp_path):
    settings = Settings(council=["phi4"], essential=True, rewrite=True)
    events, payload = _drain(evaluation.run_single(
        tmp_path / "job_listing_secretary.pdf", tmp_path / "ALEX PARK.pdf",
        settings, _deps(), out_dir=tmp_path))
    stages = [e.stage for e in events if e.state == "done"]
    assert "essential" in stages and "rewrite" in stages
    assert payload["essential"]["summary"] == "meets"
    assert payload["rewrite"]["delta"] == 2


def test_payload_carries_the_text_the_models_actually_read(tmp_path):
    """The report's whole point is showing the inputs, not just the scores."""
    _, payload = _drain(evaluation.run_single(
        tmp_path / "job_listing_secretary.pdf", tmp_path / "ALEX PARK.pdf",
        Settings(council=["phi4"]), _deps(), out_dir=tmp_path))
    assert payload["job_text"] == "job text"
    assert payload["cv_text"] == "cv text"
    assert payload["sections"] == ["experience"]


def test_the_rewritten_text_is_captured_for_the_diff(tmp_path):
    _, payload = _drain(evaluation.run_single(
        tmp_path / "job_listing_secretary.pdf", tmp_path / "ALEX PARK.pdf",
        Settings(council=["phi4"], rewrite=True), _deps(), out_dir=tmp_path))
    assert payload["rewritten_text"] == "rewritten cv"


def test_a_failed_paraphrase_leaves_no_rewritten_text_rather_than_inventing_one(tmp_path):
    """ui/report.py word-diffs rewritten_text against the original CV and
    presents it as the rewrite that produced the score_before -> score_after
    delta shown beside it. If probe's paraphrase call fails (its error path,
    where paraphrase_fn is never invoked), run_single must not manufacture a
    replacement by calling paraphrase again itself - that text was never
    scored, and showing it next to a delta from a different attempt would
    misreport the finding.
    """
    probe = lambda *a, **k: {"delta": 0, "score_before": 5, "score_after": 5,
                             "similarity": 1.0}
    _, payload = _drain(evaluation.run_single(
        tmp_path / "job_listing_secretary.pdf", tmp_path / "ALEX PARK.pdf",
        Settings(council=["phi4"], rewrite=True), _deps(probe=probe),
        out_dir=tmp_path))
    assert payload["rewritten_text"] is None


def test_the_job_title_passed_to_the_probe_comes_from_the_slug(tmp_path):
    seen = {}

    def probe(cv_text, job_title, scorer, **kwargs):
        seen["title"] = job_title
        if "paraphrase_fn" in kwargs:
            kwargs["paraphrase_fn"](cv_text, job_title)
        return {"delta": 0}

    _drain(evaluation.run_single(
        tmp_path / "job_listing_startup_fullstack_engineer.pdf",
        tmp_path / "ALEX PARK.pdf",
        Settings(council=["phi4"], rewrite=True), _deps(probe=probe),
        out_dir=tmp_path))
    assert seen["title"] == "Startup Fullstack Engineer"


def test_run_single_passes_the_settings_rewriter_model_to_the_probe(tmp_path):
    """I1: fast mode swapped council, judge and essential but left the probe
    on the 14b default because run_single never told it otherwise."""
    seen = {}

    def probe(cv_text, job_title, scorer, **kwargs):
        seen.update(kwargs)
        if "paraphrase_fn" in kwargs:
            kwargs["paraphrase_fn"](cv_text, job_title)
        return {"delta": 0}

    _drain(evaluation.run_single(
        tmp_path / "job_listing_secretary.pdf", tmp_path / "ALEX PARK.pdf",
        Settings(council=["phi4"], rewrite=True, fast=True), _deps(probe=probe),
        out_dir=tmp_path))
    assert seen["rewriter_model"] == "qwen2.5:7b"


def test_the_run_is_saved_and_the_final_event_names_the_file(tmp_path):
    events, payload = _drain(evaluation.run_single(
        tmp_path / "job_listing_secretary.pdf", tmp_path / "ALEX PARK.pdf",
        Settings(council=["phi4"]), _deps(), out_dir=tmp_path))
    final = events[-1]
    assert final.stage == "run" and final.state == "done"
    saved = json.loads(open(final.detail["path"], encoding="utf-8").read())
    assert saved["candidate"] == "ALEX PARK"
    assert saved["job"] == "job_listing_secretary"


def test_identity_is_in_the_payload_from_the_start():
    """save_run used to merge these in late, so every report was headed "? - ?"."""
    payload = evaluation.new_payload("job_listing_secretary", "ALEX PARK")
    assert payload == {"job": "job_listing_secretary", "candidate": "ALEX PARK"}


def test_result_filename_is_filesystem_safe():
    assert evaluation.result_filename(
        "job listing/secretary", "ALEX PARK") == "job_listing_secretary__ALEX_PARK.json"


def test_saved_runs_lists_newest_first(tmp_path):
    import os
    import time
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
        time.sleep(0.01)
    os.utime(tmp_path / "b.json", None)
    assert evaluation.saved_runs(tmp_path)[0].name == "b.json"


def test_saved_runs_on_a_missing_directory_is_simply_empty(tmp_path):
    assert evaluation.saved_runs(tmp_path / "nope") == []


def test_describe_run_summarises_candidate_job_and_scores(tmp_path):
    """Guards the saved-runs index line, which is the only place a reviewer
    sees a run summarised before opening it."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "job": "job_listing_secretary", "candidate": "ALEX PARK",
        "council": [{"model": "phi4", "score": 6}, {"model": "qwen2.5:14b", "score": 3}],
        "judge": {"final_score": 5},
    }), encoding="utf-8")

    line = evaluation.describe_run(path)

    assert "ALEX PARK" in line
    assert "job_listing_secretary" in line
    assert "5" in line
    assert "phi4 6" in line and "qwen2.5:14b 3" in line


def test_describe_run_survives_a_half_written_file(tmp_path):
    """An interrupted run must not stop the results list from displaying."""
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    assert "unreadable" in evaluation.describe_run(bad)
