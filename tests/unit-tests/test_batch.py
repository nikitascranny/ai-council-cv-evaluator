"""Several candidates against one listing, ranked as they land.

Batch runs are minutes long on local models, so the ordering is emitted after
every candidate rather than at the end - a reviewer watching should see the
shortlist forming.
"""

import types

from ui.pipeline.backends import Settings
from ui.pipeline.deps import Deps
from ui.pipeline import evaluation


def _fake_cv():
    cv = types.SimpleNamespace()
    cv.sections = [types.SimpleNamespace(label="experience")]
    cv.multi_column_pages = []
    cv.warnings = []
    cv.full_text = lambda: "cv text"
    cv.sections_named = lambda *n: "scoped"
    return cv


def _deps(scores):
    """scores maps candidate name -> judge final_score."""
    return Deps(load_cv=lambda p: _fake_cv(), extract_text=lambda p: "job",
                evaluate_cv=lambda *a: {"score": 5, "summary": "ok"},
                check_essential=lambda *a, **k: {},
                judge=lambda job, rows, candidate, model, essential=None: (
                    {"final_score": scores[candidate]}, None),
                probe=lambda *a, **k: {}, make_scorer=lambda m: None,
                paraphrase=lambda *a, **k: "")


def _drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def test_ranking_orders_by_the_judges_final_score(tmp_path):
    events, payloads = _drain(evaluation.run_batch(
        tmp_path / "job_listing_secretary.pdf",
        [tmp_path / "ALEX.pdf", tmp_path / "BEN.pdf", tmp_path / "CARA.pdf"],
        Settings(council=["phi4"]),
        _deps({"ALEX": 4, "BEN": 9, "CARA": 6}), out_dir=tmp_path))
    final = [e for e in events if e.stage == "rank" and e.state == "done"][-1]
    assert [r["candidate"] for r in final.detail["ranking"]] == ["BEN", "CARA", "ALEX"]
    assert len(payloads) == 3


def test_the_ranking_is_emitted_after_every_candidate(tmp_path):
    """A reviewer watching a ten-minute run should see the shortlist forming."""
    events, _ = _drain(evaluation.run_batch(
        tmp_path / "job_listing_secretary.pdf",
        [tmp_path / "ALEX.pdf", tmp_path / "BEN.pdf"],
        Settings(council=["phi4"]), _deps({"ALEX": 4, "BEN": 9}),
        out_dir=tmp_path))
    progress = [e for e in events if e.stage == "rank" and e.state == "progress"]
    assert len(progress) == 2
    assert [r["candidate"] for r in progress[0].detail["ranking"]] == ["ALEX"]


def test_a_candidate_without_a_judge_score_ranks_last_not_first():
    """None must not sort above a real score."""
    ranking = evaluation.rank([
        {"candidate": "ALEX", "judge": {"final_score": None}},
        {"candidate": "BEN", "judge": {"final_score": 3}}])
    assert [r["candidate"] for r in ranking] == ["BEN", "ALEX"]


def test_ranking_carries_the_council_rows_for_the_table():
    ranking = evaluation.rank([
        {"candidate": "ALEX", "judge": {"final_score": 7},
         "council": [{"model": "phi4", "score": 6}]}])
    assert ranking[0]["council"] == [{"model": "phi4", "score": 6}]
    assert ranking[0]["final_score"] == 7


def test_one_candidate_failing_does_not_end_the_batch(tmp_path):
    """A CV that will not parse costs its own row, like a lost model does."""
    def load_cv(path):
        if "ALEX" in str(path):
            raise RuntimeError("not a PDF")
        return _fake_cv()

    deps = _deps({"BEN": 9})
    deps.load_cv = load_cv
    events, payloads = _drain(evaluation.run_batch(
        tmp_path / "job_listing_secretary.pdf",
        [tmp_path / "ALEX.pdf", tmp_path / "BEN.pdf"],
        Settings(council=["phi4"]), deps, out_dir=tmp_path))
    failed = [e for e in events if e.stage == "run" and e.state == "failed"]
    assert failed[0].candidate == "ALEX"
    assert "not a PDF" in failed[0].detail["error"]
    assert [p["candidate"] for p in payloads] == ["BEN"]
