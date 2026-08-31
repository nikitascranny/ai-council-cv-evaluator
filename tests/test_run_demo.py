"""Tests for the demo runner.

The runner's job is orchestration, so what is worth testing is the part that
runs without a model: finding the corpus, the menu, and the order stages run
in when earlier ones fail. Every model call is injected, so nothing here needs
Ollama or the vendored integration/ code.
"""

import json
import sys
from pathlib import Path

import pytest

import run_demo

ROOT = Path(__file__).resolve().parent.parent


# --- corpus discovery -------------------------------------------------------

def test_discover_finds_cvs_across_batch1_and_the_later_batches(tmp_path):
    (tmp_path / "Batch1" / "CVs").mkdir(parents=True)
    (tmp_path / "Batch1" / "CVs" / "ALEX PARK.pdf").touch()
    (tmp_path / "Batch1" / "batch2-4" / "batch3").mkdir(parents=True)
    (tmp_path / "Batch1" / "batch2-4" / "batch3" / "bad_cv_kevin_jones.pdf").touch()

    found = run_demo.discover_cvs(tmp_path)

    assert [p.name for p in found] == ["ALEX PARK.pdf", "bad_cv_kevin_jones.pdf"]


def test_discover_ignores_non_pdfs_and_readmes(tmp_path):
    d = tmp_path / "Batch1" / "batch2-4" / "batch2"
    d.mkdir(parents=True)
    (d / "beginner_sa.pdf").touch()
    (d / "Batch2_README.txt").touch()

    assert [p.name for p in run_demo.discover_cvs(tmp_path)] == ["beginner_sa.pdf"]


def test_discover_returns_empty_when_nothing_is_there(tmp_path):
    assert run_demo.discover_cvs(tmp_path) == []
    assert run_demo.discover_jobs(tmp_path) == []


def test_label_shows_the_containing_batch_so_duplicates_are_distinguishable(tmp_path):
    """batch3 has bad_ and professional_ CVs for the same people - the menu has
    to say which batch a file came from or the two are indistinguishable."""
    p = tmp_path / "Batch1" / "batch2-4" / "batch3" / "bad_cv_kevin_jones.pdf"
    p.parent.mkdir(parents=True)
    p.touch()
    assert run_demo.label_for(p, tmp_path) == "batch3/bad_cv_kevin_jones"


# --- the menu ---------------------------------------------------------------

def test_choose_returns_the_selected_option():
    options = ["a", "b", "c"]
    assert run_demo.choose("pick", options, reader=lambda _: "2") == "b"


def test_choose_reprompts_on_a_non_number_then_accepts():
    answers = iter(["banana", "1"])
    assert run_demo.choose("pick", ["a", "b"], reader=lambda _: next(answers)) == "a"


def test_choose_reprompts_when_out_of_range():
    answers = iter(["9", "0", "2"])
    assert run_demo.choose("pick", ["a", "b"], reader=lambda _: next(answers)) == "b"


def test_choose_returns_none_on_quit():
    assert run_demo.choose("pick", ["a"], reader=lambda _: "q") is None


def test_choose_returns_none_immediately_when_there_is_nothing_to_pick():
    """No prompt at all - reader raises if called, proving we never ask."""
    def reader(_):
        raise AssertionError("should not prompt with an empty list")
    assert run_demo.choose("pick", [], reader=reader) is None


# --- stage orchestration ----------------------------------------------------

def _fake_parsed(text="CV BODY", sections=("skills", "experience")):
    class FakeSection:
        def __init__(self, label):
            self.label = label
            self.heading = label.title()
            self.raw_heading = ""
            self.nested_headings = []
        def text(self):
            return f"{self.label} content"
    class FakeCV:
        def __init__(self):
            self.sections = [FakeSection(s) for s in sections]
            self.multi_column_pages = [0]
            self.warnings = []
        def full_text(self):
            return text
        def sections_named(self, *labels):
            return " ".join(s.text() for s in self.sections if s.label in labels)
    return FakeCV()


def test_council_stage_collects_one_row_per_model():
    calls = []
    def fake_eval(job, cv, name, model):
        calls.append(model)
        return {"candidate": name, "score": 7, "strengths": [], "weaknesses": [],
                "summary": f"from {model}"}

    rows = run_demo.council_stage("JOB", "CV", "ALEX", ["m1", "m2"], evaluate=fake_eval)

    assert calls == ["m1", "m2"]
    assert [r["summary"] for r in rows] == ["from m1", "from m2"]


def test_council_stage_keeps_going_when_one_model_raises():
    """A model that dies mid-demo must not take the run with it."""
    def fake_eval(job, cv, name, model):
        if model == "broken":
            raise RuntimeError("connection refused")
        return {"candidate": name, "score": 5, "strengths": [], "weaknesses": [],
                "summary": "ok"}

    rows = run_demo.council_stage("JOB", "CV", "ALEX", ["broken", "good"], evaluate=fake_eval)

    assert len(rows) == 2
    assert rows[0]["score"] is None
    assert "connection refused" in rows[0]["summary"]
    assert rows[1]["score"] == 5


def test_rewrite_stage_reports_the_delta_and_the_guard_verdict():
    def fake_probe(cv_text, job_title, scorer):
        return {"score_before": 6, "score_after": 8, "delta": 2,
                "rejected": False, "new_facts": []}

    out = run_demo.rewrite_stage("CV", "Fullstack Engineer", scorer=object(),
                                 probe=fake_probe)

    assert out["delta"] == 2
    assert out["rejected"] is False


def test_rewrite_stage_surfaces_a_rejected_rewrite_rather_than_a_score():
    """A guard rejection is the finding, not a failure to hide."""
    def fake_probe(cv_text, job_title, scorer):
        return {"score_before": 6, "score_after": None, "delta": None,
                "rejected": True, "new_facts": ["bsc"]}

    out = run_demo.rewrite_stage("CV", "Fullstack Engineer", scorer=object(),
                                 probe=fake_probe)

    assert out["rejected"] is True
    assert out["new_facts"] == ["bsc"]


def test_result_filename_is_safe_and_names_both_sides():
    name = run_demo.result_filename("job listing: secretary", "NDLOVU KARABO REFILWE")
    assert name == "job_listing_secretary__NDLOVU_KARABO_REFILWE.json"
    assert not set(name) & set(r'<>:"/\|?*')


def test_save_run_writes_readable_json(tmp_path):
    path = run_demo.save_run(tmp_path, "job", "cand",
                             {"council": [], "judge": {"final_score": 6}})
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["judge"]["final_score"] == 6
    assert data["job"] == "job"
    assert data["candidate"] == "cand"


def test_judge_stage_returns_the_parsed_verdict():
    def fake_judge(job_text, rows, candidate, model):
        return {"candidate": candidate, "final_score": 7,
                "final_evaluation": "solid fit"}, None

    out = run_demo.judge_stage("JOB", [], "ALEX", "m", judge_fn=fake_judge)

    assert out["final_score"] == 7


def test_judge_stage_reports_a_failure_without_raising():
    """The judge is the last stage - a failure there must still leave a row."""
    def fake_judge(job_text, rows, candidate, model):
        return None, "PARSE ERROR"

    out = run_demo.judge_stage("JOB", [], "ALEX", "m", judge_fn=fake_judge)

    assert out["final_score"] is None
    assert out["error"] == "PARSE ERROR"
    assert out["candidate"] == "ALEX"


def test_title_from_slug_drops_the_job_listing_prefix():
    assert run_demo.title_from_slug(
        "job_listing_startup_fullstack_engineer") == "Startup Fullstack Engineer"


def test_parse_stage_returns_the_parsed_cv_and_survives_warnings(capsys):
    cv = _fake_parsed()
    cv.warnings = ["page 2 yielded no text"]
    out = run_demo.parse_stage("x.pdf", load_cv=lambda p: cv)
    assert out is cv
    assert "page 2 yielded no text" in capsys.readouterr().out


# --- preflight --------------------------------------------------------------

def test_available_models_parses_the_ollama_tag_list():
    payload = {"models": [{"name": "qwen2.5:14b"}, {"name": "phi4:latest"}]}
    got = run_demo.available_models(fetch=lambda: payload)
    # phi4:latest must also answer to "phi4" - that is how it is referenced.
    assert "qwen2.5:14b" in got
    assert "phi4" in got and "phi4:latest" in got


def test_available_models_returns_empty_when_ollama_is_unreachable():
    """No Ollama is a warning at startup, not a crash."""
    def fetch():
        raise OSError("connection refused")
    assert run_demo.available_models(fetch=fetch) == set()


def test_missing_models_lists_only_what_is_absent():
    available = {"qwen2.5:14b", "phi4"}
    assert run_demo.missing_models(["qwen2.5:14b", "phi4"], available) == []
    assert run_demo.missing_models(
        ["qwen2.5:14b", "deepseek-r1:8b"], available) == ["deepseek-r1:8b"]


def test_missing_models_says_nothing_when_ollama_is_unreachable():
    """An empty availability set means we could not check, not that all are gone -
    reporting every model as missing would be a false alarm."""
    assert run_demo.missing_models(["a", "b"], set()) == []
