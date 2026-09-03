"""Tests for the persistent console app layer.

The loop, the settings and the results browser are the new parts, and none of
them need a model. Actions are injected so a menu choice can be dispatched and
checked without running an evaluation.
"""

import json

import pytest

import run_demo


# --- settings ---------------------------------------------------------------

def test_settings_default_to_the_teams_real_models():
    s = run_demo.Settings()
    assert s.council == run_demo.COUNCIL_MODELS
    assert s.judge == run_demo.JUDGE_MODEL
    assert s.essential is False and s.rewrite is False


def test_fast_mode_swaps_every_model_not_just_the_council():
    """Half-swapping leaves a 14b judge behind a 3b council and the run crawls."""
    s = run_demo.Settings(fast=True)
    assert s.council == run_demo.FAST_MODELS["council"]
    assert s.judge == run_demo.FAST_MODELS["judge"]
    assert s.essential_model == run_demo.FAST_MODELS["essential"]


def test_toggling_fast_back_restores_the_real_models():
    s = run_demo.Settings(fast=True)
    s.fast = False
    assert s.council == run_demo.COUNCIL_MODELS
    assert s.judge == run_demo.JUDGE_MODEL


def test_explicit_council_overrides_fast_mode():
    """An explicit choice is the user's, and must not be undone by --fast."""
    s = run_demo.Settings(fast=True, council=["phi4"])
    assert s.council == ["phi4"]


def test_settings_summary_names_the_models_and_the_active_stages():
    s = run_demo.Settings(fast=True, essential=True)
    text = s.summary()
    assert "qwen2.5:7b" in text
    assert "essential" in text.lower()


# --- saved results ----------------------------------------------------------

def test_saved_runs_lists_newest_first(tmp_path):
    import os, time
    older = tmp_path / "a__x.json"
    newer = tmp_path / "b__y.json"
    older.write_text("{}", encoding="utf-8")
    time.sleep(0.01)
    newer.write_text("{}", encoding="utf-8")
    os.utime(older, (1, 1))

    assert [p.name for p in run_demo.saved_runs(tmp_path)] == ["b__y.json", "a__x.json"]


def test_saved_runs_is_empty_when_nothing_has_been_run(tmp_path):
    assert run_demo.saved_runs(tmp_path / "nope") == []


def test_describe_run_summarises_without_reprinting_everything(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "job": "job_listing_secretary", "candidate": "ALEX PARK",
        "council": [{"model": "phi4", "score": 6}, {"model": "qwen2.5:14b", "score": 3}],
        "judge": {"final_score": 5},
    }), encoding="utf-8")

    line = run_demo.describe_run(path)

    assert "ALEX PARK" in line
    assert "5" in line


def test_describe_run_survives_a_file_that_is_not_a_run(tmp_path):
    """A stray or half-written file must not crash the browser."""
    path = tmp_path / "junk.json"
    path.write_text("not json at all", encoding="utf-8")
    assert isinstance(run_demo.describe_run(path), str)


# --- the menu loop ----------------------------------------------------------

def test_loop_dispatches_the_chosen_action_then_quits():
    called = []
    actions = {"1": ("Do a thing", lambda settings: called.append("ran"))}
    answers = iter(["1", "q"])

    run_demo.app_loop(run_demo.Settings(), actions=actions,
                      reader=lambda _: next(answers))

    assert called == ["ran"]


def test_loop_reprompts_on_an_unknown_choice():
    called = []
    actions = {"1": ("Do a thing", lambda settings: called.append("ran"))}
    answers = iter(["banana", "1", "q"])

    run_demo.app_loop(run_demo.Settings(), actions=actions,
                      reader=lambda _: next(answers))

    assert called == ["ran"]


def test_loop_keeps_running_when_an_action_raises():
    """A failed model call must return to the menu, not kill the app mid-demo."""
    def boom(settings):
        raise RuntimeError("ollama died")
    actions = {"1": ("Explode", boom)}
    answers = iter(["1", "q"])

    run_demo.app_loop(run_demo.Settings(), actions=actions,
                      reader=lambda _: next(answers))  # must not raise


def test_loop_exits_on_eof_rather_than_looping_forever():
    """Piped input runs out; the app must exit instead of spinning."""
    def reader(_):
        raise EOFError
    run_demo.app_loop(run_demo.Settings(), actions={}, reader=reader)
