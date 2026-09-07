"""The event vocabulary every runner speaks.

Stage and state are validated because a typo'd stage name would render as a
silently missing row in the UI rather than an error.
"""

import pytest

from ui.pipeline.events import Event, STAGES, STATES


def test_event_carries_stage_state_and_detail():
    e = Event("council", "done", {"model": "phi4", "score": 6}, secs=41.24)
    assert e.stage == "council"
    assert e.state == "done"
    assert e.detail["score"] == 6


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        Event("cuncil", "done")


def test_unknown_state_is_rejected():
    with pytest.raises(ValueError, match="unknown state"):
        Event("council", "finished")


def test_to_dict_rounds_seconds_for_display():
    assert Event("judge", "done", secs=41.2449).to_dict()["secs"] == 41.2


def test_candidate_defaults_to_empty_for_single_runs():
    assert Event("parse", "start").to_dict()["candidate"] == ""


def test_every_stage_and_state_constant_is_usable():
    for stage in STAGES:
        for state in STATES:
            Event(stage, state)
