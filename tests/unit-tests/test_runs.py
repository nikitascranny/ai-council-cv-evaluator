# tests/unit-tests/test_runs.py
"""Starting a run from a request and reading its progress back.

The registry is what lets an HTTP request return immediately while a
four-minute evaluation carries on. Tests drive it with a fake runner - no
models, no sockets.
"""

import json
import threading

import pytest

from ui.pipeline.events import Event
from ui.pipeline.runs import RunBusy, RunRegistry, sse_frame


def _runner(events):
    def run():
        for e in events:
            yield e
    return run


def test_a_finished_run_keeps_every_event_in_order():
    reg = RunRegistry()
    run_id = reg.start(_runner([Event("parse", "done"), Event("judge", "done")]))
    reg.wait(run_id, timeout=5)
    assert [e.stage for e in reg.get(run_id).events] == ["parse", "judge"]


def test_events_since_returns_only_what_the_client_has_not_seen():
    reg = RunRegistry()
    run_id = reg.start(_runner([Event("parse", "done"), Event("judge", "done")]))
    reg.wait(run_id, timeout=5)
    assert [e.stage for e in reg.events_since(run_id, 1)] == ["judge"]


def test_a_runner_that_raises_marks_the_run_failed_rather_than_dying():
    def boom():
        yield Event("parse", "start")
        raise RuntimeError("ollama is not running")

    reg = RunRegistry()
    run_id = reg.start(boom)
    reg.wait(run_id, timeout=5)
    run = reg.get(run_id)
    assert run.done is True
    assert "ollama is not running" in run.error


def test_only_one_run_at_a_time():
    """Ollama serialises anyway; a queue keeps the progress readable."""
    gate = threading.Event()

    def slow():
        gate.wait(timeout=5)
        yield Event("parse", "done")

    reg = RunRegistry()
    reg.start(slow)
    with pytest.raises(RunBusy):
        reg.start(_runner([Event("parse", "done")]))
    gate.set()


def test_a_finished_run_frees_the_registry_for_the_next_one():
    reg = RunRegistry()
    first = reg.start(_runner([Event("parse", "done")]))
    reg.wait(first, timeout=5)
    second = reg.start(_runner([Event("parse", "done")]))
    assert second != first


def test_unknown_run_id_is_none_not_an_exception():
    assert RunRegistry().get("nope") is None


def test_sse_frame_is_one_json_data_line_ending_in_a_blank_line():
    frame = sse_frame(Event("council", "done", {"model": "phi4", "score": 6}))
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    payload = json.loads(frame[len("data: "):].strip())
    assert payload["stage"] == "council" and payload["detail"]["score"] == 6


def test_sse_frame_has_no_raw_newlines_that_would_split_the_message():
    """A summary containing a newline would otherwise truncate the frame."""
    frame = sse_frame(Event("judge", "done",
                            {"final_evaluation": "line one\nline two"}))
    assert frame.count("\n\n") == 1
