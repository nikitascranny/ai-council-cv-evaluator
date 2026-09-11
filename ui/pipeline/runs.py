# core/runs.py
"""Runs in flight, so an HTTP request need not wait four minutes.

A request starts a run and gets an id back. The browser then streams the events
that accumulate here. One run at a time: Ollama serialises anyway, and
interleaved progress from two evaluations is unreadable.
"""

import json
import threading
import uuid


class RunBusy(RuntimeError):
    """A run is already in flight."""


class Run:
    def __init__(self, run_id, mode: str = "reviewer", action: str = None):
        self.id = run_id
        # Which mode started this run. The live view is shared, but its stage
        # rows and its "done" link differ, and only the starting route knows
        # which kind of run this is.
        self.mode = mode
        # Within candidate mode, which of its runs this is (rewrite, evaluate,
        # both, or a re-evaluation) - each shows a different set of rows.
        self.action = action
        self.events = []
        self.payloads = []
        self.done = False
        self.error = None
        self.finished = threading.Event()


class RunRegistry:
    def __init__(self):
        self._runs = {}
        self._lock = threading.Lock()
        self._current = None

    def start(self, runner, mode: str = "reviewer", action: str = None) -> str:
        """runner is a zero-argument callable returning an Event iterator."""
        with self._lock:
            if self._current is not None and not self._runs[self._current].done:
                raise RunBusy("a run is already in progress")
            run_id = uuid.uuid4().hex[:12]
            run = Run(run_id, mode, action)
            self._runs[run_id] = run
            self._current = run_id

        def work():
            try:
                for event in runner():
                    with self._lock:
                        run.events.append(event)
                        if event.stage == "run" and event.state == "done":
                            run.payloads.append(event.detail.get("payload"))
            except Exception as exc:
                with self._lock:
                    run.error = str(exc)
            finally:
                with self._lock:
                    run.done = True
                run.finished.set()

        threading.Thread(target=work, daemon=True).start()
        return run_id

    def get(self, run_id):
        return self._runs.get(run_id)

    def events_since(self, run_id, index: int) -> list:
        run = self._runs.get(run_id)
        if run is None:
            return []
        with self._lock:
            return list(run.events[index:])

    def wait(self, run_id, timeout=None) -> bool:
        run = self._runs.get(run_id)
        return bool(run and run.finished.wait(timeout))


REGISTRY = RunRegistry()


def sse_frame(event) -> str:
    """One server-sent event.

    json.dumps escapes newlines, which matters: a model summary containing one
    would otherwise end the frame early and truncate the message.

    The run/done event's payload never goes over the wire. It rides on the
    event so the registry can lift it into run.payloads, and the browser has
    no use for it - the page reads a finished run back over HTTP. Sending it
    anyway meant streaming the whole CV text and every rewrite to a client
    that discards them, and candidate mode's payload holds RewriteResult
    objects, which json.dumps cannot encode at all: the run finished and then
    died on its last frame, taking the "done" link with it.
    """
    data = event.to_dict()
    if "payload" in data.get("detail", {}):
        data["detail"] = {k: v for k, v in data["detail"].items()
                          if k != "payload"}
    return "data: {}\n\n".format(json.dumps(data, ensure_ascii=False))
