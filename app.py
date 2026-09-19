# app.py
"""The AI CV Evaluator web app.

    python app.py            # http://localhost:8000
    python app.py --port N

Standard library only, so a marker or the client runs it with no install.

dispatch() is a pure function of (method, path, form): every route is testable
in-process, and the HTTP handler below is a shell that calls it.
"""

import argparse
import functools
import html as html_mod
import json
import re
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ui.pipeline import candidate as candidate_mod
from ui.pipeline import corpus, deps as deps_mod, evaluation
from ui.pipeline import runs as runs_mod
from ui.pipeline.backends import Settings
from ui.pipeline.deps import MissingTeamCode, team_deps
from ui.pipeline.events import Event
from ui import docx_export, pages, report

PORT = 8000


@dataclass
class Response:
    status: int
    body: "str | bytes"
    content_type: str = "text/html; charset=utf-8"
    headers: dict = field(default_factory=dict)

    def bytes(self) -> bytes:
        """A .docx body arrives already encoded - it is zip archive bytes, not
        text, and .encode() on it would raise. Pages stay str and encode here
        as before, so every existing caller is unaffected."""
        if isinstance(self.body, bytes):
            return self.body
        return self.body.encode("utf-8")


_ROUTES = []


def route(method: str, pattern: str):
    compiled = re.compile("^{}$".format(pattern))

    def decorator(fn):
        _ROUTES.append((method, compiled, fn))
        return fn
    return decorator


def dispatch(method: str, path: str, form: dict = None) -> Response:
    """Resolve a request. Pure: no socket, no state beyond the route table.

    Wrapped in an exception boundary so a route that raises still comes back
    as a Response rather than propagating out of do_GET/do_POST and killing
    the connection with a bare traceback on stderr. MissingTeamCode is caught
    specifically - it names what is missing (evaluator/ or rewriter/) rather
    than reading as a generic server error - and anything else falls through
    to a general 500 page. Still pure: every branch below returns a Response
    built from strings, none of it touches a socket or the filesystem.
    """
    form = form or {}
    path = urlparse(path).path.rstrip("/") or "/"
    matched_path = False
    try:
        for route_method, pattern, fn in _ROUTES:
            match = pattern.match(path)
            if not match:
                continue
            matched_path = True
            if route_method == method:
                return fn(form, *match.groups())
    except MissingTeamCode as exc:
        return Response(500, pages.layout(
            "Team code missing",
            "<div class='card'><h2>Team code missing</h2>"
            "<p>{}</p></div>".format(html_mod.escape(str(exc)))))
    except Exception as exc:
        return Response(500, pages.layout(
            "Something went wrong",
            "<div class='card'><h2>Something went wrong</h2>"
            "<p>{}</p></div>".format(html_mod.escape(str(exc)))))
    if matched_path:
        return Response(405, pages.layout(
            "Not allowed", "<div class='card'>Method not allowed.</div>"))
    return Response(404, pages.layout(
        "Not found", "<div class='card'>No such page.</div>"))


@route("GET", "/")
def home(form):
    return Response(200, pages.mode_picker())


@route("GET", "/reviewer")
def reviewer(form):
    return Response(200, pages.reviewer_form(
        corpus.discover_jobs(), corpus.discover_cvs(), corpus.corpus_status(),
        Settings(), corpus.label_for))


@route("GET", "/candidate")
def candidate_view(form):
    """The CV/listing picker, or an honest empty state.

    generate() is not called here - asking "is v2 importable" is what
    candidate_mod.probe() is for, and calling generate() with placeholder
    arguments to find out (as this route used to) would now start a real
    two-model rewrite the moment the seam works.
    """
    try:
        candidate_mod.probe()
    except candidate_mod.NotAvailable as exc:
        return Response(200, pages.candidate_page(str(exc)))
    jobs, cvs = corpus.discover_jobs(), corpus.discover_cvs()
    if not jobs or not cvs:
        return Response(200, pages.candidate_page(
            "No job listings or CVs were found in the corpus.", available=True))
    return Response(200, pages.candidate_form(
        jobs, cvs, corpus.label_for, candidate_mod.rewriter_choices()))


_BUSY = Response(409, pages.layout(
    "Busy", "<div class='card'>A run is already in progress. Wait for it to "
            "finish, or reload its page.</div>"))


@route("POST", "/candidate/run")
def candidate_run(form):
    job = next(iter(form.get("job", [])), None)
    cv = next(iter(form.get("cv", [])), None)
    if not job or not cv:
        return Response(400, pages.layout(
            "Nothing selected",
            "<div class='card'>Pick a job listing and your CV.</div>"))
    action = next(iter(form.get("action", [])), evaluation.EVALUATE_REWRITE)
    if action not in evaluation.CANDIDATE_ACTIONS:
        return Response(400, pages.layout(
            "Nothing selected",
            "<div class='card'>Pick whether to rewrite your CV, evaluate it, "
            "or both.</div>"))
    # The form leaves this out when the choice cannot apply (evaluate only,
    # which disables the field), so a missing value is the local default.
    rewriter = next(iter(form.get("rewriter", [])), candidate_mod.LOCAL)
    if rewriter not in candidate_mod.REWRITERS:
        return Response(400, pages.layout(
            "Nothing selected",
            "<div class='card'>Pick where to rewrite your CV: on this "
            "computer or on Ollama Cloud.</div>"))
    if action == evaluation.EVALUATE:
        rewriter = candidate_mod.LOCAL
    # The form greys the cloud out without a key; this refuses the request a
    # stale page or a hand-built one sends anyway, rather than quietly
    # running on the local models the candidate did not pick.
    unavailable = (candidate_mod.cloud_status()
                   if rewriter == candidate_mod.CLOUD else None)
    if unavailable:
        return Response(400, pages.layout(
            "Ollama Cloud not set up",
            "<div class='card'><h2>Ollama Cloud is not set up</h2><p>{}</p>"
            "<p>Go back and pick <b>This computer</b> instead.</p>"
            "</div>".format(html_mod.escape(unavailable))))
    generate = functools.partial(candidate_mod.generate, profile=rewriter)

    def runner():
        deps = deps_mod.team_deps()
        return evaluation.run_candidate(
            Path(job), Path(cv), deps, generate,
            candidate_mod.feedback, action)

    try:
        run_id = runs_mod.REGISTRY.start(runner, mode="candidate", action=action)
    except runs_mod.RunBusy:
        return _BUSY
    return Response(303, "", headers={"Location": "/run/" + run_id})


@route("GET", r"/candidate/([0-9a-f]+)")
def candidate_results(form, run_id):
    run = runs_mod.REGISTRY.get(run_id)
    if run is None or not run.payloads:
        return Response(404, pages.layout(
            "Not ready",
            "<div class='card'>That run has not finished yet.</div>"))
    return Response(200, pages.candidate_results(run_id, run.payloads[0]))


@route("POST", r"/candidate/([0-9a-f]+)/reevaluate/(\d+)")
def candidate_reevaluate(form, run_id, index):
    """Score one rewritten version with the council and the judge again.

    Refusals are decided here, before the redirect: raised on the worker
    thread instead, they would leave the live view sitting on "waiting" with
    nothing to say why.
    """
    run = runs_mod.REGISTRY.get(run_id)
    if run is None or not run.payloads:
        return Response(404, pages.layout(
            "Not ready",
            "<div class='card'>That run has not produced a rewrite yet.</div>"))
    source = run.payloads[0]
    results = source.get("rewrites") or []
    idx = int(index)
    if idx >= len(results):
        return Response(404, pages.layout(
            "Not found", "<div class='card'>No such version.</div>"))
    if not candidate_mod.is_rewrite(results[idx]):
        return Response(409, pages.layout(
            "Not a rewrite",
            "<div class='card'>That version is your original CV, not a "
            "rewrite, so there is nothing new to score.</div>"))

    def runner():
        return evaluation.run_reevaluation(source, idx, deps_mod.team_deps())

    try:
        new_id = runs_mod.REGISTRY.start(
            runner, mode="candidate", action=evaluation.REEVALUATE)
    except runs_mod.RunBusy:
        return _BUSY
    return Response(303, "", headers={"Location": "/run/" + new_id})


@route("GET", r"/candidate/([0-9a-f]+)/download/(\d+)")
def candidate_download(form, run_id, index):
    """One agent's version as a .docx.

    A fallback is refused rather than served: its text is the candidate's own
    CV, and handing that back as a download named like a rewrite is the
    confusion the guard exists to prevent. The page disables the button, and
    this refuses the hand-typed URL behind it.
    """
    run = runs_mod.REGISTRY.get(run_id)
    if run is None or not run.payloads:
        return Response(404, pages.layout(
            "Not ready",
            "<div class='card'>That run has not produced a rewrite yet.</div>"))
    results = run.payloads[0].get("rewrites") or []
    idx = int(index)
    if idx >= len(results):
        return Response(404, pages.layout(
            "Not found", "<div class='card'>No such version.</div>"))
    result = results[idx]
    if result.used_fallback:
        return Response(409, pages.layout(
            "Not a rewrite",
            "<div class='card'>That version is your original CV - the check "
            "rejected the rewrite - so there is nothing to download.</div>"))
    name = docx_export.filename_for(result)
    return Response(
        200, docx_export.docx_bytes(result),
        content_type="application/vnd.openxmlformats-officedocument"
                     ".wordprocessingml.document",
        headers={"Content-Disposition":
                 'attachment; filename="{}"'.format(name)})


@route("GET", "/runs")
def runs(form):
    rows = [("/runs/{}".format(p.stem), evaluation.describe_run(p))
            for p in evaluation.saved_runs(evaluation.DEMO_RESULTS)]
    return Response(200, pages.runs_index(rows))


_NOT_FOUND = Response(404, pages.layout(
    "Not found", "<div class='card'>No such page.</div>"))


@route("GET", r"/runs/([\w\-.]+)")
def run_detail(form, stem):
    """A saved run's report page.

    Not in the brief for this task: runs_index() links to /runs/<stem> but no
    task in the plan wired the route up, which would ship a dead link. Added
    per the task-10 ruling.

    The regex above already restricts stem to word characters, '-' and '.',
    so it cannot contain a path separator - through dispatch(), this
    containment check cannot currently fire, since a slash-bearing stem never
    matches the route at all. It is kept anyway as defence-in-depth: it would
    fire if the route regex is ever loosened, or if a regex-legal filename
    inside DEMO_RESULTS were a symlink pointing outside it.
    """
    candidate = (evaluation.DEMO_RESULTS / "{}.json".format(stem)).resolve()
    try:
        candidate.relative_to(evaluation.DEMO_RESULTS.resolve())
    except ValueError:
        return _NOT_FOUND
    if not candidate.is_file():
        return _NOT_FOUND
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        # A half-written file from an interrupted run must not produce a
        # traceback - the same spirit as evaluation.describe_run.
        return _NOT_FOUND
    return Response(200, report.render_report(payload))


def _runner_for(job_path, cv_paths, settings):
    """A zero-argument callable the registry can run on its own thread.

    Also stamps each ranking row with the position of its own candidate in
    completion order (``payload_index``) - ranking order is sorted by judge
    score, but ``/run/<id>/report/<index>`` indexes into ``run.payloads``,
    which is completion order. Those differ whenever the best candidate is
    not the first to finish, so the row cannot just use its own rank.

    completed.index(row["candidate"]) cannot miss: run_single's last yield is
    always Event("run", "done", ...) for that candidate, and run_batch only
    appends a payload and emits its rank event after that run_single call has
    returned - a candidate that raised is never appended and never ranked.
    So every candidate named in a ranking row was already recorded in
    `completed` before the rank event carrying that row was yielded.
    """
    deps = team_deps()

    def runner():
        completed = []
        if len(cv_paths) == 1:
            events = evaluation.run_single(job_path, cv_paths[0], settings, deps)
        else:
            events = evaluation.run_batch(job_path, cv_paths, settings, deps)
        for event in events:
            if event.stage == "run" and event.state == "done":
                completed.append(event.candidate)
            if event.stage == "rank":
                ranking = [
                    dict(row, payload_index=completed.index(row["candidate"]))
                    for row in event.detail.get("ranking", [])
                ]
                event = Event(event.stage, event.state,
                              dict(event.detail, ranking=ranking),
                              event.secs, event.candidate)
            yield event

    return runner


@route("POST", "/reviewer/run")
def reviewer_run(form):
    job = next(iter(form.get("job", [])), None)
    cvs = [c for c in form.get("cv", []) if c]
    if not job or not cvs:
        return Response(400, pages.layout(
            "Nothing selected",
            "<div class='card'>Pick a job listing and select at least one "
            "candidate.</div>"))
    settings = Settings.from_form(form)
    try:
        run_id = runs_mod.REGISTRY.start(
            _runner_for(Path(job), [Path(c) for c in cvs], settings))
    except runs_mod.RunBusy:
        return _BUSY
    return Response(303, "", headers={"Location": "/run/" + run_id})


@route("GET", r"/run/([0-9a-f]+)")
def run_view(form, run_id):
    run = runs_mod.REGISTRY.get(run_id)
    if run is None:
        return Response(404, pages.layout(
            "Not found", "<div class='card'>No such run.</div>"))
    return Response(200, pages.live_view(run_id, run.mode, run.action))


@route("GET", r"/run/([0-9a-f]+)/report")
@route("GET", r"/run/([0-9a-f]+)/report/(\d+)")
def run_report(form, run_id, index=None):
    """The full report for one candidate in a run.

    `/run/<id>/report` is index 0, for the single-candidate case and to keep
    existing links working. `/run/<id>/report/<index>` selects into
    run.payloads by completion order - see the ranking table's payload_index.
    """
    idx = int(index) if index is not None else 0
    run = runs_mod.REGISTRY.get(run_id)
    if run is None or not run.payloads:
        return Response(404, pages.layout(
            "Not ready",
            "<div class='card'>That run has not produced a report yet.</div>"))
    # idx < 0 cannot happen through the route (\d+) never captures a '-'
    # sign) - it guards a direct call to run_report(), not a route.
    if idx < 0 or idx >= len(run.payloads):
        return Response(404, pages.layout(
            "Not found",
            "<div class='card'>No such candidate report.</div>"))
    payload = run.payloads[idx]
    return Response(200, report.render_report(payload))


class Handler(BaseHTTPRequestHandler):
    """A shell over dispatch(). Keep logic out of here - it is untestable."""

    def _send(self, response):
        payload = response.bytes()
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _stream_events(self, run_id):
        run = runs_mod.REGISTRY.get(run_id)
        if run is None:
            self._send(Response(404, "no such run", "text/plain; charset=utf-8"))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        sent = 0
        try:
            while True:
                # Sample done BEFORE draining: the worker can append more
                # events (and flip done) while this loop is mid-write on the
                # snapshot below. Sampling first means anything appended
                # after the sample is still caught by this same drain, so the
                # terminal run/done event is never skipped.
                finished = run.done
                for event in runs_mod.REGISTRY.events_since(run_id, sent):
                    self.wfile.write(runs_mod.sse_frame(event).encode("utf-8"))
                    self.wfile.flush()
                    sent += 1
                if finished:
                    self.wfile.write(b"event: end\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                time.sleep(0.3)
        except ConnectionError:
            # The viewer closed the tab, or navigated to the results the
            # stream just told them about - the common case, not a bug.
            # ConnectionError rather than the two subclasses it used to name:
            # Windows aborts a socket the client walks away from with
            # ConnectionAbortedError (WinError 10053), which is neither, so
            # following the run's own "done" link printed a traceback.
            return

    def do_GET(self):
        stream = re.match(r"^/run/([0-9a-f]+)/events$", urlparse(self.path).path)
        if stream:
            self._stream_events(stream.group(1))
            return
        self._send(dispatch("GET", self.path,
                            parse_qs(urlparse(self.path).query)))

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        self._send(dispatch("POST", self.path, form))

    def log_message(self, fmt, *args):
        sys.stderr.write("  {} {}\n".format(self.command, self.path))


def stop_models() -> list:
    """Stop the models a run started, so Ctrl-C ends them and not only the page.

    A run is a thread in this process calling Ollama, and Ollama keeps a model
    loaded after its caller has gone. Not imported at the top: this file is
    standard library only, and ollama_utils is not loaded until a run needs it -
    when it never was, no run reached a model and there is nothing to stop.
    """
    ollama_utils = sys.modules.get("ollama_utils")
    if ollama_utils is None:
        return []
    return ollama_utils.stop_local_models()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI CV Evaluator web app")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://127.0.0.1:{}/".format(args.port)
    print("AI CV Evaluator on {}   (ctrl-c to stop)".format(url))
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
        stopped = stop_models()
        if stopped:
            print("stopped models: {}".format(", ".join(stopped)))
        print("stopped")
    return 0


if __name__ == "__main__":
    # The Ollama Cloud key from the repo's .env, before the candidate form
    # asks cloud_status() whether to offer the cloud rewriter. Here rather than
    # in main() so the tests never pick up a real key.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
    from ollama_utils import load_env_file
    load_env_file()
    raise SystemExit(main())
