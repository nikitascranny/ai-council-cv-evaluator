# tests/unit-tests/test_routing.py
"""Routing and page rendering, without binding a port.

dispatch() is a pure function of (method, path, form) so the whole web layer is
testable in-process. The handler class is a thin shell over it.
"""

import io
import re
import threading
from pathlib import Path

import app
from ui.pipeline.backends import Settings
from ui.pipeline.events import Event


def _get(path, form=None):
    return app.dispatch("GET", path, form or {})


def test_the_root_offers_both_modes_by_name():
    body = _get("/").bytes().decode("utf-8")
    assert "Reviewer" in body and "Candidate" in body


def test_an_unknown_path_is_a_404_not_a_traceback():
    assert _get("/nope").status == 404


def test_the_reviewer_page_lists_the_job_listings():
    body = _get("/reviewer").bytes().decode("utf-8")
    assert "job_listing" in body or "Secretary" in body


def test_a_missing_cv_corpus_explains_itself(monkeypatch):
    """A fresh clone of main has job listings and no CVs."""
    monkeypatch.setattr(app.corpus, "discover_cvs", lambda root=None: [])
    monkeypatch.setattr(app.corpus, "corpus_status",
                        lambda root=None: {"jobs": 9, "cvs": 0,
                                           "cv_dir_present": False,
                                           "cv_dirs": ["Batch1/CVs"]})
    body = _get("/reviewer").bytes().decode("utf-8")
    assert "Batch1/CVs" in body
    assert "No CVs" in body


def test_the_runs_index_lists_saved_runs(monkeypatch, tmp_path):
    saved = tmp_path / "job__ALEX.json"
    saved.write_text('{"candidate": "ALEX", "job": "job", "council": []}',
                     encoding="utf-8")
    monkeypatch.setattr(app.evaluation, "saved_runs", lambda d: [saved])
    body = _get("/runs").bytes().decode("utf-8")
    assert "ALEX" in body


def test_every_page_declares_utf8():
    assert "charset=utf-8" in _get("/").content_type


def test_the_reviewer_page_titles_itself_by_mode():
    assert "Reviewer" in _get("/reviewer").bytes().decode("utf-8")


def test_post_to_a_get_only_route_is_405():
    assert app.dispatch("POST", "/", {}).status == 405


# --- /runs/<stem>: not in the brief, added to avoid a dead link from
# runs_index. See app.py's run_detail() for the containment-check rationale.

def test_a_saved_run_renders_its_report(monkeypatch, tmp_path):
    saved = tmp_path / "job__ALEX.json"
    saved.write_text(
        '{"candidate": "ALEX", "job": "job", "council": []}',
        encoding="utf-8")
    monkeypatch.setattr(app.evaluation, "DEMO_RESULTS", tmp_path)
    body = _get("/runs/job__ALEX").bytes().decode("utf-8")
    assert "ALEX" in body


def test_a_saved_run_still_carries_the_facts_panel(monkeypatch, tmp_path):
    """I4: reopening a saved run by stem must deliver the facts panel too,
    not just the live /run/<id>/report route."""
    saved = tmp_path / "job__ALEX.json"
    saved.write_text(
        '{"candidate": "ALEX", "job": "job", "council": [], '
        '"cv_text": "I have a BSc."}',
        encoding="utf-8")
    monkeypatch.setattr(app.evaluation, "DEMO_RESULTS", tmp_path)
    body = _get("/runs/job__ALEX").bytes().decode("utf-8")
    assert "Facts as stated" in body


def test_an_unknown_run_stem_is_a_404(monkeypatch, tmp_path):
    monkeypatch.setattr(app.evaluation, "DEMO_RESULTS", tmp_path)
    assert _get("/runs/nope-not-here").status == 404


def test_a_corrupt_run_file_404s_rather_than_raising(monkeypatch, tmp_path):
    """A half-written file from an interrupted run must not produce a traceback."""
    bad = tmp_path / "job__BROKEN.json"
    bad.write_text("not json at all", encoding="utf-8")
    monkeypatch.setattr(app.evaluation, "DEMO_RESULTS", tmp_path)
    assert _get("/runs/job__BROKEN").status == 404


def test_a_stem_containing_slashes_never_reaches_the_route(monkeypatch, tmp_path):
    """The route regex [\\w\\-.]+ admits no '/', and urlparse does not
    percent-decode, so a %2F in the stem never becomes a real slash and the
    URL simply fails to match any route - run_detail() is never entered.
    This proves the regex keeps traversal-shaped URLs off the route at all;
    it does not exercise the containment check inside run_detail() itself.
    That is covered separately, by calling run_detail() directly, in
    test_the_containment_check_refuses_a_file_outside_the_results_directory.
    """
    monkeypatch.setattr(app.evaluation, "DEMO_RESULTS", tmp_path)
    assert _get("/runs/..%2F..%2Fsecrets").status == 404


def test_the_containment_check_refuses_a_file_outside_the_results_directory(
        tmp_path, monkeypatch):
    """The regex already blocks slashes, so this calls run_detail directly.

    Isolating the containment check needs a target that really exists -
    otherwise is_file() returns False and the 404 proves nothing about
    whether traversal was refused.
    """
    results = tmp_path / "results"
    results.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"candidate": "LEAKED", "job": "x", "council": []}',
                      encoding="utf-8")
    monkeypatch.setattr(app.evaluation, "DEMO_RESULTS", results)

    response = app.run_detail({}, "../secret")

    assert response.status == 404
    assert "LEAKED" not in response.bytes().decode("utf-8")


# --- /reviewer/run, /run/<id>, /run/<id>/report: task 11.


def _start(monkeypatch, runner):
    monkeypatch.setattr(app, "_runner_for", lambda job, cvs, settings: runner)
    return app.dispatch("POST", "/reviewer/run",
                        {"job": ["data/job-listings/job_listing_secretary.pdf"],
                         "cv": ["Batch1/CVs/ALEX PARK.pdf"]})


def test_starting_a_run_redirects_to_its_live_view(monkeypatch):
    response = _start(monkeypatch, lambda: iter([Event("parse", "done")]))
    assert response.status == 303
    assert response.headers["Location"].startswith("/run/")


def test_starting_a_run_with_no_cv_selected_is_rejected():
    response = app.dispatch("POST", "/reviewer/run",
                            {"job": ["data/job-listings/job_listing_secretary.pdf"]})
    assert response.status == 400
    assert "select" in response.bytes().decode("utf-8").lower()


def test_the_live_view_names_the_stages_it_will_run(monkeypatch):
    started = _start(monkeypatch, lambda: iter([Event("parse", "done")]))
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    body = app.dispatch("GET", "/run/" + run_id, {}).bytes().decode("utf-8")
    assert "council" in body and "judge" in body


def test_the_live_view_of_an_unknown_run_is_a_404():
    assert app.dispatch("GET", "/run/deadbeef", {}).status == 404


def test_the_report_route_404s_before_the_run_has_finished(monkeypatch):
    gate = threading.Event()

    def blocked():
        gate.wait(timeout=5)
        yield Event("parse", "done")

    started = _start(monkeypatch, blocked)
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    assert app.dispatch("GET", "/run/{}/report".format(run_id), {}).status == 404
    gate.set()


# --- /run/<id>/report/<index>: candidates need to be individually openable
# from a batch's ranking table, not just the first one to finish.


def test_the_bare_report_route_still_serves_the_first_candidate(monkeypatch):
    """`/run/<id>/report` must keep working as index 0."""
    def runner():
        yield Event("run", "done", {"payload": {"candidate": "ALEX PARK",
                                                 "cv_text": ""}})

    started = _start(monkeypatch, runner)
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    app.runs_mod.REGISTRY.wait(run_id, timeout=5)
    body = app.dispatch("GET", "/run/{}/report".format(run_id), {}) \
              .bytes().decode("utf-8")
    assert "ALEX PARK" in body


def test_the_live_report_route_carries_the_facts_panel(monkeypatch):
    """I4: /run/<id>/report must deliver the facts panel, one of the spec's
    three Reviewer outputs, not just the score tiles and the diff."""
    def runner():
        yield Event("run", "done", {"payload": {"candidate": "ALEX PARK",
                                                 "cv_text": "I have a BSc."}})

    started = _start(monkeypatch, runner)
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    app.runs_mod.REGISTRY.wait(run_id, timeout=5)
    body = app.dispatch("GET", "/run/{}/report".format(run_id), {}) \
              .bytes().decode("utf-8")
    assert "Facts as stated" in body


def test_the_indexed_report_route_serves_the_matching_payload(monkeypatch):
    def runner():
        yield Event("run", "done", {"payload": {"candidate": "FIRST",
                                                 "cv_text": ""}})
        yield Event("run", "done", {"payload": {"candidate": "SECOND",
                                                 "cv_text": ""}})

    started = _start(monkeypatch, runner)
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    app.runs_mod.REGISTRY.wait(run_id, timeout=5)

    first = app.dispatch("GET", "/run/{}/report/0".format(run_id), {}) \
               .bytes().decode("utf-8")
    second = app.dispatch("GET", "/run/{}/report/1".format(run_id), {}) \
                .bytes().decode("utf-8")
    assert "FIRST" in first and "SECOND" not in first
    assert "SECOND" in second and "FIRST" not in second


def test_an_out_of_range_report_index_is_a_404_not_a_crash(monkeypatch):
    def runner():
        yield Event("run", "done", {"payload": {"candidate": "ONLY ONE",
                                                 "cv_text": ""}})

    started = _start(monkeypatch, runner)
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    app.runs_mod.REGISTRY.wait(run_id, timeout=5)
    assert app.dispatch("GET", "/run/{}/report/7".format(run_id), {}).status == 404


# --- The ranking table must link each row to *its own* candidate's report,
# not to whatever candidate happens to share its position in the ranking.
# Ranking order (by judge score) and completion order (payload order) differ
# whenever the best candidate is not the first to finish - that's the case
# this test forces.


def test_ranking_row_payload_index_points_at_its_own_finisher_not_its_rank(
        monkeypatch):
    def fake_run_batch(job_path, cv_paths, settings, deps, out_dir=None):
        # "loser" finishes first (payload index 0) but ranks second.
        yield Event("run", "done", {"payload": {"candidate": "loser"}},
                    candidate="loser")
        # "winner" finishes second (payload index 1) but ranks first.
        yield Event("run", "done", {"payload": {"candidate": "winner"}},
                    candidate="winner")
        yield Event("rank", "done", {"ranking": [
            {"candidate": "winner", "final_score": 9, "council": [],
             "path": None},
            {"candidate": "loser", "final_score": 3, "council": [],
             "path": None},
        ]})

    monkeypatch.setattr(app.evaluation, "run_batch", fake_run_batch)
    monkeypatch.setattr(app, "team_deps", lambda: object())

    runner = app._runner_for(Path("job.pdf"),
                             [Path("loser.pdf"), Path("winner.pdf")], Settings())
    events = list(runner())

    rank_event = next(e for e in events if e.stage == "rank")
    by_candidate = {row["candidate"]: row["payload_index"]
                    for row in rank_event.detail["ranking"]}
    assert by_candidate == {"winner": 1, "loser": 0}


# --- Fix round 1: the SSE loop must re-drain after observing done, the
# stage board must name the candidate it is showing, and a client
# disconnect must not print a traceback from the streaming thread.


class _FakeStreamHandler:
    """A stand-in for Handler with just enough surface for _stream_events."""

    def __init__(self, wfile):
        self.wfile = wfile

    def send_response(self, *a, **k):
        pass

    def send_header(self, *a, **k):
        pass

    def end_headers(self):
        pass


def test_the_sse_stream_does_not_drop_events_appended_while_marking_done(
        monkeypatch):
    """Old bug: events_since() was drained once per loop iteration, and the
    loop only checked run.done AFTER writing that whole snapshot to the
    socket. If the worker appends more events (in particular the terminal
    run/done event) and marks the run done while that write is still in
    flight, the old code would see done=True right after the write and exit
    without ever re-draining - the browser gets 'end' but never the event
    that was supposed to arrive with it.

    This fakes exactly that: the first events_since() call both returns an
    event AND (as the worker would) flips run.done, before the terminal
    run/done event has been drained at all.
    """
    class FakeRun:
        done = False

    class FakeRegistry:
        def __init__(self):
            self.run = FakeRun()
            self.calls = 0

        def get(self, run_id):
            return self.run

        def events_since(self, run_id, index):
            self.calls += 1
            if self.calls == 1:
                # The worker finishes and marks done while this batch is
                # still "on the wire".
                self.run.done = True
                return [Event("parse", "done")]
            if self.calls == 2:
                return [Event("run", "done", {"payload": {}})]
            return []

    monkeypatch.setattr(app.runs_mod, "REGISTRY", FakeRegistry())
    monkeypatch.setattr(app.time, "sleep", lambda seconds: None)

    written = io.BytesIO()
    app.Handler._stream_events(_FakeStreamHandler(written), "some-run-id")

    body = written.getvalue().decode("utf-8")
    assert '"stage": "run"' in body, "the terminal run/done event was dropped"
    assert body.index('"stage": "run"') < body.index("event: end")


def test_the_sse_stream_exits_quietly_when_the_client_disconnects(monkeypatch):
    """BrokenPipeError/ConnectionResetError from a closed tab must not
    escape as an unhandled traceback from the streaming thread."""
    class FakeRun:
        done = False

    class FakeRegistry:
        def get(self, run_id):
            return FakeRun()

        def events_since(self, run_id, index):
            return [Event("parse", "done")]

    class DisconnectedWfile:
        def write(self, data):
            raise BrokenPipeError()

        def flush(self):
            pass

    monkeypatch.setattr(app.runs_mod, "REGISTRY", FakeRegistry())
    monkeypatch.setattr(app.time, "sleep", lambda seconds: None)

    # Must return quietly, not raise.
    app.Handler._stream_events(_FakeStreamHandler(DisconnectedWfile()),
                               "some-run-id")


def test_the_live_view_names_the_current_candidate_and_resets_the_board(
        monkeypatch):
    """A batch shares one set of stage rows across every candidate - ten CVs
    would give a fifty-row board if each got its own. So the board must say
    who it is currently describing, and reset the rows when the candidate
    changes, rather than silently letting one candidate's progress read as
    another's."""
    started = _start(monkeypatch, lambda: iter([Event("parse", "done")]))
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    body = app.dispatch("GET", "/run/" + run_id, {}).bytes().decode("utf-8")
    assert "id='candidate'" in body
    assert "ensureCandidate" in body and "resetBoard" in body
    assert "escapeHtml(name)" in body


def test_candidate_mode_says_v2_is_not_ready_rather_than_faking_it():
    body = app.dispatch("GET", "/candidate", {}).bytes().decode("utf-8")
    assert "Not available yet" in body
    assert "v2" in body


# --- M2: /candidate must actually call ui.pipeline.candidate.generate() ------------


def test_candidate_page_changes_once_a_backend_is_wired_up(monkeypatch):
    """The route hardcoded its message and never called generate(), so
    ui.pipeline.candidate was dead outside its own tests. Setting BACKEND must
    change what this page renders, proving the call is real."""
    default_body = app.dispatch("GET", "/candidate", {}).bytes().decode("utf-8")

    monkeypatch.setattr(app.candidate_mod, "BACKEND",
                        lambda cv_path, job_title: "out/improved.docx")
    wired_body = app.dispatch("GET", "/candidate", {}).bytes().decode("utf-8")

    assert wired_body != default_body
    assert "Not available yet" not in wired_body
    assert "wired up" in wired_body.lower()


# --- C1: a failed candidate must not vanish from the batch UI --------------


def test_the_live_view_renders_a_failures_area_that_survives_reset(monkeypatch):
    """run/failed has no matching stage-<s> row - the live view must render
    it somewhere that resetBoard() (fired on every candidate change) does not
    wipe, so a crashed candidate is still visible once the batch finishes."""
    started = _start(monkeypatch, lambda: iter([]))
    body = app.dispatch(
        "GET", "/run/" + started.headers["Location"].rsplit("/", 1)[-1], {}
    ).bytes().decode("utf-8")
    assert "id='failures'" in body
    assert "id='failures-card'" in body
    assert "resetBoard" in body
    assert "run' && e.state === 'failed'" in body
    # resetBoard() only touches '#board tr' rows, never #failures - so the
    # failures list is untouched by the reset that fires on candidate change.
    reset_body = body.split("function resetBoard")[1].split("}")[0]
    assert "failures" not in reset_body


# --- I5: a missing evaluator/rewriter must render a page, not a traceback --


def test_a_missing_team_code_error_renders_a_page_naming_it(monkeypatch):
    from ui.pipeline.deps import MissingTeamCode

    def boom():
        raise MissingTeamCode("evaluator not found under /repo.")

    monkeypatch.setattr(app, "team_deps", boom)
    response = app.dispatch(
        "POST", "/reviewer/run",
        {"job": ["data/job-listings/job_listing_secretary.pdf"],
         "cv": ["Batch1/CVs/ALEX PARK.pdf"]})
    assert response.status == 500
    body = response.bytes().decode("utf-8")
    assert "evaluator not found under /repo." in body


def test_an_unexpected_route_error_is_a_500_page_not_a_dead_socket(monkeypatch):
    def boom(form, *groups):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(app, "_ROUTES", [("GET", re.compile("^/boom$"), boom)])
    response = app.dispatch("GET", "/boom", {})
    assert response.status == 500
    assert "kaboom" in response.bytes().decode("utf-8")


def test_a_candidate_failure_event_names_the_candidate_and_the_error(
        monkeypatch):
    """The failure detail JS reads e.candidate and e.detail.error and escapes
    both - pin that the client script actually does this rather than assuming
    silently-correct JS."""
    started = _start(monkeypatch, lambda: iter([]))
    body = app.dispatch(
        "GET", "/run/" + started.headers["Location"].rsplit("/", 1)[-1], {}
    ).bytes().decode("utf-8")
    assert "escapeHtml(e.candidate" in body
    assert "escapeHtml(e.detail.error" in body
