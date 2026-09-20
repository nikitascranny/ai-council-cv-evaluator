# tests/unit-tests/test_routing.py
"""Routing and page rendering, without binding a port.

dispatch() is a pure function of (method, path, form) so the whole web layer is
testable in-process. The handler class is a thin shell over it.
"""

import io
import re
import threading
from unittest import mock
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




# --- A ranking row is the way into its candidate's run ----------------------
#
# The row rendered a link to a literal "/run/RUNID/report/<n>": live_view
# substituted the run id into its shared script and only then pasted the
# ranking renderer in, so the renderer's own placeholder was never replaced.
# Every name in the shortlist was a dead link, which is what a reviewer saw as
# "clicking a candidate does nothing".


def test_the_ranking_rows_carry_the_real_run_id_not_a_placeholder():
    body = app.pages.live_view("abc123def456", "reviewer")
    assert "RUNID" not in body
    assert "/run/abc123def456/report/" in body


def test_every_url_the_live_view_uses_is_substituted_once_and_literally():
    """The placeholder bug came from substituting into a fragment and then
    assembling. Naming each URL as its own substituted constant is what keeps
    a later fragment from carrying an unsubstituted one - so pin that the
    page declares them rather than building paths out of pieces."""
    body = app.pages.live_view("abc123def456", "reviewer")
    assert "const EVENTS_URL = '/run/abc123def456/events';" in body
    assert "const REPORT_URL = '/run/abc123def456/report/';" in body

    body = app.pages.live_view("abc123def456", "candidate", "rewrite")
    assert "const RESULTS_URL = '/candidate/abc123def456';" in body


def test_clicking_a_ranking_row_shows_that_candidates_run_here():
    """A plain click must stay on the live view and put that candidate's
    stages and report link back on the board - not navigate to the report and
    not, as before, do nothing at all. The href stays real so ctrl-click still
    opens the report in a tab, which is why the handler has to refuse the
    browser's own default only for an unmodified click."""
    body = app.pages.live_view("abc123def456", "reviewer")
    assert "a.pick" in body
    assert "event.preventDefault()" in body
    assert "dataset.candidate" in body
    # A modified click is left to the browser.
    assert "event.metaKey || event.ctrlKey" in body


def test_a_finished_candidate_keeps_its_stage_rows_for_the_ranking_to_show():
    """The board is one set of rows shared by every candidate, so showing a
    finished candidate again means the rows were kept rather than overwritten.
    This pins the per-candidate record, which is the whole reason a ranking
    row has anything to show."""
    body = app.pages.live_view("abc123def456", "reviewer")
    assert "function runFor(name)" in body
    assert "entry.stages[e.stage] = {state: e.state, detail: detailText(e.detail)}" in body


def test_candidate_mode_ships_no_ranking_card():
    """One CV, no shortlist. An empty 'Ranking' panel and a report link on a
    page that has no report would both be lies."""
    body = app.pages.live_view("abc123def456", "candidate", "rewrite")
    # The shared stylesheet carries the ranking rules on every page; what
    # candidate mode must not have is the card itself, a renderer to fill it,
    # or a report to point at.
    assert "id='ranking' hidden" not in body
    assert "function renderRanking(ranking) {}" in body
    assert "Open the full report" not in body


def test_the_reviewer_ranking_card_is_hidden_until_there_is_a_ranking():
    """A single-CV run never emits a rank event, and a permanently empty card
    reads as something that failed rather than something never asked for."""
    body = app.pages.live_view("abc123def456", "reviewer")
    assert "id='ranking' hidden" in body
    assert "card.hidden = false;" in body


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
    who it is currently describing, and show only that candidate's rows,
    rather than silently letting one candidate's progress read as another's.

    The board is drawn from a per-candidate record rather than written into
    as events land, which is what makes switching candidate and showing
    progress the same operation - and is why there is no separate reset."""
    started = _start(monkeypatch, lambda: iter([Event("parse", "done")]))
    run_id = started.headers["Location"].rsplit("/", 1)[-1]
    body = app.dispatch("GET", "/run/" + run_id, {}).bytes().decode("utf-8")
    assert "id='candidate'" in body
    # One record per candidate, and the board redrawn from whichever is shown.
    assert "function runFor(name)" in body and "function draw()" in body
    # Rows not covered by the shown candidate's record go back to 'waiting',
    # so the previous candidate's progress cannot survive the switch.
    assert "cell ? cell.state : 'waiting'" in body
    assert "escapeHtml(shown)" in body


def test_candidate_mode_offers_a_picker_now_that_v2_exists():
    """This used to assert "Not available yet". v2 landed, so the honest
    page is a form - and asserting the old message would now be asserting
    that the UI lies about a component that works."""
    body = app.dispatch("GET", "/candidate", {}).bytes().decode("utf-8")
    assert "Not available" not in body
    assert "action='/candidate/run'" in body


def test_an_unreachable_rewriter_says_so_rather_than_showing_a_dead_form():
    def boom():
        raise app.candidate_mod.NotAvailable("rewriter v2 could not be loaded")
    with mock.patch.object(app.candidate_mod, "probe", boom):
        body = app.dispatch("GET", "/candidate", {}).bytes().decode("utf-8")
    assert "Not available" in body
    assert "rewriter v2" in body


def test_rendering_the_form_does_not_start_a_rewrite():
    """The old route called generate() with placeholder arguments purely to
    ask whether the backend was wired up. Harmless when v2 did not exist;
    against the real v2 that runs two 14B models to draw a form."""
    called = []
    with mock.patch.object(app.candidate_mod, "generate",
                           lambda *a, **k: called.append(a)):
        app.dispatch("GET", "/candidate", {})
    assert called == []


def test_a_run_needs_both_a_cv_and_a_job():
    assert app.dispatch("POST", "/candidate/run", {"job": ["j.pdf"]}).status == 400
    assert app.dispatch("POST", "/candidate/run", {"cv": ["c.pdf"]}).status == 400


def _finished_run(results, action="rewrite", **extra):
    """A completed candidate run in the registry, without touching Ollama."""
    payload = {"candidate": "Jane Doe", "job_title": "Secretary",
               "rewrites": results, "action": action}
    payload.update(extra)
    run_id = app.runs_mod.REGISTRY.start(
        lambda: iter([Event("run", "done", {"payload": payload}, 0.0,
                            "Jane Doe")]), mode="candidate", action=action)
    app.runs_mod.REGISTRY.get(run_id).finished.wait(timeout=5)
    return run_id


def _rewrite(**kw):
    from types import SimpleNamespace
    base = dict(candidate="Jane Doe", job_title="Secretary", agent_model="phi4",
                original_text="orig", rewritten_text="A rewritten CV.",
                used_fallback=False, rejected=False, reasons=[], error=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_results_page_shows_both_versions_with_their_own_downloads():
    run_id = _finished_run([_rewrite(agent_model="phi4"),
                            _rewrite(agent_model="qwen2.5:14b")])
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "phi4" in body and "qwen2.5:14b" in body
    assert "/download/0" in body and "/download/1" in body


def test_a_rewrite_wraps_inside_its_card_rather_than_running_off_it():
    """Models write whole paragraphs on one line, and a <pre> keeps them that
    way unless told to wrap - the text ran straight out of the card."""
    run_id = _finished_run([_rewrite()])
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "pre.rewrite { white-space: pre-wrap; overflow-wrap: anywhere; }" in body


def test_a_fallback_version_is_labelled_and_cannot_be_downloaded():
    """used_fallback means the text IS the candidate's original CV. Offering
    it as a download named like a rewrite is the confusion the guard exists
    to prevent, so the button is disabled and the URL behind it refuses."""
    run_id = _finished_run([_rewrite(used_fallback=True)])
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "not a rewrite" in body.lower()
    assert "disabled" in body
    assert app.dispatch("GET", "/candidate/{}/download/0".format(run_id), {}).status == 409


def test_downloading_a_version_serves_a_real_docx():
    import io as _io, zipfile
    run_id = _finished_run([_rewrite()])
    resp = app.dispatch("GET", "/candidate/{}/download/0".format(run_id), {})
    assert resp.status == 200
    assert "wordprocessingml" in resp.content_type
    assert "attachment" in resp.headers["Content-Disposition"]
    assert zipfile.is_zipfile(_io.BytesIO(resp.bytes()))


def test_an_unknown_run_or_version_is_a_404_not_a_crash():
    assert app.dispatch("GET", "/candidate/deadbeef", {}).status == 404
    run_id = _finished_run([_rewrite()])
    assert app.dispatch("GET", "/candidate/{}/download/7".format(run_id), {}).status == 404


# --- C1: a failed candidate must not vanish from the batch UI --------------


def test_the_live_view_renders_a_failures_area_that_survives_reset(monkeypatch):
    """run/failed has no matching stage-<s> row - the live view must render
    it somewhere that draw() (which rewrites the board on every event and on
    every switch of candidate) does not wipe, so a crashed candidate is still
    visible once the batch finishes."""
    started = _start(monkeypatch, lambda: iter([]))
    body = app.dispatch(
        "GET", "/run/" + started.headers["Location"].rsplit("/", 1)[-1], {}
    ).bytes().decode("utf-8")
    assert "id='failures'" in body
    assert "id='failures-card'" in body
    assert "function draw()" in body
    assert "run' && e.state === 'failed'" in body
    # draw() is what rewrites the board on every event and on every switch of
    # candidate, and it touches '#board tr', '#candidate', '#done' and the
    # ranking rows - never #failures. So a crashed candidate's entry outlives
    # every candidate that runs after it.
    draw_body = body.split("function draw()")[1].split("\n}")[0]
    assert "#board tr" in draw_body
    assert "failures" not in draw_body


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
    """The failure detail JS reads the event's candidate and its error and
    escapes both - pin that the client script actually does this rather than
    assuming silently-correct JS."""
    started = _start(monkeypatch, lambda: iter([]))
    body = app.dispatch(
        "GET", "/run/" + started.headers["Location"].rsplit("/", 1)[-1], {}
    ).bytes().decode("utf-8")
    assert "e.detail.error || 'unknown error'" in body
    assert "escapeHtml(name || '?')" in body
    assert "escapeHtml(entry.error)" in body


# --- the live view serves both modes ---------------------------------------


def test_a_candidate_run_shows_its_own_stages_not_the_evaluators():
    """Reviewer's five rows would leave council, essential and judge stuck on
    'waiting' forever in a mode that never runs them - three silent failures
    to anyone reading the board."""
    run_id = _finished_run([_rewrite()])
    body = app.dispatch("GET", "/run/" + run_id, {}).bytes().decode("utf-8")
    assert "stage-rewrite_v2" in body
    assert "stage-council" not in body and "stage-judge" not in body


def test_a_candidate_run_finishes_by_pointing_at_the_rewrites():
    run_id = _finished_run([_rewrite()])
    body = app.dispatch("GET", "/run/" + run_id, {}).bytes().decode("utf-8")
    assert "/candidate/" + run_id in body
    assert "/report" not in body


def test_a_reviewer_run_still_points_at_its_report():
    run_id = app.runs_mod.REGISTRY.start(
        lambda: iter([Event("run", "done", {"payload": {}}, 0.0, "Jane")]))
    app.runs_mod.REGISTRY.get(run_id).finished.wait(timeout=5)
    body = app.dispatch("GET", "/run/" + run_id, {}).bytes().decode("utf-8")
    assert "/run/{}/report".format(run_id) in body
    assert "stage-council" in body


# --- candidate mode: rewrite, evaluate, or both, then re-evaluate -----------


def test_the_candidate_form_offers_rewrite_evaluate_and_both():
    body = app.pages.candidate_form([Path("job_listing_secretary.pdf")],
                                    [Path("ALEX PARK.pdf")], lambda p: p.stem)
    for action in ("rewrite", "evaluate", "evaluate_rewrite"):
        assert "name='action' value='{}'".format(action) in body


def test_each_candidate_action_shows_only_the_stages_it_runs():
    for action, shown, hidden in (
            ("rewrite", ["parse", "rewrite_v2", "feedback"],
             ["council", "essential", "judge"]),
            ("evaluate", ["parse", "council", "essential", "judge"],
             ["rewrite_v2", "feedback"]),
            ("evaluate_rewrite",
             ["parse", "council", "essential", "judge", "rewrite_v2",
              "feedback"],
             ["rewrite"]),
            ("reevaluate", ["council", "essential", "judge"],
             ["parse", "rewrite_v2", "feedback"])):
        body = app.pages.live_view("abc123", "candidate", action)
        assert all("stage-{}'".format(s) in body for s in shown), action
        assert not any("stage-{}'".format(s) in body for s in hidden), action


def test_starting_a_candidate_run_runs_the_action_that_was_picked(monkeypatch):
    seen = {}

    def fake_run_candidate(job, cv, deps, generate, feedback, action):
        seen["action"] = action
        yield Event("run", "done", {"payload": {}})

    monkeypatch.setattr(app.evaluation, "run_candidate", fake_run_candidate)
    monkeypatch.setattr(app.deps_mod, "team_deps", lambda: object())
    response = app.dispatch("POST", "/candidate/run",
                            {"job": ["j.pdf"], "cv": ["c.pdf"],
                             "action": ["evaluate"]})
    assert response.status == 303
    run_id = response.headers["Location"].rsplit("/", 1)[-1]
    app.runs_mod.REGISTRY.wait(run_id, timeout=5)
    assert seen["action"] == "evaluate"
    assert app.runs_mod.REGISTRY.get(run_id).action == "evaluate"


def test_an_unknown_candidate_action_is_a_400():
    response = app.dispatch("POST", "/candidate/run",
                            {"job": ["j.pdf"], "cv": ["c.pdf"],
                             "action": ["publish"]})
    assert response.status == 400


def test_an_evaluation_shows_the_scores_and_why():
    run_id = _finished_run(
        [], action="evaluate",
        council=[{"model": "phi4", "score": 6, "summary": "Solid admin record."}],
        essential={"summary": "Meets the typing requirement."},
        judge={"final_score": 7, "final_evaluation": "A good fit."})
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "class='score'>7<" in body
    assert "Solid admin record." in body and "A good fit." in body
    assert "Meets the typing requirement." in body
    assert "/download/" not in body


def test_each_real_rewrite_can_go_back_to_the_council_but_a_fallback_cannot():
    run_id = _finished_run([_rewrite(),
                            _rewrite(agent_model="qwen2.5:14b", used_fallback=True)])
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "/candidate/{}/reevaluate/0".format(run_id) in body
    assert "/reevaluate/1" not in body


def test_reevaluating_a_fallback_or_a_missing_version_is_refused():
    run_id = _finished_run([_rewrite(used_fallback=True)])
    reevaluate = "/candidate/{}/reevaluate/{}".format
    assert app.dispatch("POST", reevaluate(run_id, 0), {}).status == 409
    assert app.dispatch("POST", reevaluate(run_id, 5), {}).status == 404
    assert app.dispatch("POST", reevaluate("deadbeef", 0), {}).status == 404


def test_reevaluating_starts_a_new_run_on_the_chosen_version(monkeypatch):
    seen = {}

    def fake_reevaluation(source, index, deps):
        seen["picked"] = source["rewrites"][index].agent_model
        yield Event("run", "done", {"payload": {}})

    monkeypatch.setattr(app.evaluation, "run_reevaluation", fake_reevaluation)
    monkeypatch.setattr(app.deps_mod, "team_deps", lambda: object())
    source_id = _finished_run([_rewrite(), _rewrite(agent_model="qwen2.5:14b")])

    response = app.dispatch(
        "POST", "/candidate/{}/reevaluate/1".format(source_id), {})
    assert response.status == 303
    new_id = response.headers["Location"].rsplit("/", 1)[-1]
    assert new_id != source_id
    app.runs_mod.REGISTRY.wait(new_id, timeout=5)
    assert seen["picked"] == "qwen2.5:14b"

    board = app.dispatch("GET", "/run/" + new_id, {}).bytes().decode("utf-8")
    assert "stage-council'" in board and "stage-parse'" not in board


def test_a_reevaluation_shows_the_original_score_beside_the_new_one():
    run_id = _finished_run(
        [_rewrite(agent_model="qwen2.5:14b")], action="reevaluate",
        agent_model="qwen2.5:14b",
        council=[{"model": "phi4", "score": 8, "summary": "Clearer."}],
        essential={"summary": "Now meets all of them."},
        judge={"final_score": 8},
        before={"council": [{"model": "phi4", "score": 5}],
                "essential": {"summary": "Partly meets them."},
                "judge": {"final_score": 4}})
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "Your original CV scored <b>4</b>" in body
    assert "<td>phi4</td><td>8</td><td>5</td>" in body
    assert "Now meets all of them." in body and "Partly meets them." in body


def test_the_rewrite_judges_recommendation_roadmap_and_careers_are_shown():
    run_id = _finished_run(
        [_rewrite(agent_model="phi4"), _rewrite(agent_model="qwen2.5:14b")],
        feedback={"recommended_rewrite": "qwen2.5:14b",
                  "recommendation_reason": "It leads with the typing speed.",
                  "career_roadmap": [{"step": "Apply",
                                      "action": "Send the qwen version."}],
                  "alternative_careers": ["Office manager", "Legal secretary"]})
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "Recommended: <b>qwen2.5:14b</b>" in body
    assert "It leads with the typing speed." in body
    assert "Send the qwen version." in body and "Legal secretary" in body
    # Read before the versions it compares, not after them.
    assert body.index("Which version to send") < body.index("<pre class='rewrite'>")
    assert "qwen2.5:14b <small>&middot; recommended</small>" in body
    assert "phi4 <small>" not in body


def test_recommending_a_fallback_says_it_is_the_original_cv():
    run_id = _finished_run(
        [_rewrite(), _rewrite(agent_model="qwen2.5:14b", used_fallback=True)],
        feedback={"recommended_rewrite": "qwen2.5:14b",
                  "recommendation_reason": "Safer.", "career_roadmap": [],
                  "alternative_careers": []})
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "That version is your original CV." in body


def test_a_failed_rewrite_judge_still_shows_the_rewrites():
    run_id = _finished_run([_rewrite()], feedback={"error": "timed out"})
    body = app.dispatch("GET", "/candidate/" + run_id, {}).bytes().decode("utf-8")
    assert "The rewrite judge failed: timed out" in body
    assert "/download/0" in body
    assert "/candidate/{}/reevaluate/0".format(run_id) in body


# --- a finished run must not go on holding the machine's memory -------------
#
# Ollama keeps a model resident for its keep_alive after the request that
# loaded it ends - 30 minutes and around ten gigabytes for the rewriter's 14B.
# Nothing watched for the end of a run, so a run that finished, failed or
# timed out left that loaded until Ctrl-C.


def test_a_finished_run_unloads_the_models_it_loaded(monkeypatch):
    freed = []
    monkeypatch.setattr(app, "unload_models", lambda: freed.append(True))
    run_id = app.runs_mod.REGISTRY.start(
        lambda: iter([Event("run", "done", {"payload": {}}, 0.0, "Jane")]),
        on_finish=app.unload_models)
    app.runs_mod.REGISTRY.get(run_id).finished.wait(timeout=5)

    assert freed == [True]


def test_a_run_that_raised_still_unloads(monkeypatch):
    """The case that matters most: a rewrite that timed out has loaded the
    model and produced nothing, so nothing else will free it."""
    freed = []

    def boom():
        raise RuntimeError("model timed out")

    run_id = app.runs_mod.REGISTRY.start(
        boom, on_finish=lambda: freed.append(True))
    app.runs_mod.REGISTRY.get(run_id).finished.wait(timeout=5)

    assert freed == [True]
    assert app.runs_mod.REGISTRY.get(run_id).error == "model timed out"


def test_an_unreachable_daemon_does_not_turn_a_finished_run_into_a_failed_one():
    def unreachable():
        raise OSError("ollama is not running")

    run_id = app.runs_mod.REGISTRY.start(
        lambda: iter([Event("run", "done", {"payload": {}}, 0.0, "Jane")]),
        on_finish=unreachable)
    run = app.runs_mod.REGISTRY.get(run_id)
    run.finished.wait(timeout=5)

    assert run.done is True
    assert run.error is None
    assert len(run.payloads) == 1


def test_every_route_that_starts_a_run_asks_for_the_models_back():
    """Three routes start runs, and a rewrite left resident by any of them
    costs the same ten gigabytes."""
    source = Path(app.__file__).read_text(encoding="utf-8")
    starts = source.count("REGISTRY.start(")
    assert starts == source.count("on_finish=unload_models") == 3


def test_the_run_end_unload_does_not_shut_the_app_down():
    """unload_models(), not stop_models(): the second also refuses every later
    call, which is right for Ctrl-C and would leave the app able to run
    exactly once."""
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "on_finish=stop_models" not in source
    assert "stop_local_models()" in source   # still there, for Ctrl-C
    assert "unload_local_models()" in source
