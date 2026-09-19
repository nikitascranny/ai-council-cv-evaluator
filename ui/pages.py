# ui/pages.py
"""The app's own pages.

The run report has its own renderer in ui/report.py and its own inline CSS;
these are the pages around it - picking a mode, choosing inputs, listing runs.
"""

from ui.pipeline import candidate as candidate_mod
from ui.pipeline import evaluation
import html

_CSS = """
body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f6f7f9;
       color: #16181d; }
header { background: #16181d; color: #fff; padding: 14px 22px;
         display: flex; gap: 18px; align-items: baseline; }
header a { color: #9fb4ff; text-decoration: none; }
main { max-width: 880px; margin: 24px auto; padding: 0 18px; }
.card { background: #fff; border: 1px solid #dfe2e8; border-radius: 8px;
        padding: 20px; margin-bottom: 18px; }
.hero { text-align: center; margin: 52px 0 30px; }
.hero h1 { font-size: 38px; line-height: 1.15; letter-spacing: -.6px;
           margin: 0 0 10px; }
.hero p { margin: 0; color: #5b606b; font-size: 17px; }
.modes { display: flex; flex-wrap: wrap; gap: 20px; align-items: stretch; }
/* The anchor stretches so both cards end up the same height whatever the
   description lengths are - a short card next to a tall one reads as an
   afterthought rather than an equal choice. */
.modes a { flex: 1 1 260px; display: flex; text-decoration: none;
           color: inherit; }
.modes .card { flex: 1; display: flex; flex-direction: column; margin: 0;
               transition: border-color .15s, box-shadow .15s, transform .15s; }
.modes a:hover .card { border-color: #16181d; transform: translateY(-2px);
                       box-shadow: 0 6px 18px rgba(22, 24, 29, .12); }
.modes .card h2 { margin: 0 0 10px; }
.modes .card p { flex: 1; margin: 0 0 16px; color: #43474f; }
.modes .go { font-weight: 600; }
.modes a:hover .go { text-decoration: underline; }
.empty { background: #fff8e6; border: 1px solid #e6cf8b; border-radius: 8px;
         padding: 16px; }
label { display: block; margin: 10px 0 4px; font-weight: 600; }
select, button { font: inherit; padding: 8px; }
button { background: #16181d; color: #fff; border: 0; border-radius: 6px;
         padding: 10px 18px; cursor: pointer; }
table { border-collapse: collapse; width: 100%; }
td, th { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e6e8ec;
         vertical-align: top; }
label.choice { font-weight: 400; margin: 8px 0; }
label.choice small { display: block; margin-left: 24px; color: #5b606b; }
fieldset { border: 1px solid #dfe2e8; border-radius: 6px; margin: 18px 0 0;
           padding: 4px 14px 14px; }
legend { font-weight: 600; padding: 0 6px; }
.hint { color: #5b606b; font-size: 14px; margin: 6px 0 8px; }
.picker { border: 1px solid #c6cad2; border-radius: 6px; max-height: 260px;
          overflow-y: auto; background: #fff; }
/* Unselectable text: without this, shift-clicking a row to extend the
   selection highlights the names instead of ticking the boxes. */
.picker label { display: flex; align-items: center; gap: 10px; font-weight: 400;
                margin: 0; padding: 7px 10px; cursor: pointer;
                border-bottom: 1px solid #f0f1f4;
                -webkit-user-select: none; user-select: none; }
.picker label:last-child { border-bottom: 0; }
.picker label:hover { background: #eef2ff; }
.pickerbar { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.pickerbar button { background: #fff; color: #16181d; border: 1px solid #c6cad2;
                    padding: 5px 12px; font-size: 14px; }
.pickerbar .count { color: #5b606b; font-size: 14px; }
.actions { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
.actions form, .actions a { margin: 0; }
pre.rewrite { white-space: pre-wrap; overflow-wrap: anywhere; }
.score { font-size: 34px; font-weight: 600; margin: 4px 0; }
.score small { font-size: 15px; font-weight: 400; color: #5b606b; }
/* A ranking row is a control, not just a label - it swaps the board above to
   that candidate's run - so it has to look like one. */
#ranking tr { cursor: pointer; }
#ranking tr:hover { background: #eef2ff; }
#ranking tr.picked { background: #e7ecff; }
#ranking tr.picked td { font-weight: 600; }
#ranking a.pick { color: inherit; text-decoration: none; }
#ranking tr:hover a.pick { text-decoration: underline; }
"""


def layout(title: str, body: str, mode: str = None) -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>{}</title><style>{}</style></head><body>"
            "<header><b>AI CV Evaluator</b>"
            "<a href='/reviewer'>Reviewer</a><a href='/candidate'>Candidate</a>"
            "<a href='/runs'>Saved runs</a></header>"
            "<main>{}</main></body></html>").format(html.escape(title), _CSS, body)


def mode_picker() -> str:
    body = ("<div class='hero'><h1>AI CV Evaluator</h1>"
            "<p>Choose your desired mode.</p></div>"
            "<div class='modes'>"
            "<a href='/reviewer'><div class='card'><h2>Reviewer</h2>"
            "<p>Assess candidates against a job listing. Shows the score, how "
            "much of it was wording rather than substance, and the facts the CV "
            "states.</p><span class='go'>Open Reviewer &rarr;</span></div></a>"
            "<a href='/candidate'><div class='card'><h2>Candidate</h2>"
            "<p>Aim your own CV at a listing you are applying for: see how it "
            "scores, have it rewritten, or both.</p>"
            "<span class='go'>Open Candidate &rarr;</span></div></a></div>")
    return layout("AI CV Evaluator", body)


def _options(paths, label_for) -> str:
    return "".join("<option value='{}'>{}</option>".format(
        html.escape(str(p)), html.escape(label_for(p))) for p in paths)


def _cv_checkboxes(paths, label_for) -> str:
    """One tickable row per CV, in place of a ctrl-click multi-select.

    Every row posts under the same `cv` name, so parse_qs hands the route the
    same list of values the multi-select did.
    """
    return "".join(
        "<label><input type='checkbox' name='cv' value='{}'><span>{}</span>"
        "</label>".format(html.escape(str(p)), html.escape(label_for(p)))
        for p in paths)


# Named for what the reviewer gets out of them rather than for the stage that
# implements them: "Phrasing-sensitivity probe" told a first-time user nothing
# about what ticking it would do.
_REVIEWER_OPTIONS = (
    ("essential", True, "Check the essential requirements",
     "Reads only the experience, education and certifications sections and "
     "summarises which of the listing's essential requirements the CV clearly "
     "meets, which are a near match, and which are missing or unclear. The "
     "judge reads that summary alongside the council's when it gives the "
     "final score - untick this for a score from the council alone. Adds one "
     "model call per candidate."),
    ("rewrite", True, "Check whether rewording changes the score",
     "Scores the CV, has a model reword it without changing any facts, then "
     "scores that version too. If the two scores differ much, the ranking is "
     "reacting to how the CV is written rather than to what the candidate has "
     "done. The report shows both scores, the difference, and a highlighted "
     "diff of exactly what changed. A reworded version that fails the fact "
     "check is not scored - there is nothing to learn from the score of a "
     "rewrite that changed the facts - and the report says so instead. "
     "Roughly doubles the time per candidate. (Called the "
     "phrasing-sensitivity probe elsewhere in the project.)"),
)

_CV_PICKER_SCRIPT = """
(function () {
  const box = document.getElementById('cvs');
  const boxes = Array.from(box.querySelectorAll("input[type='checkbox']"));
  const count = document.getElementById('cv-count');
  let anchor = null;
  function refresh() {
    const n = boxes.filter((b) => b.checked).length;
    count.textContent = n === 0 ? 'None selected yet'
      : n + (n === 1 ? ' candidate selected' : ' candidates selected');
  }
  function setAll(on) { boxes.forEach((b) => { b.checked = on; }); refresh(); }
  document.getElementById('cv-all').addEventListener('click', () => setAll(true));
  document.getElementById('cv-none').addEventListener('click', () => setAll(false));
  box.addEventListener('click', (event) => {
    const hit = event.target;
    if (hit.type !== 'checkbox') return;
    // Shift-click fills in the run between the last box touched and this one,
    // so picking a whole batch is two clicks rather than twenty.
    if (event.shiftKey && anchor !== null) {
      const here = boxes.indexOf(hit);
      const [from, to] = [here, anchor].sort((a, b) => a - b);
      boxes.slice(from, to + 1).forEach((b) => { b.checked = hit.checked; });
    }
    anchor = boxes.indexOf(hit);
    refresh();
  });
  refresh();
})();
"""


def reviewer_form(jobs, cvs, status, settings, label_for) -> str:
    if not cvs:
        where = ", ".join(status["cv_dirs"])
        body = ("<div class='card'><h2>Reviewer</h2>"
                "<div class='empty'><b>No CVs found.</b><p>The CV corpus is not "
                "committed to the repository. Put PDFs under {} and reload."
                "</p></div></div>").format(html.escape(where))
        return layout("Reviewer - AI CV Evaluator", body, mode="reviewer")

    options = "".join(
        "<label class='choice'><input type='checkbox' name='{name}'{checked}> "
        "<b>{label}</b><small>{note}</small></label>".format(
            name=name, checked=" checked" if on else "",
            label=html.escape(label), note=html.escape(note))
        for name, on, label, note in _REVIEWER_OPTIONS)

    body = ("<div class='card'><h2>Reviewer</h2>"
            "<form method='post' action='/reviewer/run'>"
            "<label for='job'>Job listing</label>"
            "<select id='job' name='job'>{jobs}</select>"
            "<fieldset><legend>Candidates</legend>"
            "<p class='hint'>Tick everyone you want to compare - one click "
            "each, and shift-click to take a whole run at once.</p>"
            "<div class='picker' id='cvs'>{cvs}</div>"
            "<div class='pickerbar'>"
            "<button type='button' id='cv-all'>Select all</button>"
            "<button type='button' id='cv-none'>Clear</button>"
            "<span class='count' id='cv-count'></span></div></fieldset>"
            "<fieldset><legend>Extra checks</legend>{options}</fieldset>"
            "<p><button type='submit'>Evaluate</button></p>"
            "</form></div>").format(jobs=_options(jobs, label_for),
                                    cvs=_cv_checkboxes(cvs, label_for),
                                    options=options)
    body += "<script>{}</script>".format(_CV_PICKER_SCRIPT)
    return layout("Reviewer - AI CV Evaluator", body, mode="reviewer")


_REVIEWER_STAGES = ("parse", "council", "essential", "judge", "rewrite")
_CANDIDATE_STAGES = {
    evaluation.REWRITE: ("parse", "rewrite_v2", "feedback"),
    evaluation.EVALUATE: ("parse", "council", "essential", "judge"),
    evaluation.EVALUATE_REWRITE: ("parse", "council", "essential", "judge",
                                  "rewrite_v2", "feedback"),
    # The rewrite is already text, so there is nothing to parse.
    evaluation.REEVALUATE: ("council", "essential", "judge"),
}
_CANDIDATE_DONE_TEXT = {
    evaluation.REWRITE: "See your rewritten CV",
    evaluation.EVALUATE: "See your scores",
    evaluation.EVALUATE_REWRITE: "See your scores and rewritten CV",
    evaluation.REEVALUATE: "See the new scores",
}


def stages_for(mode: str, action: str = None) -> tuple:
    """Which rows the progress board shows.

    Candidate mode runs only some of the reviewer stages, and showing the
    rest stuck on "waiting" forever would read as things that silently
    failed rather than things that were never asked for. A candidate run
    with no action predates the choice, when rewriting was all it did.
    """
    if mode != "candidate":
        return _REVIEWER_STAGES
    return _CANDIDATE_STAGES.get(action, _CANDIDATE_STAGES[evaluation.REWRITE])


# --- The live view ---------------------------------------------------------
#
# Every placeholder in the scripts below is substituted ONCE, on the fully
# assembled script, in live_view(). An earlier version substituted RUNID into
# the shared script and only then pasted the ranking renderer in, so the
# renderer's own RUNID was never replaced and every ranking row linked to a
# literal /run/RUNID/report/<n>. Assembling first and substituting last is what
# keeps that class of bug from coming back.

_LIVE_SCRIPT = """
const EVENTS_URL = '/run/RUNID/events';
const source = new EventSource(EVENTS_URL);

// One entry per candidate: its stage rows as last reported, the index of its
// payload (completion order, which is what the report route takes) and why
// it failed if it did. A batch's board is keyed by stage, not by candidate -
// ten CVs would otherwise give a fifty-row board - so the board describes one
// candidate at a time, and this is where the rest are kept.
const runs = {};
let live = null;       // the candidate the stream is on
let shown = null;      // the candidate the board is describing
let following = true;  // whether the board follows the stream
let finished = false;
let completed = 0;     // candidates finished so far = the next payload index

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function runFor(name) {
  if (!runs[name]) runs[name] = {stages: {}, index: null, error: null};
  return runs[name];
}
function detailText(detail) {
  // A field with no value is dropped rather than printed as "delta: null",
  // which tells the reviewer nothing and crowds out the fields that do.
  return Object.entries(detail || {})
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => k + ': ' + v).join('  ');
}
function heading() {
  // A re-evaluation carries no candidate name; there is nothing to say about
  // who is running, and a bare "Evaluating: " would read as a bug.
  if (!shown) return '';
  if (shown === live && !finished) return 'Evaluating: ' + escapeHtml(shown);
  let text = 'Showing: ' + escapeHtml(shown);
  if (!finished && live && live !== shown) {
    text += " &mdash; <a class='follow' href='#'>back to " +
      escapeHtml(live) + ", running now</a>";
  }
  return text;
}
// The board is redrawn from `runs` rather than written into as events land, so
// switching candidate is the same operation as showing progress, and a
// finished candidate's rows are still there to switch back to.
function draw() {
  const entry = shown === null ? null : runs[shown];
  document.querySelectorAll('#board tr').forEach((row) => {
    const cell = entry ? entry.stages[row.dataset.stage] : null;
    row.querySelector('.state').textContent = cell ? cell.state : 'waiting';
    row.querySelector('.detail').textContent = cell ? cell.detail : '';
  });
  document.getElementById('candidate').innerHTML = heading();
  document.getElementById('done').innerHTML = entry ? doneHtml(entry) : '';
  document.querySelectorAll('#ranking tr').forEach((row) => {
    row.classList.toggle('picked', row.dataset.candidate === shown);
  });
}
document.getElementById('candidate').addEventListener('click', (event) => {
  const link = event.target.closest('a.follow');
  if (!link) return;
  event.preventDefault();
  following = true;
  shown = live;
  draw();
});
source.onmessage = (message) => {
  const e = JSON.parse(message.data);
  // rank/done carries no candidate: taking a name off it would blank the
  // board at the very end of a batch.
  if (e.stage === 'rank') {
    renderRanking(e.detail.ranking || []);
    draw();
    return;
  }
  const name = e.candidate || '';
  const entry = runFor(name);
  if (name !== live) {
    live = name;
    if (following) shown = name;
  }
  if (e.stage === 'run' && e.state === 'done') {
    entry.index = completed;
    completed += 1;
  } else if (e.stage === 'run' && e.state === 'failed') {
    entry.error = e.detail.error || 'unknown error';
    // Appended under the board on purpose rather than written into a stage
    // row: a failed candidate must not vanish when the board moves on to the
    // next one.
    const item = document.createElement('li');
    item.innerHTML = '<b>' + escapeHtml(name || '?') + '</b> failed: ' +
      escapeHtml(entry.error);
    document.getElementById('failures').appendChild(item);
    document.getElementById('failures-card').hidden = false;
  } else if (e.stage !== 'run') {
    entry.stages[e.stage] = {state: e.state, detail: detailText(e.detail)};
  }
  draw();
};
source.addEventListener('end', () => {
  source.close();
  finished = true;
  draw();
});
"""

# Reviewer mode: the ranking is the way into every candidate's run, so a row
# both swaps the board to that candidate and carries a real link to their
# report - the link so ctrl-click and middle-click still open it in a tab, the
# handler so a plain click stays on this page and shows the run instead.
_REVIEWER_SCRIPT = """
const REPORT_URL = '/run/RUNID/report/';
function doneHtml(entry) {
  if (entry.error !== null) {
    return 'This candidate failed: ' + escapeHtml(entry.error);
  }
  if (entry.index === null) return '';
  return "<a href='" + REPORT_URL + entry.index + "'>Open the full report</a>";
}
function renderRanking(ranking) {
  const rows = ranking.map((r, i) => {
    // The row's own payload_index, not its rank: the ranking is by score and
    // /report/<n> indexes by completion order, and those differ whenever the
    // best candidate was not the first to finish.
    if (r.payload_index !== undefined && r.payload_index !== null) {
      runFor(r.candidate).index = r.payload_index;
    }
    return '<tr data-candidate="' + escapeHtml(r.candidate) + '"><td>' +
      (i + 1) + '</td><td><a class="pick" href="' + REPORT_URL +
      r.payload_index + '">' + escapeHtml(r.candidate) + '</a></td><td>' +
      (r.final_score === null ? '-' : r.final_score) + '</td></tr>';
  }).join('');
  const card = document.getElementById('ranking');
  card.innerHTML = '<h3>Ranking</h3><p class="hint">Pick a candidate to put ' +
    'their run back on the board above.</p><table>' + rows + '</table>';
  card.hidden = false;
}
document.getElementById('ranking').addEventListener('click', (event) => {
  const link = event.target.closest('a.pick');
  if (!link) return;
  // A modified click keeps the browser's own meaning - open the report in a
  // new tab - and only a plain one swaps the board.
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  following = false;
  shown = link.closest('tr').dataset.candidate;
  draw();
});
"""

# Candidate mode is one CV: no shortlist, so no ranking renderer and no
# ranking card. The stub keeps the shared handler above from needing to know
# which mode it is running in.
_CANDIDATE_SCRIPT = """
const RESULTS_URL = '/candidate/RUNID';
function doneHtml(entry) {
  if (entry.error !== null) return 'This run failed: ' + escapeHtml(entry.error);
  if (entry.index === null) return '';
  return "<a href='" + RESULTS_URL + "'>DONETEXT</a>";
}
function renderRanking(ranking) {}
"""


def live_view(run_id: str, mode: str = "reviewer", action: str = None) -> str:
    """Stage list that fills in from the event stream.

    One view for both modes: the event stream, the board and the failure list
    are identical, and only the stage names and where "done" points differ.

    In reviewer mode the board is also addressable. A batch reuses one set of
    stage rows for every candidate, so progress for the candidate that
    finished ten minutes ago used to be gone for good; now every candidate's
    rows are kept, and picking a name out of the ranking puts that candidate's
    run - and the link to their full report - back on the board.
    """
    rows = "".join(
        "<tr id='stage-{s}' data-stage='{s}'><td>{s}</td>"
        "<td class='state'>waiting</td><td class='detail'></td></tr>".format(s=s)
        for s in stages_for(mode, action))
    if mode == "candidate":
        done_text = _CANDIDATE_DONE_TEXT.get(
            action, _CANDIDATE_DONE_TEXT[evaluation.REWRITE])
        mode_script = _CANDIDATE_SCRIPT.replace("DONETEXT", done_text)
        ranking_card = ""
    else:
        mode_script = _REVIEWER_SCRIPT
        # Hidden until a rank event fills it: a single-CV run never emits one,
        # and a permanently empty "Ranking" panel reads as something that
        # failed rather than something that was never asked for.
        ranking_card = "<div class='card' id='ranking' hidden></div>"
    script = (mode_script + _LIVE_SCRIPT).replace("RUNID", run_id)
    body = ("<div class='card'><h2>Running</h2>"
            "<p id='candidate'></p>"
            "<table id='board'>{rows}</table>"
            "<p id='done'></p></div>"
            "<div class='card' id='failures-card' hidden>"
            "<h2>Failed candidates</h2><ul id='failures'></ul></div>"
            "{ranking_card}"
            "<script>{script}</script>").format(rows=rows, script=script,
                                                ranking_card=ranking_card)
    return layout("Running - AI CV Evaluator", body)


def runs_index(rows) -> str:
    if not rows:
        body = "<div class='card'><h2>Saved runs</h2><p>Nothing yet.</p></div>"
    else:
        items = "".join("<tr><td><a href='{}'>{}</a></td></tr>".format(
            html.escape(href), html.escape(text)) for href, text in rows)
        body = ("<div class='card'><h2>Saved runs</h2><table>{}</table></div>"
                .format(items))
    return layout("Saved runs - AI CV Evaluator", body)


def candidate_page(message: str, available: bool = False) -> str:
    """The honest empty state: rewriter v2 could not be reached at all.

    `available` is kept for the case where v2 imports but the corpus is empty,
    so "we cannot load the rewriter" and "there is nothing to rewrite" do not
    share a heading - they are different problems with different fixes.
    """
    heading = "Nothing to rewrite." if available else "Not available."
    body = ("<div class='card'><h2>Candidate</h2>"
            "<p>Aim your CV at a listing you are applying for. The result is a "
            "Word document you can edit.</p>"
            "<div class='empty'><b>{}</b><p>{}</p></div>"
            "</div>").format(heading, html.escape(message))
    return layout("Candidate - AI CV Evaluator", body, mode="candidate")


_ACTION_CHOICES = (
    (evaluation.REWRITE, "Rewrite",
     "Two models each rewrite your CV for the listing, then a judge "
     "recommends one and suggests where your career could go next. You can "
     "have a version scored afterwards."),
    (evaluation.EVALUATE, "Evaluate",
     "A council of models scores your CV against the listing as it stands, "
     "checks it against the essential requirements, and says why."),
    (evaluation.EVALUATE_REWRITE, "Evaluate and rewrite",
     "Scores your CV first, then rewrites it. The slowest option."),
)


_REWRITER_LABELS = {
    candidate_mod.LOCAL: (
        "This computer",
        "{models}, through Ollama on this machine. Your CV never leaves it."),
    candidate_mod.CLOUD: (
        "Ollama Cloud",
        "{models} - larger models than this computer can run. Your CV text "
        "is sent to ollama.com to be rewritten. If Ollama Cloud fails (no "
        "connection, or the usage limit is reached), that version is "
        "rewritten on this computer instead, and the results say so."),
}

# Rewriting models mean nothing to an evaluate-only run, so the choice is
# disabled rather than left looking as if it mattered. A disabled fieldset is
# not submitted, and the route reads a missing choice as the local default.
_REWRITER_SCRIPT = """
(function () {
  const box = document.getElementById('rewriter');
  if (!box) return;
  function sync() {
    const picked = document.querySelector("input[name='action']:checked");
    box.disabled = !!picked && picked.value === 'EVALUATE';
  }
  document.querySelectorAll("input[name='action']").forEach(
    (radio) => radio.addEventListener('change', sync));
  sync();
})();
""".replace("EVALUATE", evaluation.EVALUATE)


def _rewriter_fieldset(rewriters) -> str:
    """Where the rewrite runs: this computer (the default) or Ollama Cloud.

    A choice that cannot run is shown disabled with the reason, rather than
    hidden - a candidate who knows the cloud option exists can ask for it to
    be set up.
    """
    rows = []
    for choice in rewriters:
        value = choice["value"]
        label, note = _REWRITER_LABELS[value]
        note = note.format(models=" and ".join(choice["models"]))
        why_not = choice.get("unavailable")
        rows.append(
            "<label class='choice'><input type='radio' name='rewriter' "
            "value='{v}'{checked}{disabled}> <b>{label}</b><small>{note}"
            "</small>{why_not}</label>".format(
                v=html.escape(value), label=html.escape(label),
                note=html.escape(note),
                why_not=("<small><b>Can't be picked yet.</b> {}</small>".format(
                    html.escape(why_not)) if why_not else ""),
                checked=" checked" if value == candidate_mod.LOCAL else "",
                disabled=" disabled" if why_not else ""))
    return ("<fieldset id='rewriter'><legend>Rewriting models</legend>"
            "<p class='hint'>Only used when rewriting. Scoring and the rewrite "
            "judge always run on this computer.</p>{}</fieldset>"
            ).format("".join(rows))


def candidate_form(jobs, cvs, label_for, rewriters=None) -> str:
    """Pick one CV, one listing, and what to do with them.

    Single-select on both, unlike reviewer mode: v2 rewrites one CV for one
    job, and the two agents it runs are the comparison here - a second CV
    would just double an already slow run.

    rewriters is candidate.rewriter_choices(); without it the form offers no
    choice and every rewrite runs locally.
    """
    choices = "".join(
        "<label class='choice'><input type='radio' name='action' value='{v}'"
        "{checked}> <b>{label}</b><small>{note}</small></label>".format(
            v=value, label=html.escape(label), note=html.escape(note),
            checked=" checked" if value == evaluation.EVALUATE_REWRITE else "")
        for value, label, note in _ACTION_CHOICES)
    where = _rewriter_fieldset(rewriters) if rewriters else ""
    body = ("<div class='card'><h2>Candidate</h2>"
            "<p>Aim your CV at a listing you are applying for: see how it "
            "scores, have it rewritten, or both.</p>"
            "<form method='post' action='/candidate/run'>"
            "<label for='job'>Job listing</label>"
            "<select id='job' name='job'>{jobs}</select>"
            "<label for='cv'>Your CV</label>"
            "<select id='cv' name='cv' size='10'>{cvs}</select>"
            "<fieldset><legend>What to do</legend>{choices}</fieldset>"
            "{where}"
            "<p><button type='submit'>Start</button></p>"
            "</form></div>").format(jobs=_options(jobs, label_for),
                                    cvs=_options(cvs, label_for),
                                    choices=choices, where=where)
    if where:
        body += "<script>{}</script>".format(_REWRITER_SCRIPT)
    return layout("Candidate - AI CV Evaluator", body, mode="candidate")


def _score(value) -> str:
    return "&mdash;" if value is None else html.escape(str(value))


def _essential_note(label: str, result) -> str:
    if result.get("error"):
        text = "The check failed: {}".format(result["error"])
    else:
        text = result.get("summary") or "The check returned no summary."
    return "<p><b>{}</b> {}</p>".format(label, html.escape(str(text)))


def _evaluation_card(payload) -> str:
    """The council's scores and the judge's verdict, in a candidate's terms.

    On a re-evaluation, each score sits beside what the original CV got from
    the same model, so the page answers the question that prompted it: did
    the rewrite do better?
    """
    council = payload.get("council") or []
    judge = payload.get("judge") or {}
    before = payload.get("before")
    again = payload.get("action") == evaluation.REEVALUATE

    if judge.get("error"):
        verdict = "The judge failed: {}".format(judge["error"])
    else:
        verdict = judge.get("final_evaluation", "")
    headline = "<p class='score'>{}<small> / 10 overall</small></p>".format(
        _score(judge.get("final_score")))

    compare = ""
    prior = {}
    if before:
        prior = {r.get("model"): r.get("score")
                 for r in before.get("council") or []}
        compare = "<p>Your original CV scored <b>{}</b> overall.</p>".format(
            _score((before.get("judge") or {}).get("final_score")))
    elif again:
        compare = ("<p>Your original CV was not scored in the run this "
                   "version came from, so there is nothing to compare it "
                   "against.</p>")

    notes = "".join(
        "<p><b>{}</b> {}</p>".format(label, html.escape(str(judge[key])))
        for key, label in (("key_strengths", "Strengths:"),
                           ("key_weaknesses", "Gaps:"))
        if judge.get(key))
    if payload.get("essential"):
        notes += _essential_note("Essential requirements:", payload["essential"])
        if payload.get("judge_saw_essential") is False:
            notes += ("<p><small>The overall score above was given without "
                      "this check.</small></p>")
        if before and before.get("essential"):
            notes += _essential_note("Essential requirements, original CV:",
                                     before["essential"])

    was = "<th>Original</th>" if before else ""
    rows = "".join(
        "<tr><td>{model}</td><td>{score}</td>{was}<td>{summary}</td></tr>".format(
            model=html.escape(str(r.get("model", "?"))),
            score=_score(r.get("score")),
            was="<td>{}</td>".format(_score(prior.get(r.get("model"))))
                if before else "",
            summary=html.escape(str(r.get("summary", ""))))
        for r in council)

    heading = ("Scores for this rewrite" if again
               else "How the council scored your CV")
    return ("<div class='card'><h3>{heading}</h3>{headline}{compare}"
            "<p>{verdict}</p>{notes}"
            "<table><tr><th>Model</th><th>Score</th>{was}<th>Why</th></tr>"
            "{rows}</table></div>").format(
                heading=heading, headline=headline, compare=compare,
                verdict=html.escape(str(verdict)), notes=notes, was=was,
                rows=rows)


def _feedback_card(feedback, results) -> str:
    """The rewrite judge's recommendation, roadmap and alternative careers.

    Placed above the versions it compares, so the candidate reads which one
    to send before pages of CV text rather than after. A recommended version
    that is really a fallback is called out: the judge is told about
    fallbacks but can still pick one, and "send this version" must not read
    as "send this rewrite" when it is the candidate's original CV.
    """
    if feedback.get("error"):
        return ("<div class='card'><h3>Which version to send</h3>"
                "<div class='empty'><b>No recommendation.</b><p>The rewrite "
                "judge failed: {}. The versions below are unaffected.</p>"
                "</div></div>").format(html.escape(str(feedback["error"])))

    pick = str(feedback.get("recommended_rewrite", ""))
    chosen = next((r for r in results if r.agent_model == pick), None)
    warning = ""
    if chosen is not None and chosen.used_fallback:
        warning = ("<div class='empty'><b>That version is your original CV."
                   "</b><p>The check rejected {}'s rewrite, so following this "
                   "recommendation means sending your CV as it stands.</p>"
                   "</div>").format(html.escape(pick))

    steps = "".join(
        "<li><b>{}</b> {}</li>".format(html.escape(str(s.get("step", ""))),
                                       html.escape(str(s.get("action", ""))))
        for s in feedback.get("career_roadmap") or [] if isinstance(s, dict))
    careers = "".join("<li>{}</li>".format(html.escape(str(c)))
                      for c in feedback.get("alternative_careers") or [])
    roadmap = "<h3>Career roadmap</h3><ol>{}</ol>".format(steps) if steps else ""
    others = ("<h3>Other careers to consider</h3><ul>{}</ul>".format(careers)
              if careers else "")

    return ("<div class='card'><h3>Which version to send</h3>"
            "<p>Recommended: <b>{pick}</b></p>{warning}<p>{reason}</p>"
            "{roadmap}{others}</div>").format(
                pick=html.escape(pick), warning=warning,
                reason=html.escape(str(feedback.get("recommendation_reason", ""))),
                roadmap=roadmap, others=others)


def _rewrite_card(run_id: str, index: int, result, recommended=False) -> str:
    """One agent's version, with the guard's verdict stated before the text.

    The verdict goes above the rewrite, not below it, because a fallback is
    not a rewrite at all and reading it as one for three paragraphs before
    finding that out is the exact confusion rewriter_v2 warns against.
    """
    error = getattr(result, "error", None)
    cloud = _cloud_note(result)
    if error:
        return ("<div class='card'><h3>{model}</h3>{cloud}"
                "<div class='empty'><b>This model failed.</b><p>{err}</p></div>"
                "</div>").format(model=html.escape(result.agent_model),
                                 cloud=cloud, err=html.escape(str(error)))

    if result.used_fallback:
        note = ("<div class='empty'><b>This is your original CV, not a "
                "rewrite.</b><p>{} It was discarded rather than shown to you "
                "as if it were sound.</p></div>").format(
                    html.escape(candidate_mod.describe_rejection(result.reasons)))
        button = "<p><button type='button' disabled>Nothing to download</button></p>"
    else:
        note = ""
        if result.rejected:
            note = ("<div class='empty'><b>Flagged.</b><p>Read this against "
                    "your original before sending it anywhere.</p></div>")
        button = ("<div class='actions'>"
                  "<a href='/candidate/{rid}/download/{idx}'>"
                  "<button type='button'>Download as Word</button></a>"
                  "<form method='post' "
                  "action='/candidate/{rid}/reevaluate/{idx}'>"
                  "<button type='submit'>Re-evaluate this version</button>"
                  "</form></div>").format(rid=html.escape(run_id), idx=index)

    reasons = ""
    if result.reasons:
        reasons = "<ul>{}</ul>".format("".join(
            "<li>{}</li>".format(html.escape(str(r))) for r in result.reasons))

    tag = _where_tag(result)
    if recommended:
        tag += " <small>&middot; recommended</small>"
    return ("<div class='card'><h3>{model}{tag}</h3>{cloud}{note}{reasons}"
            "{button}<pre class='rewrite'>{text}</pre></div>").format(
                model=html.escape(result.agent_model), tag=tag, cloud=cloud,
                note=note, reasons=reasons, button=button,
                text=html.escape(result.rewritten_text or ""))


def _where_tag(result) -> str:
    """Where a version was written, when the candidate asked for the cloud.

    A local run has no tag at all - it is where every rewrite ran before the
    cloud existed, and tagging it would clutter the common case.
    """
    if getattr(result, "replaced_cloud_model", None):
        return " <small>&middot; on this computer</small>"
    if getattr(result, "provider", None) == candidate_mod.CLOUD:
        return " <small>&middot; Ollama Cloud</small>"
    return ""


def _cloud_note(result) -> str:
    """Why a version the candidate asked the cloud for was written locally.

    Stated above the text, like the guard's verdict: the candidate picked
    Ollama Cloud, and reading a local model's rewrite as the cloud's would
    be comparing the wrong things.
    """
    replaced = getattr(result, "replaced_cloud_model", None)
    if not replaced:
        return ""
    cloud_error = getattr(result, "cloud_error", None)
    return ("<div class='empty'><b>Rewritten on this computer instead.</b>"
            "<p>{why} So {local} on this computer took the place of "
            "{cloud}.</p><p class='hint'>Details: {detail}</p></div>").format(
                why=html.escape(candidate_mod.describe_cloud_error(cloud_error)),
                local=html.escape(result.agent_model),
                cloud=html.escape(replaced),
                detail=html.escape(str(cloud_error or "none given")))


def candidate_results(run_id: str, payload) -> str:
    """Whatever the run produced: scores, rewrites side by side, or both.

    Each real rewrite carries its own download and its own re-evaluate
    button - which version to score again is the candidate's call.
    """
    results = payload.get("rewrites") or []
    action = payload.get("action")
    job = html.escape(payload.get("job_title", "the listing"))
    if action == evaluation.REEVALUATE:
        intro = "{}'s rewrite, scored again against <b>{}</b>.".format(
            html.escape(payload.get("agent_model", "A model")), job)
        title = "New scores"
    elif results:
        intro = ("Rewritten for <b>{}</b> by {} models. They differ - that is "
                 "the point. Read both and pick the one that represents you "
                 "honestly, or have one scored again.").format(job, len(results))
        replaced = sum(1 for r in results
                       if getattr(r, "replaced_cloud_model", None))
        if replaced:
            intro += (" You picked Ollama Cloud, but it failed, so {} on this "
                      "computer - see the note on {}.").format(
                          "one version was rewritten" if replaced == 1
                          else "{} versions were rewritten".format(replaced),
                          "it" if replaced == 1 else "each")
        title = "Your rewritten CV"
    else:
        intro = "Scored against <b>{}</b>.".format(job)
        title = "Your scores"

    header = "<div class='card'><h2>{}</h2><p>{}</p></div>".format(
        html.escape(payload.get("candidate", "Your CV")), intro)
    scores = (_evaluation_card(payload)
              if payload.get("council") or payload.get("judge") else "")
    feedback = payload.get("feedback")
    advice = _feedback_card(feedback, results) if feedback else ""
    pick = (feedback or {}).get("recommended_rewrite")
    cards = "".join(_rewrite_card(run_id, i, r, recommended=r.agent_model == pick)
                    for i, r in enumerate(results))
    return layout("{} - AI CV Evaluator".format(title),
                  header + scores + advice + cards, mode="candidate")
