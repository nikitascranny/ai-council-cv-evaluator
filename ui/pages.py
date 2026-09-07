# ui/pages.py
"""The app's own pages.

The run report has its own renderer in ui/report.py and its own inline CSS;
these are the pages around it - picking a mode, choosing inputs, listing runs.
"""

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
.modes { display: flex; gap: 18px; }
.modes a { flex: 1; text-decoration: none; color: inherit; }
.empty { background: #fff8e6; border: 1px solid #e6cf8b; border-radius: 8px;
         padding: 16px; }
label { display: block; margin: 10px 0 4px; font-weight: 600; }
select, button { font: inherit; padding: 8px; }
button { background: #16181d; color: #fff; border: 0; border-radius: 6px;
         padding: 10px 18px; cursor: pointer; }
table { border-collapse: collapse; width: 100%; }
td, th { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e6e8ec; }
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
    body = ("<div class='modes'>"
            "<a href='/reviewer'><div class='card'><h2>Reviewer</h2>"
            "<p>Assess candidates against a job listing. Shows the score, how "
            "much of it was wording rather than substance, and the facts the CV "
            "states.</p></div></a>"
            "<a href='/candidate'><div class='card'><h2>Candidate</h2>"
            "<p>Aim your own CV at a listing you are applying for.</p>"
            "</div></a></div>")
    return layout("AI CV Evaluator", body)


def _options(paths, label_for) -> str:
    return "".join("<option value='{}'>{}</option>".format(
        html.escape(str(p)), html.escape(label_for(p))) for p in paths)


def reviewer_form(jobs, cvs, status, settings, label_for) -> str:
    if not cvs:
        where = ", ".join(status["cv_dirs"])
        body = ("<div class='card'><h2>Reviewer</h2>"
                "<div class='empty'><b>No CVs found.</b><p>The CV corpus is not "
                "committed to the repository. Put PDFs under {} and reload."
                "</p></div></div>").format(html.escape(where))
        return layout("Reviewer - AI CV Evaluator", body, mode="reviewer")

    body = ("<div class='card'><h2>Reviewer</h2>"
            "<form method='post' action='/reviewer/run'>"
            "<label for='job'>Job listing</label>"
            "<select id='job' name='job'>{jobs}</select>"
            "<label for='cv'>Candidates <small>(select more than one to "
            "compare)</small></label>"
            "<select id='cv' name='cv' multiple size='10'>{cvs}</select>"
            "<label><input type='checkbox' name='essential'> Essential "
            "requirements</label>"
            "<label><input type='checkbox' name='rewrite' checked> "
            "Phrasing-sensitivity probe</label>"
            "<label><input type='checkbox' name='fast'> Fast models</label>"
            "<p><button type='submit'>Evaluate</button></p>"
            "</form></div>").format(jobs=_options(jobs, label_for),
                                    cvs=_options(cvs, label_for))
    return layout("Reviewer - AI CV Evaluator", body, mode="reviewer")


def live_view(run_id: str) -> str:
    """Stage list that fills in from the event stream."""
    rows = "".join(
        "<tr id='stage-{s}'><td>{s}</td><td class='state'>waiting</td>"
        "<td class='detail'></td></tr>".format(s=s)
        for s in ("parse", "council", "essential", "judge", "rewrite"))
    script = """
    const source = new EventSource('/run/RUNID/events');
    let currentCandidate = null;
    function escapeHtml(s) {
      return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    function resetBoard() {
      document.querySelectorAll('#board tr').forEach((row) => {
        row.querySelector('.state').textContent = 'waiting';
        row.querySelector('.detail').textContent = '';
      });
    }
    function ensureCandidate(name) {
      // A batch's stage rows are shared across candidates - keyed only by
      // stage, not by candidate, or ten CVs would give a fifty-row board.
      // So when the candidate changes, name it and reset the rows rather
      // than silently letting the next candidate's progress overwrite the
      // previous one's under the same label.
      if (!name || name === currentCandidate) return;
      currentCandidate = name;
      document.getElementById('candidate').innerHTML =
        'Evaluating: ' + escapeHtml(name);
      resetBoard();
    }
    source.onmessage = (message) => {
      const e = JSON.parse(message.data);
      ensureCandidate(e.candidate);
      if (e.stage === 'run' && e.state === 'done') {
        document.getElementById('done').innerHTML =
          "<a href='/run/RUNID/report'>Open the full report</a>";
        return;
      }
      if (e.stage === 'run' && e.state === 'failed') {
        // Survives resetBoard() on purpose: this appends to a list under the
        // board rather than writing into a stage row, so a failed candidate
        // does not vanish when the next candidate's progress resets the rows.
        const item = document.createElement('li');
        item.innerHTML = '<b>' + escapeHtml(e.candidate || '?') +
          '</b> failed: ' + escapeHtml(e.detail.error || 'unknown error');
        document.getElementById('failures').appendChild(item);
        document.getElementById('failures-card').hidden = false;
        return;
      }
      if (e.stage === 'rank') { renderRanking(e.detail.ranking); return; }
      const row = document.getElementById('stage-' + e.stage);
      if (!row) return;
      row.querySelector('.state').textContent = e.state;
      row.querySelector('.detail').textContent =
        Object.entries(e.detail).map(([k, v]) => k + ': ' + v).join('  ');
    };
    source.addEventListener('end', () => source.close());
    function renderRanking(ranking) {
      document.getElementById('ranking').innerHTML =
        '<h3>Ranking</h3><table>' + ranking.map((r, i) =>
          '<tr><td>' + (i + 1) + '</td><td><a href="/run/RUNID/report/' +
          r.payload_index + '">' + escapeHtml(r.candidate) +
          '</a></td><td>' + (r.final_score === null ? '-' : r.final_score) +
          '</td></tr>').join('') + '</table>';
    }
    """.replace("RUNID", run_id)
    body = ("<div class='card'><h2>Running</h2>"
            "<p id='candidate'></p>"
            "<table id='board'>{rows}</table>"
            "<p id='done'></p></div>"
            "<div class='card' id='failures-card' hidden>"
            "<h2>Failed candidates</h2><ul id='failures'></ul></div>"
            "<div class='card' id='ranking'></div>"
            "<script>{script}</script>").format(rows=rows, script=script)
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
    """`available` distinguishes "the backend is not wired up" (the honest
    default, since rewriter v2 does not exist yet) from "it is wired up but
    this page has no CV-picking form" (I2/M2's actual-adapter case) - the two
    are different messages and must not share the same "Not available yet"
    heading.
    """
    heading = "Not built yet." if available else "Not available yet."
    body = ("<div class='card'><h2>Candidate</h2>"
            "<p>Aim your CV at a listing you are applying for. The result is a "
            "Word document you can edit.</p>"
            "<div class='empty'><b>{}</b><p>{}</p></div>"
            "</div>").format(heading, html.escape(message))
    return layout("Candidate - AI CV Evaluator", body, mode="candidate")
