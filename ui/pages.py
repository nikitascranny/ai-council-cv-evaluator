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
.modes { display: flex; gap: 18px; }
.modes a { flex: 1; text-decoration: none; color: inherit; }
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
.actions { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
.actions form, .actions a { margin: 0; }
pre.rewrite { white-space: pre-wrap; overflow-wrap: anywhere; }
.score { font-size: 34px; font-weight: 600; margin: 4px 0; }
.score small { font-size: 15px; font-weight: 400; color: #5b606b; }
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


def live_view(run_id: str, mode: str = "reviewer", action: str = None) -> str:
    """Stage list that fills in from the event stream.

    One view for both modes: the event stream, the board and the failure list
    are identical, and only the stage names and where "done" points differ.
    """
    rows = "".join(
        "<tr id='stage-{s}'><td>{s}</td><td class='state'>waiting</td>"
        "<td class='detail'></td></tr>".format(s=s)
        for s in stages_for(mode, action))
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
          "<a href='DONEHREF'>DONETEXT</a>";
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
      if (e.stage === 'rank') { RANKCALL return; }
      const row = document.getElementById('stage-' + e.stage);
      if (!row) return;
      row.querySelector('.state').textContent = e.state;
      row.querySelector('.detail').textContent =
        Object.entries(e.detail).map(([k, v]) => k + ': ' + v).join('  ');
    };
    source.addEventListener('end', () => source.close());
    RANKFN
    """.replace("RUNID", run_id)
    # Candidate mode never emits a rank event - one CV, no shortlist - so its
    # page ships neither the renderer nor the card. Leaving them in would put
    # an empty "Ranking" panel and a /report link on a page that has no report.
    ranking_fn = """function renderRanking(ranking) {
      document.getElementById('ranking').innerHTML =
        '<h3>Ranking</h3><table>' + ranking.map((r, i) =>
          '<tr><td>' + (i + 1) + '</td><td><a href="/run/RUNID/report/' +
          r.payload_index + '">' + escapeHtml(r.candidate) +
          '</a></td><td>' + (r.final_score === null ? '-' : r.final_score) +
          '</td></tr>').join('') + '</table>';
    }"""
    if mode == "candidate":
        done_text = _CANDIDATE_DONE_TEXT.get(
            action, _CANDIDATE_DONE_TEXT[evaluation.REWRITE])
        script = (script.replace("DONEHREF", "/candidate/" + run_id)
                        .replace("DONETEXT", done_text)
                        .replace("RANKCALL", "").replace("RANKFN", ""))
        ranking_card = ""
    else:
        script = (script.replace("DONEHREF", "/run/" + run_id + "/report")
                        .replace("DONETEXT", "Open the full report")
                        .replace("RANKCALL", "renderRanking(e.detail.ranking);")
                        .replace("RANKFN", ranking_fn))
        ranking_card = "<div class='card' id='ranking'></div>"
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


def candidate_form(jobs, cvs, label_for) -> str:
    """Pick one CV, one listing, and what to do with them.

    Single-select on both, unlike reviewer mode: v2 rewrites one CV for one
    job, and the two agents it runs are the comparison here - a second CV
    would just double an already slow run.
    """
    choices = "".join(
        "<label class='choice'><input type='radio' name='action' value='{v}'"
        "{checked}> <b>{label}</b><small>{note}</small></label>".format(
            v=value, label=html.escape(label), note=html.escape(note),
            checked=" checked" if value == evaluation.EVALUATE_REWRITE else "")
        for value, label, note in _ACTION_CHOICES)
    body = ("<div class='card'><h2>Candidate</h2>"
            "<p>Aim your CV at a listing you are applying for: see how it "
            "scores, have it rewritten, or both.</p>"
            "<form method='post' action='/candidate/run'>"
            "<label for='job'>Job listing</label>"
            "<select id='job' name='job'>{jobs}</select>"
            "<label for='cv'>Your CV</label>"
            "<select id='cv' name='cv' size='10'>{cvs}</select>"
            "<fieldset><legend>What to do</legend>{choices}</fieldset>"
            "<p><button type='submit'>Start</button></p>"
            "</form></div>").format(jobs=_options(jobs, label_for),
                                    cvs=_options(cvs, label_for),
                                    choices=choices)
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
    if error:
        return ("<div class='card'><h3>{model}</h3>"
                "<div class='empty'><b>This model failed.</b><p>{err}</p></div>"
                "</div>").format(model=html.escape(result.agent_model),
                                 err=html.escape(str(error)))

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

    tag = " <small>&middot; recommended</small>" if recommended else ""
    return ("<div class='card'><h3>{model}{tag}</h3>{note}{reasons}{button}"
            "<pre class='rewrite'>{text}</pre></div>").format(
                model=html.escape(result.agent_model), tag=tag, note=note,
                reasons=reasons, button=button,
                text=html.escape(result.rewritten_text or ""))


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
