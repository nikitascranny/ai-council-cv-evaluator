# ui/report.py
"""The run as one self-contained HTML page.

Moved out of the console app unchanged. It stays self-contained - inline CSS,
no network - because the same markup is both served by the web app and written
to disk next to the saved JSON.
"""

import html
from pathlib import Path

from ui import facts as facts_mod


def word_diff(before: str, after: str):
    """Word-level diff as two HTML fragments: deletions left, insertions right.

    This is what makes "only the wording changed" visible rather than asserted.
    Every token is escaped - CV text is not trusted markup.
    """
    import difflib

    a, b = before.split(), after.split()
    left, right = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        cut_a = html.escape(" ".join(a[i1:i2]))
        cut_b = html.escape(" ".join(b[j1:j2]))
        if tag == "equal":
            left.append(cut_a)
            right.append(cut_b)
        else:
            if cut_a:
                left.append("<del>{}</del>".format(cut_a))
            if cut_b:
                right.append("<ins>{}</ins>".format(cut_b))
    return " ".join(left), " ".join(right)


_REPORT_CSS = """
:root { --bg:#faf9f7; --fg:#1c1b19; --dim:#6b6862; --line:#e0ddd6;
        --card:#fff; --ins:#1f7a3d; --insbg:#e4f5ea; --del:#a3382c; --delbg:#fdeae7; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16150f; --fg:#eceae4; --dim:#9a968c; --line:#33312a;
          --card:#1e1d16; --ins:#6ddc95; --insbg:#16351f; --del:#f2917f; --delbg:#3a1c17; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif; }
header { padding:22px 28px; border-bottom:1px solid var(--line); }
h1 { margin:0 0 4px; font-size:23px; letter-spacing:-.01em; }
.sub { color:var(--dim); font-size:14px; }
.strip { display:flex; flex-wrap:wrap; gap:14px; padding:18px 28px;
         border-bottom:1px solid var(--line); }
.tile { background:var(--card); border:1px solid var(--line); border-radius:9px;
        padding:11px 15px; min-width:132px; }
.tile .k { font-size:11px; text-transform:uppercase; letter-spacing:.07em;
           color:var(--dim); }
.tile .v { font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; }
.tile .n { font-size:12px; color:var(--dim); }
.panes { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
         gap:18px; padding:22px 28px; align-items:start; }
.pane { background:var(--card); border:1px solid var(--line); border-radius:11px;
        overflow:hidden; }
.pane h2 { margin:0; padding:11px 16px; font-size:12px; text-transform:uppercase;
           letter-spacing:.08em; color:var(--dim); border-bottom:1px solid var(--line); }
.body { padding:16px; white-space:pre-wrap; word-wrap:break-word;
        font:13.5px/1.6 ui-monospace, "Cascadia Code", Consolas, monospace;
        max-height:60vh; overflow:auto; }
ins { background:var(--insbg); color:var(--ins); text-decoration:none;
      border-radius:3px; padding:0 2px; }
del { background:var(--delbg); color:var(--del); text-decoration:line-through;
      border-radius:3px; padding:0 2px; }
.tags { display:flex; flex-wrap:wrap; gap:6px; padding:0 16px 14px; }
.tag { font-size:11px; border:1px solid var(--line); border-radius:20px;
       padding:2px 9px; color:var(--dim); }
.flag { margin:0 28px 6px; padding:12px 16px; border-radius:9px;
        background:var(--delbg); color:var(--del); font-size:14px; }
table { width:100%; border-collapse:collapse; }
td, th { padding:7px 16px; text-align:left; border-bottom:1px solid var(--line);
         font-size:13.5px; vertical-align:top; }
th { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); }
td.s { font-weight:600; font-variant-numeric:tabular-nums; width:46px; }
"""


def _tile(label, value, note=""):
    return ('<div class="tile"><div class="k">{}</div><div class="v">{}</div>'
            '<div class="n">{}</div></div>').format(
        html.escape(label), value, html.escape(note))


def _pane(title, body_html, tags=()):
    chips = "".join('<span class="tag">{}</span>'.format(html.escape(str(t)))
                    for t in tags)
    return ('<section class="pane"><h2>{}</h2><div class="body">{}</div>'
            '{}</section>').format(
        html.escape(title), body_html,
        '<div class="tags">{}</div>'.format(chips) if chips else "")


def render_report(payload: dict) -> str:
    """The whole run as one self-contained page.

    Local file, no server and no network: it has to open in a browser next to
    the terminal with nothing else running.
    """
    candidate = payload.get("candidate", "?")
    job = payload.get("job", "?")
    council = payload.get("council") or []
    judge = payload.get("judge") or {}
    rewrite = payload.get("rewrite") or {}
    rewritten_text = payload.get("rewritten_text") or ""
    cv_text = payload.get("cv_text") or ""

    tiles = []
    for row in council:
        score = row.get("score")
        tiles.append(_tile(row.get("model", "?"),
                           "&mdash;" if score is None else score, "council"))
    # Whether the judge read the essential-requirements check changes what its
    # score means. A run saved before that was recorded says neither.
    saw_essential = payload.get("judge_saw_essential")
    if judge:
        final = judge.get("final_score")
        if judge.get("error"):
            note = "MODEL ERROR: {}".format(judge["error"])
        elif saw_essential is True:
            note = "final verdict, with essential check"
        elif saw_essential is False:
            note = "final verdict, council only"
        else:
            note = "final verdict"
        tiles.append(_tile("judge", "&mdash;" if final is None else final, note))
    if rewrite:
        delta = rewrite.get("delta")
        shown = "&mdash;" if delta is None else ("+{}".format(delta) if delta > 0 else delta)
        note = "MODEL ERROR: {}".format(rewrite["error"]) if rewrite.get("error") \
            else "{} -> {}".format(rewrite.get("score_before"), rewrite.get("score_after"))
        tiles.append(_tile("rewrite delta", shown, note))

    similarity = rewrite.get("similarity")
    if similarity is not None:
        tiles.append(_tile("unchanged", "{:.0%}".format(similarity),
                           "word overlap"))

    flag = ""
    # A rewrite that barely changed anything makes delta meaningless: the
    # ranking did not ignore a rephrasing, there was no rephrasing. Nothing
    # else on the row shows this - a no-op has an ideal length ratio, invents
    # nothing and loses nothing.
    if similarity is not None and similarity >= 0.95:
        flag = ('<div class="flag"><b>The rewriter barely changed anything &mdash; '
                'a no-op rewrite.</b> {:.1%} of the words are unchanged, so this '
                'run says nothing about phrasing sensitivity: the delta of {} '
                'reflects an absent rewrite, not an indifferent ranking. Exclude '
                'it rather than averaging it in.</div>').format(
            similarity, rewrite.get("delta"))

    if rewrite.get("rejected"):
        reasons = ", ".join(str(r) for r in (rewrite.get("reasons") or []))
        flag = ('<div class="flag"><b>Rewrite rejected by the fact guard.</b> '
                'The paraphrase introduced material the original did not contain: '
                '{}. This is a result about the rewriting model, not a failure of '
                'the run.</div>').format(html.escape(reasons) or "unrecorded")

    panes = [_pane("Job listing", html.escape(payload.get("job_text") or ""))]

    if rewritten_text:
        left, right = word_diff(cv_text, rewritten_text)
        panes.append(_pane("Original CV (as parsed)", left,
                           payload.get("sections") or ()))
        panes.append(_pane("Rewritten CV", right))
    else:
        panes.append(_pane("CV (as parsed)", html.escape(cv_text),
                           payload.get("sections") or ()))

    rows = "".join(
        "<tr><td class='s'>{}</td><td>{}</td><td>{}</td></tr>".format(
            "&mdash;" if r.get("score") is None else r.get("score"),
            html.escape(str(r.get("model", "?"))),
            html.escape(str(r.get("summary", ""))))
        for r in council)
    essential = payload.get("essential")
    if essential:
        if essential.get("error"):
            text = "CHECK FAILED: {}".format(essential["error"])
        else:
            text = str(essential.get("summary") or "")
        if saw_essential is False:
            text = (text + " (not given to the judge)").strip()
        rows += ("<tr><td class='s'>&mdash;</td><td>essential requirements</td>"
                 "<td>{}</td></tr>").format(html.escape(text))
    if judge:
        verdict = "MODEL ERROR: {}".format(judge["error"]) if judge.get("error") \
            else judge.get("final_evaluation", "")
        rows += "<tr><td class='s'>{}</td><td>{}</td><td>{}</td></tr>".format(
            "&mdash;" if judge.get("final_score") is None else judge.get("final_score"),
            "judge", html.escape(str(verdict)))
    table = ("<section class='pane' style='margin:0 28px 28px'>"
             "<h2>Reasoning</h2><table><tr><th>Score</th><th>Model</th>"
             "<th>Verdict</th></tr>{}</table></section>").format(rows)

    facts_panel = facts_mod.render_facts_panel(cv_text)

    return ("<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>{cand} - {job}</title><style>{css}</style></head><body>"
            "<header><h1>{cand}</h1><div class='sub'>{job}</div></header>"
            "<div class='strip'>{tiles}</div>{flag}"
            "<div class='panes'>{panes}</div>{table}{facts}"
            "</body></html>").format(
        cand=html.escape(candidate), job=html.escape(job), css=_REPORT_CSS,
        tiles="".join(tiles), flag=flag, panes="".join(panes), table=table,
        facts=facts_panel)


def write_report(json_path, payload: dict) -> Path:
    """Write the page beside the run's JSON, same stem."""
    path = Path(json_path).with_suffix(".html")
    path.write_text(render_report(payload), encoding="utf-8")
    return path

