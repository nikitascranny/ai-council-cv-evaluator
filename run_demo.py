"""Console app for the CV evaluator: council, judge and the rewriter probe.

    python run_demo.py           # main menu
    python run_demo.py --fast    # start on the lighter 7b models
    python run_demo.py --once    # single evaluation, no menu (for scripting)

The menu loops until you quit, so several candidates can be run in one session
and the models and stages can be changed between runs without restarting.

Each stage prints as it finishes, so a slow or failing model is visible rather
than a hang, and stages are independent: if the council loses a model the judge
still runs on what came back. An action that fails returns to the menu.

The evaluator, judge and essential-requirements code is the team's, kept in
integration/ and imported lazily so this file still loads, and its tests still
run, on a checkout that does not have it.
"""

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTEGRATION = ROOT / "integration"
DEMO_RESULTS = ROOT / "Results" / "demo"

# The team's real models, and what --fast falls back to.
COUNCIL_MODELS = ["qwen2.5:14b", "phi4"]
ESSENTIAL_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "deepseek-r1:8b"

# 7b, not 3b. The 3b models score the same CV several points higher than the
# 14b ones and llama3.2:3b returned unparseable JSON on two of three runs, so
# anything measured on them says more about the model size than the candidate.
FAST_MODELS = {"council": ["qwen2.5:7b"],
               "essential": "qwen2.5:7b",
               "judge": "qwen2.5:7b"}


# --- corpus discovery -------------------------------------------------------

def _pdfs_under(*dirs) -> list:
    found = []
    for directory in dirs:
        if directory.is_dir():
            found.extend(sorted(directory.rglob("*.pdf")))
    return found


def discover_cvs(root: Path = ROOT) -> list:
    """Batch1's CVs plus the batch2-4 sets, which are the designed experiments."""
    base = Path(root) / "Batch1"
    return _pdfs_under(base / "CVs", base / "batch2-4")


def discover_jobs(root: Path = ROOT) -> list:
    return _pdfs_under(Path(root) / "Batch1" / "Job Listings")


def label_for(path, root: Path = ROOT) -> str:
    """Menu label: the containing folder plus the stem.

    batch3 holds bad_ and professional_ versions of the same three people, so
    the folder has to appear or two entries read identically.
    """
    path = Path(path)
    parent = path.parent.name
    if parent in ("CVs", "Job Listings"):
        return path.stem
    return f"{parent}/{path.stem}"


def title_from_slug(stem: str) -> str:
    """job_listing_startup_fullstack_engineer -> "Startup Fullstack Engineer"."""
    return re.sub(r"^job[_ ]listing[_ ]", "", stem).replace("_", " ").strip().title()


# --- the menu ---------------------------------------------------------------

def choose(prompt: str, options: list, reader=input):
    """Numbered pick from options. None means quit, or nothing to pick."""
    if not options:
        return None
    while True:
        print("\n" + prompt)
        for i, option in enumerate(options, start=1):
            text = option if isinstance(option, str) else label_for(option)
            print("  {:2}. {}".format(i, text))
        raw = reader("Number (or q to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  -> enter 1-{}, or q".format(len(options)))


# --- preflight --------------------------------------------------------------

OLLAMA_TAGS = "http://localhost:11434/api/tags"


def available_models(fetch=None) -> set:
    """Model names Ollama currently has, empty if it cannot be reached.

    Ollama reports "phi4:latest" for what everything else calls "phi4", so the
    bare name is registered alongside the tagged one.
    """
    if fetch is None:
        def fetch():
            import urllib.request
            with urllib.request.urlopen(OLLAMA_TAGS, timeout=5) as response:
                return json.load(response)
    try:
        payload = fetch()
    except Exception:
        return set()

    names = set()
    for entry in payload.get("models", []):
        name = entry.get("name", "")
        if name:
            names.add(name)
            if name.endswith(":latest"):
                names.add(name.split(":")[0])
    return names


def missing_models(wanted, available) -> list:
    """Which of `wanted` Ollama does not have.

    An empty availability set means the check could not run, not that every
    model is gone - reporting all of them would be a false alarm before a demo.
    """
    if not available:
        return []
    return [m for m in wanted if m not in available]


# --- stages -----------------------------------------------------------------

def _integration():
    """Import the team's modules, with a message rather than a traceback."""
    if not INTEGRATION.is_dir():
        sys.exit(
            "integration/ not found at {}.\n"
            "It holds the team's evaluator and judge, pulled from their GitLab "
            "branches, and is deliberately not committed here.".format(INTEGRATION))
    if str(INTEGRATION) not in sys.path:
        sys.path.insert(0, str(INTEGRATION))
    import document_utils
    import Evaluate_CV
    import judge
    import essential_requirements
    import ollama_utils
    return document_utils, Evaluate_CV, judge, essential_requirements, ollama_utils


def job_text_of(job_path) -> str:
    """Job listing PDF -> text, via the team's plain pdfplumber extraction.

    Job listings are single-column prose, so they do not need the structural
    parser - and using it here would misreport which component read what.
    """
    return _integration()[0].extract_text_from_pdf(job_path)


def parse_stage(cv_path, load_cv=None):
    """Our parser. Runs first because everything downstream reads its output."""
    if load_cv is None:
        load_cv = _integration()[0].load_cv
    started = time.time()
    cv = load_cv(cv_path)
    print("  {} sections: {}".format(len(cv.sections),
                                     ", ".join(s.label for s in cv.sections)))
    detail = "  {} chars of text".format(len(cv.full_text()))
    if cv.multi_column_pages:
        detail += ", multi-column pages {}".format(cv.multi_column_pages)
    print(detail)
    for warning in cv.warnings:
        print("  [!] " + warning)
    print("  ({:.1f}s)".format(time.time() - started))
    return cv


def council_stage(job_text, cv_text, candidate, models, evaluate=None):
    """Each evaluator in turn. One model failing must not end the run."""
    if evaluate is None:
        evaluate = _integration()[1].evaluate_cv
    rows = []
    for model in models:
        started = time.time()
        try:
            row = evaluate(job_text, cv_text, candidate, model)
        except Exception as exc:
            print("  [!] {} failed: {}".format(model, exc))
            row = {"candidate": candidate, "score": None, "strengths": [],
                   "weaknesses": [], "summary": "MODEL ERROR: {}".format(exc)}
        else:
            print("  {}: score {}  ({:.1f}s)".format(
                model, row.get("score"), time.time() - started))
            print("      " + str(row.get("summary", "")))
        row["model"] = model
        rows.append(row)
    return rows


def essential_stage(job_text, cv, candidate, model, check=None):
    """Scoped input rather than the whole CV - the point of sections_named()."""
    if check is None:
        check = _integration()[3].evaluate_essential_requirement
    scoped = cv.sections_named("experience", "education", "certifications")
    result = check(job_text, scoped, candidate, model=model)
    print("  " + str(result.get("summary", "")))
    return result


def judge_stage(job_text, rows, candidate, model, judge_fn=None):
    """Aggregate the council's summaries into one independent verdict."""
    if judge_fn is None:
        _, _, judge_mod, _, ollama_utils = _integration()

        def judge_fn(job_text, rows, candidate, model):
            summaries = judge_mod.format_summaries_for_prompt(rows)
            prompt = judge_mod.build_prompt(job_text, summaries, candidate)
            return ollama_utils.call_ollama_json(model, prompt, num_predict=1024)

    parsed, error = judge_fn(job_text, rows, candidate, model)
    if error:
        print("  [!] judge failed: " + str(error))
        return {"candidate": candidate, "final_score": None, "error": str(error)}
    print("  final score {}/10".format(parsed.get("final_score")))
    print("  " + str(parsed.get("final_evaluation", "")))
    return parsed


def rewrite_stage(cv_text, job_title, scorer, probe=None, paraphrase_fn=None):
    """The phrasing-sensitivity probe, reported as a finding.

    A rejected rewrite is a result about the rewriting model, not an error: it
    means the paraphrase introduced a fact the original did not contain.

    paraphrase_fn is passed straight through to probe(), which is how the report
    gets hold of the rewritten text without ProbeResult having to carry it.
    """
    if probe is None:
        import rewriter_v1
        extra = {"paraphrase_fn": paraphrase_fn} if paraphrase_fn else {}
        result = rewriter_v1.probe(cv_text, job_title, scorer, **extra)
        out = result if isinstance(result, dict) else vars(result)
    else:
        out = probe(cv_text, job_title, scorer)

    if out.get("rejected"):
        print("  rewrite rejected by the fact guard")
        invented = ", ".join(out.get("new_facts") or []) or "(none recorded)"
        print("  invented: " + invented)
    else:
        print("  score {} -> {}  (delta {})".format(
            out.get("score_before"), out.get("score_after"), out.get("delta")))
        if out.get("delta"):
            print("  the ranking moved on wording alone")
        else:
            print("  the ranking ignored the rephrasing")
    return out


# --- output -----------------------------------------------------------------

def _safe(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def result_filename(job: str, candidate: str) -> str:
    return "{}__{}.json".format(_safe(job), _safe(candidate))


def new_payload(job: str, candidate: str) -> dict:
    """Start a run's payload already carrying its identity.

    save_run used to merge job and candidate in on the way to disk, so the dict
    handed to write_report lacked both and every page was headed "? - ?".
    Building them in up front means one dict is saved and rendered.
    """
    return {"job": job, "candidate": candidate}


def save_run(out_dir, job: str, candidate: str, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / result_filename(job, candidate)
    body = {"job": job, "candidate": candidate}
    body.update(payload)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --- the comparison report --------------------------------------------------

def capture_paraphrase(fn):
    """Wrap a paraphrase function so the text it returned can be read back.

    probe() takes paraphrase_fn as a parameter, so this captures the rewrite
    without adding a field to ProbeResult. That matters: the batch runner writes
    result.to_dict() as one JSONL row per candidate-job, and a full CV in every
    row would bloat the research output the findings are read from.

    The guard retries on rejection, calling this a second time. Keeping the last
    call is deliberate - that is the text which was actually scored.
    """
    def wrapper(cv_text, job_title, **kwargs):
        result = fn(cv_text, job_title, **kwargs)
        wrapper.text = result
        return result

    wrapper.text = None
    return wrapper


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
    if judge:
        final = judge.get("final_score")
        tiles.append(_tile("judge", "&mdash;" if final is None else final,
                           "final verdict"))
    if rewrite:
        delta = rewrite.get("delta")
        shown = "&mdash;" if delta is None else ("+{}".format(delta) if delta > 0 else delta)
        tiles.append(_tile("rewrite delta", shown,
                           "{} -> {}".format(rewrite.get("score_before"),
                                             rewrite.get("score_after"))))

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
    if judge:
        rows += "<tr><td class='s'>{}</td><td>{}</td><td>{}</td></tr>".format(
            "&mdash;" if judge.get("final_score") is None else judge.get("final_score"),
            "judge", html.escape(str(judge.get("final_evaluation", ""))))
    table = ("<section class='pane' style='margin:0 28px 28px'>"
             "<h2>Reasoning</h2><table><tr><th>Score</th><th>Model</th>"
             "<th>Verdict</th></tr>{}</table></section>").format(rows)

    return ("<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>{cand} - {job}</title><style>{css}</style></head><body>"
            "<header><h1>{cand}</h1><div class='sub'>{job}</div></header>"
            "<div class='strip'>{tiles}</div>{flag}"
            "<div class='panes'>{panes}</div>{table}</body></html>").format(
        cand=html.escape(candidate), job=html.escape(job), css=_REPORT_CSS,
        tiles="".join(tiles), flag=flag, panes="".join(panes), table=table)


def write_report(json_path, payload: dict) -> Path:
    """Write the page beside the run's JSON, same stem."""
    path = Path(json_path).with_suffix(".html")
    path.write_text(render_report(payload), encoding="utf-8")
    return path


# --- settings ---------------------------------------------------------------

class Settings:
    """What the next run will do. Held across menu choices.

    An explicitly chosen model wins over fast mode, so switching to fast to get
    through a slow stage does not silently discard a model the user picked.
    """

    def __init__(self, fast=False, essential=False, rewrite=False,
                 council=None, judge=None):
        self.fast = fast
        self.essential = essential
        self.rewrite = rewrite
        self._council = list(council) if council else None
        self._judge = judge

    @property
    def council(self):
        if self._council:
            return self._council
        return FAST_MODELS["council"] if self.fast else COUNCIL_MODELS

    @council.setter
    def council(self, models):
        self._council = list(models) if models else None

    @property
    def judge(self):
        if self._judge:
            return self._judge
        return FAST_MODELS["judge"] if self.fast else JUDGE_MODEL

    @judge.setter
    def judge(self, model):
        self._judge = model or None

    @property
    def essential_model(self):
        return FAST_MODELS["essential"] if self.fast else ESSENTIAL_MODEL

    def stages(self) -> list:
        active = ["parse", "council"]
        if self.essential:
            active.append("essential requirements")
        active.append("judge")
        if self.rewrite:
            active.append("rewriter probe")
        return active

    def summary(self) -> str:
        return ("council: {}\n  judge:   {}\n  stages:  {}".format(
            ", ".join(self.council), self.judge, " -> ".join(self.stages())))


# --- saved results ----------------------------------------------------------

def saved_runs(out_dir) -> list:
    """Previous runs, newest first. Missing directory is simply no runs."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    return sorted(out_dir.glob("*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def describe_run(path) -> str:
    """One line per saved run.

    Anything unreadable is reported as such rather than raised: a half-written
    file from an interrupted run must not stop the list from displaying.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "{}  (unreadable)".format(path.name)

    scores = ", ".join(
        "{} {}".format(row.get("model", "?"), row.get("score"))
        for row in data.get("council", []))
    final = (data.get("judge") or {}).get("final_score")
    return "{} -> {}  |  judge {}  |  {}".format(
        data.get("candidate", path.stem), data.get("job", "?"), final, scores)


# --- the menu loop ----------------------------------------------------------

def app_loop(settings, actions=None, reader=input):
    """Main menu, until the user quits.

    An action that raises returns to the menu rather than ending the session -
    a model dying mid-demo should cost one run, not the whole app.
    """
    if actions is None:
        actions = default_actions()
    while True:
        print("\n" + "=" * 64)
        print("  AI CV EVALUATOR")
        print("=" * 64)
        print("  " + settings.summary())
        print()
        for key, (label, _) in actions.items():
            print("  {}. {}".format(key, label))
        print("  q. Quit")

        try:
            choice = reader("\nChoice: ").strip().lower()
        except EOFError:
            return 0

        if choice in ("q", "quit", "exit"):
            return 0
        entry = actions.get(choice)
        if entry is None:
            print("  -> pick one of: {}, q".format(", ".join(actions)))
            continue
        try:
            entry[1](settings)
        except KeyboardInterrupt:
            print("\n  (cancelled)")
        except Exception as exc:
            print("\n  [!] that run failed: {}".format(exc))
            print("  Back to the menu; nothing else was affected.")


# --- driver -----------------------------------------------------------------

def _stage_banners():
    """Numbers the stages as they actually run.

    The optional stages are skipped more often than not, and a run that jumps
    from stage 2 to stage 4 reads like something failed.
    """
    counter = {"n": 0}

    def banner(title):
        counter["n"] += 1
        print("\n" + "=" * 64)
        print("  STAGE {}: {}".format(counter["n"], title))
        print("=" * 64)

    return banner


def _preflight_ok(settings, needed) -> bool:
    """Warn about models Ollama does not have, before a long stage starts."""
    absent = missing_models(needed, available_models())
    if not absent:
        return True
    print("\n  Not pulled: {}".format(", ".join(absent)))
    print("  Change the models from the menu, or pull them.")
    return input("  Continue anyway? [y/N] ").strip().lower() in ("y", "yes")


# --- menu actions -----------------------------------------------------------

def run_evaluation(settings):
    """Parse, council, optional essential-requirements, judge, optional probe."""
    needed = list(settings.council) + [settings.judge]
    if settings.essential:
        needed.append(settings.essential_model)
    if not _preflight_ok(settings, needed):
        return

    job_path = choose("Job listing:", discover_jobs())
    if job_path is None:
        return
    cv_path = choose("Candidate CV:", discover_cvs())
    if cv_path is None:
        return

    candidate = cv_path.stem
    job_title = title_from_slug(job_path.stem)
    document_utils = _integration()[0]

    print("\n{}  ->  {}".format(candidate, job_title))
    banner = _stage_banners()

    banner("PARSE (structure-aware CV parser)")
    cv = parse_stage(cv_path)
    job_text = document_utils.extract_text_from_pdf(job_path)

    banner("COUNCIL ({} evaluator models)".format(len(settings.council)))
    rows = council_stage(job_text, cv.full_text(), candidate, settings.council)

    # The report needs the inputs, not just the scores: the whole point is
    # seeing the job listing and the CV the models actually read.
    payload = new_payload(job_path.stem, candidate)
    payload.update({"council": rows, "job_text": job_text,
                    "cv_text": cv.full_text(),
                    "sections": [s.label for s in cv.sections]})

    if settings.essential:
        banner("ESSENTIAL REQUIREMENTS")
        payload["essential"] = essential_stage(
            job_text, cv, candidate, settings.essential_model)

    banner("JUDGE")
    payload["judge"] = judge_stage(job_text, rows, candidate, settings.judge)

    if settings.rewrite:
        banner("REWRITER PROBE (phrasing sensitivity)")
        import rewriter_v1
        scorer = rewriter_v1.make_scorer(settings.council[0])
        capture = capture_paraphrase(rewriter_v1.paraphrase)
        payload["rewrite"] = rewrite_stage(cv.full_text(), job_title, scorer,
                                           paraphrase_fn=capture)
        payload["rewritten_text"] = capture.text

    path = save_run(DEMO_RESULTS, job_path.stem, candidate, payload)
    report = write_report(path, payload)
    print("\nSaved to  {}".format(path))
    print("Report    {}".format(report))


def run_probe_only(settings):
    """The rewriter probe on its own, without waiting for a council run."""
    if not _preflight_ok(settings, [settings.council[0]]):
        return

    job_path = choose("Job listing (title only is used):", discover_jobs())
    if job_path is None:
        return
    cv_path = choose("Candidate CV:", discover_cvs())
    if cv_path is None:
        return

    import rewriter_v1
    cv = parse_stage(cv_path)
    job_title = title_from_slug(job_path.stem)

    banner = _stage_banners()
    banner("REWRITER PROBE (phrasing sensitivity)")
    scorer = rewriter_v1.make_scorer(settings.council[0])
    capture = capture_paraphrase(rewriter_v1.paraphrase)
    result = rewrite_stage(cv.full_text(), job_title, scorer,
                           paraphrase_fn=capture)

    payload = new_payload(job_path.stem + "__probe", cv_path.stem)
    payload.update({"rewrite": result, "rewritten_text": capture.text,
                    "job_text": job_text_of(job_path),
                    "cv_text": cv.full_text(),
                    "sections": [s.label for s in cv.sections]})
    path = save_run(DEMO_RESULTS, job_path.stem + "__probe", cv_path.stem, payload)
    report = write_report(path, payload)
    print("\nSaved to  {}".format(path))
    print("Report    {}".format(report))


def edit_settings(settings):
    """Change models and toggle the optional stages, without restarting."""
    while True:
        print("\n  " + settings.summary())
        print("\n  1. Fast models {}".format("[on]" if settings.fast else "[off]"))
        print("  2. Essential-requirements stage {}".format(
            "[on]" if settings.essential else "[off]"))
        print("  3. Rewriter probe stage {}".format(
            "[on]" if settings.rewrite else "[off]"))
        print("  4. Set council models")
        print("  5. Set judge model")
        print("  b. Back")
        try:
            choice = input("\n  Choice: ").strip().lower()
        except EOFError:
            return

        if choice in ("b", "back", "q"):
            return
        if choice == "1":
            settings.fast = not settings.fast
        elif choice == "2":
            settings.essential = not settings.essential
        elif choice == "3":
            settings.rewrite = not settings.rewrite
        elif choice == "4":
            raw = input("  Comma-separated (blank to reset): ").strip()
            settings.council = [m.strip() for m in raw.split(",") if m.strip()]
        elif choice == "5":
            settings.judge = input("  Judge model (blank to reset): ").strip()
        else:
            print("  -> 1-5, or b")


def browse_results(settings):
    """Read back previous runs, so a slow run only has to happen once."""
    runs = saved_runs(DEMO_RESULTS)
    if not runs:
        print("\n  No saved runs yet - they appear here after an evaluation.")
        return

    print("\n  Saved runs, newest first:")
    for i, path in enumerate(runs, start=1):
        print("  {:2}. {}".format(i, describe_run(path)))

    try:
        raw = input("\n  Number to open in full (blank to go back): ").strip()
    except EOFError:
        return
    if raw.isdigit() and 1 <= int(raw) <= len(runs):
        print()
        print(runs[int(raw) - 1].read_text(encoding="utf-8"))


def default_actions() -> dict:
    return {
        "1": ("Run a full evaluation", run_evaluation),
        "2": ("Run the rewriter probe only", run_probe_only),
        "3": ("Settings", edit_settings),
        "4": ("Browse saved results", browse_results),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactive CV evaluator: council, judge and rewriter probe.")
    ap.add_argument("--rewrite", action="store_true",
                    help="start with the rewriter probe stage enabled")
    ap.add_argument("--essential", action="store_true",
                    help="start with the essential-requirements stage enabled")
    ap.add_argument("--fast", action="store_true",
                    help="start on the lighter 7b models instead of the team's defaults")
    ap.add_argument("--council", help="comma-separated evaluator models")
    ap.add_argument("--judge-model", help="override the judge model")
    ap.add_argument("--once", action="store_true",
                    help="run a single evaluation and exit, skipping the menu")
    args = ap.parse_args(argv)

    settings = Settings(
        fast=args.fast, essential=args.essential, rewrite=args.rewrite,
        council=[m.strip() for m in args.council.split(",")] if args.council else None,
        judge=args.judge_model)

    if args.once:
        run_evaluation(settings)
        return 0
    return app_loop(settings)


if __name__ == "__main__":
    sys.exit(main())
