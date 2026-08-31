"""Interactive demo runner: one candidate against one job, stage by stage.

    python run_demo.py                  # council + judge
    python run_demo.py --rewrite        # also run the phrasing-sensitivity probe
    python run_demo.py --essential      # also run the essential-requirements check
    python run_demo.py --fast           # 3b models, for when a big model stalls

Pick a job and a CV from the menus and each stage prints as it finishes, so a
slow or failing model is visible rather than a hang. Stages are independent: if
the council loses a model the judge still runs on what came back.

The evaluator, judge and essential-requirements code is the team's, kept in
integration/ and imported lazily so this file still loads, and its tests still
run, on a checkout that does not have it.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTEGRATION = ROOT / "integration"

# The team's real models. --fast swaps in what is pulled on a smaller machine.
COUNCIL_MODELS = ["qwen2.5:14b", "phi4"]
ESSENTIAL_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "deepseek-r1:8b"
FAST_MODELS = {"council": ["qwen2.5:3b", "llama3.2:3b"],
               "essential": "qwen2.5:3b",
               "judge": "qwen2.5:3b"}


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


def rewrite_stage(cv_text, job_title, scorer, probe=None):
    """The phrasing-sensitivity probe, reported as a finding.

    A rejected rewrite is a result about the rewriting model, not an error: it
    means the paraphrase introduced a fact the original did not contain.
    """
    if probe is None:
        import rewriter_v1
        result = rewriter_v1.probe(cv_text, job_title, scorer)
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


def save_run(out_dir, job: str, candidate: str, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / result_filename(job, candidate)
    body = {"job": job, "candidate": candidate}
    body.update(payload)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run one CV against one job, stage by stage.")
    ap.add_argument("--rewrite", action="store_true",
                    help="also run the rewriter phrasing-sensitivity probe")
    ap.add_argument("--essential", action="store_true",
                    help="also run the essential-requirements check")
    ap.add_argument("--fast", action="store_true",
                    help="use the 3b models instead of the team's defaults")
    ap.add_argument("--council", help="comma-separated evaluator models")
    ap.add_argument("--judge-model", help="override the judge model")
    args = ap.parse_args(argv)

    council = FAST_MODELS["council"] if args.fast else COUNCIL_MODELS
    essential_model = FAST_MODELS["essential"] if args.fast else ESSENTIAL_MODEL
    judge_model = FAST_MODELS["judge"] if args.fast else JUDGE_MODEL
    if args.council:
        council = [m.strip() for m in args.council.split(",") if m.strip()]
    if args.judge_model:
        judge_model = args.judge_model

    # Checked before anything is picked or parsed: the council takes minutes, and
    # discovering a missing judge model only after it has run wastes the demo.
    wanted = list(council) + [judge_model] + ([essential_model] if args.essential else [])
    absent = missing_models(wanted, available_models())
    if absent:
        print("Not pulled: {}".format(", ".join(absent)))
        print("Pull them, or pass --fast / --council / --judge-model.")
        if input("Continue anyway? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1

    job_path = choose("Job listing:", discover_jobs())
    if job_path is None:
        return 0
    cv_path = choose("Candidate CV:", discover_cvs())
    if cv_path is None:
        return 0

    candidate = cv_path.stem
    job_title = title_from_slug(job_path.stem)
    document_utils = _integration()[0]

    print("\n{}  ->  {}".format(candidate, job_title))
    print("council: {} | judge: {}".format(", ".join(council), judge_model))

    _banner = _stage_banners()

    _banner("PARSE (structure-aware CV parser)")
    cv = parse_stage(cv_path)
    job_text = document_utils.extract_text_from_pdf(job_path)

    _banner("COUNCIL ({} evaluator models)".format(len(council)))
    rows = council_stage(job_text, cv.full_text(), candidate, council)
    payload = {"council": rows}

    if args.essential:
        _banner("ESSENTIAL REQUIREMENTS")
        payload["essential"] = essential_stage(
            job_text, cv, candidate, essential_model)

    _banner("JUDGE")
    payload["judge"] = judge_stage(job_text, rows, candidate, judge_model)

    if args.rewrite:
        _banner("REWRITER PROBE (phrasing sensitivity)")
        import rewriter_v1
        scorer = rewriter_v1.make_scorer(council[0])
        payload["rewrite"] = rewrite_stage(cv.full_text(), job_title, scorer)

    path = save_run(ROOT / "Results" / "demo", job_path.stem, candidate, payload)
    print("\nSaved to {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
