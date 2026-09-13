"""
run_evaluation.py

The whole evaluation pipeline over a folder of CVs and a folder of job
listings, in one command:

  1. evaluate   Evaluate_CV.py -e     every CV against every job listing: each
                                      council model's score, plus the
                                      essential-requirements check
  2. aggregate  aggregate_results.py  one combined file per (job, candidate),
                                      with that candidate's check joined in
  3. judge      judge.py              one final verdict per (job, candidate)

Each stage runs as its own script, in a subprocess, exactly as it would be run
by hand. This file adds the order, the checks and the resuming - no evaluation
logic of its own.

Usage (relative paths are taken from the folder you run it in):

    python evaluator/run_evaluation.py --cvs data/cvs/Batch1
    python evaluator/run_evaluation.py --cvs data/cvs/Batch2 --out results/evaluation/batch2

    # the same evaluations, judged again on the council alone
    python evaluator/run_evaluation.py --out results/evaluation/batch2 --from judge --no-essential

Output, under --out:

    evaluations/<model>__job_listing_<slug>.json
    evaluations/essential_requirements/essential_requirements__job_listing_<slug>.json
    combined/<slug>/<candidate>.json
    judged/with-essential/<slug>/<candidate>.json   (judged/council-only/ with --no-essential)
    judged/<condition>/failures.log
    run.json                                        one record per run - see write_run_record
    run.log                                         everything printed, appended each run

The essential check always runs in the evaluate stage; --no-essential only
changes what the judge is shown. One evaluation therefore serves both
conditions, and the two judged/ folders differ in nothing but the prompt.

Safe to re-run. The judge stage skips any pair whose output already holds a
valid final_score, so an interrupted run picks up where it stopped. judge.py
writes its output even when the model's reply was unusable, so validity is
checked here rather than read from its exit code.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EVALUATOR_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVALUATOR_DIR.parent
for _directory in (ROOT_DIR / "common", EVALUATOR_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

STAGES = ("evaluate", "aggregate", "judge")
JOB_PREFIX = "job_listing_"


# =====================================================================
# Command line
# =====================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate, aggregate and judge a folder of CVs against a folder of "
                    "job listings. See the module docstring for the output layout.")
    parser.add_argument("--cvs", type=Path, default=ROOT_DIR / "data" / "cvs" / "Batch1",
                        help="CV PDFs, searched recursively (default: data/cvs/Batch1).")
    parser.add_argument("--jobs", type=Path, default=ROOT_DIR / "data" / "job-listings",
                        help="job_listing_<slug>.pdf files (default: data/job-listings).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where every stage writes "
                             "(default: results/evaluation/run_<timestamp>).")
    parser.add_argument("--from", dest="start", choices=STAGES, default="evaluate",
                        help="Start at this stage, reusing what --out already holds "
                             "(default: evaluate).")
    parser.add_argument("--no-essential", dest="essential", action="store_false",
                        help="Judge on the council's summaries only.")
    parser.add_argument("--judge-model", default=None,
                        help="Ollama model for the judge (default: judge.py's own, deepseek-r1:8b).")
    parser.add_argument("--force", action="store_true",
                        help="Re-judge pairs that already have a valid verdict.")
    args = parser.parse_args(argv)

    resuming = args.start != "evaluate"
    if args.out is None:
        if resuming:
            parser.error(f"--from {args.start} carries on from an earlier run, "
                         f"so it needs that run's --out folder")
        args.out = ROOT_DIR / "results" / "evaluation" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    elif resuming and not args.out.is_dir():
        parser.error(f"--out folder not found: {args.out}")
    if not args.jobs.is_dir():
        parser.error(f"--jobs folder not found: {args.jobs}")
    if not resuming and not args.cvs.is_dir():
        parser.error(f"--cvs folder not found: {args.cvs}")

    # Absolute from here on, so each stage's script reads the folders that
    # were meant, whatever it takes a relative path to be relative to.
    args.cvs, args.jobs, args.out = args.cvs.resolve(), args.jobs.resolve(), args.out.resolve()
    return args


def condition(options) -> str:
    return "with-essential" if options.essential else "council-only"


# =====================================================================
# Output and subprocesses
# =====================================================================

class Log:
    """Prints each line and appends the same line to run.log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def __call__(self, line: str = ""):
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def run_step(args, log) -> int:
    """Run one stage's script with this interpreter, streaming its output into
    the log line by line. Returns the script's exit code."""
    # Unbuffered, so progress shows as it happens rather than in one block at
    # the end of each step; UTF-8, so a candidate's name cannot crash a print
    # on a Windows console code page.
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    command = [sys.executable, *(str(arg) for arg in args)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", env=env) as process:
        for line in process.stdout:
            log(line.rstrip("\r\n"))
    return process.returncode


# =====================================================================
# Run record
# =====================================================================

def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def display_path(path: Path) -> str:
    """Repo-relative with forward slashes when inside the repo, so a committed
    run.json names data/cvs/Batch1 rather than one person's home folder;
    absolute otherwise."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def new_record(options) -> dict:
    # The options are recorded one by one rather than as the command typed:
    # an absolute path on the command line would put a home folder into a
    # file meant to be committed.
    return {
        "started": _now(),
        "finished": None,
        "exit_code": None,
        "python": platform.python_version(),
        "from": options.start,
        "condition": condition(options),
        "force": options.force,
        "cvs": display_path(options.cvs) if options.start == "evaluate" else None,
        "jobs": display_path(options.jobs),
        "out": display_path(options.out),
        "preflight": None,
        "models": {},
        "inputs": {},
        "stages": {},
    }


def models_used(options) -> dict:
    """The models this run calls. A resumed run's evaluations came from an
    earlier run, whose own record names the council."""
    import Evaluate_CV

    models = {}
    if options.start == "evaluate":
        models["council"] = list(Evaluate_CV.EVALUATOR_MODELS)
        models["essential_requirements"] = Evaluate_CV.ESSENTIAL_REQUIREMENTS_MODEL
    models["judge"] = options.judge_model
    return models


def inputs_used(options) -> dict:
    inputs = {"job_listings": sorted(p.name for p in options.jobs.glob(JOB_PREFIX + "*.pdf"))}
    if options.start == "evaluate":
        inputs["cvs"] = sorted(p.relative_to(options.cvs).as_posix()
                               for p in options.cvs.rglob("*.pdf"))
    return inputs


def write_run_record(path: Path, record: dict, log) -> None:
    """Append this run's record to run.json.

    Every run into one --out is kept - the evaluation and each later --from
    judge - so the file says how each set of results in the folder came
    about. Unlike run.log it is small enough to commit with those results.
    An unreadable run.json is moved aside rather than overwritten.
    """
    runs = []
    if path.exists():
        try:
            runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
            if not isinstance(runs, list):
                raise TypeError("'runs' is not a list")
        except (ValueError, KeyError, TypeError) as exc:
            aside = path.with_name(path.name + ".unreadable")
            path.replace(aside)
            log(f"WARN {path.name} could not be read ({exc}); moved it to {aside.name} "
                f"and started a new one")
            runs = []
    runs.append(record)
    path.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


# =====================================================================
# Preflight
# =====================================================================

def missing_dependencies() -> list:
    missing = []
    for module in ("fitz", "requests"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def check_job_listings(jobs: Path):
    """Returns (job PDFs, problems)."""
    from results_io import safe_filename_part

    pdfs = sorted(jobs.glob("*.pdf"))
    problems = []
    if not pdfs:
        problems.append(f"no PDFs in --jobs {jobs}")
    # Evaluate_CV.py names its files after the PDF, and aggregate_results.py
    # finds the job again by the job_listing_ marker - any other name cannot
    # be matched back to its listing.
    misnamed = [p.name for p in pdfs
                if not p.name.startswith(JOB_PREFIX) or safe_filename_part(p.stem) != p.stem]
    if misnamed:
        problems.append("job listings must be named job_listing_<slug>.pdf using only letters, "
                        "digits, _ and - (aggregate_results.py cannot place any other name): "
                        + ", ".join(misnamed))
    return pdfs, problems


def check_cvs(cvs: Path):
    """Returns (CV PDFs, problems).

    Candidates are grouped by name from the aggregate stage on, so two CVs
    with one name would be merged into a single candidate's file and judged
    as one person.
    """
    import aggregate_results

    pdfs = sorted(cvs.rglob("*.pdf"))
    problems = []
    if not pdfs:
        problems.append(f"no PDFs under --cvs {cvs}")
    by_name = defaultdict(list)
    for pdf in pdfs:
        by_name[aggregate_results.slugify(pdf.stem)].append(pdf)
    clashes = [group for group in by_name.values() if len(group) > 1]
    if clashes:
        listed = "\n".join("      " + ", ".join(str(p.relative_to(cvs)) for p in group)
                           for group in clashes)
        problems.append("these CVs share a candidate name, so their evaluations would be "
                        "merged and judged as one candidate - run their folders separately:\n"
                        + listed)
    return pdfs, problems


def ollama_base() -> str:
    import ollama_utils

    return ollama_utils.OLLAMA_URL.split("/api/")[0]


def fetch_ollama_models() -> set:
    """The model names Ollama has, as the scripts ask for them. Raises when
    Ollama cannot be reached."""
    with urllib.request.urlopen(ollama_base() + "/api/tags", timeout=5) as response:
        tags = json.load(response)
    names = set()
    for model in tags.get("models", []):
        name = model.get("name", "")
        names.add(name)
        if name.endswith(":latest"):
            names.add(name[: -len(":latest")])
    return names


def check_models(wanted, fetch=fetch_ollama_models) -> list:
    try:
        have = fetch()
    except Exception as exc:
        return [f"Ollama is not reachable at {ollama_base()} ({exc}) - is `ollama serve` running?"]
    missing = [m for m in dict.fromkeys(wanted) if m not in have]
    if missing:
        return ["Ollama does not have: " + ", ".join(missing) + " - `ollama pull` each one"]
    return []


def preflight(options, log, fetch=fetch_ollama_models) -> bool:
    """Every problem at once, before any model call. A missing model or a
    stopped Ollama would otherwise fail every one of hundreds of calls, one at
    a time, an hour into the run."""
    log("=== Preflight ===")
    missing = missing_dependencies()
    if missing:
        log("Preflight failed:")
        log(f"  - {sys.executable} cannot import {', '.join(missing)}. "
            f"Fix: {sys.executable} -m pip install -r requirements.txt")
        return False

    import Evaluate_CV
    import judge

    options.judge_model = options.judge_model or judge.DEFAULT_MODEL
    job_pdfs, problems = check_job_listings(options.jobs)
    found = f"{len(job_pdfs)} job listing(s)"
    wanted = [options.judge_model]
    if options.start == "evaluate":
        cv_pdfs, cv_problems = check_cvs(options.cvs)
        problems += cv_problems
        found += f", {len(cv_pdfs)} CV(s)"
        wanted = [*Evaluate_CV.EVALUATOR_MODELS, Evaluate_CV.ESSENTIAL_REQUIREMENTS_MODEL,
                  options.judge_model]
    problems += check_models(wanted, fetch)

    if problems:
        log("Preflight failed:")
        for problem in problems:
            log("  - " + problem)
        return False
    log(f"  OK: {found}; models: {', '.join(dict.fromkeys(wanted))}")
    return True


# =====================================================================
# Stages - each fills its entry in the run record with what it counted
# =====================================================================

def summarise_evaluations(folder: Path, log, stage: dict) -> bool:
    files = sorted(folder.glob("*__job_listing_*.json"))
    if not files:
        log(f"error: Evaluate_CV.py wrote no result files to {folder}")
        return False
    rows = unscored = 0
    for path in files:
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows += 1
            unscored += row.get("score") is None
    checks = sorted((folder / "essential_requirements").glob("*.json"))
    stage.update(model_files=len(files), evaluations=rows, unscored=unscored,
                 essential_files=len(checks))
    log(f"  {len(files)} model result file(s), {rows} evaluation(s), "
        f"{len(checks)} essential-requirements file(s)")
    if unscored:
        log(f"  WARN {unscored} evaluation(s) have no score (a model or parse error) - "
            f"the judge will read their error text as that reviewer's summary")
    return True


def evaluate_stage(options, log, run=run_step, stage=None) -> bool:
    stage = {} if stage is None else stage
    log()
    log("=== Stage 1/3: evaluate (council + essential-requirements check) ===")
    log(f"  CVs:  {options.cvs}")
    log(f"  Jobs: {options.jobs}")
    evaluations = options.out / "evaluations"
    code = run([EVALUATOR_DIR / "Evaluate_CV.py", "--essential-requirements",
                "--cvs", options.cvs, "--jobs", options.jobs, "--results", evaluations], log)
    if code != 0:
        log(f"error: Evaluate_CV.py failed (exit code {code})")
        return False
    return summarise_evaluations(evaluations, log, stage)


def aggregate_stage(options, log, run=run_step, stage=None) -> bool:
    stage = {} if stage is None else stage
    log()
    log("=== Stage 2/3: aggregate ===")
    evaluations = options.out / "evaluations"
    if not any(evaluations.glob("*__job_listing_*.json")):
        log(f"error: no evaluations in {evaluations} - start from the evaluate stage")
        return False
    # The check is joined in whatever the condition: the combined files are
    # the same for both, and judge.py's --no-essential decides what is shown.
    combined = options.out / "combined"
    args = [EVALUATOR_DIR / "aggregate_results.py",
            "--results-dir", evaluations, "--outdir", combined]
    if (evaluations / "essential_requirements").is_dir():
        args += ["--essential-dir", evaluations / "essential_requirements"]
    code = run(args, log)
    if code != 0:
        log(f"error: aggregate_results.py failed (exit code {code})")
        return False
    stage["combined_files"] = len(list(combined.glob("*/*.json")))
    return True


def is_valid_verdict(path: Path) -> bool:
    """
    A verdict worth keeping: JSON holding a numeric final_score in [1, 10].
    if model returns 0, logged as failure, judge again
    """
    try:
        with open(path, encoding="utf-8") as f:
            score = json.load(f).get("final_score")
    except Exception:
        return False
    return isinstance(score, (int, float)) and not isinstance(score, bool) and 1 <= score <= 10 

def judge_stage(options, log, run=run_step, stage=None) -> bool:
    stage = {} if stage is None else stage
    log()
    log(f"=== Stage 3/3: judge ({condition(options)}, {options.judge_model}) ===")
    combined = options.out / "combined"
    if not combined.is_dir():
        log(f"error: no combined files in {combined} - start from the aggregate stage")
        return False
    judged = options.out / "judged" / condition(options)
    judged.mkdir(parents=True, exist_ok=True)
    fail_log = judged / "failures.log"
    fail_log.write_text("", encoding="utf-8")

    total = ok = failed = skipped = 0
    for job_pdf in sorted(options.jobs.glob(JOB_PREFIX + "*.pdf")):
        slug = job_pdf.stem[len(JOB_PREFIX):]
        candidates = sorted((combined / slug).glob("*.json"))
        if not candidates:
            log(f"  WARN no combined results for {slug} - skipping this job listing")
            continue

        for summaries in candidates:
            verdict = judged / slug / summaries.name
            total += 1
            if not options.force and is_valid_verdict(verdict):
                skipped += 1
                continue

            log(f"  -> {slug} / {summaries.stem}")
            args = [EVALUATOR_DIR / "judge.py", "--summaries", summaries, "--job", job_pdf,
                    "--job-name", slug, "--output", verdict, "--model", options.judge_model]
            if not options.essential:
                args.append("--no-essential")
            if run(args, log) == 0 and is_valid_verdict(verdict):
                ok += 1
            else:
                failed += 1
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{slug}/{summaries.stem}\n")
                log(f"     FAILED - no valid final_score; logged to {fail_log}")
            log()
            stage.update(ok=ok, failed=failed, already_done=skipped, total=total)

    stage.update(ok=ok, failed=failed, already_done=skipped, total=total)
    log(f"Judged ({condition(options)}): {ok} ok, {failed} failed, "
        f"{skipped} already done, {total} total")
    log(f"Verdicts: {judged}")
    if total == 0:
        log(f"error: nothing to judge - no combined files match the job listings in {options.jobs}")
        return False
    if failed:
        log(f"Failures logged to: {fail_log}")
        return False
    return True


# =====================================================================
# Main
# =====================================================================

STAGE_FUNCTIONS = {"evaluate": evaluate_stage, "aggregate": aggregate_stage, "judge": judge_stage}


def run_pipeline(options, log, run, fetch, record) -> int:
    log(f"run_evaluation.py  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"  out: {options.out}")
    log(f"  from: {options.start}   condition: {condition(options)}   "
        f"python: {sys.executable}")
    log()
    if not preflight(options, log, fetch):
        record["preflight"] = "failed"
        return 1
    record["preflight"] = "ok"
    record["models"] = models_used(options)
    record["inputs"] = inputs_used(options)

    start = STAGES.index(options.start)
    for index, name in enumerate(STAGES):
        if index < start:
            record["stages"][name] = {"status": "reused"}
            continue
        stage = record["stages"][name] = {"status": "running"}
        started = time.monotonic()
        succeeded = STAGE_FUNCTIONS[name](options, log, run, stage)
        stage["status"] = "ok" if succeeded else "failed"
        stage["seconds"] = round(time.monotonic() - started, 1)
        if not succeeded:
            return 1
    return 0


def main(argv=None, run=run_step, fetch=fetch_ollama_models) -> int:
    options = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    log = Log(options.out / "run.log")
    record = new_record(options)
    code = 1
    try:
        code = run_pipeline(options, log, run, fetch, record)
    except KeyboardInterrupt:
        log()
        log("Interrupted. Run again with the same --out to carry on - add --from judge "
            "if the evaluate stage had already finished.")
        code = 130
    finally:
        # Written whatever happened, so a failed or stopped run is on record too.
        for stage in record["stages"].values():
            if stage.get("status") == "running":
                stage["status"] = "stopped"
        record["finished"] = _now()
        record["exit_code"] = code
        write_run_record(options.out / "run.json", record, log)
        log(f"Run record appended to {options.out / 'run.json'}")
        log.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
