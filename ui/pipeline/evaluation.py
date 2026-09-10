"""One candidate against one job, as a stream of events.

The caller drives this:

    for event in run_single(job, cv, settings, deps):
        ...                                  # render progress
    payload = event.detail["payload"]        # the last event carries the result

The payload is the same dict the console app built, so the saved JSON and the
HTML report are unchanged by the move to a web UI.
"""

import json
import re
import time
from pathlib import Path

from ui.pipeline import stages
from ui.pipeline.corpus import title_from_slug
from ui.pipeline.events import Event

# ui/pipeline/<module>.py -> ui/pipeline -> ui -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_RESULTS = ROOT / "Results" / "demo"


def _safe(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def result_filename(job: str, candidate: str) -> str:
    return "{}__{}.json".format(_safe(job), _safe(candidate))


def new_payload(job: str, candidate: str) -> dict:
    """Start a run's payload already carrying its identity.

    save_run used to merge job and candidate in on the way to disk, so the dict
    handed to the renderer lacked both and every page was headed "? - ?".
    """
    return {"job": job, "candidate": candidate}


def save_run(out_dir, job: str, candidate: str, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / result_filename(job, candidate)
    body = {"job": job, "candidate": candidate}
    body.update(payload)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


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

    scores = ", ".join("{} {}".format(row.get("model", "?"), row.get("score"))
                       for row in data.get("council", []))
    final = (data.get("judge") or {}).get("final_score")
    return "{} -> {}  |  judge {}  |  {}".format(
        data.get("candidate", path.stem), data.get("job", "?"), final, scores)


def run_single(job_path, cv_path, settings, deps, out_dir=None):
    """Parse, council, optional essential-requirements, judge, optional probe."""
    job_path, cv_path = Path(job_path), Path(cv_path)
    out_dir = Path(out_dir) if out_dir else DEMO_RESULTS
    candidate = cv_path.stem
    job_title = title_from_slug(job_path.stem)
    started = time.time()

    cv = yield from stages.parse_stage(cv_path, deps, candidate)
    job_text = deps.extract_text(job_path)

    rows = yield from stages.council_stage(
        job_text, cv.full_text(), candidate, settings.council, deps)

    # The report needs the inputs, not just the scores: the whole point is
    # seeing the job listing and the CV the models actually read.
    payload = new_payload(job_path.stem, candidate)
    payload.update({"council": rows, "job_text": job_text,
                    "cv_text": cv.full_text(),
                    "sections": [s.label for s in cv.sections]})

    if settings.essential:
        payload["essential"] = yield from stages.essential_stage(
            job_text, cv, candidate, settings.essential_model, deps)

    payload["judge"] = yield from stages.judge_stage(
        job_text, rows, candidate, settings.judge, deps)

    if settings.rewrite:
        scorer = deps.make_scorer(settings.council[0])
        capture = stages.capture_paraphrase(deps.paraphrase)
        payload["rewrite"] = yield from stages.rewrite_stage(
            cv.full_text(), job_title, scorer, deps, candidate,
            paraphrase_fn=capture, rewriter_model=settings.rewriter_model)
        payload["rewritten_text"] = capture.text

    path = save_run(out_dir, job_path.stem, candidate, payload)
    payload["_path"] = str(path)
    yield Event("run", "done", {"payload": payload, "path": str(path)},
                time.time() - started, candidate)
    return payload


def rank(payloads) -> list:
    """Candidates by the judge's verdict, best first.

    A missing final score sorts last rather than first: a run whose judge died
    is an unknown, and floating it to the top of a shortlist would be a lie.
    """
    rows = [{"candidate": p.get("candidate", "?"),
             "final_score": (p.get("judge") or {}).get("final_score"),
             "council": p.get("council", []),
             "path": p.get("_path")}
            for p in payloads]
    return sorted(rows, key=lambda r: (r["final_score"] is not None,
                                       r["final_score"] or 0), reverse=True)


def run_batch(job_path, cv_paths, settings, deps, out_dir=None):
    """Several candidates against one listing, ranked as each lands.

    A candidate that fails outright costs its own row, on the same principle as
    a lost council model: the reviewer keeps the shortlist they have.
    """
    payloads = []
    for cv_path in cv_paths:
        candidate = Path(cv_path).stem
        try:
            payload = yield from run_single(job_path, cv_path, settings, deps,
                                            out_dir=out_dir)
        except Exception as exc:
            yield Event("run", "failed", {"error": str(exc)}, candidate=candidate)
            continue
        payloads.append(payload)
        yield Event("rank", "progress", {"ranking": rank(payloads)},
                    candidate=candidate)

    yield Event("rank", "done", {"ranking": rank(payloads)})
    return payloads


def run_candidate(job_path, cv_path, deps, generate):
    """Candidate mode: parse the CV, then rewrite it toward one listing.

    Deliberately not a branch inside run_single. The two modes share only the
    parse - reviewer mode scores a CV against a listing and produces a report,
    candidate mode rewrites it and produces two documents - and folding them
    together would put a settings flag in front of every stage that only one
    of them runs.

    The RewriteResults ride home on the run's payload so the results page and
    the download route can both read them back without re-running two 14B
    models to recover what was already computed.
    """
    job_path, cv_path = Path(job_path), Path(cv_path)
    candidate = cv_path.stem
    job_title = title_from_slug(job_path.stem)
    started = time.time()

    cv = yield from stages.parse_stage(cv_path, deps, candidate)
    job_text = deps.extract_text(job_path)

    results = yield from stages.rewrite_v2_stage(
        cv.full_text(), job_text, job_title, candidate, generate)

    payload = new_payload(job_path.stem, candidate)
    payload.update({"mode": "candidate", "job_title": job_title,
                    "job_text": job_text, "cv_text": cv.full_text(),
                    "rewrites": results})
    yield Event("run", "done", {"payload": payload},
                time.time() - started, candidate)
    return payload
