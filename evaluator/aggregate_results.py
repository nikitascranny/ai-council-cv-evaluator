"""
aggregate_results.py

judge.py expects the following grouping: for ONE candidate applying to ONE job,
all the independent models' evaluations of that candidate, combined into a single file. 
This script does that regrouping for command line runs (i.e. preparing the data for the judge).

Evaluate_CV.py produces one JSON file per (model, job listing) pair, e.g.:

    qwen2.5_14b__job_listing_secretary.json
    phi4__job_listing_secretary.json

Each of those files is a JSON list with one entry per candidate:

    {
      "score": 8,
      "strengths": ["...", "..."],
      "weaknesses": ["...", "..."],
      "summary": "...",
      "candidate": "LERATO MUNCHE"
    }

Run with --essential-requirements, it also writes one essential-requirements
file per job listing (into results/essential_requirements/), one entry per
candidate:

    essential_requirements__job_listing_secretary.json

    {
      "summary": "...",
      "candidate": "LERATO MUNCHE"
      ("error": "..." only when the check failed)
    }


Output layout:

    <outdir>/<job_slug>/<candidate_slug>.json

Each output file is a JSON list, one entry per model that evaluated that
candidate for that job. Every entry keeps the original fields untouched and
gains a "model" field (for audit trail only -- judge.py never sends the
model name to the judge model, and never reads "score" by default either).

If an essential-requirements check exists for that candidate and job, it is
added as the last entry, with its original fields untouched plus
"type": "essential_requirements" instead of a "model" field. It is not a
reviewer's summary, and judge.py uses that marker to give it its own section
of the prompt rather than numbering it among the reviewers. A check whose
candidate has no model evaluations for that job is skipped with a warning
(judge.py has nothing to judge without them)

Usage:
    # point at a directory of *__job_listing_*.json files produced by Evaluate_CV.py
    python aggregate_results.py --results-dir results/ --outdir combined/

    # include the essential-requirements check
    python aggregate_results.py --results-dir results/ \\
        --essential-dir results/essential_requirements/ --outdir combined/

    # or list files explicitly (globs work too, shell-expanded); an
    # essential_requirements__job_listing_*.json file listed here is
    # recognised by its name and joined the same way
    python aggregate_results.py --results results/*.json --outdir combined/

    # also strip "score" at aggregation time 
    # (judge.py already ignores it by default when building its prompt)
    python aggregate_results.py --results-dir results/ --outdir combined/ --drop-scores
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# "<model>__job_listing_<job_slug>.json" -- model names may themselves
# contain underscores/dots (e.g. "qwen2.5_14b"), so split on the fixed
# "__job_listing_" marker rather than trying to regex the model name shape
SPLIT_MARKER = "__job_listing_"

REQUIRED_KEYS = {"candidate", "summary"}

# The filename prefix Evaluate_CV.py gives essential-requirements results, in
# place of a model name.
ESSENTIAL_PREFIX = "essential_requirements"

# Marks the essential-requirements entry in a combined file. Must equal
# judge.ESSENTIAL_TYPE - kept as a copy rather than imported 
# (importing judge.py pulls in the Ollama and PDF helpers this script does not use) 
# tests/unit-tests/test_aggregate_results.py pins the two together.
ESSENTIAL_TYPE = "essential_requirements"


def slugify(name: str) -> str:
    """Turn a candidate/job name into a filesystem- and shell-safe slug."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name or "unknown"


def parse_filename(path: Path):
    """Return (model, job_slug) parsed from a result filename, or (None, None)
    if the filename doesn't match the expected convention. For an
    essential-requirements file, "model" is ESSENTIAL_PREFIX."""
    stem = path.name
    if not stem.endswith(".json") or SPLIT_MARKER not in stem:
        return None, None
    model, _, rest = stem.partition(SPLIT_MARKER)
    job = rest[: -len(".json")]
    if not model or not job:
        return None, None
    return model, job


def load_entries(path: Path):
    """Load and validate one result file. Returns a list of dicts; bad
    entries/files are skipped with a warning rather than aborting the run."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"  WARN could not read {path.name}: {e}", file=sys.stderr)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  WARN skipping {path.name}: invalid JSON ({e})", file=sys.stderr)
        return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        print(
            f"  WARN skipping {path.name}: expected a JSON list (or object), "
            f"got {type(data).__name__}",
            file=sys.stderr,
        )
        return []

    entries = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"  WARN {path.name}[{i}]: not a JSON object, skipped", file=sys.stderr)
            continue
        missing = REQUIRED_KEYS - entry.keys()
        if missing:
            print(
                f"  WARN {path.name}[{i}]: missing field(s) {sorted(missing)}, skipped",
                file=sys.stderr,
            )
            continue
        if not str(entry.get("candidate", "")).strip():
            print(f"  WARN {path.name}[{i}]: empty candidate name, skipped", file=sys.stderr)
            continue
        entries.append(entry)
    return entries


def discover_files(args) -> list:
    files = []
    if args.results_dir:
        d = Path(args.results_dir)
        if not d.is_dir():
            sys.exit(f"error: --results-dir {d} is not a directory")
        files.extend(sorted(d.glob("*.json")))
    if args.essential_dir:
        d = Path(args.essential_dir)
        if not d.is_dir():
            sys.exit(f"error: --essential-dir {d} is not a directory")
        files.extend(sorted(d.glob(f"{ESSENTIAL_PREFIX}{SPLIT_MARKER}*.json")))
    if args.results:
        files.extend(Path(p) for p in args.results)
    if not files:
        sys.exit("error: no input files given (use --results-dir, --essential-dir and/or --results)")
    return files


def aggregate(files, drop_scores=False):
    """
    Regroup result files into combined[job_slug][candidate_slug] -> entries.

    Model entries are tagged with "model", in file order. Each candidate's
    essential-requirements check, if any, is appended after them, tagged
    with "type": ESSENTIAL_TYPE (NOT marked as a model)

    Returns (combined, stats); stats counts files used and skipped, and
    lists the (job_slug, candidate_slug) pairs whose check was skipped.
    """
    combined = defaultdict(lambda: defaultdict(list))
    checks = defaultdict(dict)  # job_slug -> candidate_slug -> essential entry
    stats = {"files_used": 0, "files_skipped": 0, "essential_unmatched": []}

    for path in files:
        model, job = parse_filename(path)
        if model is None:
            print(
                f"  WARN skipping {path.name}: doesn't match "
                f"'<model>{SPLIT_MARKER}<job_slug>.json'",
                file=sys.stderr,
            )
            stats["files_skipped"] += 1
            continue

        entries = load_entries(path)
        if entries:
            stats["files_used"] += 1
        for entry in entries:
            candidate = str(entry["candidate"]).strip()
            cslug = slugify(candidate)

            if model == ESSENTIAL_PREFIX:
                if cslug in checks[job]:
                    print(
                        f"  WARN {path.name}: a second essential-requirements entry for "
                        f"{candidate} ({job}), keeping the first",
                        file=sys.stderr,
                    )
                    continue
                checks[job][cslug] = dict(entry, type=ESSENTIAL_TYPE)
                continue

            tagged = dict(entry)
            tagged["model"] = model
            if drop_scores:
                tagged.pop("score", None)
            combined[job][cslug].append(tagged)

    for job, by_candidate in checks.items():
        for cslug, check in by_candidate.items():
            if cslug in combined.get(job, {}):
                combined[job][cslug].append(check)
            else:
                print(
                    f"  WARN essential-requirements entry for {job}/{cslug} has no "
                    f"council evaluations to join, skipped",
                    file=sys.stderr,
                )
                stats["essential_unmatched"].append((job, cslug))

    return combined, stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results-dir",
        help="Directory containing <model>__job_listing_<job_slug>.json files",
    )
    ap.add_argument(
        "--essential-dir",
        help="Directory containing essential_requirements__job_listing_<job_slug>.json files "
        "(Evaluate_CV.py writes them to results/essential_requirements/)",
    )
    ap.add_argument(
        "--results",
        nargs="+",
        help="Explicit list of result JSON files (in addition to/instead of --results-dir)",
    )
    ap.add_argument(
        "--outdir", required=True, help="Directory to write combined <job>/<candidate>.json files"
    )
    ap.add_argument(
        "--drop-scores",
        action="store_true",
        help="Strip the 'score' field entirely at aggregation time too "
        "(judge.py already ignores it by default when building its prompt; "
        "this additionally keeps it out of the combined files on disk).",
    )
    args = ap.parse_args()

    files = discover_files(args)
    outdir = Path(args.outdir)

    combined, stats = aggregate(files, drop_scores=args.drop_scores)

    if not combined:
        sys.exit("error: no valid entries found across input files -- nothing to write")

    n_written = 0
    for job, candidates in combined.items():
        job_dir = outdir / job
        job_dir.mkdir(parents=True, exist_ok=True)
        for cslug, entries in candidates.items():
            out_path = job_dir / f"{cslug}.json"
            out_path.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            n_written += 1

    print(f"Read {stats['files_used']} usable result file(s) ({stats['files_skipped']} skipped).")
    print(f"Wrote {n_written} combined candidate file(s) across {len(combined)} job(s) to {outdir}/\n")
    for job, candidates in sorted(combined.items()):
        models_seen = sorted({e["model"] for v in candidates.values() for e in v if "model" in e})
        n_checked = sum(
            1 for v in candidates.values() if any(e.get("type") == ESSENTIAL_TYPE for e in v)
        )
        print(
            f"  {job}: {len(candidates)} candidate(s), models={models_seen}, "
            f"essential check for {n_checked}"
        )


if __name__ == "__main__":
    main()
