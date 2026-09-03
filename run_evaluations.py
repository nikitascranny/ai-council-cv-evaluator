"""Stage 2: evaluate. The only expensive stage.

Runs (job x cv x model) through Ollama and appends one row per evaluation to
votes.jsonl. Never overwrites; safe to re-run after a crash or Ctrl-C.

Key behaviours, all deliberate:

  * MODEL-MAJOR loop order. Ollama evicts a model when another is requested, and
    on 4GB VRAM a reload costs seconds. Iterating model-major loads each model
    once; candidate-major would reload on every CV.
  * Explicit num_ctx. Ollama truncates silently unless told otherwise - measured
    on 0.32.6, the default cuts a 7,035-token prompt to 2,050. LERATO MUNCHE's CV
    alone is ~5.9k tokens, so the default drops most of it. Truncating moved her
    score from 2 to 8 on the same model, prompt and seed.
  * Truncation is recorded, not guessed - see is_truncated() for why the obvious
    test does not work.
  * The extractor is recorded too. pypdf and pymupdf disagree on word fusion and
    on the ORDER sections appear in, which is a confound if left unlabelled.
  * Raw output is always stored, so a parse failure is diagnosable instead of a
    silently dropped vote.

Usage:
    python run_evaluations.py --models qwen2.5:14b phi4
    python run_evaluations.py --dry-run          # cost estimate, no calls
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import prompts

# cv_parser.py lives in parser/, a sibling of this file at the repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "parser"))
import cv_parser  # noqa: E402

CV_DIR = ROOT / "Batch1" / "CVs"
JOB_DIR = ROOT / "Batch1" / "Job Listings"
CACHE = ROOT / ".textcache"
OUT = ROOT / "Results" / "votes.jsonl"
OLLAMA = "http://localhost:11434/api/generate"

# The models actually pulled on this machine (`ollama list`, all 3B-class, Q4).
# This list used to name qwen2.5:7b, llama3.1:8b, phi4, mistral:7b and
# deepseek-r1:7b - none of which are installed. Ollama answers an unknown model
# with an HTTP 404, urllib raises HTTPError, and HTTPError is a URLError
# subclass, so the except around call_ollama caught it and wrote an ERROR ROW.
# Running the bare documented command therefore appended 5 x 9 x 10 = 450 junk
# rows to the shared Results/votes.jsonl, which already_done() then counts as
# work completed. Naming only installed models is what stops that.
DEFAULT_MODELS = ["qwen2.5:14b", "phi4"]

# --- probe set: the minimum run that decides which models are worth using -----
# One positive job and one negative control, against four CVs chosen because each
# one tests a different failure mode. Deliberately small: model selection needs
# decisive checks, not a full ranking.
PROBE_JOBS = (
    "startup_fullstack_engineer",       # a real best candidate exists
    "aquarium_biologist_entry_level",   # negative control: nobody holds the required degree
)
PROBE_CVS = (
    "ALEX PARK",              # should top the fullstack job - can the model find the best fit?
    "JORDAN VELDMAN",         # architect of BUILDINGS - hallucination trap; deepseek-r1
                              #   credited him with React/TypeScript/Go, none of which he has
    "LERATO MUNCHE",          # 5.9k tokens - truncation trap; also lists JS/Python with no
                              #   engineering work history, so tests claim-vs-evidence
    "NDLOVU KARABO REFILWE",  # inverts between the two jobs: best aquarium fit, worst fullstack
)


def extract(pdf: Path, extractor: str = "pymupdf", parser: str = "flat") -> str:
    """PDF -> text, cached per (parser, extractor).

    Parsing is deterministic, so never redo it - but cache per parser AND per
    extractor. Sharing one cache across parsers silently reuses old text and the
    comparison stops being a comparison, which is the same trap the per-extractor
    split already guards against.

    parser="flat"        the original behaviour: cv_parser.extract()
    parser="structured"  geometry-aware parsing, which repairs reading order on
                         two-column CVs. Cached as the full ParsedCV JSON so the
                         UI and the essential-requirements agent can reuse it
                         without re-parsing.
    """
    if parser == "flat":
        directory = CACHE / extractor
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / (pdf.stem + ".txt")
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        text = cv_parser.extract(pdf, extractor)
        cached.write_text(text, encoding="utf-8")
        return text

    directory = CACHE / f"structured_v{cv_parser.PARSER_VERSION}" / extractor
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / (pdf.stem + ".json")

    cv = None
    if cached.exists():
        try:
            cv = cv_parser.ParsedCV.from_dict(
                json.loads(cached.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            # A torn cache entry must not brick every later run. write_text is
            # not atomic, so Ctrl-C during startup extraction can leave a
            # truncated file (JSONDecodeError) or one written under an older
            # schema (KeyError) - and this loop runs before any model call, so
            # the whole run would die at startup with nothing recoverable.
            # already_done() tolerates a torn final line for exactly this
            # reason. Re-parsing costs under a second per CV and honours this
            # module's promise that it is safe to re-run after a crash.
            cv = None
    if cv is None:
        cv = cv_parser.parse(pdf, extractor)
        cached.write_text(json.dumps(cv.to_dict(), indent=1), encoding="utf-8")

    # Warn on the cache HIT as well as the miss. Scanned/image-only pages are
    # an explicit non-goal - detected and warned, never handled - so this print
    # is the only signal that a CV arrived empty. Warning only on the miss meant
    # run 1 flagged it and every run after silently fed "" to the model.
    for warning in cv.warnings:
        print(f"  WARN {pdf.stem}: {warning}")
    return cv.full_text()


def key_of(row: dict) -> tuple:
    """Identity of one evaluation. Every field that can change the result
    belongs here, or a rerun is wrongly skipped as already done."""
    return (row["job"], row["cv"], row["model"], row["prompt_version"],
            row["num_ctx"], row["seed"], row.get("extractor", "pypdf"),
            row.get("parser", "flat"))


def is_truncated(prompt_tokens, prompt_chars) -> bool:
    """Detect that Ollama silently dropped part of the prompt.

    Comparing prompt_tokens against num_ctx does NOT work: Ollama truncates to
    well BELOW num_ctx, reserving the rest for generation. At num_ctx 2048 it
    cut every oversized prompt to exactly 1026 tokens, so a `>= num_ctx` test
    reports False on prompts that lost 86% of their content.

    Instead compare against the text actually sent. English tokenises at roughly
    4 chars/token across these models; requiring fewer than one token per 6 chars
    means at least a third of the text vanished. That threshold is loose enough
    to avoid false positives from tokeniser differences and still caught all
    observed truncations, including the marginal 2334 -> 1026 case.
    """
    if prompt_tokens is None or not prompt_chars:
        return False
    return prompt_tokens < prompt_chars / 6


def parse_response(raw: str):
    """Extract the JSON object. Reasoning models wrap it in <think> blocks and
    most models fence it in markdown, so strip both before parsing."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned.strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.S)  # fall back to first {...} block
    if match:
        try:
            return json.loads(match.group()), None
        except json.JSONDecodeError as exc:
            return None, f"json: {exc}"
    return None, "no JSON object found in output"


def call_ollama(model: str, prompt: str, num_ctx: int, seed: int, timeout: int):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",  # hold the model across this job's 10 CVs
        "options": {"num_ctx": num_ctx, "seed": seed, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def already_done(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn final line from an interrupted run
        keys.add(key_of(r))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--jobs", nargs="+", default=None, help="job stems; default all 9")
    ap.add_argument("--cvs", nargs="+", default=None, help="CV stems; default all 10")
    ap.add_argument("--probe", action="store_true",
                    help="model-selection subset: 2 jobs x 4 discriminative CVs")
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="8192 fits the longest CV + longest listing + output")
    ap.add_argument("--prompt-version", default="v1_plain", choices=list(prompts.VERSIONS))
    ap.add_argument("--extractor", default="pymupdf", choices=cv_parser.EXTRACTORS,
                    help="pypdf reproduces the original Batch1 text; pymupdf is cleaner")
    ap.add_argument("--parser", default="flat", choices=("flat", "structured"),
                    help="flat reproduces the original text; structured repairs "
                         "reading order on two-column CVs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cvs = {p.stem: extract(p, args.extractor, args.parser)
           for p in sorted(CV_DIR.glob("*.pdf"))}
    jobs = {p.stem.replace("job_listing_", ""): extract(p, args.extractor, args.parser)
            for p in sorted(JOB_DIR.glob("*.pdf"))}

    if args.probe:
        args.jobs = list(PROBE_JOBS)
        args.cvs = list(PROBE_CVS)
    if args.jobs:
        jobs = {k: v for k, v in jobs.items() if k in args.jobs}
    if args.cvs:
        cvs = {k: v for k, v in cvs.items() if k in args.cvs}

    total = len(args.models) * len(jobs) * len(cvs)
    print(f"{len(args.models)} models x {len(jobs)} jobs x {len(cvs)} CVs = {total} evaluations")

    if args.dry_run:
        worst = max(len(prompts.build(args.prompt_version, j, c))
                    for j in jobs.values() for c in cvs.values()) // 4
        print(f"largest prompt ~{worst} tokens vs num_ctx {args.num_ctx}: "
              f"{'OK' if worst < args.num_ctx else 'WILL TRUNCATE'}")
        print(f"at 60s/eval ~{total * 60 / 3600:.1f}h; at 120s/eval ~{total * 120 / 3600:.1f}h")
        return

    done = already_done(OUT)
    print(f"{len(done)} evaluations already recorded, skipping those\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with OUT.open("a", encoding="utf-8") as sink:
        for model in args.models:  # MODEL-MAJOR: keeps each model resident
            for job_name, job_text in jobs.items():
                for cv_name, cv_text in cvs.items():
                    row = {"job": job_name, "cv": cv_name, "model": model,
                           "prompt_version": args.prompt_version,
                           "num_ctx": args.num_ctx, "seed": args.seed,
                           "extractor": args.extractor, "parser": args.parser}
                    if key_of(row) in done:
                        continue

                    prompt = prompts.build(args.prompt_version, job_text, cv_text)

                    start = time.time()
                    try:
                        resp = call_ollama(model, prompt, args.num_ctx, args.seed, args.timeout)
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        row |= {"error": f"{type(exc).__name__}: {exc}", "parse_ok": False,
                                "latency_s": round(time.time() - start, 1)}
                        sink.write(json.dumps(row) + "\n")
                        sink.flush()
                        print(f"  FAIL {model:16s} {job_name[:22]:22s} {cv_name[:18]:18s} {exc}")
                        continue

                    raw = resp.get("response", "")
                    parsed, err = parse_response(raw)
                    ptoks = resp.get("prompt_eval_count")

                    row |= {
                        "score": parsed.get("score") if parsed else None,
                        "criteria": ({k: parsed.get(k) for k in
                                      ("required_skills", "evidence", "transferability",
                                       "findability", "customisation")}
                                     if parsed and "evidence" in parsed else None),
                        "strengths": parsed.get("strengths") if parsed else None,
                        "weaknesses": parsed.get("weaknesses") if parsed else None,
                        "summary": parsed.get("summary") if parsed else None,
                        "candidate_reported": parsed.get("candidate") if parsed else None,
                        "prompt_tokens": ptoks,
                        "prompt_chars": len(prompt),
                        "truncated": is_truncated(ptoks, len(prompt)),
                        "parse_ok": parsed is not None,
                        "parse_error": err,
                        "raw_output": raw,
                        "latency_s": round(time.time() - start, 1),
                    }
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()  # never lose work to a crash

                    n_done += 1
                    flag = "TRUNC" if row["truncated"] else ("BAD" if not row["parse_ok"] else "")
                    print(f"  [{n_done:3d}/{total}] {model:16s} {job_name[:22]:22s} "
                          f"{cv_name[:18]:18s} score={str(row['score']):>4s} "
                          f"{row['latency_s']:6.1f}s {ptoks or '?':>5} tok {flag}")

    print(f"\nwrote {n_done} new evaluations to {OUT}")


if __name__ == "__main__":
    main()
