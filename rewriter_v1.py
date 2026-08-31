"""Rewriter Variant 1 - a phrasing-sensitivity probe, not a CV you keep.

Paraphrase a CV toward a job title (the rewriter never sees the requirements),
check the paraphrase invented nothing, then re-score it with the same evaluator.
A score that moves after a rephrasing that added no information means the
ranking was reading wording, not substance.

    scorer = rewriter_v1.make_scorer("qwen2.5:14b")
    result = rewriter_v1.probe(cv_text, "Fullstack Engineer", scorer)
    result.delta        # 0 means the ranking ignored phrasing

From the shell:  python rewriter_v1.py [--verify-determinism]
"""

import argparse
import collections
import difflib
import json
import re
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

# prompts and run_evaluations live in pipeline/; this file lives at the root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))

import prompts  # noqa: E402
import run_evaluations  # noqa: E402
import cv_parser  # noqa: E402

# --- 1. Fact extraction ---
#
# The probe's whole claim is "a pure rephrasing, no new information". An
# instruction-tuned model asked to improve a CV will invent scope and upgrade
# verbs unless stopped, and a score jump caused by invented content would read
# as phrasing sensitivity - inverting the finding. These are the token classes
# an embellishing model reliably fabricates.

# Keep % attached: "72.8%" is a different claim from "72.8". Dots are kept in
# the first pass and classified in the second, because stripping them outright
# turns "72.8%" into "72 8%" - two facts where there was one.
_PUNCT = re.compile(r"[^\w\s%.]+")
_DOT_NOT_DECIMAL = re.compile(r"(?<!\d)\.|\.(?!\d)")
_SPACE = re.compile(r"\s+")

_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?")

# Two or more consecutive capitalised words: organisations, job titles, product
# names. A SINGLE capitalised word is not collected - every sentence starts with
# one, and flagging those would reject every rewrite ever produced.
_CAPS_RUN = re.compile(r"\b[A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*)+")

_QUALIFICATION = re.compile(
    r"\b(?:BSc|BA|BCom|BEng|BTech|MSc|MBA|MEng|MTech|PhD|Hons|Honours|Matric|"
    r"Matriculation|NQF|HND|Diploma|Certificate|CompTIA|AWS|Azure|Cisco|CCNA|"
    r"CCNP|ITIL|PRINCE2|PMP|Scrum)\b",
    re.I)

# An abbreviation and its expansion are one qualification. Expanding an
# abbreviation is exactly what a paraphraser does, so without this the guard
# reads "BCom Hons" -> "BCom Honours" as an invented qualification and rejects an
# honest rewrite. That false rejection fired on THE DANIEL.pdf, the first real CV
# put through the probe, and it corrupts a rejection rate we report.
#
# Only over-reporting is fixed here. The reverse - "MSc" expanding to "Master of
# Science" - is not detected, since the phrase is not in the lexicon. That is
# under-reporting, the safe direction throughout this module: a missed
# embellishment weakens one row, a false rejection discards a good one.
_QUALIFICATION_CANON = {
    "hons": "honours",
    "matriculation": "matric",
}


def normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    So that a rephrasing which only re-punctuates never registers as a new
    fact. % survives because it changes the claim, and a decimal point between
    two digits survives because "72.8" is one number, not two.
    """
    text = _PUNCT.sub(" ", text)
    text = _DOT_NOT_DECIMAL.sub(" ", text)
    return _SPACE.sub(" ", text).strip().lower()


def _extract(text: str, patterns) -> set[str]:
    facts = set()
    for pattern in patterns:
        for match in pattern.findall(text):
            fact = normalise(match)
            if fact:
                facts.add(_QUALIFICATION_CANON.get(fact, fact))
    return facts


def extract_facts(text: str) -> set[str]:
    """Every fact token: numbers, dates, entity runs, qualifications.

    The union of extract_hard_facts and extract_caps_runs. Only the hard half
    drives rejection - see check_no_new_facts and advisory_caps_runs.
    """
    return _extract(text, (_NUMBER, _CAPS_RUN, _QUALIFICATION))


def extract_hard_facts(text: str) -> set[str]:
    """Numbers and qualifications only - the unambiguous half.

    A rephrasing cannot introduce a number or a certification token without
    inventing the claim behind it, so these are safe to reject on. Capitalised
    runs are not: rephrasing moves words to sentence-initial position, re-cases
    headings and re-titles roles, so a caps run appearing in the rewrite is
    usually formatting rather than fabrication. Those go to advisory_caps_runs.
    """
    return _extract(text, (_NUMBER, _QUALIFICATION))


def extract_caps_runs(text: str) -> set[str]:
    """Capitalised multi-word runs: organisations, job titles, product names."""
    return _extract(text, (_CAPS_RUN,))


# --- 2. The guard ---


def _new_against(original: str, candidates: set[str]) -> list[str]:
    """Which candidate facts do not appear in the original.

    Word-boundary matched, not substring: `"5" in haystack` is true whenever
    any date or phone number happens to contain a 5, so an invented "5 teams"
    passed against an original mentioning 2015.

    The boundary test alone is too strict in one direction, so the original's
    own extracted facts are accepted as well. The extractor splits "3rd" into
    the fact "3" while normalise keeps the token whole, and without the union
    a rewrite that merely re-cased "3rd" would be rejected for inventing a 3.
    Accepting both is the safe direction: under-reporting weakens one row, a
    false rejection discards a good one.
    """
    haystack = normalise(original)
    own = extract_facts(original)
    return sorted(
        f for f in candidates
        if f not in own
        and not re.search(rf"(?<!\w){re.escape(f)}(?!\w)", haystack))


def numbers_lost(original: str, rewritten: str) -> list[str]:
    """Numbers present in the original and missing from the rewrite.

    The one extractor reliable in both directions, so the only loss worth
    rejecting on. Capitalised runs are not: "Certified Software Test Analyst"
    becoming "Certified as a Software Test Analyst" reads as a lost entity while
    the fact survives, and rejecting on that discards honest rewrites. Those
    stay in facts_lost, visible in the report rather than acted on.

    Added because the prompt could not do it: told "Do not omit anything. Every
    employer, job title, role, date, number ... must still be there", the
    rewriter still dropped the number 2 from ALEX PARK.
    """
    return _new_against(rewritten, _extract(original, (_NUMBER,)))


def rejection_reasons(original: str, rewritten: str, job_title: str) -> list[str]:
    """Everything that disqualifies a rewrite: a new hard fact, a borrowed job
    title, or a number the original had and the rewrite does not.

    One function so the first pass and the strict retry cannot diverge. They did:
    the title check was added to the first pass only, and a rewrite that adopted
    the role twice was retried and then accepted.
    """
    reasons = check_no_new_facts(original, rewritten)
    if title_fabricated(original, rewritten, job_title):
        reasons.append(f"job title claimed but not held: {job_title}")
    for number in numbers_lost(original, rewritten):
        reasons.append(f"number dropped from the CV: {number}")
    return reasons


def title_fabricated(original: str, rewritten: str, job_title: str) -> bool:
    """Did the rewrite adopt the role being applied for, having never held it?

    Rejecting on this specifically, rather than on capitalised runs generally,
    because the two are not comparable. A caps run in a rewrite is usually
    re-casing; the advertised job title appearing in a CV that never contained
    it is a claim to experience the candidate does not have, and the most
    consequential thing this rewriter could invent. Measured on ALEX PARK,
    where the guard passed exactly that as an advisory.
    """
    target = normalise(job_title)
    if not target:
        return False
    return target in normalise(rewritten) and target not in normalise(original)


def check_no_new_facts(original: str, rewritten: str) -> list[str]:
    """HARD facts in the rewrite that are absent from the original.

    Empty list means the rewrite invented no number and no qualification, and
    the probe is valid for this candidate. This is the only signal that
    rejects a rewrite.

    Scope is deliberately narrow. Capitalised runs were part of this check and
    dominated rejections on real CVs, because rephrasing re-cases headings and
    moves words to sentence-initial position. They are now reported by
    advisory_caps_runs and never reject.

    Membership is tested against the normalised FULL TEXT of the original as
    well as its own extracted facts - see _new_against. Testing set-against-set
    alone would fire whenever the extractor caught a phrase in the rewrite but
    missed the same phrase in the source, rejecting honest paraphrases.

    One-directional. A rewrite that DROPS a fact passes: losing content is a
    different failure and does not invalidate a phrasing probe. (probe records
    that separately as facts_lost and length_ratio.)

    What this does NOT catch, all verified against real output:

    - semantic inflation: "helped with" becoming "led" invents no token
    - word-numbers: "fourteen" is not matched by the digit pattern
    - lowercase tools: "python, kubernetes" carries no capital to catch
    - single-word organisations: "Google" alone is not a multi-word run, and
      single capitals cannot be collected - every sentence starts with one
    - vague scope: "over a decade", "a large team" state a claim with no token
    - invented multi-word entities generally, since the caps-run signal became
      advisory: "Previously at Google Cloud" is reported, not rejected

    A checker model would catch most of these, at the cost of doubling model
    calls and needing its own reliability established. Out of scope for
    variant 1, and stated as a limitation in the report rather than left to be
    discovered.
    """
    return _new_against(original, extract_hard_facts(rewritten))


def advisory_caps_runs(original: str, rewritten: str) -> list[str]:
    """Capitalised runs in the rewrite that are absent from the original.

    Advisory only: recorded on the row and reported, never used to reject.
    A real invented organisation shows up here, but so does every re-cased
    heading, so this is evidence for a human to read rather than a gate.
    """
    return _new_against(original, extract_caps_runs(rewritten))


# --- 3. The rewriting agent ---
#
# One call, plain text. It deliberately does not reuse
# run_evaluations.call_ollama, which hardcodes "format": "json" - right for a
# scorer, wrong here, as it would force the rewritten CV into a JSON object.

OLLAMA_URL = "http://localhost:11434/api/generate"
# The rewriting model. Measured on batch3/bad_cv_kevin_jones, qwen2.5:3b
# returned the CV essentially unchanged (99% word overlap) for one job
# title and a genuine paraphrase for another, at temperature 0 and a fixed
# seed - so a delta measured against it can reflect an absent rewrite
# rather than an indifferent ranking. See ProbeResult.similarity.
DEFAULT_REWRITER_MODEL = "qwen2.5:14b"
DEFAULT_NUM_CTX = 8192


class RewriteError(RuntimeError):
    """The rewriting agent returned nothing usable."""


_REWRITE_PROMPT = """Rewrite the CV below in neutral, consistent professional prose.

The point is to remove presentation advantage. Two candidates who did the same
work should read the same way, so that whoever reviews these compares what the
candidates did rather than which of them wrote the better-sounding CV.

Rules, all mandatory:
- Rewrite every sentence. Do not reuse the original phrasing.
- Change wording only. Every fact must survive exactly: employers, job titles,
  dates, numbers, tools, qualifications, and every section and every role.
- Do not omit anything. Every employer, job title, role, date, number,
  tool and qualification in the CV must still be there afterwards.
- Do not add any fact that is not already in the CV.
- Do not invent employers, job titles, dates, numbers, tools or qualifications.
- Do not upgrade a claim: "assisted with" must not become "led".
- Output the rewritten CV as plain text and nothing else. No preamble, no
  commentary, no markdown fences.

CV:
{cv}
"""

_STRICT_SUFFIX = """\

A previous attempt at this rewrite INVENTED information that was not in the \
CV. That is the single thing you must not do. Copy every name, number, date \
and qualification from the CV exactly as written. If you are unsure whether \
something is in the CV, leave it out.
"""


def build_rewrite_prompt(cv_text: str, job_title: str, *, strict: bool = False) -> str:
    """The rewriting instruction.

    Plain replacement rather than str.format, matching prompts.build: the CV
    body routinely contains literal braces that format() would read as fields.

    job_title is accepted and deliberately not used. Normalisation is
    title-independent - the same CV should normalise the same way whatever it
    is read against - and naming the target role invited the model to adopt it:
    on ALEX PARK the rewrite replaced her real title with the advertised one.
    The parameter stays so probe(), paraphrase() and every stub keep their
    signature; title_fabricated() is the backstop if a model reaches for it.
    """
    prompt = _REWRITE_PROMPT.replace("{cv}", cv_text)
    return prompt + _STRICT_SUFFIX if strict else prompt


def _ollama_text(model: str, prompt: str, num_ctx: int, seed: int, timeout: int) -> dict:
    """Plain-text Ollama call. num_ctx is explicit - it truncates silently.

    Returns the whole response body, not just the text, so that
    prompt_eval_count survives to the caller: Ollama truncates BELOW num_ctx
    without saying so, and a rewrite scored from a truncated prompt is a
    content change masquerading as a phrasing effect.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"num_ctx": num_ctx, "seed": seed, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def paraphrase(cv_text: str, job_title: str, *,
               model: str = DEFAULT_REWRITER_MODEL,
               num_ctx: int = DEFAULT_NUM_CTX,
               seed: int = 0, timeout: int = 300,
               strict: bool = False, call=None) -> str:
    """One rewriting pass. Returns the rewritten CV text.

    `call` is injectable so tests never need Ollama, and may return either the
    plain text or the whole Ollama body.

    Truncation of the REWRITE prompt is recorded on the function itself
    (`paraphrase.last_truncated`, `.last_prompt_eval_count`,
    `.last_prompt_chars`) rather than returned, so the signature stays the one
    thing probe() and every stub already agree on. None means unknown - a
    caller that returned bare text told us nothing about tokens.
    """
    call = call or _ollama_text
    prompt = build_rewrite_prompt(cv_text, job_title, strict=strict)
    raw = call(model, prompt, num_ctx, seed, timeout)
    if isinstance(raw, dict):
        text, ptoks = raw.get("response", ""), raw.get("prompt_eval_count")
    else:
        text, ptoks = raw, None
    paraphrase.last_prompt_eval_count = ptoks
    paraphrase.last_prompt_chars = len(prompt)
    paraphrase.last_truncated = (
        run_evaluations.is_truncated(ptoks, len(prompt)) if ptoks is not None else None)
    out = (text or "").strip()
    if not out:
        raise RewriteError(f"{model} returned no text")
    return out


paraphrase.last_prompt_eval_count = None
paraphrase.last_prompt_chars = None
paraphrase.last_truncated = None


# --- 4. The scorer ---
#
# A parameter, not a hardcoded dependency: the probe is a sanity check on the
# ranking a company actually runs, so it has to point at whichever evaluator
# that is. Below is the default, over the existing deterministic temp-0 path.

DEFAULT_PROMPT_VERSION = "v3_criteria"


class ScoreError(RuntimeError):
    """The evaluator returned no usable score."""


def make_scorer(model: str, prompt_version: str = DEFAULT_PROMPT_VERSION,
                num_ctx: int = DEFAULT_NUM_CTX, seed: int = 0,
                timeout: int = 300, call=None):
    """Build scorer(cv_text, job_title) -> int.

    Model and prompt version are bound here rather than passed per call, so
    probe() stays a two-argument call and the bound model can be recorded on
    every row.

    The job TITLE is passed where prompts.build expects the job text. That is
    the variant 1 design: the rewriter and the re-score both see only the
    role, keeping the context window small.

    Each call records whether Ollama truncated the prompt on the scorer
    itself (`scorer.last_truncated`, `.last_prompt_eval_count`,
    `.last_prompt_chars`). The before- and after-prompts differ in length, so
    one can truncate while the other does not - and the resulting delta would
    be read as phrasing sensitivity when it is lost text. None means unknown.
    """
    call = call or run_evaluations.call_ollama

    def scorer(cv_text: str, job_title: str) -> int:
        prompt = prompts.build(prompt_version, job_title, cv_text)
        resp = call(model, prompt, num_ctx, seed, timeout)
        ptoks = resp.get("prompt_eval_count")
        scorer.last_prompt_eval_count = ptoks
        scorer.last_prompt_chars = len(prompt)
        scorer.last_truncated = (
            run_evaluations.is_truncated(ptoks, len(prompt))
            if ptoks is not None else None)
        parsed, err = run_evaluations.parse_response(resp.get("response", ""))
        if parsed is None:
            raise ScoreError(err or "no JSON in evaluator output")
        score = parsed.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ScoreError(f"score was {score!r}, expected a number")
        # round, not int: int(6.9) is 6, so a real 0.5 move would read as 0.
        return round(score)

    scorer.model = model
    scorer.prompt_version = prompt_version
    scorer.last_prompt_eval_count = None
    scorer.last_prompt_chars = None
    scorer.last_truncated = None
    return scorer


# --- 5. The probe ---


@dataclass
class ProbeResult:
    candidate: str
    job: str
    rewriter_model: str
    scorer_model: str
    score_before: int | None = None
    score_after: int | None = None
    delta: int | None = None
    rejected: bool = False
    retried: bool = False
    reasons: list = field(default_factory=list)
    # Advisory: new capitalised runs. Reported, never rejecting - see Ruling A
    # in the final review. Rejection is driven by `reasons` alone.
    advisory: list = field(default_factory=list)
    # None means unknown (a stub scorer, or an Ollama body with no token count).
    truncated_before: bool | None = None
    truncated_after: bool | None = None
    truncated_rewrite: bool | None = None
    # Content loss. The guard passes a rewrite that DROPS facts by design, but
    # a rewrite that omits a role scores lower for content reasons, not
    # phrasing. facts_lost runs the guard with its arguments reversed.
    length_ratio: float | None = None
    facts_lost: int | None = None
    # Word overlap with the original, 1.0 meaning the rewriter handed back what
    # it was given. delta alone cannot separate "phrasing did not matter" from
    # "nothing was rephrased", and no other field can either: a no-op has an
    # ideal length_ratio, loses no facts and invents none, so it reads as a
    # healthy row. Measured on batch3/bad_cv_kevin_jones, where the rewriter
    # changed 2 words of 232 and the delta of 0 would have been reported as
    # evidence the ranking ignores wording.
    similarity: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def probe(cv_text: str, job_title: str, scorer, *,
          candidate: str = "", job: str = "",
          rewriter_model: str = DEFAULT_REWRITER_MODEL,
          paraphrase_fn=None) -> ProbeResult:
    """One candidate against one job: score, paraphrase, guard, re-score.

    Never raises on a model failure. A failed row is recorded with `error` set
    and `delta` None, so one bad candidate does not abort the full run of 270
    (90 candidate-job pairs per scorer model, three scorer models).

    ValueError covers json.JSONDecodeError from a malformed Ollama body, and
    AttributeError a body that is not a mapping: one of those uncaught would
    take the whole batch down.

    A rejected rewrite is still scored and still returned. The rejection rate
    is itself a finding about the rewriting model, and dropping those rows
    here would hide it. Reporting excludes them from the headline delta.

    Rejection is driven by check_no_new_facts alone. advisory_caps_runs is
    recorded on `advisory` and never rejects.
    """
    paraphrase_fn = paraphrase_fn or paraphrase
    result = ProbeResult(candidate=candidate, job=job,
                         rewriter_model=rewriter_model,
                         scorer_model=getattr(scorer, "model", "unknown"))
    try:
        result.score_before = scorer(cv_text, job_title)
        result.truncated_before = getattr(scorer, "last_truncated", None)

        rewritten = paraphrase_fn(cv_text, job_title, model=rewriter_model)
        reasons = rejection_reasons(cv_text, rewritten, job_title)
        if reasons:
            result.retried = True
            rewritten = paraphrase_fn(cv_text, job_title,
                                      model=rewriter_model, strict=True)
            reasons = rejection_reasons(cv_text, rewritten, job_title)
        result.truncated_rewrite = getattr(paraphrase_fn, "last_truncated", None)

        result.reasons = reasons
        result.rejected = bool(reasons)
        result.advisory = advisory_caps_runs(cv_text, rewritten)
        result.length_ratio = (len(rewritten) / len(cv_text)) if cv_text else None
        result.similarity = difflib.SequenceMatcher(
            None, cv_text.split(), rewritten.split()).ratio()
        # Arguments reversed on purpose: facts of the original missing from the
        # rewrite, which the one-directional guard would not see. Caps runs
        # count here, unlike in the rejection direction - losing a named entity
        # is unambiguous ("Umoja Bank" absent means a dropped employer), whereas
        # gaining one is usually just re-casing.
        result.facts_lost = len(_new_against(rewritten, extract_facts(cv_text)))

        result.score_after = scorer(rewritten, job_title)
        result.truncated_after = getattr(scorer, "last_truncated", None)
        result.delta = result.score_after - result.score_before
    except (RewriteError, ScoreError, OSError, TimeoutError,
            ValueError, AttributeError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


# --- 6. The determinism control ---
#
# The probe's validity rests entirely on the evaluator returning the same score
# for the same input. call_ollama sends temperature 0.0 with a fixed seed, so it
# should - but nothing here has ever checked, and GPU batching can drift a
# nominally greedy decode. All 237 rows in Results/votes.jsonl are seed 0, one
# run per (model, job, cv): no CV has ever been scored twice. If this does not
# hold, a post-paraphrase delta cannot be told apart from run-to-run noise and
# the probe measures nothing.


_INTERLEAVE_CV = (
    "Junior Analyst at Kestrel Data from 2021 to 2022. "
    "Maintained 3 dashboards and wrote the monthly report."
)


def verify_determinism(cv_texts: dict, job_title: str, scorer,
                       interleave: str = _INTERLEAVE_CV) -> list:
    """Score every CV twice, unchanged. Returns the CVs whose scores disagreed.

    Empty list means the evaluator reproduced and the probe is valid.

    A DIFFERENT text is scored between the two runs of each pair. Scoring the
    same prompt back to back is the most favourable cache condition there is,
    and the probe never runs that way - a rewrite call sits between its two
    scores. Interleaving makes the control test the condition the probe
    actually measures under, so each CV costs three calls, not two.
    """
    drift = []
    for name, text in cv_texts.items():
        first = scorer(text, job_title)
        scorer(interleave, job_title)   # discarded; exists to disturb the cache
        second = scorer(text, job_title)
        if first != second:
            drift.append((name, first, second))
    return drift


# --- 7. Batch run and reporting ---

ROOT = Path(__file__).resolve().parent
CV_DIR = ROOT / "Batch1" / "CVs"
JOB_DIR = ROOT / "Batch1" / "Job Listings"
OUT = ROOT / "Results" / "rewriter_v1.jsonl"
# The council the team actually runs. The probe is a sanity check on that
# ranking, so scoring it with smaller models measures a system nobody ships.
DEFAULT_SCORER_MODELS = ["qwen2.5:14b", "phi4"]


def title_from_slug(slug: str) -> str:
    """job_listing_startup_fullstack_engineer -> "Startup Fullstack Engineer".

    The slug, not the title printed inside the PDF. The PDF titles are messier
    ("Founding Full-Stack Engineer (#14)") and the slug is what Akram's and
    Nikita's results files are already keyed by, so this keeps keys aligned.
    """
    return slug.replace("job_listing_", "").replace("_", " ").title()


def load_cvs(cv_dir: Path = CV_DIR) -> dict:
    """Candidate name -> full CV text. Parsed once, reused across every job."""
    return {p.stem: cv_parser.parse(p).full_text()
            for p in sorted(cv_dir.glob("*.pdf"))}


def load_job_titles(job_dir: Path = JOB_DIR) -> dict:
    """Job slug -> human-readable title. No PDF is read: only the title is used."""
    return {p.stem.replace("job_listing_", ""): title_from_slug(p.stem)
            for p in sorted(job_dir.glob("*.pdf"))}


def read_rows(path: Path) -> list:
    """Every row in an output file. Missing file is an empty run, not an error.

    A torn final line from an interrupted run is skipped rather than fatal,
    matching run_evaluations.already_done.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def already_done(path: Path) -> set:
    """Keys of rows that must not be re-probed.

    Errored rows are deliberately NOT included. An Ollama restart mid-run
    writes an `error` row for every probe it interrupts; keying those as done
    would make every later resume skip them, permanently losing those probes
    from the study.
    """
    done = set()
    for row in read_rows(path):
        if row.get("error"):
            continue
        done.add((row.get("rewriter_model"), row.get("scorer_model"),
                  row.get("job"), row.get("candidate")))
    return done


def apply_filter(items: dict, wanted, label: str) -> dict:
    """Restrict `items` to `wanted`, or stop with a usable message.

    An unfiltered empty selection used to reach next(iter(jobs.values())) and
    raise StopIteration, which reads as a crash rather than a typo.
    """
    if not wanted:
        return items
    kept = {k: v for k, v in items.items() if k in wanted}
    if not kept:
        raise SystemExit(f"no {label} matched {' '.join(wanted)}; "
                         f"available: {', '.join(sorted(items)) or '(none)'}")
    return kept


_TRUNC_FIELDS = ("truncated_before", "truncated_after", "truncated_rewrite")


def row_truncated(row: dict) -> bool:
    """True if Ollama silently cut any of this row's three prompts.

    `score_before` and `score_after` are built from prompts of different
    lengths, so one can truncate while the other does not. The delta then
    measures lost text, not phrasing - this repo has already watched
    truncation move one candidate 2 -> 8 on otherwise identical inputs.
    """
    return any(row.get(k) for k in _TRUNC_FIELDS)


def summarise(rows: list, flag_threshold: int) -> str:
    """The report table.

    Rejected, truncated and errored rows are excluded from the headline delta
    but every count is reported - the rejection rate is a finding about the
    rewriting model and the truncation rate is a confound that has to be
    stated, not bookkeeping.
    """
    if not rows:
        return "no rows"

    errored = [r for r in rows if r.get("error")]
    scored = [r for r in rows if not r.get("error") and r.get("delta") is not None]
    truncated = [r for r in scored if row_truncated(r)]
    usable = [r for r in scored if not row_truncated(r)]
    # Counted over every scored row, not just untruncated ones. The rejection
    # rate is itself a reported finding about the rewriting model, and a rewrite
    # that was both rejected and truncated is still a rejection - counting over
    # `usable` biased the rate low. Truncation is an overlapping tally, reported
    # separately, not a filter on this count.
    rejected = [r for r in scored if r.get("rejected")]
    clean = [r for r in usable if not r.get("rejected")]
    flagged = [r for r in clean if abs(r["delta"]) >= flag_threshold]

    lines = [
        f"{len(clean)} clean, {len(rejected)} rejected (of {len(scored)} scored), "
        f"{len(truncated)} truncated (excluded), {len(errored)} errored",
        f"{len(flagged)} flagged (|delta| >= {flag_threshold}) of {len(clean)} clean rewrites",
    ]

    if clean:
        deltas = [r["delta"] for r in clean]
        lines.append(f"delta: min {min(deltas)}  max {max(deltas)}  "
                     f"mean {sum(deltas) / len(deltas):+.2f}  "
                     f"unchanged {sum(1 for d in deltas if d == 0)}")

        # Pooling the scorer models hides the effect: they do not share a score
        # distribution, so a mean over all three is a mean over three scales.
        lines.append("")
        lines.append("by scorer model:")
        lines.append(f"  {'model':20s}  {'n':>3}  {'mean':>7}  {'min':>4}  {'max':>4}")
        by_model = collections.defaultdict(list)
        for r in clean:
            by_model[r.get("scorer_model", "unknown")].append(r["delta"])
        for model in sorted(by_model):
            ds = by_model[model]
            lines.append(f"  {model[:20]:20s}  {len(ds):>3}  "
                         f"{sum(ds) / len(ds):>+7.2f}  {min(ds):>4}  {max(ds):>4}")

        lines.append("")
        lines.append("by starting score:")
        buckets = collections.defaultdict(list)
        for r in clean:
            buckets[r["score_before"]].append(r["delta"])
        lines.append(f"  {'before':>6}  {'n':>3}  {'mean delta':>10}  {'min':>4}  {'max':>4}")
        for before in sorted(buckets, key=lambda b: (b is None, b)):
            ds = buckets[before]
            lines.append(f"  {str(before):>6}  {len(ds):>3}  "
                         f"{sum(ds) / len(ds):>+10.2f}  {min(ds):>4}  {max(ds):>4}")

    if rejected:
        invented = collections.Counter(
            fact for r in rejected for fact in r.get("reasons", []))
        lines.append("")
        lines.append("most-invented facts: " + ", ".join(
            f"{f} ({n})" for f, n in invented.most_common(5)))

    advised = [r for r in usable if r.get("advisory")]
    if advised:
        runs = collections.Counter(
            fact for r in advised for fact in r.get("advisory", []))
        lines.append("")
        lines.append(f"{len(advised)} rows carry caps-run advisories "
                     f"(not rejections): " + ", ".join(
                         f"{f} ({n})" for f, n in runs.most_common(5)))

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Variant 1 phrasing-sensitivity probe.")
    ap.add_argument("--verify-determinism", action="store_true",
                    help="score every CV twice unchanged and stop")
    ap.add_argument("--rewriter-model", default=DEFAULT_REWRITER_MODEL)
    ap.add_argument("--scorer-models", nargs="+", default=DEFAULT_SCORER_MODELS)
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION,
                    choices=sorted(prompts.VERSIONS))
    ap.add_argument("--flag-threshold", type=int, default=2)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--jobs", nargs="+")
    ap.add_argument("--cvs", nargs="+")
    ap.add_argument("--report", action="store_true",
                    help="summarise the existing output file and stop")
    args = ap.parse_args()

    if args.report:
        # Ruling: a missing output file (no probe recorded yet) must yield an
        # empty row list, not a FileNotFoundError - summarise([], ...) already
        # answers with "no rows".
        print(summarise(read_rows(OUT), args.flag_threshold))
        return

    cvs = apply_filter(load_cvs(), args.cvs, "CVs")
    jobs = apply_filter(load_job_titles(), args.jobs, "jobs")
    if not cvs or not jobs:
        raise SystemExit("nothing to do: Batch1 has no CVs or no job listings")

    if args.verify_determinism:
        # The gate. Until this passes, every delta the probe reports is noise,
        # so a drift is exit 1, not a printed line the caller can ignore.
        failed = False
        title = next(iter(jobs.values()))
        for model in args.scorer_models:
            scorer = make_scorer(model, args.prompt_version, args.num_ctx)
            drift = verify_determinism(cvs, title, scorer)
            if drift:
                failed = True
                print(f"FAIL {model}: {len(drift)} of {len(cvs)} CVs did not reproduce")
                for name, a, b in drift:
                    print(f"  {name:28s} {a} then {b}")
            else:
                print(f"OK   {model}: all {len(cvs)} CVs reproduced exactly")
        if failed:
            raise SystemExit(1)
        return

    total = len(args.scorer_models) * len(jobs) * len(cvs)
    done = already_done(OUT)
    print(f"{len(args.scorer_models)} scorers x {len(jobs)} jobs x {len(cvs)} CVs "
          f"= {total} probes; {len(done)} already recorded\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    with OUT.open("a", encoding="utf-8") as sink:
        for model in args.scorer_models:          # model-major: keeps it resident
            scorer = make_scorer(model, args.prompt_version, args.num_ctx)
            for job_name, job_title in jobs.items():
                for candidate, cv_text in cvs.items():
                    if (args.rewriter_model, model, job_name, candidate) in done:
                        continue
                    result = probe(cv_text, job_title, scorer,
                                   candidate=candidate, job=job_name,
                                   rewriter_model=args.rewriter_model)
                    row = result.to_dict()
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()                   # never lose work to a crash
                    n += 1
                    flag = ("ERR" if result.error else
                            "REJ" if result.rejected else "")
                    print(f"  [{n:3d}/{total}] {model:16s} {job_name[:22]:22s} "
                          f"{candidate[:18]:18s} "
                          f"{str(result.score_before):>3s} -> {str(result.score_after):>3s} "
                          f"delta {str(result.delta):>4s} {flag}")

    print()
    print(f"this run: {len(rows)} new rows")
    print()
    # Summarise the WHOLE file, not just this run's rows: a resumed run holds
    # only the slice it happened to complete, and a headline over that slice
    # is a headline over an arbitrary subset.
    print(f"summary over all {OUT.name} rows:")
    print(summarise(read_rows(OUT), args.flag_threshold))


if __name__ == "__main__":
    main()
