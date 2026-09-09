"""
fact_guard.py

Shared anti-hallucination guard for CV rewriting. Extracted from
rewriter_v1.py so both rewriter variants can use the same check rather than
keeping two copies that could drift apart.

The guard's whole claim is "this rewrite is a pure rephrasing / tailoring
pass, not a rewrite that invented anything". An instruction-tuned model
asked to improve or tailor a CV will invent scope and upgrade verbs unless
stopped - these functions catch the unambiguous half of that (numbers,
qualifications, a claimed job title) and flag the ambiguous half
(capitalised entity runs) as advisory rather than rejecting on it.

This matters at least as much for a rewrite the user will actually use
(Var2) as it did for Var1's throwaway phrasing probe - arguably more,
since a fabricated certification or an "assisted with" upgraded to "led"
here becomes a real claim on a real person's CV, not just a measurement
artifact.
"""

import re

# --- Fact extraction ---
#
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


# --- The guard ---


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


# A line holding nothing but a small number is a page number, not a fact. The
# parser keeps them: ALEX PARK.pdf ends "...policies." then "2".
_STANDALONE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$", re.M)


def load_bearing_numbers(text: str) -> set[str]:
    """Numbers worth rejecting over: not page numbers, not lone digits.

    Rejecting on every number was measured too blunt. On ALEX PARK the only
    number the rewrite dropped was the trailing page number 2, so an honest
    rewrite was rejected, and the strict retry dropped it again - rejecting the
    candidate outright. A single digit is rarely a claim; a four-digit year or
    a percentage always is.

    Known gap: the extractor splits "060 000 0000" into three tokens, so a
    rewrite that reformats a phone number reads as losing three figures. Not
    seen in practice yet, recorded here rather than guessed at.
    """
    standalone = {m.group().strip() for m in _STANDALONE_NUMBER_LINE.finditer(text)}
    facts = set()
    for match in _NUMBER.findall(text):
        if match.strip() in standalone:
            continue
        if len(match.strip().rstrip("%")) < 2:
            continue
        token = normalise(match)
        if token:
            facts.add(token)
    return facts


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
    return _new_against(rewritten, load_bearing_numbers(original))


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
    calls and needing its own reliability established. Stated as a known
    limitation rather than left to be discovered.
    """
    return _new_against(original, extract_hard_facts(rewritten))


def advisory_caps_runs(original: str, rewritten: str) -> list[str]:
    """Capitalised runs in the rewrite that are absent from the original.

    Advisory only: recorded on the row and reported, never used to reject.
    A real invented organisation shows up here, but so does every re-cased
    heading, so this is evidence for a human to read rather than a gate.
    """
    return _new_against(original, extract_caps_runs(rewritten))
