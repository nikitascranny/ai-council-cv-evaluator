"""
job_req_parser.py

Pulls just the "essential requirements" section out of a job listing PDF,
instead of handing a consumer the whole job description. Evaluate_CV.py
currently extracts one job_text via document_utils.extract_text_from_pdf()
and passes that same full text to both evaluate_cv() (which genuinely needs
the whole listing) and evaluate_essential_requirement() (which only needs
the requirements) - this module is the fix for the second case.

Real listings label this section under several different headings
("Essential Requirements", "Requirements", "Person Specification" split
into "Essential"/"Desirable", ...), so this looks for the first line that
means "the essential requirements" (see REQUIREMENTS_HEADINGS) rather than
one fixed string, and stops capturing at the next section heading -
whatever that next heading is (see OTHER_SECTION_HEADINGS) - or end of
document.

Reuses cv_sections.clean() for ligature/control-glyph/NFKC cleanup - nothing
about that function is CV-specific, and a job listing extracted through the
same pdfplumber pipeline picks up the same PDF artifacts a CV does. It is
applied per finished requirement, after bullet detection, not to the raw
job text up front - see _join_body_lines for why the order matters.

Usage:
    from job_req_parser import extract_essential_requirements
    text = extract_essential_requirements("job_listing_secretary.pdf")
"""

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "common") not in sys.path:
    sys.path.insert(0, str(_ROOT / "common"))

from document_utils import extract_text_from_pdf  # noqa: E402
from cv_sections import clean  # noqa: E402


# --- Heading vocabulary ---
#
# Headings are matched as a whole normalised line, not a substring search
# over paragraph text: "3+ years' experience with client requirements" is
# body text, not a heading, and only a line that IS a heading (nothing else
# on it) should open or close a section.

# Headings that mean "the essential requirements section starts here".
REQUIREMENTS_HEADINGS = {
    "essential requirements",
    "essential criteria",
    "essential skills",
    "essential skills and experience",
    "essential experience",
    "key requirements",
    "minimum requirements",
    "required skills",
    "required skills and experience",
    "required experience",
    "requirements and qualifications",
    "candidate requirements",
    "job requirements",
    "requirements",
    "essential",  # Person Specification-style listings split into bare
                  # "Essential" / "Desirable" subheadings rather than one
                  # "Essential Requirements" heading.
    "what you'll need",
    "what you will need",
    "what we're looking for",
    "what we are looking for",
    "must haves",
    "must-haves",
}

# Headings that mean "a different section starts here" - i.e. wherever the
# requirements section ends. Includes REQUIREMENTS_HEADINGS too, since a
# second requirements-style heading further down (the "Desirable" half of
# a Person Specification split, or a listing that repeats itself) should
# also close the section rather than being swallowed into it.
OTHER_SECTION_HEADINGS = {
    "about us", "about the company", "about the role", "the role",
    "job summary", "role summary", "overview", "the opportunity",
    "key responsibilities", "responsibilities", "duties",
    "main duties", "main responsibilities",
    "desirable", "desirable criteria", "desirable skills",
    "nice to have", "nice-to-haves", "preferred qualifications",
    "preferred skills", "preferred experience",
    "what we offer", "benefits", "what you'll get", "package",
    "how to apply", "application process", "next steps",
    "closing date", "person specification",
}

_ALL_STOP_HEADINGS = REQUIREMENTS_HEADINGS | OTHER_SECTION_HEADINGS

_TRAILING_PUNCT = re.compile(r"[:\-–—]+$")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_STRAY_PUNCT = re.compile(r"[^\w\s&']+")
_WHITESPACE = re.compile(r"\s+")


def _normalise_heading(line: str) -> str:
    """Fold a heading line down to bare words for matching against the
    vocabulary sets above: case, trailing colons/dashes, parenthetical
    qualifiers ("(Not Essential)") and stray punctuation all vary between
    listings without changing what the heading means."""
    text = line.strip().lower()
    text = _TRAILING_PUNCT.sub("", text)
    text = _PARENTHETICAL.sub("", text)
    text = _STRAY_PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _is_requirements_heading(line: str) -> bool:
    return _normalise_heading(line) in REQUIREMENTS_HEADINGS


def _is_any_section_heading(line: str) -> bool:
    return _normalise_heading(line) in _ALL_STOP_HEADINGS


# A bullet glyph the source font has no Unicode mapping for extracts under
# pdfplumber as a literal "(cid:127)"-style placeholder rather than "*" or
# "-"; both forms (and the common Unicode bullet characters) are treated the
# same way here.
_BULLET_PREFIX = re.compile(r"^\s*(?:\(cid:\d+\)|[•●▪◦‣]|[-*])\s*")


def _join_body_lines(raw_lines: list) -> list:
    """Turn the section's raw lines into one cleaned string per requirement.

    pdfplumber emits one physical line per printed line, so a requirement
    that wrapped onto a second printed line ("...or a related life science
    (or equivalent" / "practical experience)") arrives as two consecutive
    entries here, the second with no bullet marker of its own. A line
    without a bullet marker is folded onto the previous requirement instead
    of becoming its own, so wrapped text reads as one requirement, not two.

    Bullet detection runs on the RAW line, before cv_sections.clean() ever
    sees it: clean() turns a "(cid:127)"-style bullet placeholder into a
    plain space, which would erase the exact marker _BULLET_PREFIX looks
    for and misread every real bullet as a wrapped continuation of the one
    before it. clean() is applied to the extracted CONTENT only, before the
    "- " marker this function adds is attached - not after - because
    clean()'s JUNK_GLYPHS currently strips every literal "-" it sees
    (see the TODO on JUNK_GLYPHS in cv_sections.py), which would otherwise
    eat the very prefix that makes this a bulleted line.
    """
    items = []
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        is_bullet = bool(_BULLET_PREFIX.match(raw))
        text = clean(_BULLET_PREFIX.sub("", raw, count=1).strip())
        if not text:
            continue
        if is_bullet or not items:
            items.append(f"- {text}")
        else:
            items[-1] = f"{items[-1]} {text}"
    return items


def extract_requirements_section(job_text: str):
    """
    Return just the essential-requirements section of an already-extracted
    job listing's text, or None if no heading in REQUIREMENTS_HEADINGS is
    found anywhere in it.
    """
    lines = job_text.splitlines()

    start = next((i for i, line in enumerate(lines) if _is_requirements_heading(line)), None)
    if start is None:
        return None

    end = next(
        (i for i in range(start + 1, len(lines)) if _is_any_section_heading(lines[i])),
        len(lines),
    )

    items = _join_body_lines(lines[start + 1:end])
    return "\n".join(items)


def extract_essential_requirements(pdf_path) -> str:
    """
    Read a job listing PDF and return just its essential-requirements
    section as plain text - one cleaned-up requirement per line.

    Falls back to the full extracted text (with a warning) if no
    recognised requirements heading is found, rather than raising: a
    listing this module can't segment is still better handed to the
    essential-requirements model whole than not evaluated at all.
    """
    job_text = extract_text_from_pdf(pdf_path)
    section = extract_requirements_section(job_text)
    if section is None:
        print(f"  WARN job_req_parser: no requirements heading found in {pdf_path} - "
              f"falling back to the full job listing text.", file=sys.stderr)
        return clean(job_text)
    return section


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Extract the essential-requirements section from a job listing PDF.")
    ap.add_argument("pdf", help="Path to the job listing PDF.")
    args = ap.parse_args()

    print(extract_essential_requirements(args.pdf))


if __name__ == "__main__":
    main()
