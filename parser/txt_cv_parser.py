"""txt_cv_parser.py - plain-text CV segmenter, for rewritten CVs with no PDF to parse.

cv_pdf_parser.py's segmentation is "typography first, vocabulary second": a
block is only a heading candidate if it's short AND set apart by bold/size,
and only becomes a SECTION boundary if that heading text also resolves
through SECTION_LEXICON. A rewritten CV (rewriter_v2.py's .txt output) has no
PDF to read bold/size from, so there is no typography signal to check first.

This module replaces ONLY that one PDF-dependent decision - "is this line
heading-shaped" - with a plain-text heuristic (short, standalone, not
bulleted, not a label:value pair, doesn't trail off like a sentence).
Everything downstream of that decision is imported directly from
cv_sections.py rather than re-implemented: heading_label() for lexicon
resolution, Section/Block/ParsedCV for the data model, and the same
first-section contact-hint upgrade. cv_sections.py is the modality-agnostic
half of what used to be one file (cv_parser.py, now split into cv_sections.py
plus the PDF-only cv_pdf_parser.py) - if its lexicon or that upgrade logic
changes, this file picks the change up automatically rather than silently
drifting out of sync with a duplicated copy.

Because only a LEXICON-MATCHED heading actually opens a new section (an
unrecognised heading nests into whichever section is already open, exactly
as it does in cv_parser), an over-permissive "heading-shaped" heuristic is
low-risk here: a false positive that isn't in the lexicon just nests instead
of corrupting anything, and a missed real heading just falls into the
previous section's body rather than being lost.

    import txt_cv_parser
    cv = txt_cv_parser.parse("Batch1_rewrites/ALEX PARK/phi4.txt")
    cv.full_text()
    cv.sections_named("experience", "education")

Validated against exactly one sample so far (rewriter_v2.py's phi4 output
for ALEX PARK) - see the __main__ block. Whether qwen2.5:14b's rewrites
format headings the same way (capitalisation, spacing, no trailing
punctuation) across many candidates is untested; run this against a
handful of real samples from both agents before trusting it the way
cv_pdf_parser is trusted (cv_pdf_parser earned that via
tests/fixtures/layout_labels.json, not assumption).
"""

from pathlib import Path

from cv_sections import (
    heading_label,
    normalise_heading,
    _implicit_label,
    _CONTACT_HINT,
    _LABEL_VALUE,
    Section,
    Block,
    ParsedCV,
    MAX_HEADING_WORDS,
)

# Bullet markers a rewriter is likely to use. A bulleted line is body text,
# never a heading, in every template this project has seen.
_BULLET_PREFIXES = ("•", "-", "*", "\u2013", "\u2014")

# A heading is a standalone label, not the start or end of a sentence -
# "Summary." or "responsibilities," never head a section in practice.
_TRAILING_SENTENCE_PUNCT = (".", ",", ";")


def is_heading_plain(line: str) -> bool:
    """Plain-text stand-in for cv_pdf_parser.is_heading(), which needs font
    size/weight this input doesn't have.

    Short, standalone, not a bullet, not "label: value", doesn't trail off
    like a sentence. Deliberately permissive rather than strict - see the
    module docstring for why an over-eager match here is low-risk.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(_BULLET_PREFIXES):
        return False
    if _LABEL_VALUE.search(stripped):
        return False
    if stripped[-1] in _TRAILING_SENTENCE_PUNCT:
        return False
    words = normalise_heading(stripped).split()
    if not words or len(words) > MAX_HEADING_WORDS:
        return False
    return True


def segment_text(text: str) -> list:
    """Plain-text counterpart to cv_pdf_parser.group_sections() - same
    lexicon-gated boundary logic, single-stream instead of column-aware
    (a rewritten CV has no columns to track).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    blocks = [
        Block(text=ln.strip(), page=0, bbox=(0.0, 0.0, 0.0, 0.0),
              max_size=0.0, bold=False, column=0, order=i)
        for i, ln in enumerate(lines)
    ]

    sections = []
    preamble = []
    current = None

    for block in blocks:
        if is_heading_plain(block.text):
            raw = block.text
            label = heading_label(raw)
            if label is not None:
                section = Section(label=label, heading=raw, raw_heading=None)
            else:
                if current is not None:
                    current.nested_headings.append(raw)
                    current.blocks.append(block)
                    continue
                section = Section(label="other", heading=raw, raw_heading=raw)
            sections.append(section)
            current = section
            continue

        if current is not None:
            current.blocks.append(block)
        else:
            preamble.append(block)

    if preamble:
        sections.insert(0, Section(label=_implicit_label(preamble),
                                    heading=None, raw_heading=None,
                                    blocks=preamble))

    # Same upgrade cv_pdf_parser applies: an opening banner (name/title) is
    # short enough to be heading-shaped but the lexicon misses it, so it
    # opens an "other" section - promote it to "contact" if it actually
    # carries an email, phone, or profile URL.
    if sections and sections[0].label == "other" and _CONTACT_HINT.search(sections[0].text()):
        sections[0].label = "contact"
        sections[0].raw_heading = None

    return sections


def parse(path) -> ParsedCV:
    """Rewritten-CV .txt file -> ParsedCV, the same class cv_pdf_parser.parse()
    returns. extractor is recorded as "plaintext" so anything reading this
    field downstream (e.g. a UI) can tell a rewrite's parse apart from an
    original PDF's.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return ParsedCV(path=path, sections=segment_text(text), extractor="plaintext")


if __name__ == "__main__":
    import sys
    cv = parse(sys.argv[1])
    print("=== labels found, in order ===")
    print([s.label for s in cv.sections])
    print("\n=== nested (sub-)headings per section ===")
    for s in cv.sections:
        if s.nested_headings:
            print(f"  {s.label}: {s.nested_headings}")
    print("\n=== sections_named('experience', 'education') ===")
    print(cv.sections_named("experience", "education"))
    print("\n=== full_text() length vs original ===")
    original_len = len(open(sys.argv[1], encoding="utf-8").read())
    print(f"original: {original_len} chars | full_text(): {len(cv.full_text())} chars")
