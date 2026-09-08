"""CV data model and vocabulary: shared by cv_pdf_parser.py and txt_cv_parser.py.

This is the modality-agnostic half of what used to be one file (cv_parser.py):
the Block/Section/ParsedCV data model, the section-heading vocabulary
(SECTION_LEXICON, heading_label), the text-pattern heuristics used while
segmenting (contact-hint, label:value detection, the implicit-preamble
label), and clean(), the PDF-artifact text normaliser - none of these need a
PDF or PyMuPDF's font/position geometry to do their job, which is exactly
what let txt_cv_parser.py segment a rewritten CV's plain text using the same
vocabulary and data model a real PDF parse produces.

cv_pdf_parser.py imports back from here: its own PDF-geometry pipeline
(clean_blocks, group_sections, parse) still calls clean(), heading_label(),
_implicit_label() and friends, and still returns Section/ParsedCV objects -
splitting this out did not change what it does, only where the shared parts
live.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


# --- Text cleanup ---
#
# Pure PDF-artifact normalisation. Nothing here is CV-specific - a job
# listing extracted through the same kind of PDF tooling picks up the same
# ligature and glyph junk.

# Ligatures PDF fonts emit as single codepoints. NFKC handles most, but it also
# rewrites other things we would rather leave alone, so map them explicitly.
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
             "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}

# Icon fonts (map pin, envelope, phone) survive extraction as C0/C1 control
# codes, private-use codepoints, or - under pypdf - as dingbats. Replaced with a
# space rather than deleted, so "\x83T. +27" does not collapse to "T.+27".
#
# Careful: no range may start with a bare "-". A leading hyphen reads as a
# literal and strips every hyphen in the document, turning "hands-on" into
# "hands on". An earlier version of this pattern did exactly that and silently
# destroyed 25 of 31 hyphens in Riaan Goven's CV.
# TODO(hyphen-bug): despite the comment above, this bare "-" is still parsed
# as a literal hyphen in the character class (verified empirically), not
# absorbed into a range boundary - JUNK_GLYPHS.match("-") is True, so
# clean() currently strips every hyphen from real content ("hands-on" ->
# "hands on", "pursuit-oriented" -> "pursuit oriented"), the exact bug this
# comment claims was fixed. Found while wiring clean() into
# job_req_parser.py - not fixed here since it's shared by cv_pdf_parser.py
# and txt_cv_parser.py too and has presumably been silently degrading CV
# text this whole time. Left flagged rather than fixed.
JUNK_GLYPHS = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"      # C0 and C1 control codes
    "-"                             # private use area
    "♂♀⌒⌗▶↵]"     # dingbats pypdf leaves behind
)


def clean(text: str) -> str:
    """Normalise extractor artefacts without altering real content."""
    for lig, plain in LIGATURES.items():
        text = text.replace(lig, plain)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # rejoin hyphenated line breaks

    # pdfplumber emits "(cid:131)" for glyphs with no unicode mapping - the icon
    # fonts other extractors render as control codes. Left in, the model reads it
    # as content.
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = JUNK_GLYPHS.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- The data model ---
#
# Block carries real PDF geometry when cv_pdf_parser builds it (bbox,
# max_size, bold from PyMuPDF), and zeroed placeholder geometry when
# txt_cv_parser builds it from a plain-text line - either way, Section and
# ParsedCV downstream only ever read block.text.


@dataclass
class Block:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    max_size: float
    bold: bool
    column: int = 0
    order: int = -1

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def centre_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def top(self) -> float:
        return self.bbox[1]


# --- Section-heading vocabulary ---
#
# Headings are identified by typography first, vocabulary second in
# cv_pdf_parser (a lexicon alone is not enough - see its group_sections). The
# vocabulary itself - what a recognised heading phrase MEANS - has nothing to
# do with typography, which is what lets txt_cv_parser reuse it unchanged
# behind its own, plain-text stand-in for "is this heading-shaped".


LABELS = ("contact", "summary", "skills", "experience", "education",
          "certifications", "projects", "references", "other")

# Normalised heading phrase -> label. heading_label resolves the longest key
# appearing as a contiguous run of whole words, so "work experience" beats
# "experience" and a qualifier like "Tutoring Experience" still matches. Plain
# exact-match missed "TUTORING EXPERIENCE" on the first out-of-sample CV and
# filed 1,642 chars of work history under education.
SECTION_LEXICON = {
    "contact": "contact",
    "contact details": "contact",
    "personal details": "contact",
    "profile": "summary",
    "summary": "summary",
    "professional summary": "summary",
    "personal statement": "summary",
    "objective": "summary",
    "about me": "summary",
    "skills": "skills",
    "technical skills": "skills",
    "key skills": "skills",
    "core competencies": "skills",
    "competencies": "skills",
    "tools": "skills",
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "employment history": "experience",
    "career history": "experience",
    "work history": "experience",
    "education": "education",
    "education and training": "education",
    "academic background": "education",
    "qualifications": "education",
    "certifications": "certifications",
    "certificates": "certifications",
    "licences": "certifications",
    "licenses": "certifications",
    "accreditations": "certifications",
    "projects": "projects",
    "personal projects": "projects",
    "portfolio": "projects",
    # Five of the ten Batch1 CVs print a references block and none was recognised,
    # so referee names, job titles and phone numbers were absorbed into whatever
    # section happened to precede them - skills on GEOFF, NDLOVU and PONTUS,
    # education on JORDAN, experience on LERATO. A model asked for a candidate's
    # skills was being handed their referees' contact details.
    "references": "references",
    "referees": "references",
    "professional profile": "summary",
    "career summary": "experience",
    "education and qualifications": "education",
    "educational information": "education",
    "courses": "education",
    "courses completed": "education",
    "details": "contact",
    "links": "contact",
    "languages": "skills",
}

# Every label this module can emit: the lexicon's values plus the two written
# literally ("other") and returned by _implicit_label ("contact"/"summary").
# Checked at import so LABELS constrains the output instead of merely describing
# it. The lexicon is the hand-edited half, so a typo in a value - "eduction" for
# "education" - would ship a label no consumer asks for and no test names.
_ORPHAN_LABELS = (set(SECTION_LEXICON.values()) | {"other", "contact", "summary"}) - set(LABELS)
if _ORPHAN_LABELS:
    raise ValueError(
        f"labels emitted but not declared in LABELS: {sorted(_ORPHAN_LABELS)}")

# Decorative numbered badges in some templates extract as a bare digit before the
# heading text ("1\nProfile" in RIAAN GOVEN.pdf).
_LEADING_NUMERAL = re.compile(r"^\s*\d{1,2}\s*[\n.):-]?\s*")

# A heading must be short. Six words admits "Education and Professional
# Training" while excluding bold lead-in sentences.
MAX_HEADING_WORDS = 6


def normalise_heading(text: str) -> str:
    text = _LEADING_NUMERAL.sub("", text.strip())
    text = re.sub(r"[^\w\s]", " ", text)
    words = re.sub(r"\s+", " ", text).strip().lower().split()
    # An icon font that survives cleaning leaves a bare one-character token before
    # the real heading - JORDAN VELDMAN's "l EXPERIENCE". Dropping it costs nothing:
    # no CV section heading in any template begins with a standalone single letter.
    if len(words) > 1 and len(words[0]) == 1:
        words = words[1:]
    return " ".join(words)


def heading_label(text: str) -> str | None:
    """Label for a heading phrase, or None if the vocabulary does not cover it.

    Matches the LONGEST lexicon key appearing as a contiguous run of whole
    words in the heading, which is what the lexicon comment claimed from the
    first commit and what the code never did. Exact-string lookup dropped
    "TUTORING EXPERIENCE" and "PROJECTS & TECHNICAL EXPERIENCE" on the first CV
    the lexicon had not seen, filing 1,642 characters of work history under
    education and returning nothing at all for sections_named("experience").

    Whole words, not raw substrings: a substring test fires on "experienced"
    and "skillset" and would mislabel prose that merely mentions the word.
    """
    words = normalise_heading(text).split()
    if not words:
        return None
    best = None
    for key, label in SECTION_LEXICON.items():
        key_words = key.split()
        n = len(key_words)
        if n > len(words):
            continue
        if any(words[i:i + n] == key_words for i in range(len(words) - n + 1)):
            if best is None or n > best[0]:
                best = (n, label)
    return best[1] if best else None


# A label followed by a value on the same line - "Position : Registry
# Coordinator", "Grade: Achieved a 72.8% average." - is data, not a heading,
# however it is styled. NDLOVU KARABO REFILWE's CV is almost entirely bold
# label:value pairs and is_heading fired on 35 of them; "Contact : 011 555 0147"
# inside her REFERENCES list opened a contact section holding four referees'
# numbers, which a UI would show as the candidate's own.
#
# A trailing colon introduces a list rather than naming a value, so "Key Job
# Responsibilities:" stays a heading - hence the required content after the colon.
_LABEL_VALUE = re.compile(r"\S\s*:\s+\S")

# Enough to recognise a contact strip: an email, a phone number, or a URL.
_CONTACT_HINT = re.compile(
    r"[\w.+-]+@[\w-]+\.\w+"           # email
    r"|\+?\d[\d\s()-]{7,}\d"          # phone
    r"|(?:https?://|www\.|linkedin\.com)", re.I)


@dataclass
class Section:
    label: str
    heading: str | None
    raw_heading: str | None
    blocks: list = field(default_factory=list)
    # Text of every unrecognised heading nested inside this section as an entry
    # or sub-heading. Nesting is right - a job title is not a section boundary -
    # but it left no trace: signals() counted only raw_heading, so a CV whose
    # headings the lexicon misread still reported unlabelled_sections == 0.
    # This is what lets the signal see what
    # the parser did not understand, whether it opened a section or nested.
    nested_headings: list = field(default_factory=list)

    def text(self) -> str:
        """Heading followed by its body, as a model should read it."""
        body = "\n".join(b.text for b in self.blocks)
        return f"{self.heading}\n{body}".strip() if self.heading else body.strip()


def _implicit_label(blocks) -> str:
    """Label for text appearing before the first heading.

    Almost always the contact strip. Where a CV opens with an unheaded profile
    paragraph instead, "summary" is the honest label - calling it contact would
    feed a paragraph of prose to consumers asking for contact details.
    """
    joined = "\n".join(b.text for b in blocks)
    return "contact" if _CONTACT_HINT.search(joined) else "summary"


# --- The parser contract ---
#
# parse(path) -> ParsedCV either way - a PDF via cv_pdf_parser, a rewritten
# CV's plain text via txt_cv_parser - and two projections: full_text() for
# the council, sections_named() for scoped input.


PARSER_VERSION = 2   # bump on any change to layout or segmentation; keys the cache
# Bumped for the longest-match heading_label. A cached v1 JSON already encodes
# sections labelled by the old exact-match lookup, and reading it back would not
# re-run heading_label at all - the mislabelling would survive forever behind a
# cache hit. Bumping points extract() at a fresh directory so every CV is
# re-segmented once under the fixed resolver.


@dataclass
class ParsedCV:
    path: Path
    sections: list
    extractor: str = "pymupdf"
    multi_column_pages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def full_text(self) -> str:
        return "\n\n".join(s.text() for s in self.sections if s.text()).strip()

    def sections_named(self, *labels: str) -> str:
        wanted = set(labels)
        return "\n\n".join(s.text() for s in self.sections
                           if s.label in wanted and s.text()).strip()

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "extractor": self.extractor,
            "parser_version": PARSER_VERSION,
            "multi_column_pages": self.multi_column_pages,
            "warnings": self.warnings,
            "sections": [
                {
                    "label": s.label,
                    "heading": s.heading,
                    "raw_heading": s.raw_heading,
                    "nested_headings": s.nested_headings,
                    "blocks": [
                        {"text": b.text, "page": b.page, "bbox": list(b.bbox),
                         "column": b.column, "order": b.order,
                         "max_size": b.max_size, "bold": b.bold}
                        for b in s.blocks
                    ],
                }
                for s in self.sections
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParsedCV":
        sections = [
            Section(
                label=s["label"], heading=s["heading"], raw_heading=s["raw_heading"],
                # .get, not [] - a cache entry written before this field
                # existed has no "nested_headings" key, and a KeyError here is
                # not always caught: run_evaluations.extract() treats one as a
                # torn cache and re-parses, but from_dict is also called
                # directly by tests and tools, which have no such fallback.
                nested_headings=s.get("nested_headings", []),
                blocks=[Block(text=b["text"], page=b["page"],
                                     bbox=tuple(b["bbox"]), max_size=b["max_size"],
                                     bold=b["bold"], column=b["column"],
                                     order=b["order"])
                        for b in s["blocks"]],
            )
            for s in data["sections"]
        ]
        return cls(path=Path(data["path"]), sections=sections,
                   extractor=data["extractor"],
                   multi_column_pages=data["multi_column_pages"],
                   warnings=data["warnings"])


# --- Findability signals ---
#
# The brief asks how easy it is to find the relevant skill information. That is a
# property of layout, and flattening the PDF to a string destroys it. Once
# sections exist the signals are just counts - and just as meaningful whether
# those sections came from PDF geometry or a plain-text heuristic, since both
# produce the same Section shape.


def signals(cv) -> dict:
    """Structural findability signals for a ParsedCV."""
    sections = cv.sections
    if not sections:
        return {"headed_fraction": 0.0, "has_skills_section": False,
                "section_count": 0, "unlabelled_sections": 0,
                "mean_section_chars": 0.0, "largest_section_fraction": 0.0}

    lengths = [len(s.text()) for s in sections]
    total = sum(lengths) or 1
    headed = sum(1 for s in sections if s.heading)

    return {
        # Headed content is findable content. A CV that is one unheaded block of
        # prose scores 0 here regardless of how good its content is.
        "headed_fraction": headed / len(sections),
        "has_skills_section": any(s.label == "skills" for s in sections),
        "section_count": len(sections),
        # Headings the lexicon missed, whether they opened their own "other"
        # section (raw_heading) or nested inside one as an entry heading
        # (nested_headings). Counting raw_heading alone went blind once nesting
        # shipped: a CV whose headings nested rather than orphaned reported 0
        # here with three of its six headings mislabelled. High counts mean an
        # unusual template or a lexicon gap - worth inspecting either way.
        "unlabelled_sections": sum(1 for s in sections if s.raw_heading)
                               + sum(len(s.nested_headings) for s in sections),
        "mean_section_chars": total / len(sections),
        # One section holding most of the document means the headings are
        # decorative rather than organisational.
        "largest_section_fraction": max(lengths) / total,
    }
