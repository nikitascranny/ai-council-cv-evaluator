"""Structure-aware CV parser: PDF in, labelled sections out.

    import cv_parser
    cv = cv_parser.parse("SOME CV.pdf")   # -> ParsedCV
    cv.full_text()                        # whole CV in reading order
    cv.sections_named("experience", "education")
    cv_parser.signals(cv)                 # structural findability signals

From the shell:  python parser/cv_parser.py "SOME CV.pdf"  -> JSON on stdout.

Five stages, in dependency order, one per section header below:

  1. Text extraction   pdf -> normalised text, with the extractor recorded
  2. Page layout       pdf page -> Blocks in reading order, columns resolved
  3. Segmentation      Blocks -> labelled Sections, typography before vocabulary
  4. Parser contract   parse() and ParsedCV - all a consumer needs
  5. Findability       structural signals over a ParsedCV

Oversized documents are not chunked: at num_ctx=8192 the largest Batch1 prompt is
~7.6k tokens, so a CV too long to fit is passed whole and truncated by the model.

One file rather than five, because two other people import this and five modules
sharing a sys.path hack is a setup problem rather than an import.

Requires PyMuPDF (fitz); pypdf and pdfplumber are imported only when requested.
"""

import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fitz


# --- 1. Text extraction ---
#
# The extractor is a controlled variable, recorded on every vote beside the model
# name: the same CV read by two extractors differs in word fusion, header junk
# and - on two-column CVs - section order, any of which can change what a model
# concludes. Measured on Batch1, pymupdf fuses 95 words across the ten CVs
# against pypdf's 145, because pypdf reads two-column layouts line by line and
# renders icon fonts as dingbats that fuse onto the next word.
#
# Neither fully solves multi-column reading order - both misplace Riaan Goven's
# Education block, which is what stage 2 exists to fix. Both run through the same
# clean() pass, so the extractor changes extraction, not normalisation.


EXTRACTORS = ("pypdf", "pymupdf", "pdfplumber")

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
JUNK_GLYPHS = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"      # C0 and C1 control codes
    "-"                             # private use area
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


def _pypdf(path) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)


def _pymupdf(path) -> str:
    import fitz
    doc = fitz.open(path)
    out = []
    for page in doc:
        # sort=True orders blocks by position, keeping a sidebar intact rather
        # than interleaving it line-by-line with the body column.
        out.extend(b[4] for b in page.get_text("blocks", sort=True))
    return "\n".join(out)


# pdfplumber's default x_tolerance of 3 is far too loose for these fonts: it
# merges adjacent words, turning "Head of Digital Product" into
# "HeadofDigitalProduct". Measured on Batch1, the default yields 609 fused words
# and only 5,743 recognisable word-tokens, against ~95 and ~8,700 for every other
# extractor tested - a third of each CV stops being readable as words. No content
# is lost; the spaces are. Tightening the tolerance recovers essentially all of it.
PDFPLUMBER_X_TOLERANCE = 1.5


def _pdfplumber(path) -> str:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n".join(pg.extract_text(x_tolerance=PDFPLUMBER_X_TOLERANCE) or ""
                         for pg in pdf.pages)


_BACKENDS = {"pypdf": _pypdf, "pymupdf": _pymupdf, "pdfplumber": _pdfplumber}


def extract(path, extractor: str = "pymupdf") -> str:
    if extractor not in EXTRACTORS:
        raise ValueError(f"unknown extractor {extractor!r}, expected one of {EXTRACTORS}")
    return clean(_BACKENDS[extractor](path))


# --- 2. Page layout and reading order ---
#
# PyMuPDF's get_text("blocks", sort=True) sorts in horizontal bands, so on a
# two-column CV it emits a main-column block then jumps to a sidebar block higher
# up the page. On RIAAN GOVEN.pdf page 0 that puts the heading "Work Experience"
# (y=296) immediately before a sidebar block beginning "Education | Jan 2024 -
# Current | Masters in Digital Business" (y=127): his entire education history
# lands under Work Experience, and every model reading it takes his degrees for
# jobs.
#
# So this stage never uses sort=True - it reads blocks with their geometry, finds
# the column gutter, and emits each column in full before the next. Font size and
# weight come along because get_text("dict") yields them alongside geometry, and
# stage 3 needs both to tell a heading from body text.


# PyMuPDF span flag bit 4 marks bold. Documented in the "dict" output schema.
_BOLD_FLAG = 1 << 4

# Blocks at least this fraction of page width span the columns rather than sitting
# in one - banners, name headers, contact strips. They confound gutter detection,
# so they are set aside and re-inserted as band separators.
FULL_WIDTH_RATIO = 0.6


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


def raw_blocks(page, page_index: int) -> list[Block]:
    """Every text block on the page, in PyMuPDF's native order.

    Ordering and column assignment happen in ordered_blocks(); this function only
    reads. Image blocks (type != 0) are skipped - they carry no text, and a scanned
    CV that consists solely of them is reported as a warning further up the stack.
    """
    out = []
    for raw in page.get_text("dict")["blocks"]:
        if raw.get("type") != 0:
            continue
        lines = raw.get("lines", [])
        spans = [s for line in lines for s in line.get("spans", [])]
        if not spans:
            continue
        text = "\n".join(
            "".join(s["text"] for s in line.get("spans", [])) for line in lines
        )
        if not text.strip():
            continue
        out.append(Block(
            text=text,
            page=page_index,
            bbox=tuple(raw["bbox"]),
            max_size=max(s.get("size", 0.0) for s in spans),
            bold=any(s.get("flags", 0) & _BOLD_FLAG for s in spans),
        ))
    return out


# A gutter must be this wide to count. Below it, ordinary inter-word or
# inter-cell whitespace qualifies and single-column pages get split.
#
# 15.0 was the starting value and measured too strict: THE DANIEL page 0 puts a
# SKILLS column at x=36.2-190.7 beside an EDUCATION column at x=203.2-564.1 - a
# ~12pt gutter a 15pt floor misses outright. Swept against
# tests/fixtures/layout_labels.json with tools/tune_gutter.py over widths 8-30pt
# and side shares 0.08-0.20: no setting gives a false positive, and <=12 paired
# with MIN_SIDE_SHARE <= 0.10 finds all 5 labelled multi-column pages out of 32.
# 12.0/0.10 ties 8.0/0.08 on precision and recall but leaves the most margin
# before an unseen CV's ordinary whitespace reads as a gutter.
MIN_GUTTER_PT = 12.0

# Each side must hold at least this share of the page's text. Without this test
# the right margin of a single-column page reads as a gutter, since no block
# crosses it - the detector then invents a second column containing nothing.
MIN_SIDE_SHARE = 0.10

# Only look for a gutter in the middle of the page. A "gutter" at x=0.9W is a
# margin; one at x=0.1W is an indent.
SEARCH_LO, SEARCH_HI = 0.25, 0.75


def detect_gutter(blocks: list[Block], page_width: float) -> tuple[float, float] | None:
    """Find the vertical band separating two text columns, or None.

    Full-width blocks are excluded before searching: a contact strip spanning
    both columns crosses the gutter and would hide it. On RIAAN GOVEN page 0 the
    contact block runs x=54.5-462.5 and does exactly that, which is why a naive
    whole-page scan reports single-column for a page that plainly is not.

    Full-width exclusion also protects ALEX PARK's career summary, where each
    row's date and job description are merged by PyMuPDF into one block
    spanning x0=53.9 x1=541.5. That layout reads as a two-column table across
    each row, not two independent columns read down; ordering it column-wise
    would emit every date and then every description, detaching each date from
    its role. Because the merged block is wide, it never reaches this search in
    the first place - see test_row_paired_date_column_is_not_a_gutter.
    """
    narrow = [b for b in blocks if b.width < FULL_WIDTH_RATIO * page_width]
    total = sum(len(b.text) for b in narrow)
    if total < 200:            # too little text to judge
        return None

    best = None
    x = SEARCH_LO * page_width
    hi = SEARCH_HI * page_width
    while x < hi:
        if any(b.bbox[0] < x < b.bbox[2] for b in narrow):
            x += 1.0
            continue
        right = x
        while right < hi + 0.03 * page_width and \
                not any(b.bbox[0] < right < b.bbox[2] for b in narrow):
            right += 1.0
        centre = (x + right) / 2
        left_chars = sum(len(b.text) for b in narrow if b.centre_x < centre)
        right_chars = total - left_chars
        wide_enough = right - x >= MIN_GUTTER_PT
        balanced = min(left_chars, right_chars) >= MIN_SIDE_SHARE * total
        if wide_enough and balanced and (best is None or right - x > best[1] - best[0]):
            best = (x, right)
        x = right + 1.0
    return best


def page_is_multi_column(page) -> bool:
    """Whether this page has a detectable column gutter.

    Delegates entirely to detect_gutter, which already measures at precision
    1.00 / recall 1.00 against tests/fixtures/layout_labels.json (see
    task-3-report.md). page_index is fixed at 0 here because the caller only
    needs a yes/no signal to pick a rendering strategy, not the geometry
    itself - ordered_blocks() re-derives blocks for the real page_index.
    """
    blocks = raw_blocks(page, 0)
    return detect_gutter(blocks, page.rect.width) is not None


def ordered_blocks(page, page_index: int) -> list[Block]:
    """Blocks in the order a human reads them.

    Single-column pages sort straight down by (top, left). Two-column pages read
    the left column in full, then the right.

    Full-width blocks are band separators rather than column members. A name or
    contact strip spans both columns, so assigning it to one would drop it into
    the middle of that column's text; instead it splits the page into horizontal
    bands, and column ordering runs within each band. Blocks are banded by their
    top edge, which is stable even where bounding boxes overlap slightly.

    A block only counts as a band separator if it straddles the detected gutter
    - its left edge sits left of the gutter and its right edge sits right of it
    - not merely because it is wide in absolute terms. FULL_WIDTH_RATIO (0.6) is
    tuned for detect_gutter's own search over the *whole page*, where excluding
    banner-width blocks keeps a contact strip from hiding the gutter; it is not
    a safe test here, once the real column boundaries are known. Measured on
    JORDAN VELDMAN page 0: his sidebar sits on the left (x=0-187) and the main
    column alone is ~63% of the 595pt page, so ordinary line-wrapped bullets
    inside it (widths up to 379pt) exceed the 0.6*page_width threshold without
    ever crossing the gutter (x=203.8-215.8). An earlier version of this function
    used the width-ratio test here too, which misclassified those bullets as
    page-spanning separators, fragmented the page into dozens of spurious bands,
    and interleaved his sidebar contact details into the middle of the
    Experience section - reproducing, on a different CV, the exact defect this
    module exists to fix. See test_jordan_sidebar_and_main_column_are_contiguous.
    """
    blocks = raw_blocks(page, page_index)
    width = page.rect.width
    gutter = detect_gutter(blocks, width)
    if gutter is None:
        out = sorted(blocks, key=lambda b: (b.top, b.bbox[0]))
        for i, b in enumerate(out):
            b.column, b.order = 0, i
        return out

    def straddles_gutter(b: Block) -> bool:
        return b.bbox[0] < gutter[0] and b.bbox[2] > gutter[1]

    narrow = [b for b in blocks if not straddles_gutter(b)]
    wide = sorted((b for b in blocks if straddles_gutter(b)), key=lambda b: b.top)

    centre = (gutter[0] + gutter[1]) / 2
    out: list[Block] = []
    boundaries = [b.top for b in wide] + [float("inf")]
    lower = float("-inf")
    for i, upper in enumerate(boundaries):
        band = [b for b in narrow if lower <= b.top < upper]
        left = sorted((b for b in band if b.centre_x < centre), key=lambda b: b.top)
        right = sorted((b for b in band if b.centre_x >= centre), key=lambda b: b.top)
        for b in left:
            b.column = 0
        for b in right:
            b.column = 1
        out.extend(left)
        out.extend(right)
        if i < len(wide):
            wide[i].column = 0
            out.append(wide[i])
            lower = wide[i].top

    for i, b in enumerate(out):
        b.order = i
    return out


def clean_blocks(page, page_index: int) -> list[Block]:
    """Ordered blocks with clean() applied to each block separately.

    Per block, not per page. clean() rejoins hyphenated line breaks with
    (\\w)-\\n(\\w) -> \\1\\2; run over concatenated columns that fuses the last word
    of one column onto the first word of the next. Blocks that clean to nothing -
    icon-only decorations - are dropped, and the surviving blocks are renumbered
    so order stays contiguous.
    """
    kept = []
    for block in ordered_blocks(page, page_index):
        block.text = clean(block.text)
        if block.text:
            kept.append(block)
    for i, block in enumerate(kept):
        block.order = i
    return kept


# --- 3. Section segmentation ---
#
# Headings are identified by typography first, vocabulary second. A lexicon alone
# is not enough: RIAAN GOVEN.pdf heads a subsection "DEVELOPMENT", which no CV
# lexicon contains but which is short, bold and above body size.
#
# Only section headings ("Work Experience") are boundaries - entry headings
# ("Senior UX Researcher, Umoja Bank") and sub-headings ("Key Responsibilities")
# are typographically similar but must not split the section. So a heading opens
# a new section only when the lexicon recognises it; anything else stays inside
# as a block. Treating every typographic heading as a boundary stranded 4,487
# chars of a Lerato Munche job entry outside Work Experience.
#
# Nothing is discarded: an unrecognised heading before any section opens an
# "other" section with raw_heading kept verbatim, so full_text() stays complete.



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

# Font must exceed body size by this factor to count as a heading on size alone.
HEADING_SIZE_RATIO = 1.15


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


def body_size(blocks) -> float:
    """Median font size across blocks - the body-text baseline.

    Median rather than mean: a CV with one very large name banner would drag a
    mean above the actual body size and hide every real heading.
    """
    sizes = [b.max_size for b in blocks if b.max_size > 0]
    return statistics.median(sizes) if sizes else 0.0


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


def is_heading(block, body_size: float) -> bool:
    """Short, set apart from body text by size or weight, and not label:value."""
    words = normalise_heading(block.text).split()
    if not words or len(words) > MAX_HEADING_WORDS:
        return False
    if _LABEL_VALUE.search(block.text.split("\n", 1)[0]):
        return False
    bigger = body_size > 0 and block.max_size >= body_size * HEADING_SIZE_RATIO
    return bool(bigger or block.bold)


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


def _sidebar_columns(blocks) -> dict[int, int]:
    """Page number -> the column index holding that page's minority of text.

    A two-column page's narrow column (contact details, skills, tools) is a
    sidebar; the wide column is the main narrative. Column 0 is not always the
    sidebar - on JORDAN VELDMAN's page 0 the sidebar is column 0 (left) while
    on LERATO MUNCHE's it is column 1 (right) - so this measures characters
    per (page, column) instead of assuming a side. A single-column page has no
    second column to compare against, so it is never a sidebar page: every
    block on it is main-column text, matching the CVs measured here (LERATO
    pages 1-6, GEOFF page 1, RIAAN page 1, JORDAN page 1, THE DANIEL's only
    page has two).
    """
    totals: dict[tuple[int, int], int] = {}
    for b in blocks:
        key = (b.page, b.column)
        totals[key] = totals.get(key, 0) + len(b.text)
    by_page: dict[int, dict[int, int]] = {}
    for (page, col), n in totals.items():
        by_page.setdefault(page, {})[col] = n
    sidebar: dict[int, int] = {}
    for page, cols in by_page.items():
        if len(cols) < 2:
            continue
        lo_col, lo_n = min(cols.items(), key=lambda kv: kv[1])
        hi_n = max(n for c, n in cols.items() if c != lo_col)
        if lo_n < hi_n:
            sidebar[page] = lo_col
    return sidebar


def group_sections(blocks) -> list[Section]:
    """Split ordered blocks into labelled sections at heading boundaries.

    Every block lands in exactly one section. Blocks before the first heading
    open an implicit section rather than being discarded, which is what keeps
    ParsedCV.full_text() complete on templates this lexicon has never seen.

    A sidebar column is read in full before the main column resumes (see
    ordered_blocks, stage 2), and a sidebar is inherently page-local - it has no
    relationship to the next page's main-column narrative. Tracking one
    "current section" for the whole document, as an earlier version of this
    function did, let a section opened inside a sidebar stay current across
    the page boundary and swallow the next page's main column. Measured on
    LERATO MUNCHE: her page-0 sidebar heading "Languages" absorbed 14,651
    chars of pages 1-4's Employment History, because those pages are
    single-column and their text was simply appended to whatever section was
    still open. The fix keeps two independent "current section" pointers, one
    per column kind (main vs sidebar, decided per page by _sidebar_columns),
    so a main-column block always resumes the last section a main-column block
    opened - never a dangling sidebar section - while a sidebar block that
    lands on a new page finds its own sidebar pointer reset, because the prior
    page's sidebar has no bearing on this one.
    """
    if not blocks:
        return []

    baseline = body_size(blocks)
    sidebar_col = _sidebar_columns(blocks)
    sections: list[Section] = []
    preamble: list = []
    current: dict[str, Section | None] = {"main": None, "sidebar": None}
    sidebar_page: int | None = None

    for block in blocks:
        kind = "sidebar" if sidebar_col.get(block.page) == block.column else "main"
        if kind == "sidebar" and current["sidebar"] is not None and block.page != sidebar_page:
            # Defensive and untested against real data: instrumented across
            # all 10 Batch1 CVs this fires zero times, because every sidebar in
            # the corpus is page 0 only. The rule is still sound - a sidebar
            # left open on a previous page has no relationship to this page's -
            # and a sidebar running down several pages is an ordinary template,
            # so it stays. The synthetic
            # test_sidebar_section_resets_at_each_page_boundary is the only
            # thing exercising it.
            current["sidebar"] = None

        if is_heading(block, baseline):
            raw = block.text.strip()
            label = heading_label(raw)
            if label is not None:
                section = Section(label=label, heading=raw, raw_heading=None)
            else:
                # An unrecognised heading is an entry or sub-heading, not a
                # section boundary, so it stays inside the section this column
                # currently has open, kept as a block so its text survives in
                # full_text(). With no section of this kind open there is
                # nothing to nest inside, so it opens an "other" section.
                target = current[kind]
                if target is not None:
                    target.nested_headings.append(raw)
                    target.blocks.append(block)
                    continue
                section = Section(label="other", heading=raw, raw_heading=raw)
            sections.append(section)
            current[kind] = section
            if kind == "sidebar":
                sidebar_page = block.page
            continue

        target = current[kind]
        if target is not None:
            target.blocks.append(block)
        elif sections:
            # The other column kind already has a section open, but this one
            # doesn't - open its own "other" bucket rather than falling back
            # to the other kind's section, which is exactly the bleed this
            # function exists to prevent.
            section = Section(label="other", heading=None, raw_heading=None)
            section.blocks.append(block)
            sections.append(section)
            current[kind] = section
            if kind == "sidebar":
                sidebar_page = block.page
        else:
            preamble.append(block)

    if preamble:
        sections.insert(0, Section(label=_implicit_label(preamble),
                                   heading=None, raw_heading=None,
                                   blocks=preamble))

    # A CV's opening banner is the candidate's name and title, short enough that
    # is_heading fires while the lexicon misses it, so it opens an "other"
    # section. On 6 of the 10 Batch1 CVs that left the candidate's own email
    # under "other" and sections_named("contact") returning nothing.
    #
    # Only the first section is reconsidered, and only if its text carries an
    # email, phone or profile URL - a contact-shaped block later in the document
    # (a referees list, typically) belongs to whatever section it sits in.
    if sections and sections[0].label == "other" and _CONTACT_HINT.search(sections[0].text()):
        sections[0].label = "contact"
        sections[0].raw_heading = None

    return sections


# --- 4. The parser contract ---
#
# Everything above is implementation detail. Consumers get parse(path) -> ParsedCV
# and two projections: full_text() for the council, sections_named() for scoped
# input. bbox and per-section blocks are carried although nothing reads them yet -
# the UI needs to point at the region a judgement came from, and changing the
# schema later would break every consumer at once.


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


def parse(path, extractor: str = "pymupdf") -> ParsedCV:
    """PDF -> ParsedCV.

    extractor is accepted and recorded so votes stay labelled with the text they
    were produced from, but structured parsing needs per-block geometry and only
    pymupdf supplies it. Any other value raises rather than silently returning
    text from a different pipeline than the label claims.
    """
    if extractor != "pymupdf":
        raise ValueError(
            f"structured parsing requires pymupdf geometry, got {extractor!r}; "
            f"use extract() for flat text from other extractors"
        )

    path = Path(path)
    doc = fitz.open(path)
    blocks, multi_column, warnings = [], [], []

    for i, page in enumerate(doc):
        page_blocks = clean_blocks(page, i)
        if not page_blocks:
            warnings.append(f"page {i} yielded no text (image-only or scanned?)")
            continue
        if page_is_multi_column(page):
            multi_column.append(i)
        blocks.extend(page_blocks)

    for i, block in enumerate(blocks):
        block.order = i

    if not blocks:
        warnings.append("no text extracted from any page - OCR is not supported")

    return ParsedCV(path=path, sections=group_sections(blocks),
                    extractor=extractor, multi_column_pages=multi_column,
                    warnings=warnings)


# --- 5. Findability signals ---
#
# The brief asks how easy it is to find the relevant skill information. That is a
# property of layout, and flattening the PDF to a string destroys it. Once
# sections exist the signals are just counts. They are inputs for the judge, not
# a score - turning them into one is the judge's call, not this stage's.


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


# --- CLI ---

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Parse a CV PDF into labelled sections.")
    ap.add_argument("pdf", help="path to a CV PDF")
    ap.add_argument("--signals", action="store_true",
                    help="print findability signals instead of the full parse")
    args = ap.parse_args()

    parsed = parse(args.pdf)
    print(json.dumps(signals(parsed) if args.signals else parsed.to_dict(),
                     indent=2, ensure_ascii=False))
