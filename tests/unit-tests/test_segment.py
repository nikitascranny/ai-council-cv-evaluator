"""Unit tests for section segmentation."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "parser"))

import cv_pdf_parser  # noqa: E402
import cv_sections  # noqa: E402
from cv_sections import Block  # noqa: E402
from fixtures.out_of_sample_headings import OUT_OF_SAMPLE_HEADINGS  # noqa: E402


def block(text, size=10.0, bold=False, top=0.0, page=0, column=0):
    return Block(text=text, page=page, bbox=(0.0, top, 200.0, top + 12.0),
                 max_size=size, bold=bold, column=column)


# --- heading_label --------------------------------------------------------

def test_known_headings_map_to_labels():
    assert cv_sections.heading_label("Work Experience") == "experience"
    assert cv_sections.heading_label("Education") == "education"
    assert cv_sections.heading_label("Certifications") == "certifications"
    assert cv_sections.heading_label("Profile") == "summary"
    assert cv_sections.heading_label("Skills") == "skills"


def test_heading_matching_is_case_and_punctuation_insensitive():
    assert cv_sections.heading_label("EDUCATION") == "education"
    assert cv_sections.heading_label("  education:  ") == "education"


def test_decorative_numerals_are_stripped():
    """Riaan's template prints a numbered badge before each heading, which
    extracts as a leading digit on its own line: "1\\nProfile".

    The TWO-DIGIT cases are what actually pin _LEADING_NUMERAL. Deleting the
    pattern entirely still passes the single-digit assertions, because Task 7b's
    single-character-glyph strip in normalise_heading (added for JORDAN
    VELDMAN's "l EXPERIENCE") drops a lone "1" for an unrelated reason and
    accidentally covers them. A badge of "12" is two characters, so that strip
    leaves it alone and only _LEADING_NUMERAL can remove it - which is why the
    pattern is written \\d{1,2} rather than \\d.
    """
    assert cv_sections.heading_label("1\nProfile") == "summary"
    assert cv_sections.heading_label("2\nWork Experience") == "experience"
    assert cv_sections.heading_label("12\nProfile") == "summary"
    assert cv_sections.heading_label("10\nWork Experience") == "experience"


def test_synonyms_collapse():
    assert cv_sections.heading_label("Employment History") == "experience"
    assert cv_sections.heading_label("Professional Experience") == "experience"
    assert cv_sections.heading_label("Technical Skills") == "skills"
    assert cv_sections.heading_label("Summary") == "summary"


def test_unknown_heading_returns_none():
    assert cv_sections.heading_label("Hobbies and Interests") is None


def test_lexicon_matching_is_whole_word_not_raw_substring():
    """Pins the mechanism heading_label actually uses (Task 14).

    This test used to pin the opposite - that a phrase merely CONTAINING a
    lexicon key, like "senior work experience role", was NOT recognised. That
    was the exact defect Task 14 fixes: exact-string lookup also missed
    "TUTORING EXPERIENCE", a heading that plainly names its own section, and
    filed 1,642 chars of real work history under education. Longest-match
    correctly resolves a heading that contains a key's words as a contiguous
    run, qualifier or not; what it must still refuse is a raw substring like
    "experienced" firing on "experience" (see test_word_runs_not_raw_substrings).
    """
    assert cv_sections.heading_label("senior work experience role") == "experience"
    assert cv_sections.heading_label("experience") == "experience"
    assert cv_sections.heading_label("work experience") == "experience"


def test_every_emitted_label_is_declared_in_labels():
    """LABELS is the contract ParsedCV publishes, so it must constrain the
    module rather than merely describe it (stage 3 enforces this at import).

    The lexicon is hand-edited - it was widened nine times in one task - so a
    typo in a value would otherwise ship a label no consumer asks for.
    """
    emitted = set(cv_sections.SECTION_LEXICON.values()) | {"other", "contact", "summary"}
    assert emitted <= set(cv_sections.LABELS), \
        f"labels outside LABELS: {sorted(emitted - set(cv_sections.LABELS))}"


# --- is_heading -----------------------------------------------------------

def test_large_short_text_is_a_heading():
    assert cv_pdf_parser.is_heading(block("Work Experience", size=14.0), body_size=10.0)


def test_bold_short_text_is_a_heading():
    assert cv_pdf_parser.is_heading(block("USER EXPERIENCE", size=10.0, bold=True), body_size=10.0)


def test_long_text_is_not_a_heading_however_styled():
    """A bold lead-in sentence is not a section heading."""
    long_text = "Responsible for the user-experience research strategy across all digital channels"
    assert not cv_pdf_parser.is_heading(block(long_text, size=14.0, bold=True), body_size=10.0)


def test_body_text_is_not_a_heading():
    assert not cv_pdf_parser.is_heading(block("Miro", size=10.0), body_size=10.0)


def test_body_size_is_the_median():
    blocks = [block("a", size=10.0), block("b", size=10.0),
              block("c", size=10.0), block("Heading", size=20.0)]
    assert cv_pdf_parser.body_size(blocks) == 10.0


# --- group_sections -------------------------------------------------------

def test_blocks_group_under_their_heading():
    blocks = [block("Work Experience", size=14.0),
              block("Feb 2022 - Current: 42th Rand Bank, Head of Web Product"),
              block("Education", size=14.0),
              block("BSc Computer Science, 2018")]
    sections = cv_pdf_parser.group_sections(blocks)
    labels = [s.label for s in sections]
    assert labels == ["experience", "education"]
    assert "Rand Bank" in sections[0].text()
    assert "BSc" in sections[1].text()


def test_unknown_heading_becomes_other_and_keeps_its_text():
    blocks = [block("Hobbies and Interests", size=14.0),
              block("Available on request")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert sections[0].label == "other"
    assert sections[0].raw_heading == "Hobbies and Interests"
    assert "Available on request" in sections[0].text()


def test_leading_contact_text_without_a_heading_is_labelled_contact():
    blocks = [block("riaan.goven@example.com  +27 76 555 0142"),
              block("Profile", size=14.0),
              block("An energetic leader")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert sections[0].label == "contact"
    assert sections[1].label == "summary"


def test_leading_non_contact_text_without_a_heading_is_summary():
    blocks = [block("An energetic leader with hands-on experience"),
              block("Skills", size=14.0),
              block("Figma")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert sections[0].label == "summary"


def test_no_block_is_ever_dropped():
    """The invariant that keeps full_text() safe on unfamiliar templates.

    A section-opening heading block is carried on Section.heading, not in
    Section.blocks (see group_sections' docstring). Since Task 7b, a heading that
    the lexicon does not recognise no longer opens its own section unless it is
    the first heading seen - it nests inside the current section as an ordinary
    block instead (that's how "Another Odd One" below is handled: it stays
    unrecognised, but a section is already open, so it lands in .blocks rather
    than starting a fresh "other" section). So the count of blocks consumed as a
    Section.heading is exactly the number of Section objects that have one, and
    what must hold is that every other block lands in exactly one section's
    blocks list.
    """
    blocks = [block("Odd Heading", size=14.0), block("body one"),
              block("Another Odd One", size=14.0), block("body two"),
              block("trailing orphan")]
    sections = cv_pdf_parser.group_sections(blocks)
    grouped = sum(len(s.blocks) for s in sections)
    consumed_as_heading = sum(1 for s in sections if s.heading is not None)
    assert grouped + consumed_as_heading == len(blocks)


def test_empty_input_yields_no_sections():
    assert cv_pdf_parser.group_sections([]) == []


# --- two-tier headings ----------------------------------------------------

def test_unrecognised_heading_stays_inside_the_current_section():
    """A job title is a heading typographically, but it is an ENTRY heading, not a
    SECTION boundary. Treating it as a boundary severs the job's own content from
    Work Experience - measured at 4,487 chars stranded for one Lerato Munche entry
    alone."""
    blocks = [block("Work Experience", size=14.0),
              block("Senior UX Researcher, Umoja Bank", size=11.0, bold=True),
              block("Led end-to-end customer research")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert [s.label for s in sections] == ["experience"]
    text = sections[0].text()
    assert "Senior UX Researcher, Umoja Bank" in text
    assert "Led end-to-end customer research" in text


def test_recognised_heading_still_opens_a_new_section():
    blocks = [block("Work Experience", size=14.0),
              block("Engineer at Acme"),
              block("Education", size=14.0),
              block("BSc Computer Science")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert [s.label for s in sections] == ["experience", "education"]
    assert "BSc" not in sections[0].text()


def test_leading_unrecognised_heading_still_opens_an_other_section():
    """Preserves the Task 7 fallback: with no section open there is nothing to
    nest inside, so an unknown heading must still start one."""
    blocks = [block("Hobbies and Interests", size=14.0), block("Available on request")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert sections[0].label == "other"
    assert sections[0].raw_heading == "Hobbies and Interests"


def test_stray_single_letter_glyph_does_not_block_the_lexicon():
    """Jordan Veldman's template renders an icon before each heading. It survives
    cleaning as a bare 'l', so the heading extracts as 'l EXPERIENCE' and misses
    the lexicon - stranding 4,490 characters of his work history."""
    assert cv_sections.heading_label("l EXPERIENCE") == "experience"


def test_widened_lexicon_covers_real_corpus_headings():
    assert cv_sections.heading_label("Career Summary") == "experience"
    assert cv_sections.heading_label("Education and Qualifications") == "education"
    assert cv_sections.heading_label("Professional Profile") == "summary"
    assert cv_sections.heading_label("COURSES COMPLETED") == "education"
    assert cv_sections.heading_label("Details") == "contact"
    assert cv_sections.heading_label("Languages") == "skills"


def test_no_heading_text_is_ever_lost():
    """Strengthens test_no_block_is_ever_dropped, which the Task 7 review found
    could not detect heading-text loss: a broken group_sections that set every
    heading to None still passed it."""
    blocks = [block("Work Experience", size=14.0),
              block("Senior Engineer at Acme", size=11.0, bold=True),
              block("body text"),
              block("Referees", size=14.0),
              block("on request")]
    joined = "\n".join(s.text() for s in cv_pdf_parser.group_sections(blocks))
    for b in blocks:
        assert b.text in joined, f"lost heading or body text: {b.text!r}"


# --- cross-page sidebar bleed (Task 7b review, round 1) -------------------

def test_sidebar_opened_section_does_not_swallow_next_pages_main_column():
    """Regression for LERATO MUNCHE: her page-0 sidebar (column 1, 60 chars
    here vs column 0's 146) opens a "Languages"/skills section that used to
    stay "current" past the page boundary and absorb page 1's main-column
    text - measured on her real CV at 14,651 chars of Employment History
    swallowed by "Languages" alone. Page 1 here is single-column, so it
    counts as main and must resume Employment History instead."""
    blocks = [
        block("Employment History", size=14.0, page=0, column=0),
        block("Chief Executive Officer overseeing daily operations and "
              "long term strategy for the organisation across units",
              page=0, column=0),
        block("Details", size=14.0, page=0, column=1),
        block("email@example.com", page=0, column=1),
        block("Languages", size=14.0, page=0, column=1),
        block("English, French, Spanish", page=0, column=1),
        block("continuing the CEO role: launched three new product lines "
              "and grew headcount by 40 percent", page=1, column=0),
    ]
    sections = cv_pdf_parser.group_sections(blocks)
    experience = next(s for s in sections if s.label == "experience")
    assert "continuing the CEO role" in experience.text()
    skills_sections = [s for s in sections if s.label == "skills"]
    assert skills_sections
    assert not any("continuing the CEO role" in s.text() for s in skills_sections)


def test_main_column_section_legitimately_continues_onto_next_page():
    """Regression for JORDAN VELDMAN: his sidebar is column 0 (LEFT, the
    minority of page-0 text), so his "l EXPERIENCE" section opens on column 1
    - the MAIN column despite the higher index - and legitimately continues
    onto page 1, which is single-column. The sidebar-closes-at-a-page-
    boundary fix must not truncate a section that was never in a sidebar,
    and must not be fooled by column 0 being the sidebar here rather than
    column 1 as in the LERATO case above."""
    blocks = [
        block("DIGITAL & TECHNICAL SOFTWARE", size=14.0, page=0, column=0),
        block("SQL, Python, AutoCAD", page=0, column=0),
        block("l EXPERIENCE", size=14.0, bold=True, page=0, column=1),
        block("Senior Architectural Technologist at Acme, managing "
              "projects through multiple design and delivery stages",
              page=0, column=1),
        block("continuation of the same job entry on page 1", page=1, column=0),
    ]
    sections = cv_pdf_parser.group_sections(blocks)
    experience = next(s for s in sections if s.label == "experience")
    assert "continuation of the same job entry on page 1" in experience.text()


@pytest.mark.parametrize("sidebar_column, main_column", [(1, 0), (0, 1)],
                         ids=["right-sidebar", "left-sidebar"])
def test_sidebar_side_is_measured_not_assumed(sidebar_column, main_column):
    """The "no hardcoded column index" rule, pinned on both orientations.

    _sidebar_columns decides which column is the sidebar by measuring text per
    (page, column) and taking the minority; column 0 is the sidebar on JORDAN
    VELDMAN and THE DANIEL, column 1 on LERATO MUNCHE, GEOFF NITSCHKE and RIAAN
    GOVEN. The two tests above cover that rule only incidentally - each is
    shaped around one real CV and asserts one orientation, so between them the
    guarantee is implicit and split across two documents. This states it
    directly: the SAME document, mirrored, must segment identically either way,
    so a constant column index fails whichever parametrisation it contradicts.
    (Measured: four hardcoding mutants - _sidebar_columns returning a constant
    0 or 1, and group_sections deriving kind from block.column directly - are
    each caught here, in the expected orientation.)

    Page 1 is single-column in both, which is what makes a wrong guess bite:
    cv_pdf_parser.ordered_blocks numbers a single column as column 0 regardless of
    which side the previous page's sidebar sat on, so a section wrongly filed
    under the sidebar pointer either gets truncated at the page boundary or
    swallows the following page.
    """
    blocks = [
        block("Work Experience", size=14.0, page=0, column=main_column),
        block("Senior Architectural Technologist at Acme, running projects "
              "through every design, tender and delivery stage since 2019",
              page=0, column=main_column),
        block("Tools", size=14.0, page=0, column=sidebar_column),
        block("AutoCAD, Revit", page=0, column=sidebar_column),
        block("continuation of the main column narrative on page 1", page=1, column=0),
    ]
    sections = cv_pdf_parser.group_sections(blocks)

    experience = next(s for s in sections if s.label == "experience")
    assert "continuation of the main column narrative" in experience.text(), \
        "the main-column section must continue onto the next page"
    assert not any("continuation of the main column narrative" in s.text()
                   for s in sections if s.label == "skills"), \
        "the sidebar section must not absorb the next page's main column"


def test_sidebar_section_resets_at_each_page_boundary():
    """Synthetic cover for the sidebar page-boundary reset in group_sections.

    That branch is defensive and, instrumented across all 10 Batch1 CVs, fires
    ZERO times: every sidebar in the corpus is page 0 only, so there is never a
    second sidebar page to reset on and deleting the branch changes no CV's
    output. A multi-page sidebar is an ordinary template though, so this builds
    the case the corpus lacks - two consecutive two-column pages, sidebar on
    the same side on both.

    Page 1's sidebar text belongs to page 1's sidebar, not to the "Languages"
    section opened in page 0's. Without the reset it is appended to that stale
    section, which is the 14,651-char LERATO MUNCHE swallow one page earlier in
    the document.
    """
    blocks = [
        # page 0: main column left, sidebar right
        block("Employment History", size=14.0, page=0, column=0),
        block("Chief Executive Officer overseeing daily operations and long "
              "term strategy across every business unit of the organisation",
              page=0, column=0),
        block("Languages", size=14.0, page=0, column=1),
        block("English, French, Spanish", page=0, column=1),
        # page 1: the same two-column layout, so the sidebar recurs
        block("continuing the CEO role: launched three new product lines and "
              "grew headcount by forty percent over two financial years",
              page=1, column=0),
        block("PAGEONESIDEBARMARKER", page=1, column=1),
    ]
    sections = cv_pdf_parser.group_sections(blocks)

    skills = [s for s in sections if s.label == "skills"]
    assert skills, "the page-0 sidebar heading must still open a skills section"
    assert not any("PAGEONESIDEBARMARKER" in s.text() for s in skills), \
        "page 1's sidebar text was absorbed by page 0's sidebar section"
    assert any("PAGEONESIDEBARMARKER" in s.text() for s in sections), \
        "no block may be dropped - it must land somewhere"


# --- longest-match heading resolution (Task 14) ----------------------------

def test_heading_naming_its_section_is_matched():
    """The out-of-sample defect. These headings plainly say what they are, and
    exact-string lookup dropped both, filing 1,642 chars of tutoring and project
    work under education."""
    assert cv_sections.heading_label("TUTORING EXPERIENCE") == "experience"
    assert cv_sections.heading_label("PROJECTS & TECHNICAL EXPERIENCE") == "experience"
    assert cv_sections.heading_label("RELEVANT WORK EXPERIENCE") == "experience"
    assert cv_sections.heading_label("Volunteer Experience") == "experience"


def test_longest_key_wins():
    """The behaviour the lexicon comment claimed from the start. A heading
    containing two keys takes the longer one, so a qualifier cannot demote a
    more specific match."""
    assert cv_sections.heading_label("Technical Skills") == "skills"
    assert cv_sections.heading_label("Projects and Technical Experience") == "experience"


def test_word_runs_not_raw_substrings():
    """Matching must be on whole words. A raw substring test would fire on
    'experienced', 'inexperience' or 'skillset' and mislabel prose."""
    assert cv_sections.heading_label("Experienced Professional Overview") is None
    assert cv_sections.heading_label("Skillset Matrix") is None


def test_unmatched_heading_still_returns_none():
    assert cv_sections.heading_label("Hobbies and Interests") is None
    assert cv_sections.heading_label("Additional") is None


def test_out_of_sample_headings_resolve_as_recorded():
    """The first CV outside Batch1 the lexicon ever saw. Batch1 tuned every
    threshold in this parser, so it cannot show that the fix generalises -
    this fixture can, without the source document's personal contact details
    entering version control."""
    for text, expected in OUT_OF_SAMPLE_HEADINGS:
        assert cv_sections.heading_label(text) == expected, text


# --- nested headings are recorded -------------------------------------------

def test_nested_unrecognised_heading_is_recorded():
    """Task 7b nests an unrecognised heading inside the current section, which
    is right, but it left no trace - so findability reported a clean bill on a
    CV whose headings it had not understood."""
    blocks = [block("Work Experience", size=14.0),
              block("Senior UX Researcher, Umoja Bank", size=11.0, bold=True),
              block("Led end-to-end customer research")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert sections[0].nested_headings == ["Senior UX Researcher, Umoja Bank"]


def test_recognised_headings_leave_nested_empty():
    blocks = [block("Work Experience", size=14.0), block("body"),
              block("Education", size=14.0), block("BSc")]
    assert all(s.nested_headings == [] for s in cv_pdf_parser.group_sections(blocks))


# --- label:value lines are not headings -----------------------------------

def test_label_value_line_is_not_a_heading():
    """NDLOVU KARABO REFILWE's CV is written as bold "Label : value" pairs.
    They are short and bold, so is_heading fired on 35 of them. Harmless while
    they nested, but "Contact : 011 555 0147" inside her REFERENCES list opened
    a contact-labelled section holding four referees' phone numbers - which a UI
    would show as the candidate's own contact details.
    """
    assert not cv_pdf_parser.is_heading(block("Position : Registry Coordinator", bold=True), 10.0)
    assert not cv_pdf_parser.is_heading(block("Contact : 011 555 0147", bold=True), 10.0)
    assert not cv_pdf_parser.is_heading(block("Grade: Achieved a 72.8% average.", bold=True), 10.0)


def test_trailing_colon_is_still_a_heading():
    """A colon with nothing after it introduces a list; it is a sub-heading,
    not a label:value pair. LERATO MUNCHE prints "Key Job Responsibilities:"."""
    assert cv_pdf_parser.is_heading(block("Key Job Responsibilities:", bold=True), 10.0)
    assert cv_pdf_parser.is_heading(block("DUTIES INVOLVES THE FOLLOWING:", bold=True), 10.0)


# --- references ------------------------------------------------------------

def test_references_is_a_label():
    """Five of the ten Batch1 CVs print a references block and none was
    recognised, so referee names and phone numbers were absorbed into whatever
    section preceded them - skills on three CVs, education on one, experience
    on one."""
    assert "references" in cv_sections.LABELS
    assert cv_sections.heading_label("REFERENCES") == "references"
    assert cv_sections.heading_label("References") == "references"
    assert cv_sections.heading_label("Referees") == "references"


# --- the header block is contact, not other -------------------------------

def test_leading_header_block_with_contact_details_is_labelled_contact():
    """The name/title banner normalises to six words or fewer, so is_heading
    fires, the lexicon misses, and it opened an "other" section - putting the
    candidate's own email under "other" on 6 of the 10 Batch1 CVs. Only text
    appearing BEFORE any heading ever reached _implicit_label."""
    blocks = [block("Alex Park - Intermediate Software Test Analyst", size=14.0, bold=True),
              block("alex.park@example.com | 060 000 0000"),
              block("Professional Profile", size=14.0),
              block("A certified test analyst.")]
    sections = cv_pdf_parser.group_sections(blocks)
    assert sections[0].label == "contact"
    assert sections[1].label == "summary"


def test_leading_header_without_contact_details_stays_other():
    blocks = [block("Some Unrecognised Banner", size=14.0, bold=True),
              block("prose with no contact details at all"),
              block("Education", size=14.0), block("BSc")]
    assert cv_pdf_parser.group_sections(blocks)[0].label == "other"


def test_referees_and_hobbies_now_matches_references():
    """Guards the fixture change that adding "referees" to the lexicon forced.
    Four tests used "Referees and Hobbies" as their canonical UNRECOGNISED
    heading; longest-match now resolves it to references, which is correct - a
    section headed that way does carry referees - so those tests moved to
    "Hobbies and Interests", which no lexicon key matches. Pinned here so the
    change is deliberate rather than something a later reader has to rediscover.
    """
    assert cv_sections.heading_label("Referees and Hobbies") == "references"
    assert cv_sections.heading_label("Hobbies and Interests") is None
