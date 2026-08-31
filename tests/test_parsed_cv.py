"""Unit tests for the ParsedCV contract - the only surface the council team uses."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv_parser  # noqa: E402

RIAAN = ROOT / "Batch1" / "CVs" / "RIAAN GOVEN.pdf"
ALEX = ROOT / "Batch1" / "CVs" / "ALEX PARK.pdf"
LERATO = ROOT / "Batch1" / "CVs" / "LERATO MUNCHE.pdf"


@pytest.fixture(scope="module")
def riaan():
    return cv_parser.parse(RIAAN)


@pytest.fixture(scope="module")
def lerato():
    return cv_parser.parse(LERATO)


def test_parse_returns_sections(riaan):
    assert riaan.sections
    assert riaan.path == RIAAN
    assert riaan.extractor == "pymupdf"


def test_expected_sections_are_found(lerato):
    """Uses LERATO MUNCHE, not RIAAN GOVEN as originally drafted.

    Riaan's sections never contain the label "education" at all - his sidebar
    heading fails is_heading() (see test_riaan_education_is_a_known_gap), so his
    degree content is grouped under "other" rather than "education". That is the
    same known, accepted limitation the task brief calls out for
    sections_named(), and it invalidates this assertion for Riaan just as surely.
    Lerato's template labels cleanly, so she is the CV that actually demonstrates
    "the expected labels are found".
    """
    labels = {s.label for s in lerato.sections}
    assert "experience" in labels
    assert "education" in labels


def test_multi_column_pages_recorded(riaan):
    assert 0 in riaan.multi_column_pages


def test_full_text_contains_the_whole_cv(riaan):
    text = riaan.full_text()
    assert "42th Rand Bank" in text
    assert "Masters in Digital Business" in text
    assert "riaan.goven@example.com" in text


def test_full_text_no_longer_files_degrees_under_work_experience(riaan):
    """End-to-end version of the layout regression."""
    text = riaan.full_text()
    heading = text.index("Work Experience")
    education = text.index("Education")
    assert not (0 < education - heading < 40)


def test_sections_named_scopes_the_output(lerato):
    """What Akram's essential-requirements agent consumes.

    RIAAN GOVEN cannot be used for this test: his sidebar "Education" heading is
    fused by clean_blocks into a 56-word block that exceeds MAX_HEADING_WORDS, and
    his "Certifications" is a lone one-word block at exactly body size with no
    bold weight, so neither is detected as a heading and his degree content lands
    in an "other" section instead of "education" (see
    test_riaan_education_is_a_known_gap below). LERATO MUNCHE's template labels
    cleanly - her education section runs 4,844 chars and her experience section
    14,947 - so she is the CV that actually exercises sections_named's scoping
    behaviour: asking for "education" must return degree content and must not
    leak her employment history.
    """
    scoped = lerato.sections_named("education")
    assert "Cape Institute of Technology" in scoped
    assert "CodeForward Women in Tech" not in scoped, "work history must not leak in"


def test_riaan_education_is_a_known_gap(riaan):
    """Pins a known, accepted limitation instead of hiding it.

    Riaan's education and certifications content is present and correctly
    ordered (see test_full_text_contains_the_whole_cv /
    test_full_text_no_longer_files_degrees_under_work_experience) but is not
    reachable via sections_named("education", "certifications"), because his
    sidebar headings fail is_heading(): "Education" is fused by clean_blocks
    into a 56-word block that exceeds MAX_HEADING_WORDS, and "Certifications" is
    a lone word at exactly body size with no bold weight. The content survives
    under label "other" instead. This is recorded as an accepted limitation in
    the plan, not fixed here - fixing it is out of scope for Task 8, which
    builds the ParsedCV contract, not the segmentation stage's heading detection.

    If Riaan's education ever becomes reachable through sections_named this
    test starts failing (the "other"-section content moves into "education"),
    which is the point: it should fail loudly so we notice and can update the
    scoping test above to also cover Riaan.
    """
    scoped = riaan.sections_named("education", "certifications")
    assert scoped == "", (
        "Riaan's education/certifications became labelled - update this test "
        "and reconsider using him in test_sections_named_scopes_the_output"
    )
    other = riaan.sections_named("other")
    assert "Masters in Digital Business" in other


def test_sections_named_unknown_label_is_empty(riaan):
    assert riaan.sections_named("nonexistent") == ""


def test_roundtrips_through_dict(riaan):
    restored = cv_parser.ParsedCV.from_dict(riaan.to_dict())
    assert restored.full_text() == riaan.full_text()
    assert restored.multi_column_pages == riaan.multi_column_pages


def test_parsing_is_deterministic():
    assert cv_parser.parse(ALEX).full_text() == cv_parser.parse(ALEX).full_text()


def test_single_column_cv_still_parses():
    cv = cv_parser.parse(ALEX)
    assert cv.full_text().strip()
    assert cv.multi_column_pages == []


def test_all_batch1_cvs_parse_without_error():
    """A parser that throws on one CV kills a multi-hour evaluation run."""
    for pdf in sorted((ROOT / "Batch1" / "CVs").glob("*.pdf")):
        cv = cv_parser.parse(pdf)
        assert cv.full_text().strip(), f"{pdf.stem} produced no text"
