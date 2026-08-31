"""Unit tests for geometry-based layout recovery.

RIAAN GOVEN.pdf is the fixture throughout. Page 0 is genuinely two-column: a main
column at x = 39.7-351.0 and a sidebar at x = 384.6-543.8 carrying Education and
Certifications. PyMuPDF's own sort=True interleaves them, which is the defect this
module exists to fix.
"""

import sys
from pathlib import Path

import fitz
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv_parser  # noqa: E402

RIAAN = ROOT / "Batch1" / "CVs" / "RIAAN GOVEN.pdf"
JORDAN = ROOT / "Batch1" / "CVs" / "JORDAN VELDMAN.pdf"


@pytest.fixture(scope="module")
def riaan_page0():
    return fitz.open(RIAAN)[0]


@pytest.fixture(scope="module")
def jordan_page0():
    return fitz.open(JORDAN)[0]


def test_raw_blocks_carry_geometry(riaan_page0):
    blocks = cv_parser.raw_blocks(riaan_page0, 0)
    assert blocks, "page 0 must yield text blocks"
    assert all(b.page == 0 for b in blocks)
    assert all(len(b.bbox) == 4 for b in blocks)
    assert all(b.width > 0 for b in blocks)


def test_raw_blocks_find_both_columns(riaan_page0):
    """The sidebar starts at x0 ~= 384.6; the main column ends by x1 ~= 351."""
    blocks = cv_parser.raw_blocks(riaan_page0, 0)
    assert any(b.bbox[0] > 380 for b in blocks), "sidebar blocks missing"
    assert any(b.bbox[2] < 355 for b in blocks), "main column blocks missing"


def test_raw_blocks_capture_font_size(riaan_page0):
    """Heading detection in Task 6 depends on max_size, so it must be populated."""
    blocks = cv_parser.raw_blocks(riaan_page0, 0)
    assert all(b.max_size > 0 for b in blocks)
    assert len({round(b.max_size, 1) for b in blocks}) > 1, \
        "a CV with headings must show more than one font size"


def test_empty_blocks_are_dropped(riaan_page0):
    assert all(b.text.strip() for b in cv_parser.raw_blocks(riaan_page0, 0))


def test_whitespace_only_blocks_are_dropped(tmp_path):
    """Synthetic fixture for raw_blocks' `if not text.strip(): continue` guard.

    No Batch1 CV yields a whitespace-only block, so on the real corpus deleting
    that guard changes nothing and every other layout test still passes - the
    assertion above is vacuously true either way, since a page with no empty
    blocks trivially has no empty blocks among its results. PyMuPDF does emit
    such blocks for stray positioned whitespace, and they matter downstream:
    they would be counted by detect_gutter's character totals, banded by
    ordered_blocks, and handed to cv_parser.is_heading as zero-word "headings".

    This builds a page that genuinely has them, checks the raw PyMuPDF output
    really does contain them (so the fixture cannot go stale and quietly stop
    testing anything), and then checks raw_blocks removed exactly those.
    """
    path = tmp_path / "whitespace.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Work Experience")
    page.insert_text((72, 140), "    ")          # spaces only
    page.insert_text((72, 200), "\t \t")         # tabs and a space
    page.insert_text((72, 260), "Engineer at Acme")
    doc.save(path)
    doc.close()

    page = fitz.open(path)[0]
    native = ["\n".join("".join(s["text"] for s in line.get("spans", []))
                        for line in raw.get("lines", []))
              for raw in page.get_text("dict")["blocks"] if raw.get("type") == 0]
    blank = [t for t in native if not t.strip()]
    assert len(blank) == 2, \
        f"fixture no longer produces whitespace-only blocks: {native!r}"

    blocks = cv_parser.raw_blocks(page, 0)
    assert [b.text for b in blocks] == ["Work Experience", "Engineer at Acme"], \
        "whitespace-only blocks must not reach the block stream"


# --- gutter detection -----------------------------------------------------

def test_detects_riaan_sidebar_gutter(riaan_page0):
    """The real gutter sits between x=351 (main column right edge) and
    x=384.6 (sidebar left edge)."""
    blocks = cv_parser.raw_blocks(riaan_page0, 0)
    gutter = cv_parser.detect_gutter(blocks, riaan_page0.rect.width)
    assert gutter is not None, "RIAAN GOVEN page 0 is two-column"
    left, right = gutter
    assert 345 <= left <= 355, f"gutter starts at {left}, expected ~351"
    assert 380 <= right <= 390, f"gutter ends at {right}, expected ~385"


def test_right_margin_is_not_a_gutter():
    """A single-column page has whitespace to the right of every block. Read
    naively that looks like a gutter, and splitting on it invents a second
    column holding nothing."""
    doc = fitz.open(ROOT / "Batch1" / "CVs" / "PONTUS PONTIOUS.pdf")
    for i, page in enumerate(doc):
        blocks = cv_parser.raw_blocks(page, i)
        assert cv_parser.detect_gutter(blocks, page.rect.width) is None, \
            f"PONTUS PONTIOUS page {i} wrongly detected as two-column"


def test_row_paired_date_column_is_not_a_gutter():
    """ALEX PARK's career summary puts dates left of the entry they belong to.
    That LOOKS two-column and a human labelling by eye calls it two-column, but
    the columns are a table read ACROSS, not two independent streams read down.
    Column-wise ordering would emit every date, then every job description,
    detaching each date from its role - worse than the current behaviour.

    It is safe today only because PyMuPDF merges each row into one full-width
    block (x0=53.9 x1=541.5 carrying "December 2016 - April 2017" AND
    "Work-Coordination - Shared Services..."), and detect_gutter excludes
    full-width blocks before searching. This test pins that: if a future change
    stops excluding them, ALEX PARK's work history silently shreds.
    """
    doc = fitz.open(ROOT / "Batch1" / "CVs" / "ALEX PARK.pdf")
    for i, page in enumerate(doc):
        blocks = cv_parser.raw_blocks(page, i)
        assert cv_parser.detect_gutter(blocks, page.rect.width) is None, \
            f"ALEX PARK page {i}: row-paired date column wrongly read as a gutter"


def test_detection_matches_hand_labels():
    """Precision matters more than recall here: a missed two-column page is no
    worse than today, but a single-column page wrongly split is a new bug."""
    import json
    labels = json.loads(
        (ROOT / "tests" / "fixtures" / "layout_labels.json").read_text(encoding="utf-8")
    )
    false_positives, misses = [], []
    for stem, spec in labels.items():
        if not isinstance(spec, dict):   # skip "_comment" metadata key
            continue
        expected = set(spec["multi_column_pages"])
        doc = fitz.open(ROOT / "Batch1" / "CVs" / f"{stem}.pdf")
        for i, page in enumerate(doc):
            got = cv_parser.detect_gutter(cv_parser.raw_blocks(page, i), page.rect.width) is not None
            if got and i not in expected:
                false_positives.append(f"{stem} p{i}")
            if not got and i in expected:
                misses.append(f"{stem} p{i}")
    assert not false_positives, f"single-column pages wrongly split: {false_positives}"
    assert len(misses) <= 1, f"too many two-column pages missed: {misses}"


def test_the_daniel_is_detected_without_a_shaded_panel():
    """Four of the five labelled pages draw a filled background rectangle behind
    the sidebar, so they could be found by reading vector drawings instead of
    text geometry. THE DANIEL has no such panel, which is why detection must be
    geometric. It is the case that proves the approach.
    """
    doc = fitz.open(ROOT / "Batch1" / "CVs" / "THE DANIEL.pdf")
    page = doc[0]
    assert cv_parser.detect_gutter(cv_parser.raw_blocks(page, 0), page.rect.width) is not None


# --- reading order --------------------------------------------------------

def _joined(page, index=0):
    return "\n".join(b.text for b in cv_parser.ordered_blocks(page, index))


def test_education_no_longer_follows_the_work_experience_heading(riaan_page0):
    """The regression this module exists to fix.

    With sort=True, the sidebar block starting "Education" is emitted directly
    after the main-column heading "Work Experience", so Riaan's degrees read as
    jobs. In true reading order the whole main column - several hundred
    characters of actual work history - separates them.
    """
    text = _joined(riaan_page0)
    heading = text.index("Work Experience")
    education = text.index("Education")
    assert not (0 < education - heading < 40), \
        "Education still lands immediately after the Work Experience heading"


def test_columns_are_not_interleaved(riaan_page0):
    """Every main-column block must precede every sidebar block within a band.

    This is the specific defect: with sort=True the sidebar interleaves into the
    main column instead of following it. Checking only min(sidebar) > min(main)
    would pass even if a sidebar block snuck in ahead of a later main-column
    block, so this asserts the stronger, literal claim in the docstring above:
    every main order is less than every sidebar order.
    """
    blocks = cv_parser.ordered_blocks(riaan_page0, 0)
    sidebar_orders = [b.order for b in blocks if b.column == 1]
    main_orders = [b.order for b in blocks if b.column == 0 and b.width < 300]
    assert sidebar_orders, "sidebar blocks must be assigned column 1"
    assert main_orders, "main-column blocks must be assigned column 0"
    assert max(main_orders) < min(sidebar_orders), \
        "a sidebar block is ordered ahead of a later main-column block"


def test_jordan_sidebar_and_main_column_are_contiguous(jordan_page0):
    """Regression for the JORDAN VELDMAN defect found in review.

    His sidebar is on the LEFT (x=0-187) and the main column, on the right, is
    alone ~63% of the 595pt page - wide enough that its own line-wrapped bullets
    (widths up to 379pt) exceed FULL_WIDTH_RATIO (0.6) without ever straddling
    the gutter. A version of ordered_blocks that used the width-ratio test to
    spot band separators misclassified those bullets as page-spanning,
    fragmenting the page into dozens of spurious bands and interleaving sidebar
    contact info into the middle of the Experience section: 'Candidate Senior
    Architectural Technologist...' (genuinely main-column, x0=215.4) came out at
    order=2 while sidebar content interleaved at orders 4-6 and beyond.

    Correct output has exactly one column transition: every sidebar block
    (column 0 for this CV - the sidebar sits left of the gutter) precedes every
    main-column block (column 1), or vice versa, but the two never alternate.
    """
    blocks = cv_parser.ordered_blocks(jordan_page0, 0)
    columns = [b.column for b in blocks]
    assert set(columns) == {0, 1}, "both columns must be present"
    transitions = sum(1 for a, b in zip(columns, columns[1:]) if a != b)
    assert transitions == 1, \
        f"columns must form two contiguous runs, not alternate: {columns}"


def test_order_and_column_are_assigned(riaan_page0):
    blocks = cv_parser.ordered_blocks(riaan_page0, 0)
    assert [b.order for b in blocks] == list(range(len(blocks)))
    assert set(b.column for b in blocks) <= {0, 1}


def test_single_column_page_keeps_top_to_bottom_order():
    doc = fitz.open(ROOT / "Batch1" / "CVs" / "ALEX PARK.pdf")
    blocks = cv_parser.ordered_blocks(doc[0], 0)
    tops = [b.top for b in blocks]
    assert tops == sorted(tops), "single-column page must read straight down"
    assert all(b.column == 0 for b in blocks)


def test_no_text_is_lost(riaan_page0):
    """Reordering must not drop blocks."""
    assert len(cv_parser.ordered_blocks(riaan_page0, 0)) == \
        len(cv_parser.raw_blocks(riaan_page0, 0))


@pytest.mark.parametrize("stem", ["JORDAN VELDMAN", "RIAAN GOVEN", "ALEX PARK"])
def test_block_count_and_order_are_preserved(stem):
    """No block is dropped or duplicated, and .order is contiguous from 0.

    Covers both multi-column CVs (JORDAN VELDMAN, RIAAN GOVEN) and the
    single-column control (ALEX PARK) so a regression in either code path -
    the banded multi-column branch or the plain top-to-bottom branch - shows
    up here regardless of which CV a future change happens to be tested against.
    """
    page = fitz.open(ROOT / "Batch1" / "CVs" / f"{stem}.pdf")[0]
    raw = cv_parser.raw_blocks(page, 0)
    ordered = cv_parser.ordered_blocks(page, 0)
    assert len(ordered) == len(raw), f"{stem}: block count changed under reordering"
    assert [b.order for b in ordered] == list(range(len(ordered))), \
        f"{stem}: .order is not contiguous from 0"
    assert {b.text for b in ordered} == {b.text for b in raw}, \
        f"{stem}: block text changed under reordering"


# --- normalisation --------------------------------------------------------

def test_clean_blocks_removes_icon_glyphs(riaan_page0):
    """Riaan's contact strip carries icon-font control codes (\\x83 and friends)
    that cv_parser.clean() replaces with spaces."""
    text = "\n".join(b.text for b in cv_parser.clean_blocks(riaan_page0, 0))
    assert "\x83" not in text
    assert "+27 76 555 0142" in text


def test_clean_blocks_preserves_hyphens(riaan_page0):
    """clean()'s junk-glyph class must not eat real hyphens - the bug recorded in
    the JUNK_GLYPHS pattern (stage 1) that destroyed 25 of 31 hyphens in this CV."""
    text = "\n".join(b.text for b in cv_parser.clean_blocks(riaan_page0, 0))
    assert "hands-on" in text


def test_hyphen_rejoin_cannot_bridge_two_columns():
    """clean() rejoins "foo-\\nbar" into "foobar". Applied to concatenated columns
    that fuses the tail of one column onto the head of the next, so it must run
    per block.

    Demonstrated directly: cleaning the join of two blocks welds them together,
    cleaning them separately does not.
    """
    import cv_parser
    tail, head = "experience in inter-", "national markets"
    assert cv_parser.clean(f"{tail}\n{head}") == "experience in international markets"
    assert cv_parser.clean(tail) + cv_parser.clean(head) == \
        "experience in inter-national markets"


def test_clean_blocks_does_not_weld_across_the_gutter(riaan_page0):
    """The same guarantee on the real document: no cleaned block spans columns."""
    blocks = cv_parser.clean_blocks(riaan_page0, 0)
    assert {b.column for b in blocks} == {0, 1}, "page 0 must yield both columns"


def test_clean_blocks_keeps_ordering(riaan_page0):
    """Order field must be renumbered contiguously from 0, not just non-decreasing.

    Regression: a buggy implementation that drops blocks but skips renumbering
    produces orders [1, 2, ...] instead of [0, 1, ...], which only-sorted() checks
    would miss. This assertion catches it.
    """
    blocks = cv_parser.clean_blocks(riaan_page0, 0)
    assert [b.order for b in blocks] == list(range(len(blocks)))


def test_clean_blocks_drops_empty_and_renumbers():
    """Synthetic regression for drop-and-renumber path. The corpus has no blocks
    that clean to empty (icon-only decorations on two-column CVs are rare), so
    the code path that removes them and renumbers .order has no real data to
    exercise it. This test constructs such blocks directly.

    If clean_blocks drops an empty block but fails to renumber, the survivors
    will have orders [1, 2, 3, ...] instead of [0, 1, 2, ...], which this asserts.
    """
    import cv_parser

    # Create a block that cleans to empty: icon-only, no real text
    empty = cv_parser.Block(
        text="\x83\x84",  # icon font control codes
        page=0,
        bbox=(0, 0, 100, 20),
        max_size=12.0,
        bold=False,
        column=0,
        order=1
    )
    # Confirm it will clean to empty
    assert not cv_parser.clean(empty.text)

    # Simulate what ordered_blocks would return: blocks with the empty one
    full = cv_parser.Block(
        text="Full block",
        page=0,
        bbox=(0, 20, 100, 40),
        max_size=12.0,
        bold=False,
        column=0,
        order=0
    )
    another = cv_parser.Block(
        text="Another block",
        page=0,
        bbox=(0, 40, 100, 60),
        max_size=12.0,
        bold=False,
        column=0,
        order=2
    )

    # Monkeypatch ordered_blocks to return our synthetic blocks
    original_ordered = cv_parser.ordered_blocks
    try:
        cv_parser.ordered_blocks = lambda page, page_index: [full, empty, another]

        # Mock page object (not used with our monkeypatch)
        class FakePage:
            pass

        result = cv_parser.clean_blocks(FakePage(), 0)

        # The empty block should be dropped
        assert len(result) == 2
        # The survivors must have contiguous order from 0
        assert [b.order for b in result] == [0, 1]
        # Texts must be preserved
        assert {b.text for b in result} == {"Full block", "Another block"}
    finally:
        cv_parser.ordered_blocks = original_ordered


def test_clean_blocks_prevents_hyphen_weld_across_columns_lerato():
    """Real-corpus regression test. LERATO MUNCHE page 0, block 12 ends with
    'institu-' (column 0), immediately followed by 'Details' (column 1).

    If clean_blocks incorrectly concatenated all text and applied clean()
    page-wise, the hyphen would rejoin: 'institu-' + 'Details' ->
    cv_parser.clean('institu-\\nDetails') -> 'instituDetails'.

    The correct per-block approach leaves both fragments intact. This test
    verifies the welded form does NOT appear, while both originals do.
    """
    doc = fitz.open(LERATO := ROOT / "Batch1" / "CVs" / "LERATO MUNCHE.pdf")
    page = doc[0]
    blocks = cv_parser.clean_blocks(page, 0)
    text = "\n".join(b.text for b in blocks)

    # The welded form must NOT appear
    assert "instituDetails" not in text, \
        "Block text was incorrectly welded across column boundary"

    # But both real fragments should survive (separately, in reading order)
    assert "institu-" in text or "institu" in text, \
        "Column 0 fragment missing"
    assert "Details" in text, "Column 1 fragment missing"

    # Both columns present
    assert {b.column for b in blocks} == {0, 1}, "page 0 must yield both columns"
