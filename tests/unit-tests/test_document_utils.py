"""
Unit tests for document_utils.py

Covers:
- extract_text_from_pdf(): reading a job description PDF page by page
- load_cv(): delegating to cv_pdf_parser.parse()

Both pdfplumber and cv_pdf_parser are mocked/stubbed here rather than
touching real files or real parsing logic.
"""

from unittest.mock import MagicMock, patch

import document_utils


# ---------------------------------------------------------------------------
# extract_text_from_pdf
# ---------------------------------------------------------------------------

def _mock_pdf(pages_text):
    """Build a fake pdfplumber.open(...) context manager yielding pages
    whose .extract_text() returns each string in `pages_text` in order."""
    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)

    mock_pdf = MagicMock()
    mock_pdf.pages = mock_pages
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = False
    return mock_pdf


@patch("document_utils.pdfplumber.open")
def test_extract_text_joins_pages_with_newline(mock_open):
    mock_open.return_value = _mock_pdf(["Page one text", "Page two text"])

    result = document_utils.extract_text_from_pdf("job.pdf")

    assert result == "Page one text\nPage two text"


@patch("document_utils.pdfplumber.open")
def test_extract_text_skips_blank_pages(mock_open):
    # A page with no extractable text (e.g. a scanned/blank page) returns
    # None from pdfplumber; it should be dropped, not turned into "None".
    mock_open.return_value = _mock_pdf(["Real content", None, "More content"])

    result = document_utils.extract_text_from_pdf("job.pdf")

    assert result == "Real content\nMore content"


@patch("document_utils.pdfplumber.open")
def test_extract_text_from_single_page_pdf(mock_open):
    mock_open.return_value = _mock_pdf(["Only page"])

    result = document_utils.extract_text_from_pdf("job.pdf")

    assert result == "Only page"


@patch("document_utils.pdfplumber.open")
def test_extract_text_all_blank_pages_returns_empty_string(mock_open):
    mock_open.return_value = _mock_pdf([None, None])

    result = document_utils.extract_text_from_pdf("job.pdf")

    assert result == ""


# ---------------------------------------------------------------------------
# load_cv
# ---------------------------------------------------------------------------

@patch("document_utils.cv_pdf_parser.parse")
def test_load_cv_calls_cv_pdf_parser_parse_with_string_path(mock_parse):
    fake_cv_object = MagicMock(name="ParsedCV")
    mock_parse.return_value = fake_cv_object

    from pathlib import Path
    cv_path = Path("cvs") / "alice.pdf"
    result = document_utils.load_cv(cv_path)

    # Compare against str(cv_path) rather than a hardcoded "cvs/alice.pdf":
    # Path joins with "/" on the class but renders with the OS-native
    # separator when stringified ("\\" on Windows, "/" on Linux/Mac), and
    # load_cv() passes str(cv_path) straight through to cv_pdf_parser.parse().
    mock_parse.assert_called_once_with(str(cv_path))
    assert result is fake_cv_object
