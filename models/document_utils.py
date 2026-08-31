"""
document_utils.py

Helpers for pulling text out of the two kinds of PDF used in this project
It reads: job description listings (plain pdfplumber extraction) and
candidate CVs (via the parsed_cv module, which understands CV layout and
section structure).
"""

import sys
from pathlib import Path
import pdfplumber

# Will run no matter what directory it's launched from
_PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
sys.path.insert(0, str(_PIPELINE_DIR))
import parsed_cv


def extract_text_from_pdf(pdf_path) -> str:
    """Pull all text out of a job description PDF, in reading order."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def load_cv(cv_path):
    """
    Parse a candidate CV PDF once via parsed_cv and return the resulting
    CV object.
    Callers should parse a given CV exactly once per run and pull
    whatever views of it they need: cv.full_text() for holistic scoring,
    cv.sections_named(...) for section-scoped checks, from that same
    object, rather than re-parsing the same PDF once per thing they need.
    """
    return parsed_cv.parse(str(cv_path))
