"""
document_utils.py

Shared helpers for pulling text out of the two kinds of PDF this project
reads: job description listings (plain pdfplumber extraction) and
candidate CVs (via cv_parser, which understands CV layout and section
structure).

Lives in common/, alongside ollama_utils.py, results_io.py and
fact_guard.py - shared by both evaluator/ and rewriter/, owned by neither.

Imports cv_parser directly from parser/. There used to be a parsed_cv.py
re-export shim at evaluator/pipeline/parsed_cv.py that existed only
because this file's old sys.path insert (relative to its old location
inside evaluator/pipeline/) needed a module literally named parsed_cv.
Now that this file lives in common/, that indirection is gone: this
imports cv_parser directly, and there is exactly one CV parser in the
project, not one plus a re-export of it. Delete parsed_cv.py once this
version is in place - nothing else should still be importing it.
"""

import sys
from pathlib import Path
import pdfplumber

# common/ -> repo root -> parser/
_ROOT = Path(__file__).resolve().parent.parent
_PARSER_DIR = _ROOT / "parser"
if str(_PARSER_DIR) not in sys.path:
    sys.path.insert(0, str(_PARSER_DIR))

import cv_parser


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
    Parse a candidate CV PDF once via cv_parser and return the resulting
    CV object.

    Parsing is the expensive step (layout analysis + section detection),
    so callers should parse a given CV exactly once per run and pull
    whatever views of it they need - cv.full_text() for holistic scoring,
    cv.sections_named(...) for section-scoped checks - from that same
    object, rather than re-parsing the same PDF once per thing they need.
    """
    return cv_parser.parse(str(cv_path))
