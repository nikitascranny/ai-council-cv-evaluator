"""
conftest.py

Shared pytest setup for the CV-EVAL test suite.

Makes sure the project's source files are importable no matter which
directory `pytest` is run from, by putting each folder that holds a module
under test on sys.path:

  - common/    document_utils.py, ollama_utils.py, results_io.py, fact_guard.py
               (shared by evaluator/ and rewriter/)
  - evaluator/ essential_requirements.py, Evaluate_CV.py, judge.py
  - rewriter/  rewriter_v1.py, rewriter_v2.py, rewriter_v2_judge.py
  - parser/    cv_pdf_parser.py, cv_sections.py, txt_cv_parser.py, job_req_parser.py

document_utils.py used to import a "parsed_cv" re-export shim that had to be
stubbed here; it now imports cv_pdf_parser directly (see common/document_utils.py's
docstring), so that stub is gone too.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluator"))
sys.path.insert(0, str(PROJECT_ROOT / "rewriter"))
sys.path.insert(0, str(PROJECT_ROOT / "parser"))
