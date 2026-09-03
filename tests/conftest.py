"""
conftest.py

Shared pytest setup for the CV-EVAL test suite.

Two jobs:

1. Make sure the project's source files (document_utils.py, ollama_utils.py,
   essential_requirements.py, Evaluate_CV.py, results_io.py in evaluator/,
   and rewriter_v1.py in rewriter/) are importable no matter which directory
   `pytest` is run from, by putting those folders and the project root on
   sys.path.

2. Stub out the "parsed_cv" module. document_utils.py does
   `sys.path.insert(0, <its own dir>/pipeline)` then `import parsed_cv`,
   but the evaluator/pipeline/ folder and its parsed_cv module are a separate
   part of the project (built by another teammate) and aren't included in
   this test suite. We register a fake "parsed_cv" module in sys.modules
   *before* document_utils gets imported, so Python's import system finds
   our stub instead of trying (and failing) to load the real thing from
   disk. Individual tests can then monkeypatch stub.parse(...) to control
   exactly what "parsing a CV" returns.
"""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# document_utils.py, ollama_utils.py, essential_requirements.py, Evaluate_CV.py
# and results_io.py live in evaluator/; without this the suite cannot import
# them and collection aborts.
sys.path.insert(0, str(PROJECT_ROOT / "evaluator"))
# rewriter_v1.py lives in rewriter/, for the same reason.
sys.path.insert(0, str(PROJECT_ROOT / "rewriter"))

if "parsed_cv" not in sys.modules:
    _stub = types.ModuleType("parsed_cv")
    _stub.parse = lambda path: None  # overridden per-test via monkeypatch
    sys.modules["parsed_cv"] = _stub
