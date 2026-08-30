"""
conftest.py

Shared pytest setup for the CV-EVAL test suite.

Two jobs:

1. Make sure the project's source files (document_utils.py, ollama_utils.py,
   essential_requirements.py, Evaluate_CV.py, results_io.py) are importable
   no matter which directory `pytest` is run from, by putting the project
   root on sys.path.

2. Stub out the "parsed_cv" module. document_utils.py does
   `sys.path.insert(0, <project_root>/pipeline)` then `import parsed_cv`,
   but the pipeline/ folder and its parsed_cv module are a separate part
   of the project (built by another teammate) and aren't included in this
   test suite. We register a fake "parsed_cv" module in sys.modules
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

if "parsed_cv" not in sys.modules:
    _stub = types.ModuleType("parsed_cv")
    _stub.parse = lambda path: None  # overridden per-test via monkeypatch
    sys.modules["parsed_cv"] = _stub
