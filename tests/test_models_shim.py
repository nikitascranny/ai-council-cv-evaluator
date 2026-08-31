"""The parser shim the team's models/document_utils.py imports.

Their file does sys.path.insert(0, <models>/pipeline) then `import parsed_cv`,
so the shim's location is part of the contract: at the repo root it would not
be found. Run in a subprocess - their conftest registers a fake parsed_cv in
sys.modules at collection time, so an in-process import would read the stub.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = ROOT / "models" / "pipeline"


def run_isolated(body: str) -> str:
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{SHIM_DIR}")
        {textwrap.indent(textwrap.dedent(body), "        ").strip()}
    """)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_shim_sits_where_their_import_looks_for_it():
    """models/pipeline/, because their insert is relative to document_utils.py."""
    assert (SHIM_DIR / "parsed_cv.py").is_file()


def test_importable_the_way_document_utils_imports_it():
    out = run_isolated("""
        import parsed_cv
        print(parsed_cv.parse.__name__, parsed_cv.ParsedCV.__name__)
    """)
    assert out.split() == ["parse", "ParsedCV"]


def test_shim_re_exports_the_real_parser_not_a_copy():
    out = run_isolated("""
        import parsed_cv, cv_parser
        print(parsed_cv.parse is cv_parser.parse)
    """)
    assert out == "True"
