"""The pipeline/parsed_cv.py shim the team's document_utils.py imports.

document_utils.py does sys.path.insert(0, <root>/pipeline) then `import
parsed_cv`, and its callers use cv.full_text() and cv.sections_named(...).
That module name is the team's, not ours - our parser is cv_parser.py at the
root. These tests pin the shim to the import shape their code already relies
on, so renaming or moving cv_parser breaks here rather than on stage.

Every check runs in a subprocess. The team's Tests/conftest.py registers a
fake "parsed_cv" in sys.modules at collection time so their suite can run
without our parser, and that fake is in place before anything here executes.
Importing the real module in-process would either read their stub or, if we
evicted it, break their tests - the two suites would silently depend on
collection order. A fresh interpreter is the only honest way to assert what a
bare `import parsed_cv` resolves to.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"


def run_isolated(body: str) -> str:
    """Run `body` in a fresh interpreter with only pipeline/ on sys.path.

    Mirrors what document_utils.py does before it imports parsed_cv, so a
    failure here is a failure of the arrangement their code actually uses.
    """
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{PIPELINE}")
        {textwrap.indent(textwrap.dedent(body), "        ").strip()}
    """)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    return result.stdout.strip()


def test_importable_the_way_document_utils_imports_it():
    """A bare `import parsed_cv` finds our shim and exposes the team's API."""
    out = run_isolated("""
        import parsed_cv
        print(parsed_cv.parse.__name__, parsed_cv.ParsedCV.__name__)
    """)
    assert out.split() == ["parse", "ParsedCV"]


def test_shim_re_exports_the_real_parser_not_a_copy():
    """parsed_cv.parse IS cv_parser.parse - no forked second implementation."""
    out = run_isolated("""
        import parsed_cv, cv_parser
        print(parsed_cv.parse is cv_parser.parse,
              parsed_cv.ParsedCV is cv_parser.ParsedCV)
    """)
    assert out.split() == ["True", "True"]


@pytest.mark.slow
def test_full_text_and_sections_named_on_a_real_cv():
    """The two methods document_utils' docstring tells callers to use."""
    out = run_isolated("""
        import parsed_cv
        cv = parsed_cv.parse("Batch1/CVs/THE DANIEL.pdf")
        # sections_named returns a string, empty when no label matches - never
        # None, which would blow up f-string interpolation in a prompt.
        print(len(cv.full_text()) > 0,
              isinstance(cv.sections_named("skills"), str),
              cv.sections_named("no_such_label_exists") == "")
    """)
    assert out.split() == ["True", "True", "True"]
