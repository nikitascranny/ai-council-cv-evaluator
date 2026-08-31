"""The name the team's document_utils.py imports the parser under.

Their code does `sys.path.insert(0, <root>/pipeline)` then `import parsed_cv`,
and calls parse(), cv.full_text() and cv.sections_named(). All of that already
exists in cv_parser.py at the repo root - only the module name and location
differ, so this re-exports rather than reimplementing anything.

Keep it a re-export. A second parse() living here would drift from the one the
tests cover, and the divergence would show up as a scoring difference nobody
could trace back to a duplicated parser.
"""

import sys
from pathlib import Path

# cv_parser lives at the repo root, one level up. document_utils only puts
# pipeline/ on the path before importing us, so the root has to be added here
# or this import fails with only their sys.path in effect.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cv_parser import ParsedCV, Section, parse, signals  # noqa: E402

__all__ = ["ParsedCV", "Section", "parse", "signals"]
