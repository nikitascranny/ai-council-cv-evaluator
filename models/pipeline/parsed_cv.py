"""The name the team's document_utils.py imports the parser under.

models/document_utils.py does sys.path.insert(0, <its own dir>/pipeline) then
`import parsed_cv`, and calls parse(), cv.full_text() and cv.sections_named().
All three already exist in cv_parser.py at the repo root, so this re-exports
rather than reimplementing, and their file needs no change.

The path matters: their insert is relative to document_utils.py, so the module
has to sit at models/pipeline/, not at the repo root.

Keep it a re-export. A second parse() here would drift from the one the tests
cover, and the divergence would surface as a scoring difference nobody could
trace back to a duplicated parser.
"""

import sys
from pathlib import Path

# models/pipeline/parsed_cv.py -> up three to the repo root, where cv_parser is.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cv_parser import ParsedCV, Section, parse, signals  # noqa: E402

__all__ = ["ParsedCV", "Section", "parse", "signals"]
