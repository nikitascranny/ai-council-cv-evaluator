"""What the CV states, for the reviewer to read directly.

rewriter_v1 extracts these token classes to catch a rewrite that invented
something. Nothing else in the pipeline shows a reviewer the CV's own claims
laid out, so the same extractors are read forwards here.

This is a claim about the CV's content. It is deliberately separate from the
phrasing-sensitivity delta, which is a claim about the evaluator.
"""

import html
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "rewriter") not in sys.path:
    sys.path.insert(0, str(_ROOT / "rewriter"))

import rewriter_v1  # noqa: E402


def _verbatim(cv_text: str, token: str) -> str:
    """The token as the CV writes it.

    The extractors lowercase and strip punctuation so the rewrite guard can
    match case-insensitively. That is right for rejection and wrong here: a
    reviewer reading "bsc" sees a degraded version of their own document.
    Falls back to the normalised token when the original cannot be located,
    so this can only ever improve on showing the normalised form.
    """
    pattern = r"\b" + r"[\s\-]+".join(re.escape(w) for w in token.split()) + r"\b"
    match = re.search(pattern, cv_text, re.IGNORECASE)
    return match.group() if match else token


def facts_of(cv_text: str) -> dict:
    """The CV's own numbers, qualifications and named organisations.

    Numbers come from load_bearing_numbers rather than the raw extractor, so
    page numbers and lone digits stay out: a reviewer does not want them either.
    Qualifications and organisations are recased to the CV's own wording -
    the extractors normalise for the guard's case-insensitive matching, but a
    reviewer should see their own document, not a lowercased version of it.
    """
    if not cv_text:
        return {"numbers": [], "qualifications": [], "organisations": []}
    qualifications = {_verbatim(cv_text, q)
                       for q in rewriter_v1.extract_qualifications(cv_text)}
    organisations = {_verbatim(cv_text, o)
                      for o in rewriter_v1.extract_caps_runs(cv_text)}
    return {"numbers": sorted(rewriter_v1.load_bearing_numbers(cv_text)),
            "qualifications": sorted(qualifications, key=str.casefold),
            "organisations": sorted(organisations, key=str.casefold)}


def _list(title: str, items) -> str:
    if not items:
        return ""
    entries = "".join("<li>{}</li>".format(html.escape(i)) for i in items)
    return "<h4>{}</h4><ul>{}</ul>".format(html.escape(title), entries)


def render_facts_panel(cv_text: str) -> str:
    got = facts_of(cv_text)
    body = (_list("Numbers", got["numbers"])
            + _list("Qualifications", got["qualifications"])
            + _list("Organisations", got["organisations"]))
    if not body:
        body = "<p>No numbers, qualifications or named organisations found.</p>"
    return '<section class="facts"><h3>Facts as stated</h3>{}</section>'.format(body)
