"""The rewritten CV as a Word document the candidate can edit.

rewriter_v2 returns plain text plus guard metadata and deliberately stops
there - its docstring says the export step exists so that data need not be
reshaped to build it. This is that step.

Pure by design: bytes in memory, no filesystem. The web UI serves the bytes
as a download; rewriter_v2's own CLI can write them to a file next to its
.txt and metadata.csv. Neither one is baked in here, and a test can assert on
the document without Ollama or a temp directory.

The fallback banner is written INTO the document rather than shown only on the
page that offered it. A .docx outlives the browser session that produced it,
and the whole point of the guard is that a rewrite it rejected must never be
mistaken for one it passed - least of all by whoever opens this file a week
later with no memory of the warning.
"""

import io
import re

from docx import Document

from ui.pipeline import candidate
from docx.shared import Pt

# The guard rejected this agent's rewrite twice, so rewritten_text is the
# candidate's ORIGINAL CV. Saying that plainly is the entire job of this
# paragraph - "fallback" is our word, not something a candidate would parse.
_FALLBACK_PREFIX = "This is your original CV, not a rewrite. "
_FALLBACK_SUFFIX = (" It was discarded rather than shown to you as if it were "
                    "sound.")

_FLAGGED_WARNING = (
    "The automated check flagged this rewrite. Read it against your original "
    "before sending it anywhere.")


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")


def _add_runs(paragraph, text: str):
    """Write text into a paragraph, turning **bold** into actual bold runs.

    Split rather than stripped: the markers have to become formatting, not
    vanish and not survive as literal asterisks. Anything that is not a
    matched pair is left exactly as written - a lone ** in a CV is far more
    likely to be the candidate's own punctuation than a formatting
    instruction we should silently eat.
    """
    position = 0
    for match in _BOLD.finditer(text):
        if match.start() > position:
            paragraph.add_run(text[position:match.start()])
        paragraph.add_run(match.group(1)).bold = True
        position = match.end()
    if position < len(text):
        paragraph.add_run(text[position:])
    return paragraph


def _add_body(document, text: str):
    """The rewritten text as Word paragraphs, honouring the markdown a model
    emits whether or not it was asked for.

    Both agents are told to output plain text with no markdown. phi4 emits it
    anyway - 26 markers in the first CV that passed the guard, so a
    candidate's finished document read **Jason Muller** with the asterisks
    visible. qwen2.5 emitted none from the same prompt, so this is one
    model's habit rather than a fault in the instruction, and no wording will
    reliably stop it. Converting beats stripping: what the model marks up is
    headings and job titles, which is real structure the candidate wants.

    One paragraph per line, not per blank-line block. Models emit one logical
    line at a time - a heading, a role, a bullet - and rarely hard-wrap prose,
    so splitting on blank lines collapsed a whole CV into a couple of
    paragraphs with newlines buried inside them, which Word does not render
    as breaks at all.
    """
    for line in text.replace("\r\n", "\n").split("\n"):
        # A markdown hard break is two trailing spaces. Every line is already
        # its own paragraph here, so it carries no meaning and would survive
        # only as trailing whitespace in Word.
        line = line.rstrip()
        if not line.strip():
            continue

        heading = _HEADING.match(line)
        if heading:
            level = min(len(heading.group(1)), 4)
            _add_runs(document.add_heading("", level=level), heading.group(2))
            continue

        bullet = _BULLET.match(line)
        if bullet:
            _add_runs(document.add_paragraph("", style="List Bullet"),
                      bullet.group(1))
            continue

        _add_runs(document.add_paragraph(), line)


def build_document(result) -> "Document":
    """A Document for one agent's RewriteResult.

    Returned rather than saved so callers can serialise it however they need
    and tests can read the paragraphs back without touching a disk.
    """
    document = Document()
    document.styles["Normal"].font.size = Pt(11)

    heading = result.candidate or "Rewritten CV"
    document.add_heading(heading, level=1)
    if result.job_title:
        document.add_paragraph("Tailored for: {}".format(result.job_title))

    if result.used_fallback:
        document.add_paragraph(
            _FALLBACK_PREFIX
            + candidate.describe_rejection(getattr(result, "reasons", None))
            + _FALLBACK_SUFFIX)
    elif result.rejected:
        document.add_paragraph(_FLAGGED_WARNING)

    _add_body(document, result.rewritten_text or "")
    return document


def docx_bytes(result) -> bytes:
    """The document as the bytes of a .docx file."""
    buffer = io.BytesIO()
    build_document(result).save(buffer)
    return buffer.getvalue()


def filename_for(result) -> str:
    """A download name that says whose CV, for what job, from which agent.

    Three CVs in a downloads folder all called "cv.docx" are useless, and the
    agent belongs in the name because the two versions differ - that is the
    whole reason the user was shown both.
    """
    parts = [result.candidate or "cv", result.job_title or "", result.agent_model or ""]
    keep = "-_(). "
    cleaned = []
    for part in parts:
        part = "".join(c if (c.isalnum() or c in keep) else "_" for c in part).strip()
        if part:
            cleaned.append(part)
    return " - ".join(cleaned).strip().strip(".") + ".docx"
