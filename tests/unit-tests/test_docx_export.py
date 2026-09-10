"""The rewritten CV as a .docx.

The guard's verdict has to survive into the file itself - a document outlives
the page that offered it, and a rejected rewrite must never read as a sound one.
"""

import io
import zipfile
from types import SimpleNamespace

from docx import Document

from ui import docx_export


def _result(**kw):
    base = dict(candidate="Jane Doe", job_title="Secretary", agent_model="phi4",
                original_text="orig", rewritten_text="First block.\n\nSecond block.",
                used_fallback=False, rejected=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _paragraphs(result):
    return [p.text for p in docx_export.build_document(result).paragraphs]


def test_the_bytes_are_a_real_docx():
    """A .docx is a zip. Asserting that catches a truncated or empty write
    that a paragraph-level check would sail past."""
    raw = docx_export.docx_bytes(_result())
    assert zipfile.is_zipfile(io.BytesIO(raw))
    assert Document(io.BytesIO(raw)).paragraphs


def test_the_rewritten_text_is_split_into_paragraphs_not_one_slab():
    assert "First block." in _paragraphs(_result())
    assert "Second block." in _paragraphs(_result())


def test_a_fallback_says_in_the_document_that_it_is_the_original_cv():
    """used_fallback means the guard rejected the rewrite twice and v2
    substituted the candidate's own text. Someone opening this file later
    must not read it as a rewrite that happened not to change much."""
    text = " ".join(_paragraphs(_result(used_fallback=True)))
    assert "not a rewrite" in text
    assert "original CV" in text


def test_a_clean_rewrite_carries_no_warning():
    text = " ".join(_paragraphs(_result()))
    assert "not a rewrite" not in text
    assert "flagged" not in text


def test_a_flagged_but_kept_rewrite_says_so():
    text = " ".join(_paragraphs(_result(rejected=True)))
    assert "flagged" in text


def test_the_job_title_is_named_so_the_file_says_what_it_was_aimed_at():
    assert any("Secretary" in p for p in _paragraphs(_result()))


def test_the_filename_names_candidate_job_and_agent():
    """Two versions of one CV land in the same downloads folder, so the
    agent has to be in the name or the second silently overwrites nothing
    and the user cannot tell them apart."""
    name = docx_export.filename_for(_result())
    assert name.endswith(".docx")
    assert "Jane Doe" in name and "Secretary" in name and "phi4" in name


def test_a_colon_in_the_model_tag_does_not_reach_the_filename():
    """qwen2.5:14b is a valid Ollama tag. On NTFS a ":" in a filename opens
    an alternate data stream, and in a Content-Disposition header it breaks
    the parse."""
    name = docx_export.filename_for(_result(agent_model="qwen2.5:14b"))
    assert ":" not in name
    assert "qwen2.5" in name


def test_empty_rewritten_text_still_produces_a_readable_document():
    raw = docx_export.docx_bytes(_result(rewritten_text=""))
    assert zipfile.is_zipfile(io.BytesIO(raw))


# --- markdown a model emits despite being told not to -----------------------


def _doc(text):
    return docx_export.build_document(_result(rewritten_text=text))


def _visible(doc):
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def test_bold_markers_become_bold_not_literal_asterisks():
    """phi4 emitted 26 markdown markers from a prompt that asked for plain
    text, so a candidate's finished CV read **Jason Muller** with the
    asterisks showing. qwen2.5 emitted none from the same prompt - no
    wording reliably stops it, so the export has to cope."""
    doc = _doc("**Jason Muller**\nPretoria, South Africa")
    assert "**" not in _visible(doc)
    bolded = [r.text for p in doc.paragraphs for r in p.runs if r.bold]
    assert "Jason Muller" in bolded


def test_text_around_a_bold_run_survives():
    doc = _doc("**Acting Assistant Manager** | Uncle Dave's | 2019-2020")
    text = _visible(doc)
    assert "Uncle Dave's" in text and "2019-2020" in text
    assert "**" not in text


def test_markdown_bullets_become_real_list_items():
    doc = _doc("- Supported daily site operations\n- Handled escalations")
    bullets = [p for p in doc.paragraphs if p.style.name == "List Bullet"]
    assert len(bullets) == 2
    assert bullets[0].text == "Supported daily site operations"


def test_hash_headings_become_word_headings():
    doc = _doc("## Professional Experience\nAssistant Manager")
    styles = [p.style.name for p in doc.paragraphs if p.text.strip()]
    assert any(s.startswith("Heading") for s in styles)


def test_each_line_is_its_own_paragraph():
    """Splitting on blank lines buried single newlines inside one paragraph,
    and Word does not render those as breaks - a whole CV arrived as two
    unbroken slabs."""
    doc = _doc("JASON MULLER\nPretoria\nPROFESSIONAL PROFILE\nAdaptable.")
    assert len([p for p in doc.paragraphs if p.text.strip()]) == 4 + 2


def test_an_unpaired_asterisk_is_left_alone():
    """More likely the candidate's own punctuation than a formatting
    instruction, and eating it would silently alter their CV."""
    assert "2 * 3" in _visible(_doc("Throughput was 2 * 3 units"))


def test_plain_text_is_unchanged_by_the_markdown_handling():
    """qwen2.5 emits no markdown; its output must pass through untouched."""
    doc = _doc("JASON MULLER\nCustomer service professional.")
    assert "JASON MULLER" in _visible(doc)
    assert not any(r.bold for p in doc.paragraphs for r in p.runs)
