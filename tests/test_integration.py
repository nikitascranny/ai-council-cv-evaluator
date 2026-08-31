"""Integration tests for the parser's entry into the evaluation pipeline."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import cv_parser  # noqa: E402
import run_evaluations  # noqa: E402

RIAAN = ROOT / "Batch1" / "CVs" / "RIAAN GOVEN.pdf"


def test_extract_flat_matches_the_existing_extractor(tmp_path, monkeypatch):
    """--parser flat must reproduce today's text exactly, or the parser switch
    is not a controlled comparison."""
    monkeypatch.setattr(run_evaluations, "CACHE", tmp_path)
    import cv_parser
    assert run_evaluations.extract(RIAAN, "pymupdf", parser="flat") == \
        cv_parser.extract(RIAAN, "pymupdf")


def test_extract_structured_returns_repaired_text(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluations, "CACHE", tmp_path)
    text = run_evaluations.extract(RIAAN, "pymupdf", parser="structured")
    heading = text.index("Work Experience")
    education = text.index("Education")
    assert not (0 < education - heading < 40)


def test_structured_cache_is_keyed_separately(tmp_path, monkeypatch):
    """Reusing flat text under the structured key would silently invalidate the
    comparison - the same trap the per-extractor cache split already guards."""
    monkeypatch.setattr(run_evaluations, "CACHE", tmp_path)
    flat = run_evaluations.extract(RIAAN, "pymupdf", parser="flat")
    structured = run_evaluations.extract(RIAAN, "pymupdf", parser="structured")
    assert flat != structured
    assert run_evaluations.extract(RIAAN, "pymupdf", parser="flat") == flat


def test_structured_cache_is_reused_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluations, "CACHE", tmp_path)
    first = run_evaluations.extract(RIAAN, "pymupdf", parser="structured")
    cached = list(tmp_path.rglob("*.json"))
    assert cached, "structured parse must be cached as JSON"
    assert run_evaluations.extract(RIAAN, "pymupdf", parser="structured") == first


def _structured_cache_file(tmp_path):
    """The single JSON file extract() wrote under the structured cache key."""
    cached = list(tmp_path.rglob("*.json"))
    assert len(cached) == 1, f"expected exactly one cache entry, got {cached}"
    return cached[0]


def test_torn_structured_cache_is_reparsed_instead_of_crashing(tmp_path, monkeypatch):
    """A Ctrl-C during startup extraction must not brick every later run.

    write_text is not atomic, so an interrupt can leave a truncated JSON file
    behind. extract() runs over every CV before any model is called, so an
    unhandled JSONDecodeError there kills the run at startup - contradicting
    this module's own docstring ("safe to re-run after a crash or Ctrl-C") and
    its own precedent in already_done(), which deliberately tolerates a torn
    final line.
    """
    monkeypatch.setattr(run_evaluations, "CACHE", tmp_path)
    expected = run_evaluations.extract(RIAAN, "pymupdf", parser="structured")

    cached = _structured_cache_file(tmp_path)
    whole = cached.read_text(encoding="utf-8")
    cached.write_text(whole[: len(whole) // 2], encoding="utf-8")  # torn mid-write

    assert run_evaluations.extract(RIAAN, "pymupdf", parser="structured") == expected
    assert cached.read_text(encoding="utf-8") == whole, \
        "a torn cache entry must be rewritten, not left torn for the next run"


def test_partial_schema_structured_cache_is_reparsed(tmp_path, monkeypatch):
    """Valid JSON missing a key the schema needs is the other torn-file shape.

    from_dict reads data["extractor"] directly, so a file written by an older
    or interrupted schema raises KeyError rather than JSONDecodeError. Both
    must fall through to a fresh parse.
    """
    monkeypatch.setattr(run_evaluations, "CACHE", tmp_path)
    expected = run_evaluations.extract(RIAAN, "pymupdf", parser="structured")

    cached = _structured_cache_file(tmp_path)
    data = json.loads(cached.read_text(encoding="utf-8"))
    del data["extractor"]
    cached.write_text(json.dumps(data), encoding="utf-8")

    assert run_evaluations.extract(RIAAN, "pymupdf", parser="structured") == expected


def _two_page_pdf_with_one_image_only_page(path):
    """A 2-page PDF whose second page carries no text at all.

    No Batch1 CV is scanned, so the warned-but-not-handled path the spec's
    non-goal describes has no real fixture. This builds the smallest document
    that triggers it.
    """
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Profile\nAn energetic leader")
    doc.new_page()          # deliberately empty: stands in for a scanned page
    doc.save(path)
    doc.close()
    return path


def test_scanned_page_warning_is_printed_on_a_cache_hit(tmp_path, monkeypatch, capsys):
    """The warned half of "detected, warned, not handled" must survive caching.

    The warning loop used to sit only in the cache-miss branch, so run 1
    printed WARN and every run afterwards silently fed the model a CV with a
    missing page and said nothing. from_dict restores warnings, so the hit
    path has the same information available - it just wasn't printing it.
    """
    monkeypatch.setattr(run_evaluations, "CACHE", tmp_path)
    pdf = _two_page_pdf_with_one_image_only_page(tmp_path / "SCANNED SAMPLE.pdf")

    run_evaluations.extract(pdf, "pymupdf", parser="structured")     # miss
    first = capsys.readouterr().out
    assert "WARN SCANNED SAMPLE" in first, "the cache-miss path must warn"

    assert _structured_cache_file(tmp_path).exists()
    run_evaluations.extract(pdf, "pymupdf", parser="structured")     # hit
    second = capsys.readouterr().out
    assert "WARN SCANNED SAMPLE" in second, \
        "a cached scanned PDF still has a missing page; the warning must repeat"
    assert "page 1 yielded no text" in second


def test_default_models_are_installed_locally():
    """Every DEFAULT_MODELS entry must be a model this machine can actually run.

    An uninstalled model does not fail loudly: Ollama answers 404, urllib
    raises HTTPError, HTTPError subclasses URLError, and the except around
    call_ollama catches it and WRITES AN ERROR ROW to the shared
    Results/votes.jsonl. already_done() then treats those rows as work
    completed. The bare documented command previously appended 5 x 9 x 10 =
    450 such rows.

    Asserted against the three 3B-class models recorded as pulled here rather
    than by querying Ollama, so the test needs no daemon and no network.
    """
    installed = {"qwen2.5:3b", "llama3.2:3b", "phi3.5:3.8b"}
    assert set(run_evaluations.DEFAULT_MODELS) <= installed, (
        "DEFAULT_MODELS names a model that is not installed: "
        f"{sorted(set(run_evaluations.DEFAULT_MODELS) - installed)}"
    )
    assert run_evaluations.__doc__ is not None
    for model in run_evaluations.DEFAULT_MODELS[:2]:
        assert model in run_evaluations.__doc__, \
            f"the documented usage line must name runnable models, missing {model}"


def test_already_done_distinguishes_parsers():
    """Rows from the two parsers must not collide, or a structured run is skipped
    as already complete."""
    row = {"job": "j", "cv": "c", "model": "m", "prompt_version": "v1_plain",
           "num_ctx": 8192, "seed": 0, "extractor": "pymupdf", "parser": "flat"}
    keys = run_evaluations.key_of(row)
    other = run_evaluations.key_of({**row, "parser": "structured"})
    assert keys != other
