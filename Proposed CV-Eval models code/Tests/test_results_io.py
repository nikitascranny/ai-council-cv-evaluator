"""
Unit tests for results_io.py

Covers:
- safe_filename_part(): turning arbitrary text into a filesystem-safe string
- save_json(): writing results to disk, including creating missing folders
"""

import json

from results_io import safe_filename_part, save_json


# ---------------------------------------------------------------------------
# safe_filename_part
# ---------------------------------------------------------------------------

def test_safe_filename_part_replaces_spaces_and_punctuation():
    result = safe_filename_part("Senior ML Engineer (Remote)!")
    assert result == "Senior_ML_Engineer_Remote_"


def test_safe_filename_part_keeps_word_characters_and_hyphens():
    # Letters, digits, underscores, and hyphens should all survive untouched.
    result = safe_filename_part("job-2024_v2")
    assert result == "job-2024_v2"


def test_safe_filename_part_collapses_runs_of_bad_characters():
    # A run of consecutive unsafe characters should become a single underscore,
    # not one underscore per character.
    result = safe_filename_part("a///b   c")
    assert result == "a_b_c"


def test_safe_filename_part_empty_string_stays_empty():
    assert safe_filename_part("") == ""


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

def test_save_json_creates_missing_directories(tmp_path):
    output_dir = tmp_path / "results" / "nested"
    assert not output_dir.exists()

    save_json([{"candidate": "Alice", "score": 9}], output_dir, "alice.json")

    assert output_dir.exists()


def test_save_json_writes_expected_content(tmp_path):
    data = [{"candidate": "Bob", "score": 7, "summary": "Solid fit."}]

    filepath = save_json(data, tmp_path, "bob.json")

    with open(filepath, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_save_json_returns_full_path(tmp_path):
    filepath = save_json([], tmp_path, "empty.json")

    assert filepath == tmp_path / "empty.json"


def test_save_json_preserves_non_ascii_characters(tmp_path):
    # ensure_ascii=False in the implementation means names like "Zoë" should
    # be written as-is, not escaped as \u00eb.
    data = [{"candidate": "Zoë Müller"}]

    filepath = save_json(data, tmp_path, "unicode.json")

    raw_text = filepath.read_text(encoding="utf-8")
    assert "Zoë Müller" in raw_text
