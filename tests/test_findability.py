"""Unit tests for deterministic findability signals."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv_parser  # noqa: E402
from cv_parser import Block  # noqa: E402

RIAAN = ROOT / "Batch1" / "CVs" / "RIAAN GOVEN.pdf"


def section(label, heading, body, raw_heading=None):
    return cv_parser.Section(
        label=label, heading=heading, raw_heading=raw_heading,
        blocks=[Block(text=body, page=0, bbox=(0.0, 0.0, 100.0, 10.0),
                      max_size=10.0, bold=False)])


def test_well_structured_cv_scores_fully_headed():
    cv = cv_parser.ParsedCV(path=RIAAN, sections=[
        section("skills", "Skills", "Python, SQL"),
        section("experience", "Work Experience", "Engineer at Acme"),
    ])
    s = cv_parser.signals(cv)
    assert s["headed_fraction"] == 1.0
    assert s["has_skills_section"] is True
    assert s["section_count"] == 2
    assert s["unlabelled_sections"] == 0


def test_wall_of_prose_scores_unheaded():
    """A CV with no headings is the worst case for findability, and today the
    models have to infer that from the text alone."""
    cv = cv_parser.ParsedCV(path=RIAAN, sections=[
        section("summary", None, "I have done many things across many years"),
    ])
    s = cv_parser.signals(cv)
    assert s["headed_fraction"] == 0.0
    assert s["has_skills_section"] is False


def test_unlabelled_headings_are_counted():
    cv = cv_parser.ParsedCV(path=RIAAN, sections=[
        section("skills", "Skills", "Python"),
        section("other", "Referees and Hobbies", "On request",
                raw_heading="Referees and Hobbies"),
    ])
    assert cv_parser.signals(cv)["unlabelled_sections"] == 1


def test_largest_section_fraction_flags_a_single_dominant_block():
    cv = cv_parser.ParsedCV(path=RIAAN, sections=[
        section("summary", "Profile", "x" * 900),
        section("skills", "Skills", "y" * 100),
    ])
    s = cv_parser.signals(cv)
    assert s["largest_section_fraction"] > 0.8


def test_length_signals_are_measured_not_constant():
    """Pins the two numeric signals to exact values.

    Every other assertion in this file is one-sided or absent:
    largest_section_fraction is only ever checked as "> 0.8" on a deliberately
    lopsided CV, and mean_section_chars is never checked at all. A mutant
    returning a constant 1.0 for the fraction and a constant 0.0 for the mean
    passes the whole file. Both sections here are built to exactly 101 chars
    ("Profile\\n" + 93, "Skills\\n" + 94), so the correct answers are 101.0 and
    0.5 - values a constant cannot produce, in either direction.
    """
    even = cv_parser.ParsedCV(path=RIAAN, sections=[
        section("summary", "Profile", "x" * 93),
        section("skills", "Skills", "y" * 94),
    ])
    assert [len(s.text()) for s in even.sections] == [101, 101], \
        "fixture drifted: the two sections must be equal length"
    s = cv_parser.signals(even)
    assert s["mean_section_chars"] == 101.0
    assert s["largest_section_fraction"] == 0.5

    lopsided = cv_parser.ParsedCV(path=RIAAN, sections=[
        section("summary", "Profile", "x" * 900),
        section("skills", "Skills", "y" * 100),
    ])
    t = cv_parser.signals(lopsided)
    assert t["mean_section_chars"] == 507.5
    assert t["largest_section_fraction"] == pytest.approx(908 / 1015)
    assert t["largest_section_fraction"] < 1.0, \
        "a two-section CV can never have one section holding all of it"


def test_empty_cv_does_not_divide_by_zero():
    cv = cv_parser.ParsedCV(path=RIAAN, sections=[])
    s = cv_parser.signals(cv)
    assert s["section_count"] == 0
    assert s["headed_fraction"] == 0.0
    assert s["largest_section_fraction"] == 0.0


def test_signals_on_a_real_cv():
    cv = cv_parser.parse(RIAAN)
    s = cv_parser.signals(cv)
    assert 0.0 <= s["headed_fraction"] <= 1.0
    assert s["section_count"] > 0


def test_nested_headings_count_as_unlabelled():
    """The signal must see what the parser did not understand. Before this,
    a CV with three mislabelled headings reported unlabelled_sections == 0."""
    sec = section("experience", "Work Experience", "body text")
    sec.nested_headings = ["ADDITIONAL", "TUTORING EXPERIENCE"]
    cv = cv_parser.ParsedCV(path=RIAAN, sections=[sec])
    assert cv_parser.signals(cv)["unlabelled_sections"] == 2
