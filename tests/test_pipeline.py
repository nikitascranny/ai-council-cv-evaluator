"""Unit tests for the pure stages.

These cover the parsing and aggregation logic without invoking a model, which is
the point of separating stage 2 (evaluate) from stage 3 (aggregate): the logic
that decides rankings is deterministically testable.

Run:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "analysis"))

from run_evaluations import parse_response          # noqa: E402
from compare_models import rank_from_scores, normalise_model_name  # noqa: E402


# --- parse_response -------------------------------------------------------

def test_plain_json():
    parsed, err = parse_response('{"score": 7, "candidate": "ALEX PARK"}')
    assert err is None
    assert parsed["score"] == 7


def test_markdown_fenced():
    """Most instruct models wrap JSON in a code fence despite being told not to."""
    parsed, err = parse_response('```json\n{"score": 3, "candidate": "THE DANIEL"}\n```')
    assert err is None
    assert parsed["score"] == 3


def test_reasoning_model_think_block():
    """deepseek-r1 emits <think> before the answer. This is the leading suspect
    for its null scores in the original Batch1 runs."""
    raw = '<think>\nThe candidate lists Go... wait, no he does not.\n</think>\n{"score": 2}'
    parsed, err = parse_response(raw)
    assert err is None
    assert parsed["score"] == 2


def test_prose_wrapped_json():
    raw = 'Here is my assessment:\n{"score": 5, "summary": "Mixed fit"}\nHope that helps!'
    parsed, err = parse_response(raw)
    assert err is None
    assert parsed["score"] == 5


def test_unparseable_reports_error_rather_than_raising():
    """A bad response must become a recorded, diagnosable row - never an
    exception that kills a multi-hour run, and never a silently dropped vote."""
    parsed, err = parse_response("I cannot evaluate this candidate.")
    assert parsed is None
    assert "no JSON object" in err


def test_truncated_json_reports_error():
    parsed, err = parse_response('{"score": 8, "strengths": ["good at')
    assert parsed is None
    assert err is not None


# --- rank_from_scores -----------------------------------------------------

def test_higher_score_ranks_first():
    assert rank_from_scores({"A": 9, "B": 5, "C": 1}) == {"A": 1, "B": 2, "C": 3}


def test_ties_share_average_rank():
    """mistral:7b gave many candidates identical scores; tied candidates must not
    be ordered arbitrarily, or rank correlation rewards dictionary insertion order."""
    assert rank_from_scores({"A": 5, "B": 5, "C": 1}) == {"A": 1.5, "B": 1.5, "C": 3}


def test_all_tied():
    """llama-3.3-70b gave all 10 aquarium candidates a 2."""
    assert rank_from_scores({"A": 2, "B": 2, "C": 2}) == {"A": 2, "B": 2, "C": 2}


def test_single_candidate():
    assert rank_from_scores({"A": 4}) == {"A": 1}


# --- normalise_model_name -------------------------------------------------

def test_inconsistent_qwen_filenames_map_together():
    """The two job folders spell the same model differently; if these do not
    unify, qwen appears as two models each with half its data."""
    assert normalise_model_name("qwen2.57b.json") == "qwen2.5:7b"
    assert normalise_model_name("qwen2.5_7b.json") == "qwen2.5:7b"


def test_other_models_get_colon_form():
    assert normalise_model_name("llama3.1_8b.json") == "llama3.1:8b"
    assert normalise_model_name("deepseek-r1_7b.json") == "deepseek-r1:7b"
