"""
Unit tests for essential_requirements.py

Covers:
- build_essential_requirement_prompt(): the prompt sent to the model
- evaluate_essential_requirement(): success and failure paths, with the
  actual model call (call_ollama_json) mocked out
- to_judge_payload(): reshaping results for a later step
"""

from unittest.mock import patch

import essential_requirements as er


# ---------------------------------------------------------------------------
# build_essential_requirement_prompt
# ---------------------------------------------------------------------------

def test_prompt_includes_job_text_and_cv_sections():
    prompt = er.build_essential_requirement_prompt(
        job_text="Looking for a Python developer with 3+ years experience.",
        cv_sections_text="Experience: 4 years building Python services.",
    )

    assert "Looking for a Python developer with 3+ years experience." in prompt
    assert "Experience: 4 years building Python services." in prompt


def test_prompt_asks_for_json_only_response():
    prompt = er.build_essential_requirement_prompt("job", "cv")
    assert '"summary"' in prompt


# ---------------------------------------------------------------------------
# evaluate_essential_requirement
# ---------------------------------------------------------------------------

@patch("essential_requirements.call_ollama_json")
def test_evaluate_essential_requirement_success_adds_candidate_name(mock_call):
    mock_call.return_value = ({"summary": "Meets most requirements."}, None)

    result = er.evaluate_essential_requirement(
        job_text="job text", cv_sections_text="cv text", candidate_name="Alice"
    )

    assert result == {"summary": "Meets most requirements.", "candidate": "Alice"}


@patch("essential_requirements.call_ollama_json")
def test_evaluate_essential_requirement_uses_default_model(mock_call):
    mock_call.return_value = ({"summary": "ok"}, None)

    er.evaluate_essential_requirement("job text", "cv text", "Alice")

    called_model = mock_call.call_args.args[0]
    assert called_model == er.DEFAULT_MODEL


@patch("essential_requirements.call_ollama_json")
def test_evaluate_essential_requirement_respects_explicit_model(mock_call):
    mock_call.return_value = ({"summary": "ok"}, None)

    er.evaluate_essential_requirement("job text", "cv text", "Alice", model="phi4")

    called_model = mock_call.call_args.args[0]
    assert called_model == "phi4"


@patch("essential_requirements.call_ollama_json")
def test_evaluate_essential_requirement_on_model_failure_returns_error_dict(mock_call):
    mock_call.return_value = ({}, "PARSE ERROR")

    result = er.evaluate_essential_requirement("job text", "cv text", "Bob")

    assert result["candidate"] == "Bob"
    assert result["summary"] == ""
    assert result["error"] == "PARSE ERROR"


# ---------------------------------------------------------------------------
# to_judge_payload
# ---------------------------------------------------------------------------

def test_to_judge_payload_keeps_only_candidate_and_summary():
    results = [
        {"candidate": "Alice", "summary": "Good fit.", "error": None},
        {"candidate": "Bob", "summary": "Partial fit."},
    ]

    payload = er.to_judge_payload(results)

    assert payload == [
        {"candidate": "Alice", "summary": "Good fit."},
        {"candidate": "Bob", "summary": "Partial fit."},
    ]


def test_to_judge_payload_defaults_missing_summary_to_empty_string():
    results = [{"candidate": "Carol", "error": "CV PARSE ERROR: bad file"}]

    payload = er.to_judge_payload(results)

    assert payload == [{"candidate": "Carol", "summary": ""}]


def test_to_judge_payload_handles_empty_list():
    assert er.to_judge_payload([]) == []
