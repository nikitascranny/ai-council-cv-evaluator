"""
Unit tests for ollama_utils.py

Covers:
- strip_think_tags(): removing <think>...</think> reasoning blocks
- extract_json(): pulling a JSON object out of a noisy model reply
- call_ollama_json(): the full request/parse round trip, with the actual
  HTTP call mocked out so these tests don't need a real Ollama server
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from ollama_utils import strip_think_tags, extract_json, call_ollama_json


# ---------------------------------------------------------------------------
# strip_think_tags
# ---------------------------------------------------------------------------

def test_strip_think_tags_removes_single_block():
    text = "<think>reasoning about the candidate</think>{\"score\": 8}"
    assert strip_think_tags(text) == '{"score": 8}'


def test_strip_think_tags_removes_multiple_blocks():
    text = "<think>first</think>middle<think>second</think>end"
    assert strip_think_tags(text) == "middleend"


def test_strip_think_tags_handles_multiline_reasoning():
    text = "<think>\nline one\nline two\n</think>\n{\"ok\": true}"
    assert strip_think_tags(text) == '{"ok": true}'


def test_strip_think_tags_no_tags_returns_stripped_text():
    text = "  just a plain reply with no reasoning block  "
    assert strip_think_tags(text) == "just a plain reply with no reasoning block"


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_extract_json_parses_clean_object():
    result = extract_json('{"score": 7, "summary": "solid"}')
    assert result == {"score": 7, "summary": "solid"}


def test_extract_json_ignores_surrounding_commentary():
    text = 'Sure, here is the result:\n{"score": 5}\nHope that helps!'
    result = extract_json(text)
    assert result == {"score": 5}


def test_extract_json_raises_when_no_object_present():
    with pytest.raises(ValueError):
        extract_json("no JSON here at all")


# ---------------------------------------------------------------------------
# call_ollama_json
# ---------------------------------------------------------------------------

def _mock_response(payload_content: str):
    """Build a fake `requests` Response whose .json() matches the shape
    Ollama's /api/chat endpoint returns."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": payload_content}}
    return mock_resp


@patch("ollama_utils.requests.post")
def test_call_ollama_json_success_returns_parsed_dict(mock_post):
    mock_post.return_value = _mock_response('{"score": 9, "summary": "great fit"}')

    parsed, error = call_ollama_json("qwen2.5:7b", "some prompt")

    assert error is None
    assert parsed == {"score": 9, "summary": "great fit"}


@patch("ollama_utils.requests.post")
def test_call_ollama_json_strips_think_tags_before_parsing(mock_post):
    mock_post.return_value = _mock_response(
        '<think>let me consider this</think>{"score": 4}'
    )

    parsed, error = call_ollama_json("qwen2.5:7b", "some prompt")

    assert error is None
    assert parsed == {"score": 4}


@patch("ollama_utils.requests.post")
def test_call_ollama_json_sends_expected_request_body(mock_post):
    mock_post.return_value = _mock_response('{"ok": true}')

    call_ollama_json("phi4", "evaluate this candidate", num_predict=512, keep_alive=30)

    sent_kwargs = mock_post.call_args.kwargs
    assert sent_kwargs["json"]["model"] == "phi4"
    assert sent_kwargs["json"]["messages"] == [
        {"role": "user", "content": "evaluate this candidate"}
    ]
    assert sent_kwargs["json"]["options"]["num_predict"] == 512
    assert sent_kwargs["json"]["keep_alive"] == 30
    assert sent_kwargs["json"]["stream"] is False


@patch("ollama_utils.requests.post")
def test_call_ollama_json_http_error_is_reported_not_raised(mock_post):
    mock_post.side_effect = requests.exceptions.HTTPError("connection refused")

    parsed, error = call_ollama_json("qwen2.5:7b", "some prompt")

    assert parsed == {}
    assert error is not None
    assert "REQUEST ERROR" in error


@patch("ollama_utils.requests.post")
def test_call_ollama_json_malformed_reply_gives_parse_error(mock_post):
    mock_post.return_value = _mock_response("this is not JSON at all")

    parsed, error = call_ollama_json("qwen2.5:7b", "some prompt")

    assert parsed == {}
    assert error == "PARSE ERROR"


# ---------------------------------------------------------------------------
# load_env_file
# ---------------------------------------------------------------------------

def test_load_env_file_sets_unset_variables_only(tmp_path, monkeypatch):
    import os
    from ollama_utils import load_env_file

    env = tmp_path / ".env"
    env.write_text('# a comment\n\nOLLAMA_API_KEY="abc123"\nexport OTHER_TEST_VAR=x\n'
                   "ALREADY_SET_TEST_VAR=from-file\nnot a variable\n", encoding="utf-8")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_TEST_VAR", raising=False)
    monkeypatch.setenv("ALREADY_SET_TEST_VAR", "from-terminal")

    loaded = load_env_file(env)

    assert sorted(loaded) == ["OLLAMA_API_KEY", "OTHER_TEST_VAR"]
    assert os.environ["OLLAMA_API_KEY"] == "abc123"
    assert os.environ["OTHER_TEST_VAR"] == "x"
    assert os.environ["ALREADY_SET_TEST_VAR"] == "from-terminal"


def test_load_env_file_without_a_file_does_nothing(tmp_path):
    from ollama_utils import load_env_file

    assert load_env_file(tmp_path / "missing.env") == []
