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
# stop_local_models (Ctrl-C in the web app)
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_stop_state():
    """stop_local_models() flips a process-wide flag; put it back afterwards."""
    import ollama_utils
    ollama_utils._used_models.clear()
    ollama_utils._stopping.clear()
    yield ollama_utils
    ollama_utils._used_models.clear()
    ollama_utils._stopping.clear()


def _ps(*names):
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": n, "model": n} for n in names]}
    return resp


@patch("ollama_utils.requests.post")
def test_only_local_calls_are_remembered_for_shutdown(mock_post, fresh_stop_state):
    mock_post.return_value = _mock_response("ok")

    fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")
    fresh_stop_state.call_ollama_text("gpt-oss:120b", "p", keep_alive=None,
                                      host="https://ollama.com/api/chat")

    assert fresh_stop_state._used_models == {"http://localhost:11434": {"phi4"}}


@patch("ollama_utils.requests.get")
@patch("ollama_utils.requests.post")
def test_stop_unloads_only_our_loaded_models(mock_post, mock_get, fresh_stop_state):
    mock_post.return_value = _mock_response("ok")
    fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")
    fresh_stop_state.call_ollama_text("qwen2.5:14b", "p", keep_alive="30m")
    mock_post.reset_mock()
    # phi4 is loaded (as phi4:latest), qwen2.5:14b is not, and llama3 is someone else's.
    mock_get.return_value = _ps("phi4:latest", "llama3:8b")

    stopped = fresh_stop_state.stop_local_models()

    assert stopped == ["phi4:latest"]
    mock_get.assert_called_once()
    assert mock_get.call_args.args[0] == "http://localhost:11434/api/ps"
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "http://localhost:11434/api/generate"
    assert mock_post.call_args.kwargs["json"] == {"model": "phi4:latest", "keep_alive": 0}


@patch("ollama_utils.requests.post")
def test_calls_after_stop_are_cancelled_without_reaching_ollama(mock_post, fresh_stop_state):
    with patch("ollama_utils.requests.get", return_value=_ps()):
        fresh_stop_state.stop_local_models()

    text, error = fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")

    assert text == ""
    assert error.startswith("CANCELLED")
    mock_post.assert_not_called()


@patch("ollama_utils.requests.get")
@patch("ollama_utils.requests.post")
def test_stop_skips_a_daemon_that_cannot_be_reached(mock_post, mock_get, fresh_stop_state):
    mock_post.return_value = _mock_response("ok")
    fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")
    mock_get.side_effect = requests.exceptions.ConnectionError("down")

    assert fresh_stop_state.stop_local_models() == []




# ---------------------------------------------------------------------------
# unload_local_models (the end of an ordinary run)
# ---------------------------------------------------------------------------
#
# A finished run used to leave its models resident for their keep_alive - half
# an hour and about ten gigabytes for the rewriter's 14B - because nothing
# unloaded them until Ctrl-C. stop_local_models() could not be reused for it:
# it also refuses every later call, which would leave the app unable to run
# anything a second time.


@patch("ollama_utils.requests.get")
@patch("ollama_utils.requests.post")
def test_unload_frees_our_loaded_models(mock_post, mock_get, fresh_stop_state):
    mock_post.return_value = _mock_response("ok")
    fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")
    mock_post.reset_mock()
    mock_get.return_value = _ps("phi4:latest", "llama3:8b")

    assert fresh_stop_state.unload_local_models() == ["phi4:latest"]
    assert mock_post.call_args.kwargs["json"] == {"model": "phi4:latest",
                                                  "keep_alive": 0}


@patch("ollama_utils.requests.post")
def test_unload_leaves_the_app_able_to_run_again(mock_post, fresh_stop_state):
    """The difference from stop_local_models(). A run that ends must not take
    the next one with it."""
    mock_post.return_value = _mock_response("ok")
    with patch("ollama_utils.requests.get", return_value=_ps()):
        fresh_stop_state.unload_local_models()

    text, error = fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")

    assert error is None and text == "ok"


@patch("ollama_utils.requests.get")
@patch("ollama_utils.requests.post")
def test_stop_still_unloads_and_still_cancels(mock_post, mock_get,
                                              fresh_stop_state):
    """Ctrl-C keeps both halves: unload, and refuse everything after."""
    mock_post.return_value = _mock_response("ok")
    fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")
    mock_get.return_value = _ps("phi4:latest")

    assert fresh_stop_state.stop_local_models() == ["phi4:latest"]

    _, error = fresh_stop_state.call_ollama_text("phi4", "p", keep_alive="30m")
    assert error.startswith("CANCELLED")


def test_note_local_model_records_a_call_this_module_did_not_make(
        fresh_stop_state):
    """The rewriter's probe reaches Ollama through its own urllib call, so it
    says so here. Without this its models were invisible to the unloader, and
    Ctrl-C reported stopping the run's models while leaving the largest one
    resident."""
    fresh_stop_state.note_local_model("qwen2.5:14b",
                                      "http://localhost:11434/api/generate")

    assert fresh_stop_state._used_models == {
        "http://localhost:11434": {"qwen2.5:14b"}}


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
