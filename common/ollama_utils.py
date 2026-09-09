"""
ollama_utils.py

Shared low-level helpers for talking to a local Ollama model and getting a
clean JSON object back out of its reply. Both Evaluate_CV.py (recruiter-
style scoring) and essential_requirements.py (essential-requirements
summary) call through this module instead of each keeping their own copy
of the same request/parse/error-handling logic.
"""

import re
import json
import requests
from typing import Union

OLLAMA_URL = "http://localhost:11434/api/chat"


def strip_think_tags(text: str) -> str:
    """DeepSeek-R1 (and similar reasoning models) wrap thinking in
    <think>...</think>. Remove that so we're left with just the final
    output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model's reply, in case it
    added extra commentary before or after it."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(match.group(0))


def call_ollama_json(model: str,
                     prompt: str,
                     num_predict: int = 1024,
                     keep_alive=0, # keep_alive can be str or int. int= in seconds. str e.g. 5m = 5 minutes
                     host: str = OLLAMA_URL,
                     temperature=None, # float
                     timeout: int=3600, # guard against hung request (alive but stuck process). To refactor in evaluate files.
                     num_ctx: int = None, # context window size in tokens; None leaves Ollama's default (usually 4096)
                    ):

    """
    Send a single user-turn prompt to a local Ollama model and return the
    JSON object parsed out of its reply.

    Returns a (parsed, error) tuple:
      - On success: (dict, None)
      - On failure: ({}, "<human-readable error message>")

    keep_alive=0 unloads the model immediately after this request instead
    of leaving it resident for Ollama's default 5-minute keep-alive - useful
    when a pipeline cycles through several different models back to back.

    Callers are responsible for building a schema-appropriate fallback
    result around the error message - this function only knows about
    talking to Ollama, not about any particular caller's expected JSON
    shape (score/strengths/weaknesses vs. summary, etc).
    """
    
    parsed_text, error = call_ollama_text(
        model=model,
        prompt=prompt,
        num_predict=num_predict,
        keep_alive=keep_alive,
        host=host,
        temperature=temperature,
        timeout=timeout,
        num_ctx=num_ctx,
    )

    if error:
        return parsed_text, error

    try:
        parsed = extract_json(parsed_text)
    except (ValueError, json.JSONDecodeError):
        error = "PARSE ERROR"
        return ({},error)

    return (parsed,None)



def call_ollama_text(model: str,
                     prompt: str,
                     num_predict: int = 1024,
                     keep_alive=0, # keep_alive can be str or int. int= in seconds. str e.g. 5m = 5 minutes
                     host: str = OLLAMA_URL,
                     temperature=None, # float
                     timeout: int=3600, # guard against hung request (alive but stuck process). To refactor in evaluate files.
                     num_ctx: int = None, # context window size in tokens; None leaves Ollama's default (usually 4096)
                    ):

    """
    Send a single user-turn prompt to a local Ollama model and return its
    raw text reply - no JSON parsing. For tasks where the model's output
    IS the deliverable (e.g. a rewritten CV), not a JSON envelope around it.

    Returns a (text, error) tuple, matching call_ollama_json's shape:
      - On success: (str, None)
      - On failure: ("", "<human-readable error message>")

    An empty reply (e.g. a reasoning model that used its whole budget on
    <think> tags and produced nothing after) is treated as a failure
    rather than a valid empty rewrite - silently accepting "" here would
    make a blank CV look like it passed every downstream check.
    """
    options: dict[str, Union[int, float]] = {"num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    try:
        # gets the models reply from the local Ollama API and raises an exception if the request fails
        response = requests.post(
            host,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
                "keep_alive": keep_alive,
            },
            timeout=timeout ## Use to avoid hung request
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e: # includes HTTPError
        # Catches connection failures (Ollama not running), timeouts, and HTTP error statuses alike, 
        # so callers always get back a (dict,str) pair instead of an uncaught exception killing the run.
        error = f"REQUEST ERROR: {e}"
        return ({}, error)

    raw_reply = response.json()["message"]["content"]
    cleaned = strip_think_tags(raw_reply)

    if not cleaned:
        return "", "EMPTY RESPONSE"

    return cleaned, None
