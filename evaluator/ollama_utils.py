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


def call_ollama_json(model: str, prompt: str, num_predict: int = 1024, keep_alive: int = 0):
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
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": num_predict},
                "keep_alive": keep_alive,
            },
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return {}, f"REQUEST ERROR: {e}"

    raw_reply = response.json()["message"]["content"]
    cleaned = strip_think_tags(raw_reply)

    try:
        return extract_json(cleaned), None
    except (ValueError, json.JSONDecodeError):
        return {}, "PARSE ERROR"


def call_ollama_text(model: str, prompt: str, num_predict: int = 1024, keep_alive: int = 0):
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
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": num_predict},
                "keep_alive": keep_alive,
            },
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return "", f"REQUEST ERROR: {e}"

    raw_reply = response.json()["message"]["content"]
    cleaned = strip_think_tags(raw_reply)

    if not cleaned:
        return "", "EMPTY RESPONSE"

    return cleaned, None
