"""
ollama_utils.py

Shared low level helpers for talking to a local Ollama model and getting a
clean JSON object back out of its reply.
"""

import re
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat" # The local Ollama API endpoint
# Ollama must be running  and the model must be loaded for this to work


def strip_think_tags(text: str) -> str:
    """DeepSeek-R1 (and similar reasoning models) wrap thinking in
    <think>...</think>. Remove that so we're left with just the final
    output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model's reply, just in case it
    added extra commentary before or after it."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(match.group(0))


def call_ollama_json(model: str, prompt: str, num_predict: int = 1024, keep_alive: int = 0):
    """
    Send a single user prompt to a local Ollama model and return the
    JSON object parsed out of its reply.

    Returns a (parsed, error) tuple:
      - On success: (dict, None)
      - On failure: ({}, "<human-readable error message>")

    keep_alive=0 unloads the model immediately after this request instead
    of leaving it resident which is its default behavior. This comes in handy
    when running many different models in a row so that we don't run out of RAM on the host machine.
    """
    try:
        # gets the models reply from the local Ollama API and raises an exception if the request fails
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
