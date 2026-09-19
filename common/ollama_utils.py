"""
ollama_utils.py

Shared low-level helpers for talking to a local Ollama model and getting a
clean JSON object back out of its reply. Both Evaluate_CV.py (recruiter-
style scoring) and essential_requirements.py (essential-requirements
summary) call through this module instead of each keeping their own copy
of the same request/parse/error-handling logic.
"""

import os
import re
import json
import threading
import requests
from pathlib import Path
from typing import Union

OLLAMA_URL = "http://localhost:11434/api/chat"

# What stop_local_models() needs to shut this process's models down: the
# daemons this process has asked to run a local model, and which models. Cloud
# calls (keep_alive=None) are not recorded - there is nothing loaded locally.
_used_models: dict = {}
_used_lock = threading.Lock()
_stopping = threading.Event()

# Where the command-line scripts look for an Ollama Cloud key kept in a file
# rather than set in the terminal. Git-ignored (.gitignore: .env).
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path = ENV_FILE) -> list:
    """Copy KEY=VALUE lines from a .env file into os.environ, for variables
    not already set - one set in the terminal wins over the file.

    Blank lines and # comments are skipped, an "export " prefix is allowed,
    and quotes around a value are removed. A missing file is not an error:
    the key may be set in the terminal instead. Returns the names it set.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    loaded = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


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


def _error_detail(response) -> str:
    """The "error" field of a failed Ollama reply, or "" if there is none."""
    if response is None:
        return ""
    try:
        detail = response.json().get("error", "")
    except Exception:
        return ""
    return detail if isinstance(detail, str) else ""


def _daemon_base(host: str) -> str:
    """http://localhost:11434 from http://localhost:11434/api/chat."""
    return host.split("/api/")[0].rstrip("/")


def _tagged(name: str) -> str:
    """Ollama lists "phi4" as "phi4:latest"; compare on the tagged form."""
    return name if ":" in name else name + ":latest"


def stop_local_models(timeout: float = 3) -> list:
    """Stop this process's models on the local Ollama daemon, for Ctrl-C.

    Exiting the process drops any request in flight and Ollama cancels it, but
    the model itself stays loaded for its keep_alive (30 minutes for the
    rewriter) - what `ollama ps` shows as still running. keep_alive=0 is what
    `ollama stop` sends; a model still busy unloads as soon as its request
    ends. Only models this process asked for and that are loaded right now are
    touched, so another program's models are left alone and nothing gets
    loaded just to be unloaded.

    Also makes every later call return a CANCELLED error at once. Returns the
    names it stopped; a daemon that cannot be reached is skipped, not an error.
    """
    _stopping.set()
    with _used_lock:
        used = {base: set(models) for base, models in _used_models.items()}
    stopped = []
    for base, models in used.items():
        wanted = {_tagged(m) for m in models}
        try:
            loaded = requests.get(base + "/api/ps", timeout=timeout).json().get("models", [])
        except (requests.exceptions.RequestException, ValueError):
            continue
        for entry in loaded:
            name = entry.get("model") or entry.get("name")
            if not name or _tagged(name) not in wanted:
                continue
            try:
                requests.post(base + "/api/generate", timeout=timeout,
                              json={"model": name, "keep_alive": 0})
            except requests.exceptions.RequestException:
                continue
            stopped.append(name)
    return stopped


def call_ollama_json(model: str,
                     prompt: str,
                     num_predict: int = 1024,
                     keep_alive=0, # keep_alive can be str or int. int= in seconds. str e.g. 5m = 5 minutes
                     host: str = OLLAMA_URL,
                     timeout: int=3600, # guard against hung request (alive but stuck process). To refactor in evaluate files.
                     num_ctx: int = None, # context window size in tokens; None leaves Ollama's default (usually 4096)
                     headers: dict = None, # passed through to call_ollama_text, e.g. for Ollama Cloud
                     think=None,
                     stats: dict = None,
                    ):

    """
    Send a single user-turn prompt to an Ollama model and return the
    JSON object parsed out of its reply. headers/think/stats are as in
    call_ollama_text, so a cloud call works the same way here.

    Returns a (parsed, error) tuple:
      - On success: (dict, None)
      - On failure: ({}, "<human-readable error message>")

    keep_alive=0 unloads the model immediately after this request instead
    of leaving it resident for Ollama's default 5-minute keep-alive.
    This is useful when the pipeline cycles through several different models back to back.

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
        timeout=timeout,
        num_ctx=num_ctx,
        headers=headers,
        think=think,
        stats=stats,
    )

    if error:
        return {}, error

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
                     timeout: int=3600, # guard against hung request (alive but stuck process). To refactor in evaluate files.
                     num_ctx: int = None, # context window size in tokens; None leaves Ollama's default (usually 4096)
                     headers: dict = None, # e.g. {"Authorization": "Bearer <key>"} for Ollama Cloud
                     think=None, # reasoning level for models that take one, e.g. "low" for gpt-oss; None sends nothing
                     stats: dict = None, # if given, filled with the reply's token counts and done_reason
                    ):

    """
    Send a single user-turn prompt to an Ollama model and return its
    raw text reply - no JSON parsing. For tasks where the model's output
    IS the deliverable (e.g. a rewritten CV), not a JSON envelope around it.

    host defaults to the local daemon. Ollama Cloud (https://ollama.com/api/chat)
    takes the same request body, so a cloud call is the same call with a
    different host and an Authorization header - and keep_alive=None, since
    there is no local model to keep loaded.

    Returns a (text, error) tuple, matching call_ollama_json's shape:
      - On success: (str, None)
      - On failure: ("", "<human-readable error message>")

    An empty reply (e.g. a reasoning model that used its whole budget on
    <think> tags and produced nothing after) is treated as a failure
    rather than a valid empty rewrite - silently accepting "" here would
    make a blank CV look like it passed every downstream check.
    """
    if _stopping.is_set():
        # Ctrl-C arrived while a run was between two model calls. Starting the
        # next one would load a model just after stop_local_models() unloaded it.
        return "", "CANCELLED: the app is shutting down"
    options: dict[str, Union[int, float]] = {"num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
        with _used_lock:
            _used_models.setdefault(_daemon_base(host), set()).add(model)
    if think is not None:
        body["think"] = think
    extra = {"headers": headers} if headers else {}
    try:
        # gets the models reply from the Ollama API and raises an exception if the request fails
        response = requests.post(
            host,
            json=body,
            timeout=timeout, ## Use to avoid hung request
            **extra,
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e: # includes HTTPError
        # Catches connection failures (Ollama not running), timeouts, and HTTP error statuses alike,
        # so callers always get back a (dict,str) pair instead of an uncaught exception killing the run.
        error = f"REQUEST ERROR: {e}"
        detail = _error_detail(getattr(e, "response", None))
        if detail:
            # Ollama says why in the body ({"error": "..."}), which the status
            # line alone does not - "429 Too Many Requests" could be the plan's
            # concurrency limit or its usage running out.
            error += f" ({detail})"
        return "", error

    reply = response.json()
    if stats is not None:
        stats.update({"prompt_eval_count": reply.get("prompt_eval_count"),
                      "eval_count": reply.get("eval_count"),
                      "done_reason": reply.get("done_reason")})
    raw_reply = reply["message"]["content"]
    cleaned = strip_think_tags(raw_reply)

    if not cleaned:
        return "", "EMPTY RESPONSE"

    return cleaned, None
