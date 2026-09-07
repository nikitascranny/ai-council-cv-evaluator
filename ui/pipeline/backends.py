"""Which models a run uses, and where they are called.

Every model name and every host in the app resolves through this module, so
pointing a batch run at a hosted Ollama is a config change rather than a
rewrite. Nothing here calls a cloud endpoint today.
"""

import json

OLLAMA_HOST = "http://localhost:11434"

# The team's real models, and what fast mode falls back to.
COUNCIL_MODELS = ["qwen2.5:14b", "phi4"]
ESSENTIAL_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "deepseek-r1:8b"
REWRITER_MODEL = "qwen2.5:14b"

# 7b, not 3b. The 3b models score the same CV several points higher than the
# 14b ones and llama3.2:3b returned unparseable JSON on two of three runs, so
# anything measured on them says more about the model size than the candidate.
# The rewriter follows the same reasoning: it is the paraphrase itself, not a
# scorer, but it is still a qwen2.5 checkpoint sized the same way, and a
# memory-constrained machine that cannot load the 14b for the council cannot
# load it for the rewrite either - fast mode has to swap it too or the probe
# just fails instead of running small.
FAST_MODELS = {"council": ["qwen2.5:7b"],
               "essential": "qwen2.5:7b",
               "judge": "qwen2.5:7b",
               "rewriter": "qwen2.5:7b"}


def tags_url(host: str = None) -> str:
    return (host or OLLAMA_HOST).rstrip("/") + "/api/tags"


def available_models(fetch=None, host: str = None) -> set:
    """Model names Ollama currently has, empty if it cannot be reached."""
    if fetch is None:
        def fetch():
            import urllib.request
            with urllib.request.urlopen(tags_url(host), timeout=5) as response:
                return json.load(response)
    try:
        payload = fetch()
    except Exception:
        return set()

    names = set()
    for entry in payload.get("models", []):
        name = entry.get("name", "")
        if name:
            names.add(name)
            if name.endswith(":latest"):
                names.add(name.split(":")[0])
    return names


def missing_models(wanted, available) -> list:
    """Which of `wanted` Ollama does not have.

    An empty availability set means the check could not run, not that every
    model is gone - reporting all of them would be a false alarm before a demo.
    """
    if not available:
        return []
    return [m for m in wanted if m not in available]


class Settings:
    """What the next run will do.

    An explicitly chosen model wins over fast mode, so switching to fast to get
    through a slow stage does not silently discard a model the user picked.
    """

    def __init__(self, fast=False, essential=False, rewrite=False,
                 council=None, judge=None, host=None):
        self.fast = fast
        self.essential = essential
        self.rewrite = rewrite
        self.host = host or OLLAMA_HOST
        self._council = list(council) if council else None
        self._judge = judge

    @classmethod
    def from_form(cls, form: dict):
        """Build from a parsed query string, where every value is a list."""
        council = [m for m in form.get("council", []) if m]
        judge = next((j for j in form.get("judge", []) if j), None)
        return cls(fast=bool(form.get("fast")),
                   essential=bool(form.get("essential")),
                   rewrite=bool(form.get("rewrite")),
                   council=council, judge=judge)

    @property
    def council(self):
        if self._council:
            return self._council
        return FAST_MODELS["council"] if self.fast else COUNCIL_MODELS

    @property
    def judge(self):
        if self._judge:
            return self._judge
        return FAST_MODELS["judge"] if self.fast else JUDGE_MODEL

    @property
    def essential_model(self):
        return FAST_MODELS["essential"] if self.fast else ESSENTIAL_MODEL

    @property
    def rewriter_model(self):
        return FAST_MODELS["rewriter"] if self.fast else REWRITER_MODEL

    def stages(self) -> list:
        active = ["parse", "council"]
        if self.essential:
            active.append("essential requirements")
        active.append("judge")
        if self.rewrite:
            active.append("rewriter probe")
        return active

    def needed_models(self) -> list:
        models = list(self.council) + [self.judge]
        if self.essential:
            models.append(self.essential_model)
        if self.rewrite:
            models.append(self.rewriter_model)
        return models

    def to_dict(self) -> dict:
        return {"fast": self.fast, "essential": self.essential,
                "rewrite": self.rewrite, "council": self.council,
                "judge": self.judge, "stages": self.stages()}
