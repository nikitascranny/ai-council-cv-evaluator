"""Which models a run uses, and where they are called.

Every model name and every host in reviewer mode resolves through this module,
so pointing a batch run at a hosted Ollama is a config change rather than a
rewrite. Nothing here calls a cloud endpoint. Candidate mode's rewriter v2 is
the exception: its local agents and its Ollama Cloud agents are named in
rewriter/rewriter_v2.py (AGENT_MODELS, CLOUD_AGENTS), next to the code that
calls them.
"""

import json

OLLAMA_HOST = "http://localhost:11434"

# The production models.
COUNCIL_MODELS = ["qwen2.5:14b", "phi4"]
ESSENTIAL_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "deepseek-r1:8b"
REWRITER_MODEL = "qwen2.5:14b"


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

    An explicitly chosen council or judge wins over the defaults.
    """

    def __init__(self, essential=False, rewrite=False,
                 council=None, judge=None, host=None):
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
        return cls(essential=bool(form.get("essential")),
                   rewrite=bool(form.get("rewrite")),
                   council=council, judge=judge)

    @property
    def council(self):
        return self._council or COUNCIL_MODELS

    @property
    def judge(self):
        return self._judge or JUDGE_MODEL

    @property
    def essential_model(self):
        return ESSENTIAL_MODEL

    @property
    def rewriter_model(self):
        return REWRITER_MODEL

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
        return {"essential": self.essential,
                "rewrite": self.rewrite, "council": self.council,
                "judge": self.judge, "stages": self.stages()}
