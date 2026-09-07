"""Where the pipeline's collaborators come from.

The console app's _integration() looked for an integration/ directory holding
the team's code vendored from their branches. The September reorg committed
that code to evaluator/ instead, and left _integration() pointing at a path
that no longer exists. This is its replacement.

Runners take a Deps rather than importing any of this directly, so the test
suite never loads the team's modules and never reaches Ollama.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ui/pipeline/<module>.py -> ui/pipeline -> ui -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent


class MissingTeamCode(RuntimeError):
    """evaluator/ or rewriter/ is not where it should be."""


@dataclass
class Deps:
    load_cv: Callable
    extract_text: Callable
    evaluate_cv: Callable
    check_essential: Callable
    judge: Callable
    probe: Callable
    make_scorer: Callable
    paraphrase: Callable


def team_deps(root: Path = ROOT) -> Deps:
    root = Path(root)
    evaluator = root / "evaluator"
    rewriter = root / "rewriter"
    for directory in (evaluator, rewriter):
        if not directory.is_dir():
            raise MissingTeamCode(
                "{} not found under {}. The evaluator, judge and rewriter live "
                "there since the September reorg.".format(directory.name, root))

    for directory in (evaluator, rewriter, root):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    import document_utils
    import Evaluate_CV
    import essential_requirements
    import judge as judge_mod
    import ollama_utils
    import rewriter_v1

    def judge_fn(job_text, rows, candidate, model):
        summaries = judge_mod.format_summaries_for_prompt(rows)
        prompt = judge_mod.build_prompt(job_text, summaries, candidate)
        return ollama_utils.call_ollama_json(model, prompt, num_predict=1024)

    return Deps(load_cv=document_utils.load_cv,
                extract_text=document_utils.extract_text_from_pdf,
                evaluate_cv=Evaluate_CV.evaluate_cv,
                check_essential=essential_requirements.evaluate_essential_requirement,
                judge=judge_fn,
                probe=rewriter_v1.probe,
                make_scorer=rewriter_v1.make_scorer,
                paraphrase=rewriter_v1.paraphrase)
