"""Where the pipeline's collaborators come from.

The console app's _integration() looked for an integration/ directory holding
the evaluator code vendored from feature branches. The September reorg committed
that code to evaluator/ instead, and left _integration() pointing at a path
that no longer exists. This is its replacement.

The second reorg then split that code again: the modules shared by both
evaluator/ and rewriter/ - document_utils, ollama_utils, results_io,
fact_guard - moved to common/, and the CV parser to parser/cv_pdf_parser.py.
So the modules this resolves against now come from four directories, not two,
and common/ is the one that must exist for anything here to import at all.

Runners take a Deps rather than importing any of this directly, so the test
suite never loads the evaluator modules and never reaches Ollama.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ui/pipeline/<module>.py -> ui/pipeline -> ui -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent


class MissingTeamCode(RuntimeError):
    """One of common/, evaluator/, parser/ or rewriter/ is not where it should be."""


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
    common = root / "common"
    evaluator = root / "evaluator"
    parser = root / "parser"
    rewriter = root / "rewriter"
    for directory in (common, evaluator, parser, rewriter):
        if not directory.is_dir():
            raise MissingTeamCode(
                "{} not found under {}. The shared modules, evaluator, judge, "
                "parser and rewriter live there since the September "
                "reorg.".format(directory.name, root))

    for directory in (common, evaluator, parser, rewriter, root):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    import document_utils
    import Evaluate_CV
    import essential_requirements
    import judge as judge_mod
    import rewriter_v1

    def judge_fn(job_text, rows, candidate, model, essential=None):
        # The command line's prompt and call settings, through the one
        # function both use - so a web app score and a batch score of the
        # same CV come from the same judging. It returns an error rather than
        # exiting when Ollama is unreachable.
        return judge_mod.judge_candidate(job_text, rows, candidate, essential,
                                         model=model)

    return Deps(load_cv=document_utils.load_cv,
                extract_text=document_utils.extract_text_from_pdf,
                evaluate_cv=Evaluate_CV.evaluate_cv,
                check_essential=essential_requirements.evaluate_essential_requirement,
                judge=judge_fn,
                probe=rewriter_v1.probe,
                make_scorer=rewriter_v1.make_scorer,
                paraphrase=rewriter_v1.paraphrase)
