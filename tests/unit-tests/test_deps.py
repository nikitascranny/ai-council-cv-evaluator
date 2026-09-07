"""The seam between our runner and the team's modules.

The console app looked for integration/, which the September reorg replaced
with the committed evaluator/. Every runner takes a Deps so tests never import
the team's code or reach a model.
"""

import pytest

from ui.pipeline.deps import Deps, MissingTeamCode, team_deps


def test_deps_can_be_built_entirely_from_fakes():
    """This is the shape every runner test relies on."""
    d = Deps(load_cv=lambda p: "cv", extract_text=lambda p: "job",
             evaluate_cv=lambda *a: {}, check_essential=lambda *a, **k: {},
             judge=lambda *a: ({}, None), probe=lambda *a, **k: {},
             make_scorer=lambda m: (lambda t: 0), paraphrase=lambda *a, **k: "")
    assert d.load_cv("x") == "cv"
    assert d.extract_text("x") == "job"


def test_team_deps_points_at_evaluator_not_integration(tmp_path):
    """The reorg moved the team's code; the old path is a dead directory."""
    with pytest.raises(MissingTeamCode, match="evaluator"):
        team_deps(root=tmp_path)


def test_the_error_names_the_directory_it_looked_in(tmp_path):
    with pytest.raises(MissingTeamCode) as exc:
        team_deps(root=tmp_path)
    assert str(tmp_path) in str(exc.value)


def test_team_deps_loads_the_real_modules_on_this_checkout():
    """Guards the import wiring itself - no model is called."""
    d = team_deps()
    assert callable(d.evaluate_cv) and callable(d.load_cv)
