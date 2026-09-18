"""Which models a run uses, and whether Ollama has them.

Ported from the console app's Settings and preflight. from_form() is new: the
web form posts strings, and a half-parsed form must not silently produce a
different model set than the one shown on screen.
"""

from ui.pipeline.backends import (Settings, COUNCIL_MODELS, JUDGE_MODEL, ESSENTIAL_MODEL,
                           REWRITER_MODEL, available_models, missing_models,
                           tags_url)


def test_defaults_are_the_teams_real_models():
    s = Settings()
    assert s.council == COUNCIL_MODELS
    assert s.judge == JUDGE_MODEL
    assert s.essential is False and s.rewrite is False


def test_the_essential_and_rewriter_models_are_the_teams_real_models():
    assert Settings().essential_model == ESSENTIAL_MODEL
    assert Settings().rewriter_model == REWRITER_MODEL


def test_an_explicit_council_keeps_the_default_judge():
    s = Settings(council=["phi4"])
    assert s.council == ["phi4"]
    assert s.judge == JUDGE_MODEL


def test_stages_list_reflects_the_optional_toggles():
    assert Settings().stages() == ["parse", "council", "judge"]
    assert "essential requirements" in Settings(essential=True).stages()
    assert "rewriter probe" in Settings(rewrite=True).stages()


def test_needed_models_includes_the_essential_model_only_when_enabled():
    assert Settings(council=["a"], judge="b").needed_models() == ["a", "b"]
    assert Settings(council=["a"], judge="b", essential=True).needed_models() == [
        "a", "b", "qwen2.5:7b"]


def test_needed_models_includes_the_rewriter_model_only_when_rewrite_is_on():
    """A future preflight has to be able to see the rewriter model coming,
    or it will check Ollama for everything except the model that actually
    failed to load in production."""
    assert Settings(council=["a"], judge="b").needed_models() == ["a", "b"]
    assert Settings(council=["a"], judge="b", rewrite=True).needed_models() == [
        "a", "b", REWRITER_MODEL]


def test_from_form_reads_checkbox_and_multi_select_fields():
    s = Settings.from_form({"rewrite": ["on"],
                            "council": ["phi4", "qwen2.5:14b"], "judge": ["phi4"]})
    assert s.rewrite is True and s.essential is False
    assert s.council == ["phi4", "qwen2.5:14b"]
    assert s.judge == "phi4"


def test_from_form_with_nothing_selected_gives_the_defaults():
    s = Settings.from_form({})
    assert s.council == COUNCIL_MODELS and s.judge == JUDGE_MODEL


def test_available_models_registers_the_bare_name_alongside_the_tag():
    """Ollama says "phi4:latest" for what everything else calls "phi4"."""
    got = available_models(fetch=lambda: {"models": [{"name": "phi4:latest"}]})
    assert got == {"phi4:latest", "phi4"}


def test_unreachable_ollama_is_an_empty_set_not_an_exception():
    def boom():
        raise OSError("connection refused")
    assert available_models(fetch=boom) == set()


def test_missing_models_stays_quiet_when_the_check_could_not_run():
    """Empty availability means the check failed, not that every model is gone."""
    assert missing_models(["phi4"], set()) == []


def test_missing_models_lists_what_is_absent():
    assert missing_models(["phi4", "gone"], {"phi4"}) == ["gone"]


def test_tags_url_follows_the_configured_host():
    assert tags_url("http://cloud.example:9999") == "http://cloud.example:9999/api/tags"
