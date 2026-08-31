"""Tests for the variant 1 phrasing-sensitivity probe."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import rewriter_v1  # noqa: E402


def test_normalise_folds_case_and_punctuation():
    assert rewriter_v1.normalise("Head of  Digital, Product.") == "head of digital product"


def test_normalise_keeps_percentages():
    assert rewriter_v1.normalise("Grew revenue 72.8%") == "grew revenue 72.8%"


def test_extract_facts_finds_numbers_and_years():
    facts = rewriter_v1.extract_facts("Managed 14 staff from 2019 to 2023, up 72.8%.")
    assert "14" in facts
    assert "2019" in facts
    assert "2023" in facts
    assert "72.8%" in facts


def test_extract_facts_finds_capitalised_runs():
    facts = rewriter_v1.extract_facts("Senior Engineer at Umoja Bank in Cape Town.")
    assert "umoja bank" in facts
    assert "senior engineer" in facts


def test_extract_facts_finds_qualifications():
    facts = rewriter_v1.extract_facts("Holds a BSc and an AWS certification.")
    assert "bsc" in facts
    assert "aws" in facts


def test_extract_facts_ignores_single_capitalised_words():
    """A sentence-initial capital is not an entity - it would flag every rewrite.

    The positive assertion matters: without it this test passes against an
    extract_facts that always returns the empty set.
    """
    facts = rewriter_v1.extract_facts(
        "Managed the team. Delivered the project at Umoja Bank in 2019.")
    assert "managed" not in facts
    assert "delivered" not in facts
    assert {"umoja bank", "2019"} <= facts


def test_extract_hard_facts_are_numbers_and_qualifications_only():
    hard = rewriter_v1.extract_hard_facts(
        "Senior Engineer at Umoja Bank holds a BSc, managed 14 staff.")
    assert hard == {"14", "bsc"}


def test_extract_caps_runs_are_the_multi_word_entities():
    runs = rewriter_v1.extract_caps_runs(
        "Senior Engineer at Umoja Bank holds a BSc, managed 14 staff.")
    assert "umoja bank" in runs
    assert "senior engineer" in runs
    assert "14" not in runs


ORIGINAL = (
    "Senior Engineer at Umoja Bank from 2019 to 2023. "
    "Managed 14 staff and cut costs 12%."
)


def test_pure_paraphrase_passes():
    rewritten = (
        "Worked as a Senior Engineer with Umoja Bank between 2019 and 2023, "
        "leading 14 people and reducing costs by 12%."
    )
    assert rewriter_v1.check_no_new_facts(ORIGINAL, rewritten) == []


def test_invented_organisation_is_advisory_not_a_rejection():
    """Ruling A: caps runs are reported, never rejecting. Rephrasing re-cases
    headings and moves words to sentence-initial position, so this signal fired
    on honest paraphrases of real CVs and dominated the rejections."""
    rewritten = ORIGINAL + " Previously at Google Cloud."
    assert rewriter_v1.check_no_new_facts(ORIGINAL, rewritten) == []
    assert "google cloud" in rewriter_v1.advisory_caps_runs(ORIGINAL, rewritten)


def test_advisory_is_empty_for_a_pure_paraphrase():
    rewritten = (
        "Worked as a Senior Engineer with Umoja Bank between 2019 and 2023, "
        "leading 14 people and reducing costs by 12%."
    )
    assert rewriter_v1.advisory_caps_runs(ORIGINAL, rewritten) == []


def test_a_short_invented_number_is_not_masked_by_a_longer_one():
    """Substring membership passed "5" because "2019 to 2023" contains no 5 but
    "12%" contains a 2: any digit inside a date or a percentage used to satisfy
    the check. Whole-word matching is the fix."""
    assert "2" in rewriter_v1.check_no_new_facts(
        "Managed 12 staff over 2019.", "Managed 12 staff over 2019, across 2 sites.")
    assert "1" in rewriter_v1.check_no_new_facts(
        "Managed 14 staff.", "Managed 14 staff on 1 project.")


def test_a_number_that_really_is_in_the_original_still_passes():
    assert rewriter_v1.check_no_new_facts(
        "Managed 12 staff over 2019.", "Over 2019, managed 12 staff.") == []


def test_invented_number_is_flagged():
    rewritten = "Senior Engineer at Umoja Bank. Managed 40 staff."
    assert "40" in rewriter_v1.check_no_new_facts(ORIGINAL, rewritten)


def test_invented_qualification_is_flagged():
    rewritten = ORIGINAL + " Holds an MBA."
    assert "mba" in rewriter_v1.check_no_new_facts(ORIGINAL, rewritten)


def test_dropping_a_fact_is_not_flagged():
    """One-directional on purpose: losing a fact is a different problem and
    does not invalidate a phrasing probe."""
    rewritten = "Senior Engineer at Umoja Bank."
    assert rewriter_v1.check_no_new_facts(ORIGINAL, rewritten) == []


def test_reformatting_alone_is_not_a_new_fact():
    rewritten = "SENIOR ENGINEER, Umoja Bank (2019-2023). Managed 14 staff; cut costs 12%."
    assert rewriter_v1.check_no_new_facts(ORIGINAL, rewritten) == []


def test_result_is_sorted_and_deduplicated():
    rewritten = ORIGINAL + " Holds an MBA and an MSc. Also an MBA from 40 courses."
    reasons = rewriter_v1.check_no_new_facts(ORIGINAL, rewritten)
    assert reasons == sorted(set(reasons))
    assert reasons == ["40", "mba", "msc"]


def _plausible_rephrasing(text: str) -> str:
    """A rewrite that adds no fact but re-cases and re-headings everything.

    This is what an instruction-tuned model actually does to a CV: it adds
    section headings and title-cases lines. Every capitalised run it produces
    is new relative to the source, which is exactly the pattern that made the
    old caps-run guard reject honest paraphrases of real CVs.
    """
    body = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return ("Professional Summary\n"
            "Key Achievements And Responsibilities\n"
            f"{body.title()}\n"
            "Core Technical Skills\n")


REAL_CV = ROOT / "Batch1" / "CVs" / "ALEX PARK.pdf"


@pytest.mark.slow
@pytest.mark.skipif(not REAL_CV.exists(), reason="Batch1 not present")
def test_guard_does_not_reject_a_plausible_rephrasing_of_a_real_cv():
    """Regression for the reason finding 2 survived to final review.

    The other guard tests use two hand-written sentences whose capitalised runs
    are all genuine entities, so they never exercised the case that broke: a
    real CV, hundreds of lines, re-cased by the rewriter. Needs no model.
    """
    import cv_parser

    original = cv_parser.parse(REAL_CV).full_text()
    assert len(original) > 500, "expected a real, non-trivial CV"

    rewritten = _plausible_rephrasing(original)
    assert rewriter_v1.check_no_new_facts(original, rewritten) == []
    # The caps-run signal is still recorded, and it is non-empty: that is
    # exactly what the old guard rejected this rephrasing on.
    assert rewriter_v1.advisory_caps_runs(original, rewritten)
    old_guard_reasons = rewriter_v1._new_against(
        original, rewriter_v1.extract_facts(rewritten))
    assert old_guard_reasons, "fixture no longer reproduces the reported bug"


def test_rewrite_prompt_names_the_role_and_forbids_invention():
    prompt = rewriter_v1.build_rewrite_prompt("CV BODY", "Fullstack Engineer")
    assert "Fullstack Engineer" in prompt
    assert "CV BODY" in prompt
    assert "do not add" in prompt.lower()


def test_strict_prompt_is_more_emphatic_than_the_normal_one():
    normal = rewriter_v1.build_rewrite_prompt("CV", "Role")
    strict = rewriter_v1.build_rewrite_prompt("CV", "Role", strict=True)
    assert strict != normal
    assert len(strict) > len(normal)


def test_paraphrase_passes_the_prompt_to_the_caller_and_returns_text():
    seen = {}

    def fake_call(model, prompt, num_ctx, seed, timeout):
        seen.update(model=model, prompt=prompt, num_ctx=num_ctx, seed=seed)
        return "  rewritten cv  "

    out = rewriter_v1.paraphrase("CV BODY", "Fullstack Engineer",
                                 model="test-model", call=fake_call)
    assert out == "rewritten cv"
    assert seen["model"] == "test-model"
    assert seen["num_ctx"] == rewriter_v1.DEFAULT_NUM_CTX
    assert seen["seed"] == 0
    assert "Fullstack Engineer" in seen["prompt"]


def test_paraphrase_raises_on_empty_output():
    def empty_call(model, prompt, num_ctx, seed, timeout):
        return "   "

    with pytest.raises(rewriter_v1.RewriteError):
        rewriter_v1.paraphrase("CV", "Role", call=empty_call)


def test_paraphrase_accepts_a_full_ollama_body_and_records_truncation():
    def fat_call(model, prompt, num_ctx, seed, timeout):
        return {"response": "rewritten cv", "prompt_eval_count": 3}

    out = rewriter_v1.paraphrase("x" * 6000, "Role", call=fat_call)
    assert out == "rewritten cv"
    assert rewriter_v1.paraphrase.last_prompt_eval_count == 3
    assert rewriter_v1.paraphrase.last_truncated is True


def test_paraphrase_records_unknown_truncation_when_no_token_count_comes_back():
    def bare_call(model, prompt, num_ctx, seed, timeout):
        return "rewritten cv"

    rewriter_v1.paraphrase("CV", "Role", call=bare_call)
    assert rewriter_v1.paraphrase.last_truncated is None


def _fake_ollama(score, prompt_eval_count=None):
    def call(model, prompt, num_ctx, seed, timeout):
        body = {"response": '{"score": %s, "summary": "ok"}' % score}
        if prompt_eval_count is not None:
            body["prompt_eval_count"] = prompt_eval_count
        return body
    return call


def test_scorer_returns_the_score_as_an_int():
    scorer = rewriter_v1.make_scorer("m", call=_fake_ollama(7))
    assert scorer("cv text", "Fullstack Engineer") == 7


def test_scorer_records_its_model_and_prompt_version():
    scorer = rewriter_v1.make_scorer("qwen2.5:3b", prompt_version="v2_anchored",
                                     call=_fake_ollama(3))
    assert scorer.model == "qwen2.5:3b"
    assert scorer.prompt_version == "v2_anchored"


def test_scorer_sends_the_job_title_as_the_job_text():
    seen = {}

    def call(model, prompt, num_ctx, seed, timeout):
        seen["prompt"] = prompt
        return {"response": '{"score": 5}'}

    rewriter_v1.make_scorer("m", call=call)("CV BODY", "Aquarium Biologist")
    assert "Aquarium Biologist" in seen["prompt"]
    assert "CV BODY" in seen["prompt"]


def test_scorer_raises_when_no_score_comes_back():
    def call(model, prompt, num_ctx, seed, timeout):
        return {"response": "the candidate seems fine"}

    with pytest.raises(rewriter_v1.ScoreError):
        rewriter_v1.make_scorer("m", call=call)("cv", "role")


def test_scorer_raises_when_score_is_not_a_number():
    def call(model, prompt, num_ctx, seed, timeout):
        return {"response": '{"score": "very good"}'}

    with pytest.raises(rewriter_v1.ScoreError):
        rewriter_v1.make_scorer("m", call=call)("cv", "role")


def test_scorer_rounds_a_float_score_instead_of_flooring_it():
    """int(6.9) is 6, so a genuine 6.5 -> 6.9 move would have read as 0."""
    assert rewriter_v1.make_scorer("m", call=_fake_ollama(6.9))("cv", "role") == 7
    assert rewriter_v1.make_scorer("m", call=_fake_ollama(6.4))("cv", "role") == 6


def test_scorer_records_truncation_of_its_prompt():
    scorer = rewriter_v1.make_scorer("m", call=_fake_ollama(5, prompt_eval_count=2))
    scorer("x" * 4000, "role")
    assert scorer.last_prompt_eval_count == 2
    assert scorer.last_truncated is True


def test_scorer_records_an_untruncated_prompt_as_false():
    scorer = rewriter_v1.make_scorer("m", call=_fake_ollama(5, prompt_eval_count=900))
    scorer("short cv", "role")
    assert scorer.last_truncated is False


def test_scorer_records_unknown_truncation_without_a_token_count():
    scorer = rewriter_v1.make_scorer("m", call=_fake_ollama(5))
    scorer("short cv", "role")
    assert scorer.last_truncated is None


def _stub_scorer(scores, truncated=None):
    """Returns successive scores from `scores` on each call.

    `truncated` is a matching list of last_truncated values, if the test cares.
    """
    it = iter(scores)
    trunc = iter(truncated) if truncated is not None else None

    def scorer(cv_text, job_title):
        if trunc is not None:
            scorer.last_truncated = next(trunc)
        return next(it)

    scorer.model = "stub-scorer"
    scorer.prompt_version = "v3_criteria"
    if truncated is not None:
        scorer.last_truncated = None
    return scorer


def _stub_paraphraser(*outputs):
    """Returns successive rewrites; records how many times it was called."""
    it = iter(outputs)
    calls = []

    def fn(cv_text, job_title, *, model, strict=False):
        calls.append({"strict": strict, "model": model})
        return next(it)

    fn.calls = calls
    return fn


def test_probe_reports_the_delta():
    result = rewriter_v1.probe(
        ORIGINAL, "Fullstack Engineer", _stub_scorer([4, 7]),
        candidate="ALEX PARK", job="startup_fullstack_engineer",
        paraphrase_fn=_stub_paraphraser("Senior Engineer at Umoja Bank, 2019 to 2023."),
    )
    assert result.score_before == 4
    assert result.score_after == 7
    assert result.delta == 3
    assert result.rejected is False
    assert result.candidate == "ALEX PARK"
    assert result.scorer_model == "stub-scorer"


def test_probe_records_zero_delta_when_phrasing_does_not_matter():
    result = rewriter_v1.probe(
        ORIGINAL, "Role", _stub_scorer([5, 5]),
        paraphrase_fn=_stub_paraphraser("Senior Engineer, Umoja Bank, 2019-2023."),
    )
    assert result.delta == 0
    assert result.rejected is False


def test_probe_retries_once_when_the_rewrite_invents_a_fact():
    para = _stub_paraphraser(
        ORIGINAL + " Managed a budget of 40 million.",   # invents a number
        "Senior Engineer at Umoja Bank, 2019 to 2023.",  # clean on retry
    )
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4, 6]), paraphrase_fn=para)
    assert result.retried is True
    assert result.rejected is False
    assert result.delta == 2
    assert para.calls[0]["strict"] is False
    assert para.calls[1]["strict"] is True


def test_probe_marks_rejected_when_the_retry_also_invents():
    para = _stub_paraphraser(
        ORIGINAL + " Managed a budget of 40 million.",
        ORIGINAL + " Managed a budget of 40 million.",
    )
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4, 9]), paraphrase_fn=para)
    assert result.rejected is True
    assert "40" in result.reasons
    assert result.delta == 5, "a rejected row is still scored, just excluded later"


def test_probe_records_a_caps_run_as_advisory_and_does_not_reject():
    """Ruling A: an invented multi-word entity is recorded, never rejecting,
    and no retry is spent on it."""
    para = _stub_paraphraser(ORIGINAL + " Previously at Google Cloud.")
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4, 6]), paraphrase_fn=para)
    assert result.rejected is False
    assert result.retried is False
    assert result.reasons == []
    assert "google cloud" in result.advisory
    assert len(para.calls) == 1


def test_probe_records_truncation_of_both_scores_and_the_rewrite():
    scorer = _stub_scorer([4, 6], truncated=[False, True])
    para = _stub_paraphraser("Senior Engineer, Umoja Bank, 2019-2023.")
    para.last_truncated = False
    result = rewriter_v1.probe(ORIGINAL, "Role", scorer, paraphrase_fn=para)
    assert result.truncated_before is False
    assert result.truncated_after is True
    assert result.truncated_rewrite is False


def test_probe_leaves_truncation_unknown_for_a_scorer_that_does_not_report_it():
    result = rewriter_v1.probe(
        ORIGINAL, "Role", _stub_scorer([4, 4]),
        paraphrase_fn=_stub_paraphraser("Senior Engineer, Umoja Bank, 2019-2023."))
    assert result.truncated_before is None
    assert result.truncated_after is None
    assert result.truncated_rewrite is None


def test_probe_records_length_ratio_and_facts_lost():
    """A rewrite that drops a role passes the guard but scores lower for
    content reasons, so the loss has to be visible on the row."""
    short = "Senior Engineer at Umoja Bank."      # loses 2019, 2023, 14, 12%
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4, 2]),
                               paraphrase_fn=_stub_paraphraser(short))
    assert result.rejected is False
    assert result.facts_lost == 4
    assert result.length_ratio == pytest.approx(len(short) / len(ORIGINAL))
    assert result.length_ratio < 0.5


def test_probe_records_no_loss_for_a_faithful_rewrite():
    rewritten = ("Worked as a Senior Engineer with Umoja Bank between 2019 and "
                 "2023, leading 14 people and reducing costs by 12%.")
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4, 4]),
                               paraphrase_fn=_stub_paraphraser(rewritten))
    assert result.facts_lost == 0
    assert result.length_ratio > 1.0


def test_probe_survives_a_malformed_ollama_body():
    """A torn JSON body raises json.JSONDecodeError - a ValueError - and used
    to abort the whole 270-probe run."""
    import json as _json

    def bad_scorer(cv_text, job_title):
        raise _json.JSONDecodeError("Expecting value", "", 0)

    bad_scorer.model = "stub-scorer"
    result = rewriter_v1.probe(ORIGINAL, "Role", bad_scorer,
                               paraphrase_fn=_stub_paraphraser("x"))
    assert result.error is not None
    assert "JSONDecodeError" in result.error
    assert result.delta is None


def test_probe_survives_a_body_that_is_not_a_mapping():
    def bad_scorer(cv_text, job_title):
        raise AttributeError("'str' object has no attribute 'get'")

    bad_scorer.model = "stub-scorer"
    result = rewriter_v1.probe(ORIGINAL, "Role", bad_scorer,
                               paraphrase_fn=_stub_paraphraser("x"))
    assert result.error is not None
    assert "AttributeError" in result.error


def test_probe_records_a_rewrite_failure_without_raising():
    def boom(cv_text, job_title, *, model, strict=False):
        raise rewriter_v1.RewriteError("model returned no text")

    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4]), paraphrase_fn=boom)
    assert result.error is not None
    assert result.delta is None
    assert result.score_before == 4


def test_probe_records_a_scoring_failure_without_raising():
    def bad_scorer(cv_text, job_title):
        raise rewriter_v1.ScoreError("no score in output")

    bad_scorer.model = "stub-scorer"
    bad_scorer.prompt_version = "v3_criteria"
    result = rewriter_v1.probe(ORIGINAL, "Role", bad_scorer,
                               paraphrase_fn=_stub_paraphraser("x"))
    assert result.error is not None
    assert result.delta is None


def test_probe_result_serialises_to_a_flat_dict():
    result = rewriter_v1.probe(
        ORIGINAL, "Role", _stub_scorer([4, 4]),
        paraphrase_fn=_stub_paraphraser("Senior Engineer, Umoja Bank, 2019-2023."),
    )
    row = result.to_dict()
    assert row["delta"] == 0
    assert row["rejected"] is False
    assert isinstance(row["reasons"], list)


def test_determinism_control_passes_when_scores_reproduce():
    #                       A     mid   A     B     mid   B
    scorer = _stub_scorer([5, 9, 5, 3, 9, 3])
    drift = rewriter_v1.verify_determinism(
        {"A": "cv a", "B": "cv b"}, "Role", scorer)
    assert drift == []


def test_determinism_control_reports_every_drifting_cv():
    scorer = _stub_scorer([5, 9, 6, 3, 9, 3, 8, 9, 2])
    drift = rewriter_v1.verify_determinism(
        {"A": "cv a", "B": "cv b", "C": "cv c"}, "Role", scorer)
    assert drift == [("A", 5, 6), ("C", 8, 2)]


def test_determinism_control_interleaves_a_different_call_between_the_pair():
    """Scoring the same prompt back to back is the most favourable cache
    condition there is, and the probe never runs that way - a rewrite call sits
    between its two scores."""
    calls = []

    def scorer(cv_text, job_title):
        calls.append(cv_text)
        return 4

    scorer.model = "stub-scorer"
    rewriter_v1.verify_determinism({"A": "cv a", "B": "cv b"}, "Role", scorer,
                                   interleave="OTHER")
    assert calls == ["cv a", "OTHER", "cv a", "cv b", "OTHER", "cv b"]


def test_determinism_control_default_interleave_is_not_one_of_the_cvs():
    assert rewriter_v1._INTERLEAVE_CV not in ("cv a", "")
    calls = []

    def scorer(cv_text, job_title):
        calls.append(cv_text)
        return 4

    scorer.model = "stub-scorer"
    rewriter_v1.verify_determinism({"A": "cv a"}, "Role", scorer)
    assert calls == ["cv a", rewriter_v1._INTERLEAVE_CV, "cv a"]


def test_title_from_slug_is_human_readable():
    assert rewriter_v1.title_from_slug("startup_fullstack_engineer") == \
        "Startup Fullstack Engineer"
    assert rewriter_v1.title_from_slug("aquarium_biologist_entry_level") == \
        "Aquarium Biologist Entry Level"


def test_already_done_keys_on_models_job_and_cv(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text(
        '{"rewriter_model": "r", "scorer_model": "s", "job": "j", "candidate": "c"}\n',
        encoding="utf-8")
    assert ("r", "s", "j", "c") in rewriter_v1.already_done(out)


def test_already_done_on_a_missing_file_is_empty(tmp_path):
    assert rewriter_v1.already_done(tmp_path / "nope.jsonl") == set()


def test_already_done_skips_errored_rows(tmp_path):
    """An Ollama restart writes an error row for every probe it interrupts.
    Keying those as done would make every later resume skip them for good."""
    out = tmp_path / "rows.jsonl"
    out.write_text(
        '{"rewriter_model": "r", "scorer_model": "s", "job": "j", '
        '"candidate": "ok", "error": null}\n'
        '{"rewriter_model": "r", "scorer_model": "s", "job": "j", '
        '"candidate": "boom", "error": "OSError: connection refused"}\n',
        encoding="utf-8")
    done = rewriter_v1.already_done(out)
    assert ("r", "s", "j", "ok") in done
    assert ("r", "s", "j", "boom") not in done


def test_read_rows_tolerates_a_torn_final_line(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text('{"delta": 1}\n\n{"delta": 2}\n{"delta"', encoding="utf-8")
    assert rewriter_v1.read_rows(out) == [{"delta": 1}, {"delta": 2}]


def test_read_rows_on_a_missing_file_is_empty(tmp_path):
    assert rewriter_v1.read_rows(tmp_path / "nope.jsonl") == []


def test_apply_filter_keeps_the_named_items():
    assert rewriter_v1.apply_filter({"a": 1, "b": 2}, ["b"], "jobs") == {"b": 2}


def test_apply_filter_without_a_filter_keeps_everything():
    assert rewriter_v1.apply_filter({"a": 1}, None, "jobs") == {"a": 1}


def test_apply_filter_stops_cleanly_when_nothing_matches():
    """--jobs with a typo used to reach next(iter(jobs.values())) and raise
    StopIteration, which reads as a crash rather than a bad argument."""
    with pytest.raises(SystemExit) as exc:
        rewriter_v1.apply_filter({"a": 1, "b": 2}, ["typo"], "jobs")
    assert "typo" in str(exc.value)
    assert "a, b" in str(exc.value)


def test_summarise_excludes_rejected_rows_from_the_headline():
    rows = [
        {"delta": 0, "rejected": False, "score_before": 2, "error": None},
        {"delta": 4, "rejected": False, "score_before": 2, "error": None},
        {"delta": 9, "rejected": True, "score_before": 2, "error": None},
    ]
    text = rewriter_v1.summarise(rows, flag_threshold=2)
    assert "2 clean" in text
    assert "1 rejected" in text
    # the +9 rejected row must not be counted as a flagged move
    assert "1 flagged" in text


def test_summarise_reports_deltas_split_by_starting_score():
    rows = [
        {"delta": 5, "rejected": False, "score_before": 2, "error": None},
        {"delta": 0, "rejected": False, "score_before": 7, "error": None},
    ]
    text = rewriter_v1.summarise(rows, flag_threshold=2)
    assert "by starting score" in text


def test_summarise_handles_an_empty_run():
    assert "no rows" in rewriter_v1.summarise([], flag_threshold=2)


def test_summarise_excludes_truncated_rows_from_the_headline_but_counts_them():
    rows = [
        {"delta": 0, "rejected": False, "score_before": 2, "error": None,
         "scorer_model": "m"},
        {"delta": 6, "rejected": False, "score_before": 2, "error": None,
         "scorer_model": "m", "truncated_after": True},
    ]
    text = rewriter_v1.summarise(rows, flag_threshold=2)
    assert "1 clean" in text
    assert "1 truncated" in text
    # the +6 truncated row must not reach the headline
    assert "0 flagged" in text
    assert "mean +0.00" in text


def test_summarise_counts_truncation_of_the_rewrite_prompt_too():
    rows = [{"delta": 3, "rejected": False, "score_before": 2, "error": None,
             "scorer_model": "m", "truncated_rewrite": True}]
    assert "0 clean" in rewriter_v1.summarise(rows, flag_threshold=2)
    assert "1 truncated" in rewriter_v1.summarise(rows, flag_threshold=2)


def test_summarise_splits_by_scorer_model():
    """The three scorers do not share a score distribution, so a pooled mean is
    a mean over three scales."""
    rows = [
        {"delta": 4, "rejected": False, "score_before": 2, "error": None,
         "scorer_model": "qwen2.5:3b"},
        {"delta": 0, "rejected": False, "score_before": 2, "error": None,
         "scorer_model": "llama3.2:3b"},
    ]
    text = rewriter_v1.summarise(rows, flag_threshold=2)
    assert "by scorer model" in text
    assert "qwen2.5:3b" in text
    assert "llama3.2:3b" in text
    assert "+4.00" in text


def test_summarise_reports_min_and_max_per_starting_score_bucket():
    rows = [
        {"delta": -1, "rejected": False, "score_before": 2, "error": None,
         "scorer_model": "m"},
        {"delta": 5, "rejected": False, "score_before": 2, "error": None,
         "scorer_model": "m"},
    ]
    text = rewriter_v1.summarise(rows, flag_threshold=2)
    lines = text.splitlines()
    bucket = lines[lines.index("by starting score:") + 2]
    assert bucket.split() == ["2", "2", "+2.00", "-1", "5"]


def test_summarise_reports_advisories_without_calling_them_rejections():
    rows = [{"delta": 1, "rejected": False, "score_before": 2, "error": None,
             "scorer_model": "m", "advisory": ["google cloud"]}]
    text = rewriter_v1.summarise(rows, flag_threshold=2)
    assert "1 clean" in text
    assert "0 rejected" in text
    assert "google cloud" in text
    assert "advisories" in text


def _patch_batch(monkeypatch, scores, tmp_path):
    """Point main() at a two-CV, one-job fake batch with a stub scorer."""
    monkeypatch.setattr(rewriter_v1, "load_cvs", lambda *a, **k: {"A": "cv a", "B": "cv b"})
    monkeypatch.setattr(rewriter_v1, "load_job_titles", lambda *a, **k: {"j": "Role"})
    monkeypatch.setattr(rewriter_v1, "make_scorer",
                        lambda *a, **k: _stub_scorer(scores))
    monkeypatch.setattr(rewriter_v1, "OUT", tmp_path / "rewriter_v1.jsonl")


def test_determinism_gate_exits_nonzero_on_drift(monkeypatch, tmp_path, capsys):
    """It printed FAIL and returned 0, so a CI step or a shell && would treat a
    failed control as a pass."""
    _patch_batch(monkeypatch, [5, 9, 6, 3, 9, 3], tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["rewriter_v1.py", "--verify-determinism", "--scorer-models", "m"])
    with pytest.raises(SystemExit) as exc:
        rewriter_v1.main()
    assert exc.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_determinism_gate_returns_normally_when_it_reproduces(monkeypatch, tmp_path, capsys):
    _patch_batch(monkeypatch, [5, 9, 5, 3, 9, 3], tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["rewriter_v1.py", "--verify-determinism", "--scorer-models", "m"])
    rewriter_v1.main()
    assert "OK" in capsys.readouterr().out


def test_final_summary_covers_the_whole_file_not_just_this_run(monkeypatch, tmp_path, capsys):
    """A resumed run holds only the slice it completed; a headline over that
    slice is a headline over an arbitrary subset."""
    out = tmp_path / "rewriter_v1.jsonl"
    out.write_text(json.dumps({
        "rewriter_model": "r", "scorer_model": "m", "job": "j", "candidate": "OLD",
        "score_before": 2, "score_after": 2, "delta": 0, "rejected": False,
        "error": None}) + "\n", encoding="utf-8")

    _patch_batch(monkeypatch, [4, 6, 4, 6], tmp_path)
    monkeypatch.setattr(rewriter_v1, "paraphrase",
                        lambda cv, title, *, model, strict=False: "cv a rewritten")
    monkeypatch.setattr(sys, "argv",
                        ["rewriter_v1.py", "--scorer-models", "m", "--rewriter-model", "r"])
    rewriter_v1.main()
    text = capsys.readouterr().out
    assert "this run: 2 new rows" in text
    assert "summary over all" in text
    # 2 new + the pre-existing OLD row
    assert "3 clean" in text


def test_probe_docstring_states_the_real_batch_size():
    doc = rewriter_v1.probe.__doc__
    assert "270" in doc
    assert "batch of 90" not in doc


def test_guard_docstring_lists_every_verified_miss():
    """The docstring is a claim in a research write-up, so an understated
    limitation list is a defect."""
    doc = rewriter_v1.check_no_new_facts.__doc__.lower()
    for miss in ("semantic inflation", "fourteen", "kubernetes", "google",
                 "over a decade", "multi-word entities"):
        assert miss in doc, miss


@pytest.mark.ollama
def test_paraphrase_against_real_ollama():
    """Skipped by default: pytest -m ollama to run. Needs qwen2.5:3b pulled."""
    out = rewriter_v1.paraphrase(ORIGINAL, "Fullstack Engineer")
    assert out.strip()
    assert "Umoja" in out or "umoja" in out.lower()


# --- post-smoke-test fixes -------------------------------------------------
# Three defects found by putting the first REAL CV through the probe rather
# than a hand-written fixture. THE DANIEL.pdf says "BCom Hons"; qwen2.5:3b
# rewrote it as "BCom Honours" and the guard called that an invented
# qualification - a false rejection in the hard channel, the same class of
# defect the caps-run split fixed in the advisory channel.

def test_expanding_an_abbreviation_is_not_an_invented_qualification():
    """Hons -> Honours is what a paraphraser does. Measured on THE DANIEL.pdf."""
    assert rewriter_v1.check_no_new_facts(
        "BCom Hons Business Information Systems",
        "BCom Honours Business Information Systems") == []


def test_canonicalised_pairs_collapse_to_one_fact():
    assert rewriter_v1.extract_hard_facts("BCom Hons") == \
        rewriter_v1.extract_hard_facts("BCom Honours")
    assert rewriter_v1.extract_hard_facts("Matric") == \
        rewriter_v1.extract_hard_facts("Matriculation")


def test_a_genuinely_invented_qualification_is_still_rejected():
    """The canonicalisation must not blunt the guard it sits inside."""
    reasons = rewriter_v1.check_no_new_facts("Worked 2019 to 2023",
                                             "Led 5 teams and holds an MBA")
    assert "mba" in reasons and "5" in reasons


def test_qualification_tokens_need_word_boundaries():
    """Guards the regex's \b anchors.

    Written after they were destroyed by an editing slip that turned them into
    literal backspace characters, leaving the pattern matching NOTHING - the
    guard silently stopped rejecting any qualification at all. Asserting a
    positive AND a negative pins both failure directions: a pattern with no
    anchors matches inside words, a pattern with broken anchors matches nothing.
    """
    assert rewriter_v1.extract_hard_facts("Holds an MBA and a BSc") == {"mba", "bsc"}
    assert rewriter_v1.extract_hard_facts("He was awesome and unbatched") == set()


def test_facts_lost_sees_a_dropped_employer():
    """facts_lost uses hard facts AND caps runs, unlike the rejection direction.

    Losing "Umoja Bank" is a dropped employer and an alternative explanation
    for a score drop that has nothing to do with phrasing. Reversing the
    rejection guard - hard facts only - left this at 0.
    """
    original = "Senior Engineer at Umoja Bank from 2019 to 2023"
    stripped = "Senior Engineer, 2019-2023"
    lost = rewriter_v1._new_against(stripped, rewriter_v1.extract_facts(original))
    assert "umoja bank" in lost


def test_rejection_rate_counts_truncated_rejections():
    """The rejection rate is a reported finding, so it counts every scored row.

    Counting it over the untruncated subset silently biased it low: a rewrite
    that was rejected AND truncated is still a rejection.
    """
    rows = [
        {"delta": 0, "rejected": False, "score_before": 2, "error": None,
         "truncated_before": False, "truncated_after": False, "truncated_rewrite": False},
        {"delta": 3, "rejected": True, "score_before": 2, "error": None,
         "truncated_before": True, "truncated_after": False, "truncated_rewrite": False},
    ]
    text = rewriter_v1.summarise(rows, flag_threshold=2)
    assert "1 rejected (of 2 scored)" in text
    assert "1 truncated (excluded)" in text


# --- similarity: telling a real rewrite from a no-op -------------------------

def test_probe_records_similarity_against_the_original():
    """delta alone cannot distinguish "phrasing did not matter" from "the model
    barely rewrote anything". Measured on batch3/bad_cv_kevin_jones: the
    rewriter changed 2 words of 232 and returned length_ratio 1.002, which
    looks like a perfectly ordinary rewrite on every other field."""
    rewritten = "Senior Engineer, Umoja Bank, 2019-2023, 14 people, 12%."
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4, 4]),
                               paraphrase_fn=_stub_paraphraser(rewritten))
    assert result.similarity is not None
    assert 0.0 <= result.similarity <= 1.0


def test_similarity_is_one_when_the_rewriter_returns_the_input_unchanged():
    """The no-op case, which every other field on the row reports as healthy."""
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4, 4]),
                               paraphrase_fn=_stub_paraphraser(ORIGINAL))
    assert result.similarity == pytest.approx(1.0)
    assert result.delta == 0
    assert result.rejected is False
    assert result.length_ratio == pytest.approx(1.0)


def test_similarity_falls_when_the_wording_genuinely_changes():
    barely = rewriter_v1.probe(
        ORIGINAL, "Role", _stub_scorer([4, 4]),
        paraphrase_fn=_stub_paraphraser(ORIGINAL.replace("Senior", "Sr"))).similarity
    heavily = rewriter_v1.probe(
        ORIGINAL, "Role", _stub_scorer([4, 4]),
        paraphrase_fn=_stub_paraphraser(
            "Led a team of 14 at Umoja Bank from 2019 to 2023, lifting "
            "throughput by 12%.")).similarity
    assert heavily < barely


def test_similarity_is_none_when_the_probe_errored_before_rewriting():
    """No rewrite happened, so there is nothing to compare - not a 0.0, which
    would read as a completely different rewrite."""
    def explode(cv_text, job_title, **kwargs):
        raise OSError("ollama down")      # what an unreachable host actually raises
    result = rewriter_v1.probe(ORIGINAL, "Role", _stub_scorer([4]),
                               paraphrase_fn=explode)
    assert result.similarity is None
    assert result.error
