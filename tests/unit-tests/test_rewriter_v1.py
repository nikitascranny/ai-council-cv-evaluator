"""
Unit tests for rewriter/rewriter_v1.py.

prompts.py, run_evaluations.py and parser/cv_parser.py are stubbed at the
repo root purely so rewriter_v1's top-level imports succeed (see the stub
files themselves for why). Every test that depends on their behaviour
monkeypatches the specific function it needs rather than relying on the
stub bodies, which all deliberately raise NotImplementedError if called
unpatched - that way a test silently depending on stub behaviour fails
loudly instead of passing for the wrong reason.

fact_guard itself is real (not mocked): rewriter_v1 imports
rejection_reasons / advisory_caps_runs / extract_facts / _new_against
directly, and using the genuine implementations means probe() tests also
catch a fact_guard contract break, not just a stale mock's assumptions.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rewriter"))

import rewriter_v1 as rv1


# ---------------- build_rewrite_prompt ----------------

class TestBuildRewritePrompt:
    def test_substitutes_cv_text(self):
        prompt = rv1.build_rewrite_prompt("MY CV BODY", "Some Role")
        assert "MY CV BODY" in prompt

    def test_job_title_deliberately_not_inserted(self):
        # Per the function's own docstring: naming the role invited the
        # model to adopt it (regression seen on ALEX PARK), so job_title
        # is accepted but never substituted into the prompt.
        prompt = rv1.build_rewrite_prompt("cv text", "UNIQUE_TITLE_TOKEN")
        assert "UNIQUE_TITLE_TOKEN" not in prompt

    def test_curly_braces_in_cv_survive_literally(self):
        cv_text = "Skills: {Python, Go}"
        prompt = rv1.build_rewrite_prompt(cv_text, "Role")
        assert cv_text in prompt

    def test_strict_appends_suffix(self):
        plain = rv1.build_rewrite_prompt("cv", "Role", strict=False)
        strict = rv1.build_rewrite_prompt("cv", "Role", strict=True)
        assert "INVENTED information" not in plain
        assert "INVENTED information" in strict
        assert strict.startswith(plain)


# ---------------- paraphrase ----------------

class TestParaphrase:
    def test_plain_string_response(self, monkeypatch):
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)
        result = rv1.paraphrase("cv text", "Role", call=lambda *a, **k: "  rewritten  ")
        assert result == "rewritten"
        assert rv1.paraphrase.last_prompt_eval_count is None
        assert rv1.paraphrase.last_truncated is None  # ptoks is None -> unknown

    def test_dict_response_extracts_text_and_token_count(self, monkeypatch):
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda ptoks, chars: False)
        fake = {"response": "rewritten text", "prompt_eval_count": 42}
        result = rv1.paraphrase("cv text", "Role", call=lambda *a, **k: fake)
        assert result == "rewritten text"
        assert rv1.paraphrase.last_prompt_eval_count == 42
        assert rv1.paraphrase.last_truncated is False

    def test_truncation_flag_uses_is_truncated_when_ptoks_known(self, monkeypatch):
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda ptoks, chars: True)
        fake = {"response": "text", "prompt_eval_count": 5}
        rv1.paraphrase("cv", "Role", call=lambda *a, **k: fake)
        assert rv1.paraphrase.last_truncated is True

    def test_empty_response_raises_rewrite_error(self, monkeypatch):
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)
        import pytest
        with pytest.raises(rv1.RewriteError):
            rv1.paraphrase("cv", "Role", call=lambda *a, **k: "   ")

    def test_strict_flag_reaches_the_prompt_sent_to_call(self, monkeypatch):
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)
        captured = {}

        def fake_call(model, prompt, num_ctx, seed, timeout):
            captured["prompt"] = prompt
            return "ok"

        rv1.paraphrase("cv", "Role", call=fake_call, strict=True)
        assert "INVENTED information" in captured["prompt"]

    def test_records_prompt_char_length(self, monkeypatch):
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)
        rv1.paraphrase("cv", "Role", call=lambda *a, **k: "ok")
        expected_len = len(rv1.build_rewrite_prompt("cv", "Role"))
        assert rv1.paraphrase.last_prompt_chars == expected_len


# ---------------- make_scorer / scorer ----------------

class TestMakeScorer:
    def test_binds_model_and_prompt_version(self):
        scorer = rv1.make_scorer("phi4", prompt_version="v2_basic", call=lambda *a, **k: {})
        assert scorer.model == "phi4"
        assert scorer.prompt_version == "v2_basic"

    def test_scorer_calls_prompts_build_with_title_in_job_text_slot(self, monkeypatch):
        # Variant 1's design: the job TITLE goes where prompts.build
        # expects the job text, keeping the context window small.
        captured = {}

        def fake_build(prompt_version, job_title, cv_text):
            captured["args"] = (prompt_version, job_title, cv_text)
            return "PROMPT"

        monkeypatch.setattr(rv1.prompts, "build", fake_build)
        monkeypatch.setattr(rv1.run_evaluations, "parse_response", lambda text: ({"score": 5}, None))
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)

        scorer = rv1.make_scorer(
            "phi4", prompt_version="v3_criteria",
            call=lambda model, prompt, num_ctx, seed, timeout: {"response": "{}", "prompt_eval_count": 1},
        )
        scorer("CV TEXT", "Role Title")

        assert captured["args"] == ("v3_criteria", "Role Title", "CV TEXT")

    def test_rounds_rather_than_truncates_score(self, monkeypatch):
        monkeypatch.setattr(rv1.prompts, "build", lambda *a, **k: "PROMPT")
        monkeypatch.setattr(rv1.run_evaluations, "parse_response", lambda text: ({"score": 6.9}, None))
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)

        scorer = rv1.make_scorer(
            "phi4", call=lambda *a, **k: {"response": "{}", "prompt_eval_count": 1},
        )
        # int(6.9) would be 6 - round() must be used, or a real 0.5+ move
        # in score gets silently truncated away.
        assert scorer("cv", "role") == 7

    def test_raises_score_error_when_parse_response_fails(self, monkeypatch):
        monkeypatch.setattr(rv1.prompts, "build", lambda *a, **k: "PROMPT")
        monkeypatch.setattr(rv1.run_evaluations, "parse_response", lambda text: (None, "no JSON found"))
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)

        scorer = rv1.make_scorer("phi4", call=lambda *a, **k: {"response": "not json"})
        import pytest
        with pytest.raises(rv1.ScoreError):
            scorer("cv", "role")

    def test_raises_score_error_when_score_missing(self, monkeypatch):
        monkeypatch.setattr(rv1.prompts, "build", lambda *a, **k: "PROMPT")
        monkeypatch.setattr(rv1.run_evaluations, "parse_response", lambda text: ({"not_score": 1}, None))
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)

        scorer = rv1.make_scorer("phi4", call=lambda *a, **k: {"response": "{}"})
        import pytest
        with pytest.raises(rv1.ScoreError):
            scorer("cv", "role")

    def test_raises_score_error_when_score_is_bool(self, monkeypatch):
        # bool is a subclass of int in Python - isinstance(True, int) is
        # True - so this must be excluded explicitly or a bool "score"
        # silently passes as 0 or 1.
        monkeypatch.setattr(rv1.prompts, "build", lambda *a, **k: "PROMPT")
        monkeypatch.setattr(rv1.run_evaluations, "parse_response", lambda text: ({"score": True}, None))
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda *a, **k: False)

        scorer = rv1.make_scorer("phi4", call=lambda *a, **k: {"response": "{}"})
        import pytest
        with pytest.raises(rv1.ScoreError):
            scorer("cv", "role")

    def test_records_truncation_state_on_scorer(self, monkeypatch):
        monkeypatch.setattr(rv1.prompts, "build", lambda *a, **k: "PROMPT")
        monkeypatch.setattr(rv1.run_evaluations, "parse_response", lambda text: ({"score": 3}, None))
        monkeypatch.setattr(rv1.run_evaluations, "is_truncated", lambda ptoks, chars: True)

        scorer = rv1.make_scorer(
            "phi4", call=lambda *a, **k: {"response": "{}", "prompt_eval_count": 999},
        )
        scorer("cv", "role")
        assert scorer.last_truncated is True
        assert scorer.last_prompt_eval_count == 999


# ---------------- probe ----------------

def _fake_scorer(scores, model="fake-scorer", truncations=None):
    """A minimal scorer stand-in matching what probe() actually reads off
    it: call it like a function, and it exposes .model plus
    .last_truncated (probe reads the latter via getattr after each call)."""
    state = {"calls": 0}

    def scorer(cv_text, job_title):
        i = state["calls"]
        state["calls"] += 1
        scorer.last_truncated = (truncations[i] if truncations else None)
        return scores[i]

    scorer.model = model
    scorer.last_truncated = None
    return scorer


def _fake_paraphrase(rewrites, raises_on=None, truncated=False):
    """A minimal paraphrase_fn stand-in. `rewrites` is a list consumed in
    call order (so index 0 is the first, non-strict attempt and index 1,
    if present, is the strict retry). raises_on, if given, is a set of
    call indices that raise RewriteError instead of returning text."""
    state = {"calls": 0}

    def fn(cv_text, job_title, *, model=None, strict=False):
        i = state["calls"]
        state["calls"] += 1
        fn.last_truncated = truncated
        if raises_on and i in raises_on:
            raise rv1.RewriteError("boom")
        return rewrites[i]

    fn.last_truncated = truncated
    return fn


class TestProbe:
    def test_clean_rewrite_no_rejection(self):
        cv_text = "Assisted with support tickets."
        scorer = _fake_scorer([3, 3])
        paraphrase_fn = _fake_paraphrase(["Provided assistance with support tickets."])

        result = rv1.probe(cv_text, "Support Technician", scorer,
                            candidate="Alex", job="support_tech",
                            paraphrase_fn=paraphrase_fn)

        assert result.error is None
        assert result.rejected is False
        assert result.retried is False
        assert result.score_before == 3
        assert result.score_after == 3
        assert result.delta == 0
        assert result.facts_lost == 0
        assert 0.0 <= result.similarity <= 1.0
        assert result.candidate == "Alex"
        assert result.job == "support_tech"
        assert result.scorer_model == "fake-scorer"

    def test_rejected_first_try_retries_and_recomputes_reasons(self):
        cv_text = "Assisted with support tickets."
        scorer = _fake_scorer([3, 5])
        # First attempt invents a number; strict retry is clean.
        paraphrase_fn = _fake_paraphrase([
            "Resolved 500 support tickets.",
            "Provided assistance with support tickets.",
        ])

        result = rv1.probe(cv_text, "Support Technician", scorer,
                            paraphrase_fn=paraphrase_fn)

        assert result.retried is True
        assert result.rejected is False  # the retry was clean
        assert result.delta == 2

    def test_still_rejected_after_retry_stays_rejected(self):
        cv_text = "Assisted with support tickets."
        scorer = _fake_scorer([3, 3])
        paraphrase_fn = _fake_paraphrase([
            "Resolved 500 support tickets.",
            "Resolved 900 support tickets.",  # strict retry still invents a number
        ])

        result = rv1.probe(cv_text, "Support Technician", scorer,
                            paraphrase_fn=paraphrase_fn)

        assert result.retried is True
        assert result.rejected is True
        assert any("900" in r for r in result.reasons)
        # Rejected rows are still scored and still returned, per the
        # module's own docstring - not dropped.
        assert result.score_after == 3
        assert result.delta == 0

    def test_paraphrase_error_is_caught_and_recorded(self):
        cv_text = "Assisted with support tickets."
        scorer = _fake_scorer([3])
        paraphrase_fn = _fake_paraphrase([], raises_on={0})

        result = rv1.probe(cv_text, "Support Technician", scorer,
                            paraphrase_fn=paraphrase_fn)

        assert result.error is not None
        assert "RewriteError" in result.error
        assert result.delta is None
        assert result.score_after is None

    def test_score_error_from_scorer_is_caught_and_recorded(self):
        def raising_scorer(cv_text, job_title):
            raise rv1.ScoreError("evaluator returned garbage")
        raising_scorer.model = "broken-scorer"

        result = rv1.probe("cv text", "Role", raising_scorer,
                            paraphrase_fn=_fake_paraphrase(["irrelevant"]))

        assert result.error is not None
        assert "ScoreError" in result.error
        assert result.score_before is None

    def test_unlisted_exception_propagates(self):
        # probe() only catches a specific tuple of exception types -
        # anything else (e.g. a plain bug) should NOT be silently
        # swallowed into an `error` row.
        def broken_scorer(cv_text, job_title):
            raise KeyError("unexpected")
        broken_scorer.model = "broken-scorer"

        import pytest
        with pytest.raises(KeyError):
            rv1.probe("cv text", "Role", broken_scorer,
                      paraphrase_fn=_fake_paraphrase(["irrelevant"]))

    def test_facts_lost_counts_dropped_originals(self):
        # facts_lost runs the guard reversed: numbers/qualifications in
        # the ORIGINAL that are absent from the rewrite.
        cv_text = "Certified in AWS with 5 years experience."
        scorer = _fake_scorer([3, 3])
        paraphrase_fn = _fake_paraphrase(["Experienced professional."])  # drops both facts

        result = rv1.probe(cv_text, "Role", scorer, paraphrase_fn=paraphrase_fn)

        assert result.facts_lost >= 2

    def test_advisory_recorded_but_never_rejects(self):
        cv_text = "Worked in retail."
        scorer = _fake_scorer([3, 3])
        # New multi-word caps run, no numbers/qualifications invented.
        paraphrase_fn = _fake_paraphrase(["Previously at Bright Future Retailers."])

        result = rv1.probe(cv_text, "Role", scorer, paraphrase_fn=paraphrase_fn)

        assert result.rejected is False
        assert any("bright" in a for a in result.advisory)

    # --- score_rejected=False: a failed rewrite is not scored --------------
    #
    # A rewrite the guard threw out is a rewrite that failed. Scoring it is a
    # second full evaluator call - minutes, locally - for a number the web app
    # never shows. The batch CLI still wants it (see the default), so the
    # caller chooses.

    def test_score_rejected_false_does_not_score_a_rejected_rewrite(self):
        cv_text = "Assisted with support tickets."
        # Only one score is available: asking for a second would raise
        # IndexError off the end of the list, which is the point.
        scorer = _fake_scorer([3])
        paraphrase_fn = _fake_paraphrase([
            "Resolved 500 support tickets.",
            "Resolved 900 support tickets.",
        ])

        result = rv1.probe(cv_text, "Support Technician", scorer,
                           paraphrase_fn=paraphrase_fn, score_rejected=False)

        assert result.rejected is True
        assert result.error is None          # declined, not broken
        assert result.score_before == 3
        assert result.score_after is None
        assert result.delta is None

    def test_score_rejected_false_still_records_every_free_measurement(self):
        """Only the scoring call is skipped. Everything the guard and the text
        comparison already computed is still on the row, so the report can say
        what was wrong with the rewrite."""
        cv_text = "Assisted with support tickets."
        scorer = _fake_scorer([3])
        paraphrase_fn = _fake_paraphrase([
            "Resolved 500 support tickets.",
            "Resolved 900 support tickets.",
        ])

        result = rv1.probe(cv_text, "Support Technician", scorer,
                           paraphrase_fn=paraphrase_fn, score_rejected=False)

        assert result.retried is True
        assert any("900" in r for r in result.reasons)
        assert result.similarity is not None
        assert result.length_ratio is not None
        assert result.facts_lost is not None

    def test_score_rejected_false_still_scores_a_clean_rewrite(self):
        """The switch turns off the scoring of a FAILED rewrite, not scoring.
        A rewrite that passes the guard is what the probe exists to measure."""
        cv_text = "Assisted with support tickets."
        scorer = _fake_scorer([3, 5])
        paraphrase_fn = _fake_paraphrase(
            ["Provided assistance with support tickets."])

        result = rv1.probe(cv_text, "Support Technician", scorer,
                           paraphrase_fn=paraphrase_fn, score_rejected=False)

        assert result.rejected is False
        assert result.score_after == 5
        assert result.delta == 2

    def test_score_rejected_false_still_scores_a_retry_that_came_back_clean(self):
        """A first attempt that was rejected and a strict retry that was not
        is a successful rewrite, and the delta is the finding. Keying the skip
        off `retried` rather than off the final verdict would throw it away."""
        cv_text = "Assisted with support tickets."
        scorer = _fake_scorer([3, 4])
        paraphrase_fn = _fake_paraphrase([
            "Resolved 500 support tickets.",
            "Provided assistance with support tickets.",
        ])

        result = rv1.probe(cv_text, "Support Technician", scorer,
                           paraphrase_fn=paraphrase_fn, score_rejected=False)

        assert result.retried is True and result.rejected is False
        assert result.delta == 1


# ---------------- verify_determinism ----------------

class TestVerifyDetermism:
    def test_empty_drift_when_scores_reproduce(self):
        cvs = {"Alex": "cv A", "Sam": "cv B"}

        def scorer(text, job_title):
            return {"cv A": 3, "cv B": 5, rv1._INTERLEAVE_CV: 1}[text]

        drift = rv1.verify_determinism(cvs, "Role", scorer)
        assert drift == []

    def test_flags_a_candidate_whose_score_changed(self):
        cvs = {"Flaky": "cv A"}
        state = {"n": 0}

        def scorer(text, job_title):
            if text == rv1._INTERLEAVE_CV:
                return 0
            state["n"] += 1
            return 3 if state["n"] == 1 else 4  # drifts on the second scoring

        drift = rv1.verify_determinism(cvs, "Role", scorer)
        assert len(drift) == 1
        assert drift[0][0] == "Flaky"
        assert drift[0][1:] == (3, 4)

    def test_interleave_call_happens_between_the_two_scorings(self):
        cvs = {"Alex": "cv A"}
        calls = []

        def scorer(text, job_title):
            calls.append(text)
            return 1

        rv1.verify_determinism(cvs, "Role", scorer)
        assert calls == ["cv A", rv1._INTERLEAVE_CV, "cv A"]


# ---------------- title_from_slug ----------------

class TestTitleFromSlug:
    def test_strips_prefix_and_title_cases(self):
        assert rv1.title_from_slug(
            "job_listing_startup_fullstack_engineer"
        ) == "Startup Fullstack Engineer"

    def test_slug_without_prefix(self):
        assert rv1.title_from_slug("secretary") == "Secretary"


# ---------------- read_rows / already_done ----------------

class TestReadRows:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert rv1.read_rows(tmp_path / "does_not_exist.jsonl") == []

    def test_parses_valid_jsonl(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
        assert rv1.read_rows(path) == [{"a": 1}, {"a": 2}]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        assert rv1.read_rows(path) == [{"a": 1}, {"a": 2}]

    def test_skips_torn_final_line(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        path.write_text('{"a": 1}\n{"a": 2, "unterminat', encoding="utf-8")
        assert rv1.read_rows(path) == [{"a": 1}]


class TestAlreadyDone:
    def test_excludes_errored_rows(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        rows = [
            {"rewriter_model": "m", "scorer_model": "s", "job": "j", "candidate": "c", "error": None},
            {"rewriter_model": "m", "scorer_model": "s", "job": "j2", "candidate": "c2", "error": "boom"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        done = rv1.already_done(path)
        assert ("m", "s", "j", "c") in done
        assert ("m", "s", "j2", "c2") not in done

    def test_missing_file_gives_empty_set(self, tmp_path):
        assert rv1.already_done(tmp_path / "missing.jsonl") == set()


# ---------------- apply_filter ----------------

class TestApplyFilter:
    def test_no_filter_returns_everything(self):
        items = {"a": 1, "b": 2}
        assert rv1.apply_filter(items, None, "things") == items
        assert rv1.apply_filter(items, [], "things") == items

    def test_filter_keeps_only_wanted_keys(self):
        items = {"a": 1, "b": 2, "c": 3}
        assert rv1.apply_filter(items, ["a", "c"], "things") == {"a": 1, "c": 3}

    def test_filter_matching_nothing_raises_system_exit(self):
        import pytest
        items = {"a": 1, "b": 2}
        with pytest.raises(SystemExit):
            rv1.apply_filter(items, ["nope"], "things")


# ---------------- row_truncated ----------------

class TestRowTruncated:
    def test_false_when_all_fields_falsy(self):
        row = {"truncated_before": False, "truncated_after": None, "truncated_rewrite": False}
        assert rv1.row_truncated(row) is False

    def test_true_when_any_field_truthy(self):
        row = {"truncated_before": False, "truncated_after": True, "truncated_rewrite": False}
        assert rv1.row_truncated(row) is True

    def test_missing_fields_treated_as_falsy(self):
        assert rv1.row_truncated({}) is False


# ---------------- summarise ----------------

class TestSummarise:
    def test_no_rows(self):
        assert rv1.summarise([], flag_threshold=2) == "no rows"

    def test_counts_split_by_category(self):
        rows = [
            {"error": "boom"},
            {"error": None, "delta": 0, "rejected": False, "scorer_model": "m1",
             "score_before": 3, "truncated_before": False, "truncated_after": False,
             "truncated_rewrite": False, "advisory": []},
            {"error": None, "delta": 5, "rejected": False, "scorer_model": "m1",
             "score_before": 2, "truncated_before": False, "truncated_after": False,
             "truncated_rewrite": False, "advisory": []},
            {"error": None, "delta": None, "rejected": True, "scorer_model": "m1",
             "reasons": ["invented: 500"], "score_before": 3,
             "truncated_before": False, "truncated_after": False,
             "truncated_rewrite": False, "advisory": []},
        ]
        summary = rv1.summarise(rows, flag_threshold=2)

        # 2 clean scored rows (delta not None, not rejected), 1 rejected
        # (delta is None so not in `scored`... use a rejected-but-scored
        # row instead to hit the intended branch)
        assert "1 errored" in summary

    def test_rejected_row_with_a_delta_is_counted_in_scored_and_rejected(self):
        rows = [
            {"error": None, "delta": 0, "rejected": True, "scorer_model": "m1",
             "reasons": ["number dropped from the CV: 5"], "score_before": 3,
             "truncated_before": False, "truncated_after": False,
             "truncated_rewrite": False, "advisory": []},
        ]
        summary = rv1.summarise(rows, flag_threshold=2)
        assert "0 clean, 1 rejected (of 1 scored)" in summary

    def test_flag_threshold_controls_flagged_count(self):
        rows = [
            {"error": None, "delta": 1, "rejected": False, "scorer_model": "m1",
             "score_before": 3, "truncated_before": False, "truncated_after": False,
             "truncated_rewrite": False, "advisory": []},
            {"error": None, "delta": 5, "rejected": False, "scorer_model": "m1",
             "score_before": 3, "truncated_before": False, "truncated_after": False,
             "truncated_rewrite": False, "advisory": []},
        ]
        summary = rv1.summarise(rows, flag_threshold=2)
        assert "1 flagged (|delta| >= 2) of 2 clean rewrites" in summary

    def test_truncated_rows_excluded_from_clean_but_counted(self):
        rows = [
            {"error": None, "delta": 3, "rejected": False, "scorer_model": "m1",
             "score_before": 3, "truncated_before": True, "truncated_after": False,
             "truncated_rewrite": False, "advisory": []},
        ]
        summary = rv1.summarise(rows, flag_threshold=2)
        assert "0 clean, 0 rejected (of 1 scored), 1 truncated (excluded)" in summary


# --- the probe's timeout and the models it leaves loaded --------------------
#
# The default was 300s, which is a budget for a fast machine rather than a
# guard against a hung one. A 14B model that does not fit in VRAM spills to
# system RAM and generates at a few tokens a second; a CV-length rewrite is
# about a thousand output tokens, so every rewrite on such a machine died at
# five minutes with "TimeoutError: timed out".


def test_the_rewriter_timeout_is_a_hung_daemon_guard_not_a_speed_budget():
    assert rv1.DEFAULT_TIMEOUT >= 1800
    # Both halves of the probe: the paraphrase and the re-score.
    import inspect
    for fn in (rv1.paraphrase, rv1.make_scorer):
        assert (inspect.signature(fn).parameters["timeout"].default
                == rv1.DEFAULT_TIMEOUT), fn.__name__


def _urlopen_times_out(monkeypatch):
    monkeypatch.setattr(
        rv1.urllib.request, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out")))


def test_a_rewrite_timeout_names_the_model_and_the_budget(monkeypatch):
    """"TimeoutError: timed out" names neither, so a reviewer reading it on
    the report cannot tell a dead daemon from a model too large for the
    machine - the common cause, and the one with a different fix."""
    _urlopen_times_out(monkeypatch)

    with pytest.raises(rv1.RewriteError) as caught:
        rv1._ollama_text("qwen2.5:14b", "prompt", 8192, 0, 1800)

    assert "qwen2.5:14b" in str(caught.value)
    assert "1800s" in str(caught.value)
    assert "rewriting" in str(caught.value)


def test_a_scoring_timeout_says_so_too():
    """The re-score reaches Ollama through run_evaluations.call_ollama, which
    raises the same bare TimeoutError. Both halves of the probe can be the one
    that runs out of time, and the report has to say which."""
    def times_out(model, prompt, num_ctx, seed, timeout):
        raise TimeoutError("timed out")

    scorer = rv1.make_scorer("phi4", call=times_out, timeout=1800)

    with pytest.raises(rv1.ScoreError) as caught:
        scorer("cv", "Role")

    assert "phi4" in str(caught.value)
    assert "1800s" in str(caught.value)
    assert "scoring" in str(caught.value)


def test_a_timed_out_rewrite_is_still_recorded_as_a_row_not_a_crash():
    """RewriteError is already in probe()'s caught tuple, so the new message
    must not change probe()'s contract: one bad candidate cannot end a batch."""
    def times_out(cv_text, job_title, **kwargs):
        raise rv1.RewriteError("qwen2.5:14b did not finish rewriting within 1800s.")

    result = rv1.probe("cv text", "Role", _fake_scorer([3]),
                       paraphrase_fn=times_out)

    assert result.error is not None and "1800s" in result.error
    assert result.score_before == 3
    assert result.score_after is None and result.delta is None


def test_the_rewriting_model_is_recorded_for_the_unloader(monkeypatch):
    """rewriter_v1 talks to Ollama through its own urllib call, so nothing in
    ollama_utils sees it. Unrecorded, the probe's 14B stayed resident for its
    30-minute keep_alive after every run and after Ctrl-C."""
    import ollama_utils
    noted = []
    monkeypatch.setattr(ollama_utils, "note_local_model",
                        lambda model, host=None: noted.append((model, host)))
    _urlopen_times_out(monkeypatch)

    with pytest.raises(rv1.RewriteError):
        rv1._ollama_text("qwen2.5:14b", "prompt", 8192, 0, 1800)

    # Recorded BEFORE the call, so a request that timed out - which has still
    # loaded the model - is exactly the case the unloader can see.
    assert noted == [("qwen2.5:14b", rv1.OLLAMA_URL)]


def test_the_scoring_model_is_recorded_for_the_unloader(monkeypatch):
    """The scorer goes through run_evaluations.call_ollama, which predates the
    web app, sends keep_alive 30m and records nothing."""
    import ollama_utils
    noted = []
    monkeypatch.setattr(ollama_utils, "note_local_model",
                        lambda model, host=None: noted.append(model))

    scorer = rv1.make_scorer(
        "phi4", call=lambda *a: {"response": '{"score": 5}',
                                 "prompt_eval_count": 10})
    scorer("cv", "Role")

    assert noted == ["phi4"]
