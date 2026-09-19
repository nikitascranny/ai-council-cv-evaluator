"""
Unit tests for rewriter/rewriter_v2.py.

document_utils and ollama_utils are stubbed (see common/document_utils.py
and common/ollama_utils.py) purely so the module's top-level imports
succeed - these tests never call the stubs. call_ollama_text is
monkeypatched per test to drive _run_agent through each branch of the
guard -> retry -> fallback flow without touching a real model.

fact_guard itself is NOT mocked in most of these tests: rewriter_v2 imports
rejection_reasons/advisory_caps_runs directly into its own namespace, and
using the real functions means these tests also catch integration breaks
(e.g. a fact_guard signature change) rather than just testing against a
mock that assumes the old contract still holds.
"""

import csv
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rewriter"))

import rewriter_v2 as rv2


# ---------------- build_rewrite_prompt ----------------

class TestBuildRewritePrompt:
    def test_substitutes_cv_text_and_job_title(self):
        prompt = rv2.build_rewrite_prompt("MY CV BODY", "Some Role")
        assert "MY CV BODY" in prompt
        assert "Some Role" in prompt

    def test_curly_braces_in_cv_text_survive_literally(self):
        # Plain string.replace is used specifically because a CV can
        # contain literal "{" / "}" (template artifacts) that .format()
        # would misparse as fields. Confirm they pass through untouched.
        cv_text = "Skills: {Python, C++}, salary expectation: {negotiable}"
        prompt = rv2.build_rewrite_prompt(cv_text, "Role")
        assert cv_text in prompt

    def test_job_title_is_inserted_directly(self):
        # Per the function's own docstring: v2 has no job description to
        # infer the role from, so job_title IS substituted into the prompt
        # (unlike rewriter_v1, which withholds it).
        prompt = rv2.build_rewrite_prompt("cv", "UNIQUE_TITLE_TOKEN")
        assert "UNIQUE_TITLE_TOKEN" in prompt
        assert "{job_title}" not in prompt

    def test_strict_false_omits_strict_suffix(self):
        prompt = rv2.build_rewrite_prompt("cv", "Role", strict=False)
        assert "INVENTED information" not in prompt

    def test_strict_true_appends_strict_suffix(self):
        prompt = rv2.build_rewrite_prompt("cv", "Role", strict=True)
        assert "INVENTED information" in prompt
        # Strict suffix must come after the base prompt content, not replace it
        assert prompt.index("cv") < prompt.index("INVENTED information") or "cv" in prompt


# ---------------- safe_path_part (results_io, as rewriter_v2 uses it) ----------------

class TestSafeFolderName:
    def test_leaves_simple_name_untouched(self):
        assert rv2.safe_path_part("Alex Park") == "Alex Park"

    def test_replaces_colon(self):
        # The exact NTFS alternate-data-stream bug: a colon in a model
        # tag like "qwen2.5:14b" must not survive into a filename.
        assert ":" not in rv2.safe_path_part("qwen2.5:14b")

    def test_replaces_slash_and_backslash(self):
        result = rv2.safe_path_part("Path/With\\Slashes")
        assert "/" not in result and "\\" not in result

    def test_keeps_hyphen_underscore_parens_dots_and_spaces(self):
        name = "Full-Stack (Startup) Engineer_v2.1"
        assert rv2.safe_path_part(name) == name

    def test_keeps_version_dot_in_model_name(self):
        # Dots must survive - only ":" (the NTFS trap) needs stripping.
        assert rv2.safe_path_part("qwen2.5") == "qwen2.5"
        assert rv2.safe_path_part("qwen2.5:14b") == "qwen2.5_14b"

    def test_strips_leading_and_trailing_whitespace(self):
        assert rv2.safe_path_part("  Trailing Dots ") == "Trailing Dots"

    def test_strips_trailing_dot(self):
        # A trailing "." is invalid in a Windows filename/foldername even
        # though a mid-string "." is fine.
        assert rv2.safe_path_part("Trailing Dot.") == "Trailing Dot"

    def test_empty_name_falls_back(self):
        assert rv2.safe_path_part("") == "unnamed"

    def test_whitespace_only_name_falls_back(self):
        assert rv2.safe_path_part("   ") == "unnamed"

    def test_colon_only_name_does_not_fall_back(self):
        # Colons become underscores, which are valid characters - the
        # result is non-empty, so this should NOT hit the "unnamed" path.
        assert rv2.safe_path_part("::::") == "____"


# ---------------- RewriteResult ----------------

class TestRewriteResult:
    def test_to_dict_round_trips_fields(self):
        r = rv2.RewriteResult(
            candidate="Alex Park", job_title="Team Lead", agent_model="phi4",
            original_text="orig", rewritten_text="new",
        )
        d = r.to_dict()
        assert d["candidate"] == "Alex Park"
        assert d["rewritten_text"] == "new"
        assert d["reasons"] == []
        assert d["used_fallback"] is False


# ---------------- _run_agent ----------------

class TestRunAgent:
    def test_clean_first_try_is_accepted(self, monkeypatch):
        monkeypatch.setattr(rv2, "call_ollama_text", lambda model, prompt, num_predict: ("REWRITTEN TEXT", None))
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: [])
        monkeypatch.setattr(rv2, "advisory_caps_runs", lambda *a, **k: [])

        result = rv2._run_agent("original cv", "Role", "Candidate", "phi4")

        assert result.rewritten_text == "REWRITTEN TEXT"
        assert result.rejected is False
        assert result.retried is False
        assert result.used_fallback is False
        assert result.error is None

    def test_error_on_first_call_short_circuits(self, monkeypatch):
        called = {"count": 0}

        def fake_call(model, prompt, num_predict):
            called["count"] += 1
            return None, "connection refused"

        monkeypatch.setattr(rv2, "call_ollama_text", fake_call)

        result = rv2._run_agent("original cv", "Role", "Candidate", "phi4")

        assert result.error == "connection refused"
        assert result.rewritten_text == "original cv"  # falls back to init value
        assert called["count"] == 1  # never reaches the retry

    def test_flagged_first_try_then_clean_retry_uses_retry_text(self, monkeypatch):
        calls = []

        def fake_call(model, prompt, num_predict):
            calls.append(prompt)
            if len(calls) == 1:
                return "BAD FIRST DRAFT", None
            return "GOOD RETRY DRAFT", None

        def fake_reasons(original, rewritten, job_title):
            return ["invented fact"] if rewritten == "BAD FIRST DRAFT" else []

        monkeypatch.setattr(rv2, "call_ollama_text", fake_call)
        monkeypatch.setattr(rv2, "rejection_reasons", fake_reasons)
        monkeypatch.setattr(rv2, "advisory_caps_runs", lambda *a, **k: [])

        result = rv2._run_agent("original cv", "Role", "Candidate", "phi4")

        assert len(calls) == 2
        assert result.retried is True
        assert result.rejected is False
        assert result.used_fallback is False
        assert result.rewritten_text == "GOOD RETRY DRAFT"

    def test_flagged_after_retry_falls_back_to_original(self, monkeypatch):
        monkeypatch.setattr(rv2, "call_ollama_text", lambda model, prompt, num_predict: ("STILL BAD", None))
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: ["invented fact"])
        monkeypatch.setattr(rv2, "advisory_caps_runs", lambda *a, **k: [])

        result = rv2._run_agent("ORIGINAL CV TEXT", "Role", "Candidate", "phi4")

        assert result.retried is True
        assert result.rejected is True
        assert result.used_fallback is True
        assert result.rewritten_text == "ORIGINAL CV TEXT"

    def test_error_on_retry_call_is_recorded(self, monkeypatch):
        calls = {"count": 0}

        def fake_call(model, prompt, num_predict):
            calls["count"] += 1
            if calls["count"] == 1:
                return "BAD FIRST DRAFT", None
            return None, "timeout"

        monkeypatch.setattr(rv2, "call_ollama_text", fake_call)
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: ["invented fact"])

        result = rv2._run_agent("original cv", "Role", "Candidate", "phi4")

        assert result.error == "timeout"
        assert result.retried is True
        assert calls["count"] == 2

    def test_length_ratio_and_similarity_are_computed_on_success(self, monkeypatch):
        monkeypatch.setattr(rv2, "call_ollama_text", lambda model, prompt, num_predict: ("word word word", None))
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: [])
        monkeypatch.setattr(rv2, "advisory_caps_runs", lambda *a, **k: [])

        result = rv2._run_agent("word word", "Role", "Candidate", "phi4")

        assert result.length_ratio == len("word word word") / len("word word")
        assert 0.0 <= result.similarity <= 1.0

    def test_length_ratio_is_none_for_empty_original(self, monkeypatch):
        monkeypatch.setattr(rv2, "call_ollama_text", lambda model, prompt, num_predict: ("anything", None))
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: [])
        monkeypatch.setattr(rv2, "advisory_caps_runs", lambda *a, **k: [])

        result = rv2._run_agent("", "Role", "Candidate", "phi4")

        assert result.length_ratio is None


# ---------------- rewrite_cv ----------------

class TestRewriteCv:
    def test_runs_one_result_per_agent_model(self, monkeypatch):
        monkeypatch.setattr(rv2, "AGENT_MODELS", ["phi4", "qwen2.5:14b"])
        monkeypatch.setattr(rv2, "call_ollama_text", lambda model, prompt, num_predict: (f"text from {model}", None))
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: [])
        monkeypatch.setattr(rv2, "advisory_caps_runs", lambda *a, **k: [])

        results = rv2.rewrite_cv("cv text", "Role", candidate_name="Candidate")

        assert [r.agent_model for r in results] == ["phi4", "qwen2.5:14b"]
        assert results[0].rewritten_text == "text from phi4"
        assert results[1].rewritten_text == "text from qwen2.5:14b"


# ---------------- save_results ----------------

class TestSaveResults:
    def _make_results(self, **overrides):
        base = dict(
            candidate="Alex Park", job_title="Computer Repair Technician",
            agent_model="phi4", original_text="orig", rewritten_text="rewritten text",
        )
        base.update(overrides)
        return rv2.RewriteResult(**base)

    def test_returns_none_for_empty_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)
        assert rv2.save_results([]) is None

    def test_nests_by_job_title_then_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)
        results = [self._make_results()]

        saved_dir = rv2.save_results(results)

        expected = tmp_path / "Computer Repair Technician" / "Alex Park"
        assert saved_dir == expected
        assert expected.is_dir()

    def test_model_name_with_colon_is_sanitized_on_disk(self, tmp_path, monkeypatch):
        # Regression test for the NTFS alternate-data-stream bug: writing
        # to a path containing ":" must not happen.
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)
        results = [self._make_results(agent_model="qwen2.5:14b", rewritten_text="hello world")]

        saved_dir = rv2.save_results(results)

        bad_path = Path(str(saved_dir / "qwen2.5:14b.txt"))
        good_path = saved_dir / "qwen2.5_14b.txt"
        assert not bad_path.exists()
        assert good_path.exists()
        assert good_path.read_text(encoding="utf-8") == "hello world"

    def test_writes_one_txt_file_per_result_with_correct_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)
        results = [
            self._make_results(agent_model="phi4", rewritten_text="phi4 output"),
            self._make_results(agent_model="qwen2.5", rewritten_text="qwen output"),
        ]

        saved_dir = rv2.save_results(results)

        assert (saved_dir / "phi4.txt").read_text(encoding="utf-8") == "phi4 output"
        assert (saved_dir / "qwen2.5.txt").read_text(encoding="utf-8") == "qwen output"

    def test_writes_metadata_csv_with_expected_columns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)
        results = [self._make_results(rejected=True, reasons=["bad"], used_fallback=True)]

        saved_dir = rv2.save_results(results)
        csv_path = saved_dir / "metadata.csv"
        assert csv_path.exists()

        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        row = rows[0]
        assert row["candidate"] == "Alex Park"
        assert row["agent_model"] == "phi4"
        assert row["used_fallback"] == "True"
        # Large text fields are deliberately excluded from the CSV
        assert "rewritten_text" not in row
        assert "original_text" not in row

    def test_unsafe_job_title_and_candidate_are_sanitized_in_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)
        results = [self._make_results(
            job_title="Weird/Job:Title",
            candidate="Weird:Candidate/Name",
        )]

        saved_dir = rv2.save_results(results)

        assert ":" not in str(saved_dir.relative_to(tmp_path))
        assert "/" not in saved_dir.name  # candidate folder name itself has no separator
        assert saved_dir.is_dir()

    def test_second_job_for_same_candidate_does_not_overwrite_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)

        job_a = [self._make_results(job_title="Job A", rewritten_text="result for job A")]
        job_b = [self._make_results(job_title="Job B", rewritten_text="result for job B")]

        dir_a = rv2.save_results(job_a)
        dir_b = rv2.save_results(job_b)

        assert dir_a != dir_b
        assert (dir_a / "phi4.txt").read_text(encoding="utf-8") == "result for job A"
        assert (dir_b / "phi4.txt").read_text(encoding="utf-8") == "result for job B"


# ---------------- rewrite_cv on Ollama Cloud ----------------

class TestRewriteCvCloud:
    """profile="cloud": the cloud pair, with each slot falling back to its
    local partner when its cloud call fails."""

    @pytest.fixture(autouse=True)
    def _guard_passes(self, monkeypatch):
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: [])
        monkeypatch.setattr(rv2, "advisory_caps_runs", lambda *a, **k: [])
        monkeypatch.setenv(rv2.CLOUD_API_KEY_ENV, "test-key")

    def _record_calls(self, monkeypatch, fail=()):
        """call_ollama_text that records every call and fails the models named."""
        calls = []

        def fake_call(model, prompt, **kw):
            calls.append((model, kw))
            if model in fail:
                return "", "REQUEST ERROR: 429 Client Error: Too Many Requests"
            if "stats" in kw:
                kw["stats"].update(prompt_eval_count=100, eval_count=50)
            return f"text from {model}", None

        monkeypatch.setattr(rv2, "call_ollama_text", fake_call)
        return calls

    def test_the_default_profile_is_still_the_local_pair(self, monkeypatch):
        calls = self._record_calls(monkeypatch)

        results = rv2.rewrite_cv("cv", "Role", "Candidate")

        assert [r.agent_model for r in results] == ["phi4", "qwen2.5:14b"]
        assert [r.provider for r in results] == ["local", "local"]
        # The local call is exactly the one v2 always made.
        assert [kw for _, kw in calls] == [{"num_predict": rv2.REWRITE_NUM_PREDICT}] * 2

    def test_an_unknown_profile_is_refused(self):
        with pytest.raises(ValueError):
            rv2.rewrite_cv("cv", "Role", "Candidate", profile="gpu-farm")

    def test_cloud_runs_the_cloud_pair_on_ollama_com_with_the_key(self, monkeypatch):
        calls = self._record_calls(monkeypatch)

        results = rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")

        assert [r.agent_model for r in results] == ["gpt-oss:120b", "gemma4:31b"]
        assert [r.provider for r in results] == ["cloud", "cloud"]
        assert all(r.replaced_cloud_model is None for r in results)
        for _, kw in calls:
            assert kw["host"] == "https://ollama.com/api/chat"
            assert kw["headers"] == {"Authorization": "Bearer test-key"}
            assert kw["keep_alive"] is None
            assert kw["num_predict"] == rv2.CLOUD_NUM_PREDICT

    def test_only_gpt_oss_is_told_how_hard_to_think(self, monkeypatch):
        calls = self._record_calls(monkeypatch)

        rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")

        think = {model: kw.get("think") for model, kw in calls}
        assert think == {"gpt-oss:120b": "low", "gemma4:31b": None}

    def test_cloud_tokens_are_added_up_across_the_retry(self, monkeypatch):
        self._record_calls(monkeypatch)
        checked = []

        def first_draft_rejected(original, rewritten, job_title):
            checked.append(rewritten)
            return ["invented"] if len(checked) == 1 else []

        monkeypatch.setattr(rv2, "rejection_reasons", first_draft_rejected)

        result = rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")[0]

        assert result.retried is True
        assert (result.tokens_in, result.tokens_out) == (200, 100)

    def test_a_failed_cloud_agent_is_replaced_by_its_local_partner(self, monkeypatch):
        self._record_calls(monkeypatch, fail={"gemma4:31b"})

        results = rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")

        assert [r.agent_model for r in results] == ["gpt-oss:120b", "qwen2.5:14b"]
        assert results[0].provider == "cloud"
        stand_in = results[1]
        assert stand_in.provider == "local"
        assert stand_in.replaced_cloud_model == "gemma4:31b"
        assert "429" in stand_in.cloud_error
        assert stand_in.error is None
        assert stand_in.rewritten_text == "text from qwen2.5:14b"

    def test_once_the_cloud_fails_the_rest_of_the_run_stays_local(self, monkeypatch):
        """A dead connection or a spent plan fails the next call the same way,
        and a timeout would make the candidate wait for it twice."""
        calls = self._record_calls(monkeypatch, fail={"gpt-oss:120b"})

        results = rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")

        assert [model for model, _ in calls] == ["gpt-oss:120b", "phi4", "qwen2.5:14b"]
        assert [r.agent_model for r in results] == ["phi4", "qwen2.5:14b"]
        assert [r.replaced_cloud_model for r in results] == ["gpt-oss:120b", "gemma4:31b"]
        assert all("429" in r.cloud_error for r in results)

    def test_no_key_means_no_cloud_call_and_both_slots_run_locally(self, monkeypatch):
        monkeypatch.delenv(rv2.CLOUD_API_KEY_ENV)
        calls = self._record_calls(monkeypatch)

        results = rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")

        assert [model for model, _ in calls] == ["phi4", "qwen2.5:14b"]
        assert all(r.cloud_error == "OLLAMA_API_KEY is not set" for r in results)

    def test_a_guard_rejection_on_the_cloud_is_a_result_not_an_outage(self, monkeypatch):
        """The model answered - the guard just would not vouch for it. That is
        a finding about the cloud model, and running a local one instead would
        hide it."""
        calls = self._record_calls(monkeypatch)
        monkeypatch.setattr(rv2, "rejection_reasons", lambda *a, **k: ["invented"])

        results = rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")

        assert {model for model, _ in calls} == {"gpt-oss:120b", "gemma4:31b"}
        assert all(r.used_fallback and r.provider == "cloud" for r in results)
        assert all(r.replaced_cloud_model is None for r in results)

    def test_every_cloud_agent_falls_back_to_a_real_local_agent(self):
        assert [local for _, local in rv2.CLOUD_AGENTS] == rv2.AGENT_MODELS

    def test_metadata_csv_records_where_each_version_ran(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv2, "REWRITE_RESULTS_FOLDER", tmp_path)
        self._record_calls(monkeypatch, fail={"gemma4:31b"})
        results = rv2.rewrite_cv("cv", "Role", "Candidate", profile="cloud")

        saved_dir = rv2.save_results(results)

        with (saved_dir / "metadata.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["provider"] for r in rows] == ["cloud", "local"]
        assert rows[1]["replaced_cloud_model"] == "gemma4:31b"
        assert rows[0]["tokens_in"] == "100"
