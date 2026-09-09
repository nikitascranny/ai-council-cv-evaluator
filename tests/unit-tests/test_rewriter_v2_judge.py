"""
test_rewriter_v2_judge.py

Unit tests for rewriter_v2_judge.py. Mocks out ollama_utils.call_ollama_json,
results_io.save_json, and document_utils.extract_text_from_pdf at the points
where rewriter_v2_judge.py imports them, so these tests never touch a real
Ollama server, the real filesystem outside tmp_path, or a real PDF.

Run with:
    pytest test_rewriter_v2_judge.py -v
"""

import csv
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import rewriter_v2_judge as rvj


# ---------------------------------------------------------------------------
# _parse_list_field / _parse_bool_field
# ---------------------------------------------------------------------------

class TestParseListField:
    def test_empty_string_returns_empty_list(self):
        assert rvj._parse_list_field("") == []

    def test_valid_repr_list_round_trips(self):
        assert rvj._parse_list_field("['a', 'b']") == ["a", "b"]

    def test_malformed_syntax_returns_empty_list_not_raises(self):
        assert rvj._parse_list_field("[unterminated") == []

    def test_valid_python_literal_that_is_not_a_list_returns_empty(self):
        # e.g. a stray dict or string ending up in the column
        assert rvj._parse_list_field("'just a string'") == []
        assert rvj._parse_list_field("{'a': 1}") == []


class TestParseBoolField:
    @pytest.mark.parametrize("raw", ["True", "true", " TRUE ", "tRuE"])
    def test_truthy_variants(self, raw):
        assert rvj._parse_bool_field(raw) is True

    @pytest.mark.parametrize("raw", ["False", "false", "", "0", "yes", "1"])
    def test_falsy_variants(self, raw):
        assert rvj._parse_bool_field(raw) is False


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------

class TestLoadMetadata:
    def _write_csv(self, tmp_path, rows, fieldnames):
        path = tmp_path / "metadata.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_converts_bool_and_list_columns_per_row(self, tmp_path):
        fieldnames = ["agent_model", "used_fallback", "rejected", "retried", "reasons", "advisory"]
        rows = [
            {"agent_model": "phi4", "used_fallback": "True", "rejected": "False",
             "retried": "True", "reasons": "['dropped a fact']", "advisory": "[]"},
            {"agent_model": "qwen2.5:14b", "used_fallback": "False", "rejected": "False",
             "retried": "False", "reasons": "[]", "advisory": "['re-casing']"},
        ]
        path = self._write_csv(tmp_path, rows, fieldnames)

        result = rvj.load_metadata(path)

        assert result[0]["used_fallback"] is True
        assert result[0]["rejected"] is False
        assert result[0]["retried"] is True
        assert result[0]["reasons"] == ["dropped a fact"]
        assert result[0]["advisory"] == []
        assert result[1]["advisory"] == ["re-casing"]

    def test_missing_optional_columns_default_safely(self, tmp_path):
        # DictReader gives "" for a column present in header but empty; here
        # we omit the columns entirely to check .get() defaults are used.
        path = self._write_csv(tmp_path, [{"agent_model": "phi4"}], ["agent_model"])

        result = rvj.load_metadata(path)

        row = result[0]
        assert row["used_fallback"] is False
        assert row["rejected"] is False
        assert row["retried"] is False
        assert row["reasons"] == []
        assert row["advisory"] == []

    def test_empty_csv_body_returns_empty_list(self, tmp_path):
        path = self._write_csv(tmp_path, [], ["agent_model"])

        assert rvj.load_metadata(path) == []


# ---------------------------------------------------------------------------
# find_job_listing_pdf
# ---------------------------------------------------------------------------

class TestFindJobListingPdf:
    def test_finds_pdf_via_lowercase_underscored_slug(self, tmp_path):
        pdf = tmp_path / "job_listing_computer_repair_technician.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        result = rvj.find_job_listing_pdf("Computer Repair Technician", tmp_path)

        assert result == pdf

    def test_returns_none_when_no_matching_pdf(self, tmp_path):
        result = rvj.find_job_listing_pdf("Nonexistent Job", tmp_path)

        assert result is None

    def test_strips_surrounding_whitespace_in_title(self, tmp_path):
        pdf = tmp_path / "job_listing_cashier.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        result = rvj.find_job_listing_pdf("  Cashier  ", tmp_path)

        assert result == pdf


# ---------------------------------------------------------------------------
# load_submission
# ---------------------------------------------------------------------------

class TestLoadSubmission:
    def _make_submission_dir(self, tmp_path, rows, write_texts=True):
        sub_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        sub_dir.mkdir(parents=True)
        fieldnames = ["agent_model", "used_fallback", "rejected", "retried",
                      "reasons", "advisory", "length_ratio", "similarity"]
        csv_path = sub_dir / "metadata.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if write_texts:
            for row in rows:
                text_path = sub_dir / f"{rvj.safe_filename_part(row['agent_model'])}.txt"
                text_path.write_text(f"CV text for {row['agent_model']}", encoding="utf-8")
        return sub_dir

    def _two_agent_rows(self):
        return [
            {"agent_model": "phi4", "used_fallback": "False", "rejected": "False",
             "retried": "False", "reasons": "[]", "advisory": "[]",
             "length_ratio": "1.1", "similarity": "0.8"},
            {"agent_model": "qwen2.5:14b", "used_fallback": "False", "rejected": "False",
             "retried": "False", "reasons": "[]", "advisory": "[]",
             "length_ratio": "0.9", "similarity": "0.7"},
        ]

    def test_missing_metadata_csv_returns_none_with_reason(self, tmp_path):
        sub_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        sub_dir.mkdir(parents=True)

        submission, reason = rvj.load_submission("Cashier", "ALEX PARK", sub_dir)

        assert submission is None
        assert "no metadata.csv" in reason

    @patch("rewriter_v2_judge.find_job_listing_pdf", return_value=None)
    def test_missing_job_listing_pdf_returns_none_with_reason(self, mock_find, tmp_path):
        sub_dir = self._make_submission_dir(tmp_path, self._two_agent_rows())

        submission, reason = rvj.load_submission("Cashier", "ALEX PARK", sub_dir)

        assert submission is None
        assert "no job listing PDF" in reason

    @patch("rewriter_v2_judge.find_job_listing_pdf")
    def test_empty_metadata_returns_none_with_reason(self, mock_find, tmp_path):
        mock_find.return_value = tmp_path / "job.pdf"
        sub_dir = self._make_submission_dir(tmp_path, [])

        submission, reason = rvj.load_submission("Cashier", "ALEX PARK", sub_dir)

        assert submission is None
        assert "empty" in reason

    @patch("rewriter_v2_judge.find_job_listing_pdf")
    def test_missing_text_file_returns_none_with_reason(self, mock_find, tmp_path):
        mock_find.return_value = tmp_path / "job.pdf"
        sub_dir = self._make_submission_dir(tmp_path, self._two_agent_rows(), write_texts=False)

        submission, reason = rvj.load_submission("Cashier", "ALEX PARK", sub_dir)

        assert submission is None
        assert "missing" in reason

    @patch("rewriter_v2_judge.find_job_listing_pdf")
    def test_fewer_than_two_rewrites_returns_none_with_reason(self, mock_find, tmp_path):
        mock_find.return_value = tmp_path / "job.pdf"
        sub_dir = self._make_submission_dir(tmp_path, self._two_agent_rows()[:1])

        submission, reason = rvj.load_submission("Cashier", "ALEX PARK", sub_dir)

        assert submission is None
        assert "need at least 2" in reason

    @patch("rewriter_v2_judge.find_job_listing_pdf")
    def test_successful_load_populates_text_and_fields(self, mock_find, tmp_path):
        job_pdf = tmp_path / "job.pdf"
        mock_find.return_value = job_pdf
        sub_dir = self._make_submission_dir(tmp_path, self._two_agent_rows())

        submission, reason = rvj.load_submission("Cashier", "ALEX PARK", sub_dir)

        assert reason is None
        assert submission.job_title == "Cashier"
        assert submission.candidate == "ALEX PARK"
        assert submission.job_pdf == job_pdf
        assert len(submission.rewrites) == 2
        assert submission.rewrites[0]["text"] == "CV text for phi4"
        assert submission.rewrites[1]["text"] == "CV text for qwen2.5:14b"


# ---------------------------------------------------------------------------
# discover_submissions
# ---------------------------------------------------------------------------

class TestDiscoverSubmissions:
    def test_nonexistent_results_dir_returns_empty_list(self, tmp_path):
        assert rvj.discover_submissions(tmp_path / "nope") == []

    def test_finds_all_job_candidate_pairs_sorted(self, tmp_path):
        (tmp_path / "Cashier" / "ALEX PARK").mkdir(parents=True)
        (tmp_path / "Cashier" / "ZOE ADAMS").mkdir(parents=True)
        (tmp_path / "Baker" / "SAM LEE").mkdir(parents=True)

        result = rvj.discover_submissions(tmp_path)

        job_titles = [job for job, _, _ in result]
        assert job_titles == sorted(job_titles)
        assert ("Baker", "SAM LEE", tmp_path / "Baker" / "SAM LEE") in result
        assert ("Cashier", "ALEX PARK", tmp_path / "Cashier" / "ALEX PARK") in result
        assert ("Cashier", "ZOE ADAMS", tmp_path / "Cashier" / "ZOE ADAMS") in result
        assert len(result) == 3

    def test_ignores_non_directory_entries(self, tmp_path):
        (tmp_path / "Cashier" / "ALEX PARK").mkdir(parents=True)
        (tmp_path / "stray_file.txt").write_text("not a job dir")

        result = rvj.discover_submissions(tmp_path)

        assert result == [("Cashier", "ALEX PARK", tmp_path / "Cashier" / "ALEX PARK")]


# ---------------------------------------------------------------------------
# _format_rewrite_block -- logic branches only (conditional lines, tolerance
# of a missing/invalid length_ratio), not exact wording/layout.
# ---------------------------------------------------------------------------

class TestFormatRewriteBlock:
    def _base_row(self, **overrides):
        row = {
            "agent_model": "phi4",
            "rejected": False,
            "used_fallback": False,
            "retried": False,
            "reasons": [],
            "advisory": [],
            "length_ratio": "1.0",
            "similarity": "0.5",
            "text": "some cv text",
        }
        row.update(overrides)
        return row

    @pytest.mark.parametrize("reasons, expect_present", [
        ([], False),
        (["dropped a fact"], True),
    ], ids=["omitted_when_empty", "included_when_present"])
    def test_reasons_line_presence(self, reasons, expect_present):
        block = rvj._format_rewrite_block("A", self._base_row(reasons=reasons))
        assert ("flagged issues" in block) is expect_present
        if expect_present:
            assert "dropped a fact" in block

    @pytest.mark.parametrize("advisory, expect_present", [
        ([], False),
        (["re-casing"], True),
    ], ids=["omitted_when_empty", "included_when_present"])
    def test_advisory_line_presence(self, advisory, expect_present):
        block = rvj._format_rewrite_block("A", self._base_row(advisory=advisory))
        assert ("advisory notes" in block) is expect_present
        if expect_present:
            assert "re-casing" in block

    def test_non_numeric_length_ratio_does_not_raise(self):
        # covers the shared except clause (KeyError/TypeError/ValueError) --
        # a missing key and a non-numeric value both land here.
        block = rvj._format_rewrite_block("A", self._base_row(length_ratio="not-a-number"))
        assert "length vs original" not in block

    def test_valid_length_ratio_produces_line(self):
        block = rvj._format_rewrite_block("A", self._base_row(length_ratio="1.25", similarity="0.6"))
        assert "length vs original" in block


# ---------------------------------------------------------------------------
# build_prompt -- structural correctness only (label count, agent order),
# not exact wording.
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def _submission(self, n_rewrites):
        models = ["phi4", "qwen2.5:14b", "llama3", "mistral"][:n_rewrites]
        rewrites = [
            {"agent_model": m, "rejected": False, "used_fallback": False,
             "retried": False, "reasons": [], "advisory": [],
             "length_ratio": "1.0", "similarity": "0.5", "text": f"text-{m}"}
            for m in models
        ]
        return rvj.RewriteSubmission(
            job_title="Cashier", candidate="ALEX PARK",
            job_pdf=Path("job.pdf"), rewrites=rewrites,
        )

    def test_uses_one_label_per_rewrite_for_two_rewrites(self):
        prompt = rvj.build_prompt("job text", self._submission(2))
        assert "REWRITE A" in prompt
        assert "REWRITE B" in prompt
        assert "REWRITE C" not in prompt

    def test_uses_one_label_per_rewrite_for_four_rewrites(self):
        prompt = rvj.build_prompt("job text", self._submission(4))
        for label in ("A", "B", "C", "D"):
            assert f"REWRITE {label}" in prompt

    def test_agent_names_appear_in_rewrite_order(self):
        prompt = rvj.build_prompt("job text", self._submission(3))
        assert prompt.index("phi4") < prompt.index("qwen2.5:14b") < prompt.index("llama3")


# ---------------------------------------------------------------------------
# _valid
# ---------------------------------------------------------------------------

class TestValid:
    def _agent_names(self):
        return ["phi4", "qwen2.5:14b"]

    def _valid_payload(self):
        return {
            "candidate": "ALEX PARK",
            "job_title": "Cashier",
            "recommended_rewrite": "phi4",
            "recommendation_reason": "reason text",
            "career_roadmap": [{"step": "Apply", "action": "Submit rewrite A"}],
            "alternative_careers": ["Inventory Analyst", "Logistics Coordinator"],
        }

    def test_valid_payload_passes(self):
        assert rvj._valid(self._valid_payload(), self._agent_names()) is True

    @pytest.mark.parametrize("mutate", [
        lambda p: p.pop("recommendation_reason"),
        lambda p: p.__setitem__("recommended_rewrite", "gpt-nonexistent"),
        lambda p: p.__setitem__("career_roadmap", "just a string"),
        lambda p: p.__setitem__("career_roadmap", []),
        lambda p: p.__setitem__("career_roadmap", [{"step": "Apply"}]),
        lambda p: p.__setitem__("career_roadmap", ["just apply"]),
        lambda p: p.__setitem__("alternative_careers", "just a string"),
        lambda p: p.__setitem__("alternative_careers", []),
        lambda p: p.__setitem__("alternative_careers", [{"career": "Inventory Analyst"}]),
        lambda p: p.__setitem__("alternative_careers", ["   "]),
    ], ids=[
        "missing_required_key",
        "recommended_rewrite_not_in_agent_names",
        "career_roadmap_not_a_list",
        "empty_career_roadmap",
        "career_roadmap_step_missing_action",
        "career_roadmap_item_not_a_dict",
        "alternative_careers_not_a_list",
        "empty_alternative_careers",
        "alternative_careers_with_non_string_entry",
        "alternative_careers_with_blank_string",
    ])
    def test_invalid_payloads_fail(self, mutate):
        payload = self._valid_payload()
        mutate(payload)
        assert rvj._valid(payload, self._agent_names()) is False

    def test_career_roadmap_with_multiple_valid_steps_passes(self):
        payload = self._valid_payload()
        payload["career_roadmap"] = [
            {"step": "Apply", "action": "Submit rewrite A"},
            {"step": "Upskill", "action": "Take a course"},
        ]
        assert rvj._valid(payload, self._agent_names()) is True

    def test_single_alternative_career_passes(self):
        payload = self._valid_payload()
        payload["alternative_careers"] = ["Inventory Analyst"]
        assert rvj._valid(payload, self._agent_names()) is True


# ---------------------------------------------------------------------------
# call_judge_with_retries
# ---------------------------------------------------------------------------

class TestCallJudgeWithRetries:
    AGENT_NAMES = ["phi4", "qwen2.5:14b"]
    VALID_REPLY = {
        "candidate": "ALEX PARK", "job_title": "Cashier",
        "recommended_rewrite": "phi4", "recommendation_reason": "why",
        "career_roadmap": [{"step": "Apply", "action": "do it"}],
        "alternative_careers": ["Inventory Analyst"],
    }

    def _call(self, prompt="prompt", max_retries=3):
        return rvj.call_judge_with_retries(
            prompt, self.AGENT_NAMES, model="deepseek-r1:8b",
            host="http://localhost:11434/api/chat", num_predict=4096,
            temperature=0.2, timeout=3600, max_retries=max_retries,
        )

    @patch("rewriter_v2_judge.call_ollama_json")
    def test_succeeds_on_first_attempt(self, mock_call):
        mock_call.return_value = (self.VALID_REPLY, None)

        parsed, error = self._call()

        assert parsed == self.VALID_REPLY
        assert error is None
        assert mock_call.call_count == 1

    @patch("rewriter_v2_judge.call_ollama_json")
    def test_retries_on_parse_error_then_succeeds(self, mock_call):
        mock_call.side_effect = [
            ({}, "PARSE ERROR"),
            (self.VALID_REPLY, None),
        ]

        parsed, error = self._call()

        assert parsed == self.VALID_REPLY
        assert error is None
        assert mock_call.call_count == 2

    @patch("rewriter_v2_judge.call_ollama_json")
    def test_retries_when_recommended_rewrite_names_unknown_agent(self, mock_call):
        bad_reply = dict(self.VALID_REPLY, recommended_rewrite="not-a-real-agent")
        mock_call.side_effect = [
            (bad_reply, None),
            (self.VALID_REPLY, None),
        ]

        parsed, error = self._call()

        assert parsed == self.VALID_REPLY
        assert error is None
        assert mock_call.call_count == 2

    @patch("rewriter_v2_judge.call_ollama_json")
    def test_exhausts_max_retries_and_returns_last_failure(self, mock_call):
        mock_call.return_value = ({}, "PARSE ERROR")

        parsed, error = self._call(max_retries=3)

        assert parsed == {}
        assert error == "PARSE ERROR"
        assert mock_call.call_count == 3

    @patch("rewriter_v2_judge.call_ollama_json")
    def test_connection_error_exits_immediately_without_retrying(self, mock_call):
        mock_call.return_value = ({}, "REQUEST ERROR: connection refused")

        with pytest.raises(SystemExit):
            self._call()

        assert mock_call.call_count == 1


# ---------------------------------------------------------------------------
# write_feedback
# ---------------------------------------------------------------------------

class TestWriteFeedback:
    @patch("rewriter_v2_judge.save_json")
    def test_calls_save_json_with_job_title_and_candidate_subpath(self, mock_save, tmp_path):
        submission = rvj.RewriteSubmission(
            job_title="Cashier", candidate="ALEX PARK",
            job_pdf=Path("job.pdf"), rewrites=[],
        )
        parsed = {"candidate": "ALEX PARK"}

        result = rvj.write_feedback(parsed, submission, tmp_path)

        expected_dir = tmp_path / "Cashier" / "ALEX PARK"
        mock_save.assert_called_once_with(parsed, expected_dir, "feedback.json")
        assert result == expected_dir


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def _set_argv(self, monkeypatch, *extra):
        monkeypatch.setattr(sys, "argv", ["rewriter_v2_judge.py", *extra])

    def test_defaults(self, monkeypatch):
        self._set_argv(monkeypatch)
        args = rvj._parse_args()

        assert args.job_title is None
        assert args.candidate is None
        assert args.model == rvj.DEFAULT_MODEL
        assert args.host == rvj.DEFAULT_HOST
        assert args.num_predict == rvj.DEFAULT_NUM_PREDICT
        assert args.temperature == 0.2
        assert args.timeout == 3600
        assert args.max_retries == 3
        assert args.force is False

    def test_job_title_without_candidate_errors(self, monkeypatch):
        self._set_argv(monkeypatch, "--job-title", "Cashier")
        with pytest.raises(SystemExit):
            rvj._parse_args()

    def test_candidate_without_job_title_errors(self, monkeypatch):
        self._set_argv(monkeypatch, "--candidate", "ALEX PARK")
        with pytest.raises(SystemExit):
            rvj._parse_args()

    def test_job_title_and_candidate_together_is_allowed(self, monkeypatch):
        self._set_argv(monkeypatch, "--job-title", "Cashier", "--candidate", "ALEX PARK")
        args = rvj._parse_args()

        assert args.job_title == "Cashier"
        assert args.candidate == "ALEX PARK"

    def test_force_flag(self, monkeypatch):
        self._set_argv(monkeypatch, "--force")
        args = rvj._parse_args()
        assert args.force is True


# ---------------------------------------------------------------------------
# main() -- thin integration tests with the pipeline mocked out; verifying
# side effects (what write_feedback is called with, and the tallied
# counters) rather than printed text.
# ---------------------------------------------------------------------------

class TestMain:
    def _argv(self, monkeypatch, *extra):
        monkeypatch.setattr(sys, "argv", ["rewriter_v2_judge.py", *extra])

    def _submission(self):
        return rvj.RewriteSubmission(
            job_title="Cashier", candidate="ALEX PARK", job_pdf=Path("job.pdf"),
            rewrites=[
                {"agent_model": "phi4", "rejected": False, "used_fallback": False,
                 "retried": False, "reasons": [], "advisory": [], "text": "t1"},
                {"agent_model": "qwen2.5:14b", "rejected": False, "used_fallback": False,
                 "retried": False, "reasons": [], "advisory": [], "text": "t2"},
            ],
        )

    def test_skips_existing_feedback_unless_forced(self, monkeypatch, tmp_path):
        job_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        job_dir.mkdir(parents=True)
        out_feedback = tmp_path / "out" / "Cashier" / "ALEX PARK" / "feedback.json"
        out_feedback.parent.mkdir(parents=True)
        out_feedback.write_text("{}")

        monkeypatch.setattr(rvj, "discover_submissions", lambda results_dir: [("Cashier", "ALEX PARK", job_dir)])
        load_mock = MagicMock()
        monkeypatch.setattr(rvj, "load_submission", load_mock)
        self._argv(monkeypatch, "--results-dir", str(tmp_path / "results"), "--outdir", str(tmp_path / "out"))

        rvj.main()

        load_mock.assert_not_called()

    def test_force_reprocesses_existing_feedback(self, monkeypatch, tmp_path):
        job_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        job_dir.mkdir(parents=True)
        out_feedback = tmp_path / "out" / "Cashier" / "ALEX PARK" / "feedback.json"
        out_feedback.parent.mkdir(parents=True)
        out_feedback.write_text("{}")

        monkeypatch.setattr(rvj, "discover_submissions", lambda results_dir: [("Cashier", "ALEX PARK", job_dir)])
        monkeypatch.setattr(rvj, "load_submission", lambda jt, c, d: (self._submission(), None))
        monkeypatch.setattr(rvj, "extract_text_from_pdf", lambda path: "job text")
        monkeypatch.setattr(
            rvj, "call_judge_with_retries",
            lambda *a, **kw: ({"recommended_rewrite": "phi4"}, None),
        )
        write_mock = MagicMock(return_value=tmp_path / "out" / "Cashier" / "ALEX PARK")
        monkeypatch.setattr(rvj, "write_feedback", write_mock)
        self._argv(monkeypatch, "--results-dir", str(tmp_path / "results"),
                   "--outdir", str(tmp_path / "out"), "--force")

        rvj.main()

        write_mock.assert_called_once()

    def test_missing_submission_is_counted_and_does_not_write_feedback(self, monkeypatch, tmp_path):
        job_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        job_dir.mkdir(parents=True)

        monkeypatch.setattr(rvj, "discover_submissions", lambda results_dir: [("Cashier", "ALEX PARK", job_dir)])
        monkeypatch.setattr(rvj, "load_submission", lambda jt, c, d: (None, "no metadata.csv"))
        write_mock = MagicMock()
        monkeypatch.setattr(rvj, "write_feedback", write_mock)
        self._argv(monkeypatch, "--results-dir", str(tmp_path / "results"), "--outdir", str(tmp_path / "out"))

        rvj.main()

        write_mock.assert_not_called()

    def test_successful_submission_writes_feedback_with_parsed_result(self, monkeypatch, tmp_path):
        job_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        job_dir.mkdir(parents=True)
        submission = self._submission()

        monkeypatch.setattr(rvj, "discover_submissions", lambda results_dir: [("Cashier", "ALEX PARK", job_dir)])
        monkeypatch.setattr(rvj, "load_submission", lambda jt, c, d: (submission, None))
        monkeypatch.setattr(rvj, "extract_text_from_pdf", lambda path: "job text")
        parsed_result = {"recommended_rewrite": "phi4", "career_roadmap": []}
        monkeypatch.setattr(rvj, "call_judge_with_retries", lambda *a, **kw: (parsed_result, None))
        write_mock = MagicMock(return_value=tmp_path / "out" / "Cashier" / "ALEX PARK")
        monkeypatch.setattr(rvj, "write_feedback", write_mock)
        self._argv(monkeypatch, "--results-dir", str(tmp_path / "results"), "--outdir", str(tmp_path / "out"))

        rvj.main()

        write_mock.assert_called_once_with(parsed_result, submission, tmp_path / "out")

    def test_error_from_judge_still_writes_feedback_as_is(self, monkeypatch, tmp_path):
        job_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        job_dir.mkdir(parents=True)
        submission = self._submission()

        monkeypatch.setattr(rvj, "discover_submissions", lambda results_dir: [("Cashier", "ALEX PARK", job_dir)])
        monkeypatch.setattr(rvj, "load_submission", lambda jt, c, d: (submission, None))
        monkeypatch.setattr(rvj, "extract_text_from_pdf", lambda path: "job text")
        monkeypatch.setattr(rvj, "call_judge_with_retries", lambda *a, **kw: ({}, "PARSE ERROR"))
        write_mock = MagicMock(return_value=tmp_path / "out" / "Cashier" / "ALEX PARK")
        monkeypatch.setattr(rvj, "write_feedback", write_mock)
        self._argv(monkeypatch, "--results-dir", str(tmp_path / "results"), "--outdir", str(tmp_path / "out"))

        rvj.main()

        write_mock.assert_called_once_with({}, submission, tmp_path / "out")

    def test_single_target_mode_uses_given_job_title_and_candidate(self, monkeypatch, tmp_path):
        job_dir = tmp_path / "results" / "Cashier" / "ALEX PARK"
        job_dir.mkdir(parents=True)
        submission = self._submission()

        discover_mock = MagicMock()
        monkeypatch.setattr(rvj, "discover_submissions", discover_mock)
        load_mock = MagicMock(return_value=(submission, None))
        monkeypatch.setattr(rvj, "load_submission", load_mock)
        monkeypatch.setattr(rvj, "extract_text_from_pdf", lambda path: "job text")
        monkeypatch.setattr(rvj, "call_judge_with_retries", lambda *a, **kw: ({"recommended_rewrite": "phi4"}, None))
        monkeypatch.setattr(rvj, "write_feedback", MagicMock(return_value=tmp_path / "out"))
        self._argv(monkeypatch, "--job-title", "Cashier", "--candidate", "ALEX PARK",
                   "--results-dir", str(tmp_path / "results"), "--outdir", str(tmp_path / "out"))

        rvj.main()

        discover_mock.assert_not_called()
        load_mock.assert_called_once_with("Cashier", "ALEX PARK", job_dir)

    def test_single_target_mode_missing_dir_exits(self, monkeypatch, tmp_path):
        self._argv(monkeypatch, "--job-title", "Cashier", "--candidate", "GHOST",
                   "--results-dir", str(tmp_path / "results"))

        with pytest.raises(SystemExit):
            rvj.main()

    def test_no_submissions_found_exits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rvj, "discover_submissions", lambda results_dir: [])
        self._argv(monkeypatch, "--results-dir", str(tmp_path / "results"))

        with pytest.raises(SystemExit):
            rvj.main()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
