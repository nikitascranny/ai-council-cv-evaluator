"""
test_judge.py

Unit tests for judge.py. Mocks out ollama_utils.call_ollama_json and
results_io.save_json at the points where judge.py imports them, so these
tests never touch a real Ollama server. document_utils.extract_text_from_pdf
is mocked via judge.extract_text_from_pdf in every test that reaches it.

Assumes document_utils.py (providing extract_text_from_pdf) exists alongside
judge.py, ollama_utils.py, and results_io.py in your project -- judge.py
imports it at module load time, so it must be importable even though these
tests never call the real implementation.

Run with:
    pytest test_judge.py -v
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import judge


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def _set_argv(self, monkeypatch, *extra):
        monkeypatch.setattr(
            sys, "argv",
            ["judge.py", "--summaries", "s.json", "--job", "job.pdf", *extra],
        )

    def test_required_args_are_captured(self, monkeypatch):
        self._set_argv(monkeypatch)
        args = judge._parse_args()
        assert args.summaries == ["s.json"]
        assert args.job == "job.pdf"

    def test_missing_required_arg_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["judge.py", "--job", "job.pdf"])
        with pytest.raises(SystemExit):
            judge._parse_args()

    def test_defaults(self, monkeypatch):
        self._set_argv(monkeypatch)
        args = judge._parse_args()

        assert args.model == "deepseek-r1:8b"
        assert args.host == "http://localhost:11434/api/chat"
        assert args.num_predict == 4096
        assert args.max_retries == 3
        assert args.keep_alive == "0m"
        assert args.temperature == 0.2
        assert args.timeout == 3600
        assert args.keep_scores is False
        assert args.candidate is None
        assert args.output is None
        assert args.job_name is None
        assert args.no_essential is False

    def test_temperature_and_timeout_have_correct_types_when_passed_on_cli(self, monkeypatch):
        # Regression check: these must have type=float / type=int in the
        # parser, or CLI-supplied values arrive as raw strings and break
        # requests.post(timeout=...) downstream.
        self._set_argv(monkeypatch, "--temperature", "0.9", "--timeout", "120")
        args = judge._parse_args()

        assert args.temperature == pytest.approx(0.9)
        assert isinstance(args.temperature, float)
        assert args.timeout == 120
        assert isinstance(args.timeout, int)

    def test_max_retries_can_be_overridden(self, monkeypatch):
        self._set_argv(monkeypatch, "--max-retries", "7")
        args = judge._parse_args()
        assert args.max_retries == 7

    def test_keep_scores_flag(self, monkeypatch):
        self._set_argv(monkeypatch, "--keep-scores")
        args = judge._parse_args()
        assert args.keep_scores is True

    def test_output_and_candidate_and_job_name_pass_through(self, monkeypatch):
        self._set_argv(
            monkeypatch,
            "--output", "out.json",
            "--candidate", "Lerato Munche",
            "--job-name", "secretary",
        )
        args = judge._parse_args()
        assert args.output == "out.json"
        assert args.candidate == "Lerato Munche"
        assert args.job_name == "secretary"


# ---------------------------------------------------------------------------
# load_summaries
# ---------------------------------------------------------------------------

class TestLoadSummaries:
    def test_single_file_with_dict(self, tmp_path):
        f = tmp_path / "one.json"
        f.write_text(json.dumps({"candidate": "Lerato", "score": 7}))

        result = judge.load_summaries([f])

        assert result == [{"candidate": "Lerato", "score": 7}]

    def test_single_file_with_list(self, tmp_path):
        f = tmp_path / "many.json"
        f.write_text(json.dumps([{"candidate": "A"}, {"candidate": "B"}]))

        result = judge.load_summaries([f])

        assert result == [{"candidate": "A"}, {"candidate": "B"}]

    def test_multiple_files_are_flattened_together(self, tmp_path):
        f1 = tmp_path / "f1.json"
        f1.write_text(json.dumps({"candidate": "A"}))
        f2 = tmp_path / "f2.json"
        f2.write_text(json.dumps([{"candidate": "B"}, {"candidate": "C"}]))

        result = judge.load_summaries([f1, f2])

        assert result == [{"candidate": "A"}, {"candidate": "B"}, {"candidate": "C"}]

    def test_missing_file_exits(self, tmp_path):
        missing = tmp_path / "nope.json"

        with pytest.raises(SystemExit) as exc:
            judge.load_summaries([missing])

        assert "not found" in str(exc.value)

    def test_invalid_json_exits(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")

        with pytest.raises(SystemExit) as exc:
            judge.load_summaries([f])

        assert "could not parse" in str(exc.value)

    def test_unexpected_top_level_type_exits(self, tmp_path):
        f = tmp_path / "weird.json"
        f.write_text(json.dumps("just a string"))

        with pytest.raises(SystemExit) as exc:
            judge.load_summaries([f])

        assert "unexpected JSON structure" in str(exc.value)

    def test_all_files_empty_list_exits(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text(json.dumps([]))

        with pytest.raises(SystemExit) as exc:
            judge.load_summaries([f])

        assert "no summaries loaded" in str(exc.value)


# ---------------------------------------------------------------------------
# _as_text
# ---------------------------------------------------------------------------

class TestAsText:
    @pytest.mark.parametrize("value, expected", [
        ("Strong communicator", "Strong communicator"),
        ("", ""),
        (None, ""),
    ], ids=["plain_string_passthrough", "empty_string_returns_empty", "none_returns_empty_string"])
    def test_scalar_passthrough(self, value, expected):
        assert judge._as_text(value) == expected

    def test_list_becomes_bulleted_lines(self):
        result = judge._as_text(["Good typing speed", "Bilingual"])
        assert result == "  - Good typing speed\n  - Bilingual"

    def test_list_filters_blank_items(self):
        result = judge._as_text(["Real point", "   ", ""])
        assert result == "  - Real point"


# ---------------------------------------------------------------------------
# format_summaries_for_prompt
# ---------------------------------------------------------------------------

class TestFormatSummariesForPrompt:
    def test_single_summary_default_fields(self):
        summaries = [{
            "candidate": "Lerato Munche",
            "strengths": "Fast learner",
            "weaknesses": "Limited Excel experience",
            "summary": "Solid overall fit.",
        }]

        text = judge.format_summaries_for_prompt(summaries)

        assert "--- Grader summary #1 (candidate: Lerato Munche) ---" in text
        assert "Strengths:\nFast learner" in text
        assert "Weaknesses:\nLimited Excel experience" in text
        assert "Summary: Solid overall fit." in text

    def test_missing_candidate_defaults_to_unknown(self):
        summaries = [{"strengths": "x", "weaknesses": "y"}]

        text = judge.format_summaries_for_prompt(summaries)

        assert "candidate: Unknown candidate" in text

    def test_scores_excluded_by_default_even_if_present(self):
        summaries = [{"candidate": "A", "score": 9}]

        text = judge.format_summaries_for_prompt(summaries, keep_scores=False)

        assert "Prior score given" not in text

    def test_keep_scores_true_includes_score_line(self):
        summaries = [{"candidate": "A", "score": 9}]

        text = judge.format_summaries_for_prompt(summaries, keep_scores=True)

        assert "Prior score given: 9/10" in text

    def test_keep_scores_true_but_no_score_key_omits_line(self):
        summaries = [{"candidate": "A"}]

        text = judge.format_summaries_for_prompt(summaries, keep_scores=True)

        assert "Prior score given" not in text

    def test_multiple_summaries_numbered_in_order(self):
        summaries = [{"candidate": "A"}, {"candidate": "B"}, {"candidate": "C"}]

        text = judge.format_summaries_for_prompt(summaries)

        assert text.index("#1") < text.index("#2") < text.index("#3")


# ---------------------------------------------------------------------------
# essential-requirements check: split_essential, format_essential_for_prompt,
# and its section in build_prompt
# ---------------------------------------------------------------------------

class TestSplitEssential:
    def test_entries_without_a_type_are_all_reviewer_summaries(self):
        entries = [{"candidate": "A", "summary": "x"}, {"candidate": "A", "summary": "y"}]

        assert judge.split_essential(entries) == (entries, None)

    def test_the_essential_entry_is_separated_from_the_summaries(self):
        council = [{"candidate": "A", "summary": "x", "model": "phi4"}]
        check = {"candidate": "A", "summary": "meets most", "type": judge.ESSENTIAL_TYPE}

        summaries, essential = judge.split_essential(council + [check])

        assert summaries == council
        assert essential == check

    def test_more_than_one_check_uses_the_first_and_warns(self, capsys):
        first = {"candidate": "A", "summary": "first", "type": judge.ESSENTIAL_TYPE}
        second = {"candidate": "A", "summary": "second", "type": judge.ESSENTIAL_TYPE}

        _, essential = judge.split_essential([first, second])

        assert essential == first
        assert "using the first" in capsys.readouterr().out


class TestFormatEssentialForPrompt:
    def test_the_summary_is_used(self):
        assert judge.format_essential_for_prompt({"summary": " Meets all. "}) == "Meets all."

    @pytest.mark.parametrize("entry", [
        None,
        {"summary": ""},
        {"summary": "   "},
        {"summary": "", "error": "PARSE ERROR"},
        {"summary": "partial", "error": "CV PARSE ERROR: boom"},
    ], ids=["no_entry", "empty_summary", "blank_summary", "failed_check", "error_with_text"])
    def test_nothing_usable_is_none(self, entry):
        # A failed check must never reach the judge as if it were an assessment.
        assert judge.format_essential_for_prompt(entry) is None


class TestBuildPrompt:
    def test_without_the_check_there_is_no_essential_section(self):
        prompt = judge.build_prompt("job", "summaries", "Lerato")

        assert "ESSENTIAL REQUIREMENTS" not in prompt
        assert "essential requirements check" not in prompt
        assert "3. Produce ONE final, independent evaluation." in prompt

    def test_an_empty_check_is_the_same_as_no_check(self):
        assert judge.build_prompt("job", "s", "L", essential_text="") == \
            judge.build_prompt("job", "s", "L")

    def test_the_check_gets_its_own_section_after_the_reviewer_summaries(self):
        prompt = judge.build_prompt("job", "summaries", "Lerato",
                                    essential_text="Meets the typing requirement.")

        assert (prompt.index("REVIEWER SUMMARIES:")
                < prompt.index("ESSENTIAL REQUIREMENTS CHECK:")
                < prompt.index("Your task:"))
        assert "Meets the typing requirement." in prompt
        assert "3. Use the essential requirements check" in prompt
        assert "4. Produce ONE final, independent evaluation." in prompt

    def test_the_check_is_not_among_the_reviewer_summaries(self):
        prompt = judge.build_prompt("job", "summaries", "Lerato", essential_text="Meets.")

        reviewer_block = prompt.split("REVIEWER SUMMARIES:")[1].split("ESSENTIAL REQUIREMENTS CHECK:")[0]
        assert "Meets." not in reviewer_block

    def test_braces_in_the_check_survive_literally(self):
        prompt = judge.build_prompt("job", "s", "L", essential_text="uses {curly} braces")

        assert "uses {curly} braces" in prompt


# ---------------------------------------------------------------------------
# call_judge
# ---------------------------------------------------------------------------

class TestCallJudge:
    @patch("judge.call_ollama_json")
    def test_success_returns_parsed_and_none(self, mock_call):
        mock_call.return_value = ({"final_score": 8}, None)

        parsed, error = judge.call_judge(
            "prompt text", model="deepseek-r1:8b", host="http://localhost:11434/api/chat",
            num_predict=4096, keep_alive="0m",
        )

        assert parsed == {"final_score": 8}
        assert error is None
        mock_call.assert_called_once_with(
            "deepseek-r1:8b", "prompt text",
            num_predict=4096, keep_alive="0m",
            host="http://localhost:11434/api/chat",
            temperature=0.2, timeout=3600,
        )

    @patch("judge.call_ollama_json")
    def test_request_error_exits(self, mock_call):
        mock_call.return_value = ({}, "REQUEST ERROR: Connection refused")

        with pytest.raises(SystemExit) as exc:
            judge.call_judge(
                "prompt", model="deepseek-r1:8b", host="http://localhost:11434/api/chat",
                num_predict=4096, keep_alive="0m",
            )

        message = str(exc.value)
        assert "could not get a response from ollama" in message
        assert "deepseek-r1:8b" in message

    @patch("judge.call_ollama_json")
    def test_parse_error_is_returned_not_raised(self, mock_call):
        mock_call.return_value = ({}, "PARSE ERROR")

        parsed, error = judge.call_judge(
            "prompt", model="deepseek-r1:8b", host="http://localhost:11434/api/chat",
            num_predict=4096, keep_alive="0m",
        )

        assert parsed == {}
        assert error == "PARSE ERROR"


# ---------------------------------------------------------------------------
# call_judge_with_retries
# ---------------------------------------------------------------------------

class TestCallJudgeWithRetries:
    @patch("judge.call_ollama_json")
    def test_succeeds_on_first_attempt_no_retry(self, mock_call, capsys):
        mock_call.return_value = ({"final_score": 9}, None)

        parsed, error = judge.call_judge_with_retries(
            "prompt", "deepseek-r1:8b", "http://localhost:11434/api/chat",
            num_predict=4096, keep_alive="0m", max_retries=3,
        )

        assert parsed == {"final_score": 9}
        assert error is None
        assert mock_call.call_count == 1
        assert "retrying" not in capsys.readouterr().out

    @patch("judge.call_ollama_json")
    def test_succeeds_after_two_parse_failures(self, mock_call):
        mock_call.side_effect = [
            ({}, "PARSE ERROR"),
            ({}, "PARSE ERROR"),
            ({"final_score": 6}, None),
        ]

        parsed, error = judge.call_judge_with_retries(
            "prompt", "deepseek-r1:8b", "http://localhost:11434/api/chat",
            num_predict=4096, keep_alive="0m", max_retries=3,
        )

        assert parsed == {"final_score": 6}
        assert error is None
        assert mock_call.call_count == 3

    @patch("judge.call_ollama_json")
    def test_exhausts_max_retries_and_returns_last_failure(self, mock_call, capsys):
        mock_call.return_value = ({}, "PARSE ERROR")

        parsed, error = judge.call_judge_with_retries(
            "prompt", "deepseek-r1:8b", "http://localhost:11434/api/chat",
            num_predict=4096, keep_alive="0m", max_retries=3,
        )

        assert parsed == {}
        assert error == "PARSE ERROR"
        assert mock_call.call_count == 3  # capped, did not loop forever
        assert "giving up after max retries" in capsys.readouterr().out

    @patch("judge.call_ollama_json")
    def test_connection_error_exits_immediately_without_retrying(self, mock_call):
        mock_call.return_value = ({}, "REQUEST ERROR: connection refused")

        with pytest.raises(SystemExit):
            judge.call_judge_with_retries(
                "prompt", "deepseek-r1:8b", "http://localhost:11434/api/chat",
                num_predict=4096, keep_alive="0m", max_retries=3,
            )

        assert mock_call.call_count == 1  # no retries burned on a dead connection

    @patch("judge.call_ollama_json")
    def test_max_retries_of_one_means_a_single_attempt(self, mock_call):
        mock_call.return_value = ({}, "PARSE ERROR")

        parsed, error = judge.call_judge_with_retries(
            "prompt", "deepseek-r1:8b", "http://localhost:11434/api/chat",
            num_predict=4096, keep_alive="0m", max_retries=1,
        )

        assert error == "PARSE ERROR"
        assert mock_call.call_count == 1


# ---------------------------------------------------------------------------
# write_json_output
# ---------------------------------------------------------------------------

class TestWriteJsonOutput:
    @patch("judge.save_json")
    def test_calls_save_json_with_split_path_and_reports_success(self, mock_save, tmp_path, capsys):
        out_path = tmp_path / "judged" / "lerato.json"
        result = {"candidate": "Lerato", "final_score": 8}

        return_value = judge.write_json_output(out_path, result)

        mock_save.assert_called_once_with(result, out_path.parent, out_path.name)
        assert return_value is True
        assert "Saved final evaluation" in capsys.readouterr().out

    @patch("judge.save_json")
    def test_writes_empty_dict_as_is_on_parse_failure(self, mock_save, tmp_path):
        out_path = tmp_path / "lerato.json"

        judge.write_json_output(out_path, {})

        mock_save.assert_called_once_with({}, out_path.parent, out_path.name)


# ---------------------------------------------------------------------------
# main() -- thin integration tests with everything below it mocked out
# ---------------------------------------------------------------------------

class TestMain:
    def _patch_pipeline(self, monkeypatch, parsed=None, error=None):
        monkeypatch.setattr(judge, "load_summaries", lambda paths: [{"candidate": "Lerato"}])
        monkeypatch.setattr(judge, "extract_text_from_pdf", lambda path: "job text")
        monkeypatch.setattr(
            judge, "call_judge_with_retries",
            lambda *a, **kw: (parsed if parsed is not None else {
                "candidate": "Lerato", "final_score": 8, "final_evaluation": "ok",
            }, error),
        )

    def _argv(self, monkeypatch, *extra):
        monkeypatch.setattr(
            sys, "argv",
            ["judge.py", "--summaries", "s.json", "--job", "job.pdf", *extra],
        )

    def test_main_without_output_does_not_write(self, monkeypatch, capsys):
        self._patch_pipeline(monkeypatch)
        write_mock = MagicMock()
        monkeypatch.setattr(judge, "write_json_output", write_mock)
        self._argv(monkeypatch)

        judge.main()

        write_mock.assert_not_called()
        assert "Final score for Lerato: 8/10" in capsys.readouterr().out

    def test_main_with_output_calls_write_json_output(self, monkeypatch, tmp_path):
        self._patch_pipeline(monkeypatch)
        write_mock = MagicMock(return_value=True)
        monkeypatch.setattr(judge, "write_json_output", write_mock)
        out_path = str(tmp_path / "out.json")
        self._argv(monkeypatch, "--output", out_path)

        judge.main()

        write_mock.assert_called_once()
        assert write_mock.call_args.args[0] == out_path
        assert write_mock.call_args.args[1] == {
            "candidate": "Lerato", "final_score": 8, "final_evaluation": "ok",
        }

    def test_explicit_candidate_flag_overrides_summary_candidate(self, monkeypatch, capsys):
        self._patch_pipeline(monkeypatch)
        monkeypatch.setattr(judge, "write_json_output", MagicMock(return_value=True))
        self._argv(monkeypatch, "--candidate", "Override Name")

        judge.main()

        assert "Final score for Override Name" in capsys.readouterr().out

    def test_main_prints_warning_when_error_survives_retries(self, monkeypatch, capsys):
        self._patch_pipeline(monkeypatch, parsed={}, error="PARSE ERROR")
        monkeypatch.setattr(judge, "write_json_output", MagicMock(return_value=True))
        self._argv(monkeypatch)

        judge.main()

        out = capsys.readouterr().out
        assert "Warning: PARSE ERROR" in out
        assert "Final score for Lerato: ?/10" in out

    def test_main_writes_empty_dict_when_output_given_and_parse_never_succeeds(self, monkeypatch, tmp_path):
        self._patch_pipeline(monkeypatch, parsed={}, error="PARSE ERROR")
        write_mock = MagicMock(return_value=True)
        monkeypatch.setattr(judge, "write_json_output", write_mock)
        out_path = str(tmp_path / "out.json")
        self._argv(monkeypatch, "--output", out_path)

        judge.main()

        write_mock.assert_called_once_with(out_path, {})

    def test_main_passes_max_retries_through_to_call_judge_with_retries(self, monkeypatch):
        monkeypatch.setattr(judge, "load_summaries", lambda paths: [{"candidate": "Lerato"}])
        monkeypatch.setattr(judge, "extract_text_from_pdf", lambda path: "job text")
        monkeypatch.setattr(judge, "write_json_output", MagicMock(return_value=True))

        captured_kwargs = {}

        def fake_retries(prompt, model, host, num_predict, keep_alive, temperature, timeout, max_retries):
            captured_kwargs["max_retries"] = max_retries
            return ({"candidate": "Lerato", "final_score": 8, "final_evaluation": "ok"}, None)

        monkeypatch.setattr(judge, "call_judge_with_retries", fake_retries)
        self._argv(monkeypatch, "--max-retries", "9")

        judge.main()

        assert captured_kwargs["max_retries"] == 9

    def test_job_name_defaults_to_pdf_stem(self, monkeypatch, capsys):
        self._patch_pipeline(monkeypatch)
        monkeypatch.setattr(judge, "write_json_output", MagicMock(return_value=True))
        self._argv(monkeypatch)  # --job job.pdf, no --job-name

        judge.main()

        assert "(job)" in capsys.readouterr().out  # Path("job.pdf").stem == "job"

    def _capture_prompt(self, monkeypatch, entries):
        monkeypatch.setattr(judge, "load_summaries", lambda paths: entries)
        monkeypatch.setattr(judge, "extract_text_from_pdf", lambda path: "job text")
        monkeypatch.setattr(judge, "write_json_output", MagicMock(return_value=True))
        seen = {}

        def fake_retries(prompt, *a, **kw):
            seen["prompt"] = prompt
            return ({"candidate": "Lerato", "final_score": 8, "final_evaluation": "ok"}, None)

        monkeypatch.setattr(judge, "call_judge_with_retries", fake_retries)
        return seen

    def test_an_essential_entry_in_the_summaries_reaches_the_prompt(self, monkeypatch, capsys):
        seen = self._capture_prompt(monkeypatch, [
            {"candidate": "Lerato", "summary": "Good fit."},
            {"candidate": "Lerato", "summary": "Meets the typing requirement.",
             "type": judge.ESSENTIAL_TYPE},
        ])
        self._argv(monkeypatch)

        judge.main()

        assert "ESSENTIAL REQUIREMENTS CHECK:" in seen["prompt"]
        assert "Meets the typing requirement." in seen["prompt"]
        assert "Grader summary #2" not in seen["prompt"]
        assert "Essential-requirements check: included" in capsys.readouterr().out

    def test_no_essential_flag_leaves_the_check_out(self, monkeypatch):
        seen = self._capture_prompt(monkeypatch, [
            {"candidate": "Lerato", "summary": "Good fit."},
            {"candidate": "Lerato", "summary": "Meets.", "type": judge.ESSENTIAL_TYPE},
        ])
        self._argv(monkeypatch, "--no-essential")

        judge.main()

        assert "ESSENTIAL REQUIREMENTS" not in seen["prompt"]
        assert "Meets." not in seen["prompt"]

    def test_a_failed_check_is_left_out_and_said_so(self, monkeypatch, capsys):
        seen = self._capture_prompt(monkeypatch, [
            {"candidate": "Lerato", "summary": "Good fit."},
            {"candidate": "Lerato", "summary": "", "error": "PARSE ERROR",
             "type": judge.ESSENTIAL_TYPE},
        ])
        self._argv(monkeypatch)

        judge.main()

        assert "ESSENTIAL REQUIREMENTS" not in seen["prompt"]
        assert "not usable (PARSE ERROR)" in capsys.readouterr().out

    def test_summaries_holding_only_an_essential_entry_exit(self, monkeypatch):
        self._capture_prompt(monkeypatch, [
            {"candidate": "Lerato", "summary": "Meets.", "type": judge.ESSENTIAL_TYPE},
        ])
        self._argv(monkeypatch)

        with pytest.raises(SystemExit) as exc:
            judge.main()

        assert "no reviewer summaries" in str(exc.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))