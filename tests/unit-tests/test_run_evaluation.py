"""
test_run_evaluation.py

Unit tests for evaluator/run_evaluation.py: the argument rules, the preflight
checks, and the order and resuming of the three stages. Each stage script is
replaced by a fake that writes the files the real one would, so no subprocess
is started and no model is called.

Run with:
    pytest test_run_evaluation.py -v
"""

import json
from pathlib import Path

import pytest

import run_evaluation as runner

ALL_MODELS = {"qwen2.5:14b", "phi4", "qwen2.5:7b", "deepseek-r1:8b"}


def _pdf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def folders(tmp_path):
    cvs, jobs = tmp_path / "cvs", tmp_path / "jobs"
    _pdf(cvs / "ALEX PARK.pdf")
    _pdf(jobs / "job_listing_secretary.pdf")
    return cvs, jobs, tmp_path / "out"


class Log(list):
    def __call__(self, line=""):
        self.append(line)

    def text(self):
        return "\n".join(self)


class FakeScripts:
    """Stands in for run_step: records each command, and writes what the real
    script would have written."""

    def __init__(self, verdict=None, fail=()):
        self.calls = []
        self.verdict = {"final_score": 6} if verdict is None else verdict
        self.fail = set(fail)

    def __call__(self, args, log):
        args = [str(a) for a in args]
        script = Path(args[0]).name
        self.calls.append((script, args[1:]))
        if script in self.fail:
            return 1

        def option(name):
            return Path(args[args.index(name) + 1])

        if script == "Evaluate_CV.py":
            out = option("--results")
            (out / "essential_requirements").mkdir(parents=True, exist_ok=True)
            (out / "phi4__job_listing_secretary.json").write_text(
                json.dumps([{"candidate": "ALEX PARK", "score": 6, "summary": "ok"}]),
                encoding="utf-8")
            (out / "essential_requirements" / "essential_requirements__job_listing_secretary.json"
             ).write_text(json.dumps([{"candidate": "ALEX PARK", "summary": "meets"}]),
                          encoding="utf-8")
        elif script == "aggregate_results.py":
            out = option("--outdir") / "secretary"
            out.mkdir(parents=True, exist_ok=True)
            (out / "alex_park.json").write_text("[]", encoding="utf-8")
        elif script == "judge.py":
            out = option("--output")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self.verdict), encoding="utf-8")
        return 0

    def scripts(self):
        return [script for script, _ in self.calls]

    def args_of(self, script):
        return next(args for name, args in self.calls if name == script)


def _run(folders, *extra, scripts=None, models=ALL_MODELS):
    cvs, jobs, out = folders
    scripts = scripts or FakeScripts()
    code = runner.main(["--cvs", str(cvs), "--jobs", str(jobs), "--out", str(out), *extra],
                       run=scripts, fetch=lambda: set(models))
    return code, scripts


# ---------------------------------------------------------------------------
# is_valid_verdict
# ---------------------------------------------------------------------------

class TestIsValidVerdict:
    @pytest.mark.parametrize("content, valid", [
        ({"final_score": 7}, True),
        ({"final_score": 0}, True),
        ({"final_score": 10}, True),
        ({"final_score": 7.5}, True),
        ({"final_score": 11}, False),
        ({"final_score": -1}, False),
        ({"final_score": "7"}, False),
        ({"final_score": True}, False),
        ({"final_score": None}, False),
        ({}, False),
        ([], False),
    ])
    def test_only_a_numeric_score_out_of_ten_counts(self, tmp_path, content, valid):
        path = tmp_path / "verdict.json"
        path.write_text(json.dumps(content), encoding="utf-8")

        assert runner.is_valid_verdict(path) is valid

    def test_a_missing_or_unreadable_file_is_not_valid(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")

        assert runner.is_valid_verdict(tmp_path / "missing.json") is False
        assert runner.is_valid_verdict(broken) is False


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_resuming_needs_the_earlier_runs_out_folder(self, folders):
        _, jobs, _ = folders

        with pytest.raises(SystemExit):
            runner.parse_args(["--jobs", str(jobs), "--from", "judge"])

    def test_resuming_from_an_out_folder_that_does_not_exist_is_refused(self, folders):
        _, jobs, out = folders

        with pytest.raises(SystemExit):
            runner.parse_args(["--jobs", str(jobs), "--from", "judge", "--out", str(out)])

    def test_missing_input_folders_are_refused(self, folders, tmp_path):
        cvs, jobs, out = folders

        with pytest.raises(SystemExit):
            runner.parse_args(["--cvs", str(cvs), "--jobs", str(tmp_path / "nope"), "--out", str(out)])
        with pytest.raises(SystemExit):
            runner.parse_args(["--cvs", str(tmp_path / "nope"), "--jobs", str(jobs), "--out", str(out)])

    def test_resuming_does_not_need_the_cvs(self, folders, tmp_path):
        _, jobs, out = folders
        out.mkdir()

        args = runner.parse_args(["--cvs", str(tmp_path / "nope"), "--jobs", str(jobs),
                                  "--out", str(out), "--from", "aggregate"])

        assert args.start == "aggregate"

    def test_defaults_and_flags(self, folders):
        cvs, jobs, out = folders

        args = runner.parse_args(["--cvs", str(cvs), "--jobs", str(jobs), "--out", str(out)])
        assert args.start == "evaluate" and args.essential is True and args.force is False
        assert args.out.is_absolute() and args.cvs.is_absolute()

        args = runner.parse_args(["--cvs", str(cvs), "--jobs", str(jobs), "--out", str(out),
                                  "--no-essential", "--force"])
        assert args.essential is False and args.force is True


# ---------------------------------------------------------------------------
# preflight checks
# ---------------------------------------------------------------------------

class TestPreflightChecks:
    def test_a_job_listing_without_the_naming_convention_is_a_problem(self, tmp_path):
        _pdf(tmp_path / "job_listing_secretary.pdf")
        _pdf(tmp_path / "Secretary Job.pdf")

        pdfs, problems = runner.check_job_listings(tmp_path)

        assert len(pdfs) == 2
        assert len(problems) == 1 and "Secretary Job.pdf" in problems[0]

    def test_two_cvs_with_one_candidate_name_are_a_problem(self, tmp_path):
        _pdf(tmp_path / "A" / "Alex Turner.pdf")
        _pdf(tmp_path / "B" / "Alex Turner.pdf")
        _pdf(tmp_path / "A" / "Someone Else.pdf")

        pdfs, problems = runner.check_cvs(tmp_path)

        assert len(pdfs) == 3
        assert len(problems) == 1 and "Alex Turner.pdf" in problems[0]
        assert "Someone Else" not in problems[0]

    def test_a_model_ollama_does_not_have_is_named(self):
        problems = runner.check_models(["phi4", "not-a-model:1b"], fetch=lambda: {"phi4"})

        assert len(problems) == 1 and "not-a-model:1b" in problems[0]
        assert "phi4" not in problems[0]

    def test_an_unreachable_ollama_is_a_problem_not_a_crash(self):
        def refused():
            raise OSError("connection refused")

        problems = runner.check_models(["phi4"], fetch=refused)

        assert len(problems) == 1 and "not reachable" in problems[0]

    def test_every_problem_is_reported_before_anything_runs(self, folders):
        cvs, jobs, out = folders
        _pdf(jobs / "Secretary Job.pdf")
        scripts = FakeScripts()

        code, _ = _run(folders, scripts=scripts, models={"phi4"})

        assert code == 1
        assert scripts.calls == []
        log = (out / "run.log").read_text(encoding="utf-8")
        assert "Secretary Job.pdf" in log and "Ollama does not have" in log


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_a_full_run_evaluates_aggregates_and_judges_in_order(self, folders):
        _, _, out = folders

        code, scripts = _run(folders)

        assert code == 0
        assert scripts.scripts() == ["Evaluate_CV.py", "aggregate_results.py", "judge.py"]
        assert "--essential-requirements" in scripts.args_of("Evaluate_CV.py")
        assert "--essential-dir" in scripts.args_of("aggregate_results.py")
        assert "--no-essential" not in scripts.args_of("judge.py")
        assert runner.is_valid_verdict(out / "judged" / "with-essential" / "secretary" / "alex_park.json")
        assert "Judged (with-essential): 1 ok" in (out / "run.log").read_text(encoding="utf-8")

    def test_a_rerun_from_judge_skips_a_valid_verdict(self, folders):
        _run(folders)

        code, scripts = _run(folders, "--from", "judge")

        assert code == 0
        assert scripts.calls == []

    def test_force_judges_a_valid_verdict_again(self, folders):
        _run(folders)

        code, scripts = _run(folders, "--from", "judge", "--force")

        assert code == 0
        assert scripts.scripts() == ["judge.py"]

    def test_no_essential_judges_into_its_own_folder(self, folders):
        _, _, out = folders
        _run(folders)

        code, scripts = _run(folders, "--from", "judge", "--no-essential")

        assert code == 0
        assert "--no-essential" in scripts.args_of("judge.py")
        assert runner.is_valid_verdict(out / "judged" / "council-only" / "secretary" / "alex_park.json")
        assert runner.is_valid_verdict(out / "judged" / "with-essential" / "secretary" / "alex_park.json")

    def test_an_unusable_verdict_is_a_logged_failure(self, folders):
        _, _, out = folders

        code, _ = _run(folders, scripts=FakeScripts(verdict={}))

        assert code == 1
        fail_log = out / "judged" / "with-essential" / "failures.log"
        assert fail_log.read_text(encoding="utf-8") == "secretary/alex_park\n"

    def test_a_failed_evaluation_stops_before_aggregating(self, folders):
        code, scripts = _run(folders, scripts=FakeScripts(fail={"Evaluate_CV.py"}))

        assert code == 1
        assert scripts.scripts() == ["Evaluate_CV.py"]

    def test_aggregating_with_no_evaluations_says_where_to_start(self, folders):
        _, _, out = folders
        out.mkdir()
        scripts = FakeScripts()

        code, _ = _run(folders, "--from", "aggregate", scripts=scripts)

        assert code == 1
        assert scripts.calls == []
        assert "start from the evaluate stage" in (out / "run.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------

def _runs(out):
    return json.loads((out / "run.json").read_text(encoding="utf-8"))["runs"]


class TestRunRecord:
    def test_a_full_run_records_what_produced_its_results(self, folders):
        _, _, out = folders

        _run(folders)

        (run,) = _runs(out)
        assert run["exit_code"] == 0 and run["preflight"] == "ok"
        assert run["from"] == "evaluate" and run["condition"] == "with-essential"
        assert run["models"]["judge"] == "deepseek-r1:8b"
        assert run["models"]["council"] and run["models"]["essential_requirements"]
        assert run["inputs"] == {"job_listings": ["job_listing_secretary.pdf"],
                                 "cvs": ["ALEX PARK.pdf"]}
        assert [stage["status"] for stage in run["stages"].values()] == ["ok", "ok", "ok"]
        assert run["stages"]["evaluate"]["evaluations"] == 1
        assert run["stages"]["aggregate"]["combined_files"] == 1
        assert run["stages"]["judge"]["ok"] == 1
        assert "command" not in run

    def test_each_later_run_into_the_same_out_is_appended(self, folders):
        _, _, out = folders
        _run(folders)

        _run(folders, "--from", "judge", "--no-essential")

        first, second = _runs(out)
        assert first["condition"] == "with-essential"
        assert second["condition"] == "council-only"
        assert second["stages"]["evaluate"] == {"status": "reused"}
        assert second["stages"]["judge"]["ok"] == 1
        assert "council" not in second["models"] and second["cvs"] is None

    def test_a_failed_run_is_recorded_too(self, folders):
        _, _, out = folders

        _run(folders, scripts=FakeScripts(verdict={}))

        (run,) = _runs(out)
        assert run["exit_code"] == 1
        assert run["stages"]["judge"]["status"] == "failed"
        assert run["stages"]["judge"]["failed"] == 1

    def test_a_run_stopped_by_preflight_records_that_nothing_ran(self, folders):
        _, _, out = folders

        _run(folders, models={"phi4"})

        (run,) = _runs(out)
        assert run["preflight"] == "failed"
        assert run["stages"] == {} and run["exit_code"] == 1

    def test_an_interrupted_stage_is_recorded_as_stopped(self, folders):
        _, _, out = folders

        def interrupt(args, log):
            raise KeyboardInterrupt

        code, _ = _run(folders, scripts=interrupt)

        assert code == 130
        (run,) = _runs(out)
        assert run["stages"]["evaluate"]["status"] == "stopped"
        assert run["exit_code"] == 130

    def test_an_unreadable_run_json_is_set_aside_not_overwritten(self, folders):
        _, _, out = folders
        out.mkdir()
        (out / "run.json").write_text("{not json", encoding="utf-8")

        _run(folders)

        assert (out / "run.json.unreadable").read_text(encoding="utf-8") == "{not json"
        assert len(_runs(out)) == 1

    def test_paths_inside_the_repo_are_recorded_relative(self, tmp_path):
        # A committed run.json should not carry one person's home folder.
        assert runner.display_path(runner.ROOT_DIR / "data" / "cvs" / "Batch1") == "data/cvs/Batch1"
        assert runner.display_path(tmp_path) == str(tmp_path.resolve())
