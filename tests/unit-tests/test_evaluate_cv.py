"""
Unit tests for Evaluate_CV.py

Covers:
- evaluate_cv(): success and failure paths, with the actual model call
  (call_ollama_json) mocked out so no real Ollama server is needed
- parse_args(): the --essential-requirements / -e command-line flag

Note: main() itself is an integration-level function (it walks real
folders of PDFs and writes real files) rather than a unit, so it isn't
covered here. evaluate_cv() and parse_args() are the two pieces of real
logic that are safe and useful to unit test in isolation.
"""

from unittest.mock import patch

import Evaluate_CV as ecv


# ---------------------------------------------------------------------------
# evaluate_cv
# ---------------------------------------------------------------------------

@patch("Evaluate_CV.call_ollama_json")
def test_evaluate_cv_success_attaches_candidate_name(mock_call):
    mock_call.return_value = (
        {
            "score": 8,
            "strengths": ["Strong Python background"],
            "weaknesses": ["No cloud experience"],
            "summary": "Good overall fit.",
        },
        None,
    )

    result = ecv.evaluate_cv(
        job_text="job description",
        cv_text="candidate cv",
        candidate_name="Alice",
        model="qwen2.5:14b",
    )

    assert result["candidate"] == "Alice"
    assert result["score"] == 8
    assert result["strengths"] == ["Strong Python background"]


@patch("Evaluate_CV.call_ollama_json")
def test_evaluate_cv_passes_job_and_cv_text_into_prompt(mock_call):
    mock_call.return_value = ({"score": 5, "strengths": [], "weaknesses": [], "summary": "ok"}, None)

    ecv.evaluate_cv(
        job_text="UNIQUE_JOB_MARKER",
        cv_text="UNIQUE_CV_MARKER",
        candidate_name="Bob",
        model="phi4",
    )

    sent_prompt = mock_call.call_args.args[1]
    assert "UNIQUE_JOB_MARKER" in sent_prompt
    assert "UNIQUE_CV_MARKER" in sent_prompt


@patch("Evaluate_CV.call_ollama_json")
def test_evaluate_cv_on_model_failure_returns_safe_fallback(mock_call):
    mock_call.return_value = ({}, "REQUEST ERROR: connection refused")

    result = ecv.evaluate_cv(
        job_text="job description",
        cv_text="candidate cv",
        candidate_name="Carol",
        model="qwen2.5:14b",
    )

    assert result["candidate"] == "Carol"
    assert result["score"] is None
    assert result["strengths"] == []
    assert result["weaknesses"] == []
    assert result["summary"] == "REQUEST ERROR: connection refused"


@patch("Evaluate_CV.call_ollama_json")
def test_evaluate_cv_without_a_key_calls_the_local_daemon_as_before(mock_call):
    mock_call.return_value = ({"score": 5, "strengths": [], "weaknesses": [], "summary": "ok"}, None)

    result = ecv.evaluate_cv("job", "cv", "Dan", "phi4")

    assert mock_call.call_args.kwargs == {"num_predict": 1024}
    assert "tokens_in" not in result


@patch("Evaluate_CV.call_ollama_json")
def test_evaluate_cv_with_a_key_sends_the_same_prompt_to_ollama_cloud(mock_call):
    def reply(model, prompt, **kwargs):
        kwargs["stats"].update(prompt_eval_count=900, eval_count=120)
        return {"score": 7, "strengths": [], "weaknesses": [], "summary": "ok"}, None
    mock_call.side_effect = reply

    result = ecv.evaluate_cv("job", "cv", "Eve", "gpt-oss:120b", api_key="secret")

    kwargs = mock_call.call_args.kwargs
    assert kwargs["host"] == ecv.CLOUD_HOST
    assert kwargs["headers"] == {"Authorization": "Bearer secret"}
    assert kwargs["keep_alive"] is None
    assert kwargs["think"] == "low"
    assert (result["tokens_in"], result["tokens_out"]) == (900, 120)

    mock_call.side_effect = None
    mock_call.return_value = ({"score": 7, "strengths": [], "weaknesses": [], "summary": "ok"}, None)
    ecv.evaluate_cv("job", "cv", "Eve", "phi4")
    assert mock_call.call_args.args[1] == mock_call.call_args_list[0].args[1]


@patch("Evaluate_CV.call_ollama_json")
def test_a_failed_cloud_evaluation_is_recorded_not_replaced_by_a_local_one(mock_call):
    mock_call.return_value = ({}, "REQUEST ERROR: 429 Too Many Requests")

    result = ecv.evaluate_cv("job", "cv", "Fay", "gemma4:31b", api_key="secret")

    assert mock_call.call_count == 1
    assert result["score"] is None
    assert result["summary"].startswith("REQUEST ERROR")


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults_essential_requirements_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py"])

    args = ecv.parse_args()

    assert args.essential_requirements is False


def test_parse_args_long_flag_enables_essential_requirements(monkeypatch):
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py", "--essential-requirements"])

    args = ecv.parse_args()

    assert args.essential_requirements is True


def test_parse_args_short_flag_enables_essential_requirements(monkeypatch):
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py", "-e"])

    args = ecv.parse_args()

    assert args.essential_requirements is True


def test_parse_args_folders_default_to_the_module_constants(monkeypatch):
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py"])

    args = ecv.parse_args()

    assert args.cvs == ecv.CVS_FOLDER
    assert args.jobs == ecv.JOB_DESCRIPTIONS_FOLDER
    assert args.results == ecv.RESULTS_FOLDER


def test_parse_args_folders_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py", "--cvs", str(tmp_path / "cvs"),
                                     "--jobs", str(tmp_path / "jobs"),
                                     "--results", str(tmp_path / "out")])

    args = ecv.parse_args()

    assert args.cvs == tmp_path / "cvs"
    assert args.jobs == tmp_path / "jobs"
    assert args.results == tmp_path / "out"


# ---------------------------------------------------------------------------
# main - the folders and filenames run_evaluation.py and aggregate_results.py
# depend on, with every PDF read and model call faked
# ---------------------------------------------------------------------------

def test_main_reads_and_writes_the_folders_it_is_given(monkeypatch, tmp_path):
    from types import SimpleNamespace

    jobs = tmp_path / "jobs"
    cvs = tmp_path / "cvs" / "batch"
    jobs.mkdir()
    cvs.mkdir(parents=True)
    (jobs / "job_listing_secretary.pdf").write_bytes(b"")
    (cvs / "ALEX PARK.pdf").write_bytes(b"")
    out = tmp_path / "out"
    seen = {}

    def check(job_text, sections_text, candidate_name, model):
        seen["sections"] = sections_text
        return {"candidate": candidate_name, "summary": "meets"}

    cv = SimpleNamespace(full_text=lambda: "cv text",
                         sections_named=lambda *labels: "scoped to " + ",".join(labels))
    monkeypatch.setattr(ecv, "extract_text_from_pdf", lambda path: "job text")
    monkeypatch.setattr(ecv, "load_cv", lambda path: cv)
    monkeypatch.setattr(ecv, "evaluate_cv", lambda job, cv_text, name, model: {
        "candidate": name, "score": 6, "strengths": [], "weaknesses": [], "summary": "ok"})
    monkeypatch.setattr(ecv.essential_requirements, "evaluate_essential_requirement", check)
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py", "-e", "--cvs", str(tmp_path / "cvs"),
                                     "--jobs", str(jobs), "--results", str(out)])

    ecv.main()

    for model in ecv.EVALUATOR_MODELS:
        assert (out / f"{model.replace(':', '_')}__job_listing_secretary.json").is_file()
    assert (out / "essential_requirements" / "essential_requirements__job_listing_secretary.json").is_file()
    assert seen["sections"] == "scoped to experience,education,certifications"


def test_main_with_cloud_runs_the_cloud_council_with_the_key(monkeypatch, tmp_path):
    from types import SimpleNamespace

    jobs = tmp_path / "jobs"
    cvs = tmp_path / "cvs"
    jobs.mkdir()
    cvs.mkdir()
    (jobs / "job_listing_secretary.pdf").write_bytes(b"")
    (cvs / "ALEX PARK.pdf").write_bytes(b"")
    out = tmp_path / "out"
    keys = []

    def evaluate(job, cv_text, name, model, api_key=None):
        keys.append(api_key)
        return {"candidate": name, "score": 6, "strengths": [], "weaknesses": [],
                "summary": "ok", "tokens_in": 10, "tokens_out": 2}

    monkeypatch.setenv(ecv.CLOUD_API_KEY_ENV, "secret")
    monkeypatch.setattr(ecv, "extract_text_from_pdf", lambda path: "job text")
    monkeypatch.setattr(ecv, "load_cv", lambda path: SimpleNamespace(full_text=lambda: "cv"))
    monkeypatch.setattr(ecv, "evaluate_cv", evaluate)
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py", "--cloud", "--cvs", str(cvs),
                                     "--jobs", str(jobs), "--results", str(out)])

    ecv.main()

    assert keys == ["secret"] * len(ecv.CLOUD_EVALUATOR_MODELS)
    for model in ecv.CLOUD_EVALUATOR_MODELS:
        assert (out / f"{model.replace(':', '_')}__job_listing_secretary.json").is_file()
    for model in ecv.EVALUATOR_MODELS:
        assert not (out / f"{model.replace(':', '_')}__job_listing_secretary.json").exists()


def test_main_with_cloud_and_no_key_stops_before_any_evaluation(monkeypatch, tmp_path):
    import pytest

    monkeypatch.delenv(ecv.CLOUD_API_KEY_ENV, raising=False)
    monkeypatch.setattr(ecv, "evaluate_cv", lambda *a, **k: pytest.fail("evaluated"))
    monkeypatch.setattr("sys.argv", ["Evaluate_CV.py", "--cloud", "--results", str(tmp_path)])

    with pytest.raises(SystemExit) as stopped:
        ecv.main()

    assert ecv.CLOUD_API_KEY_ENV in str(stopped.value)
