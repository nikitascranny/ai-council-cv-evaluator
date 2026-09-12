"""
test_aggregate_results.py

Unit tests for evaluator/aggregate_results.py: regrouping per-model result
files into one combined file per (job, candidate), with the
essential-requirements check joined in as its own marked entry.

Run with:
    pytest test_aggregate_results.py -v
"""

import json
import sys

import aggregate_results as agg


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _council(directory, model, job="secretary", entries=None):
    if entries is None:
        entries = [{"candidate": "LERATO MUNCHE", "score": 7, "strengths": ["a"],
                    "weaknesses": ["b"], "summary": f"{model} says ok"}]
    return _write(directory / f"{model}__job_listing_{job}.json", entries)


def _essential(directory, job="secretary", entries=None):
    if entries is None:
        entries = [{"candidate": "LERATO MUNCHE", "summary": "Meets the typing requirement."}]
    return _write(directory / f"essential_requirements__job_listing_{job}.json", entries)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_model_entries_are_grouped_per_job_and_candidate_and_tagged(self, tmp_path):
        files = [_council(tmp_path, "phi4"), _council(tmp_path, "qwen2.5_14b")]

        combined, stats = agg.aggregate(files)

        entries = combined["secretary"]["lerato_munche"]
        assert [e["model"] for e in entries] == ["phi4", "qwen2.5_14b"]
        assert stats["files_used"] == 2

    def test_the_check_is_joined_last_and_marked_rather_than_tagged_as_a_model(self, tmp_path):
        files = [_council(tmp_path, "phi4"), _essential(tmp_path), _council(tmp_path, "qwen2.5_14b")]

        combined, _ = agg.aggregate(files)

        entries = combined["secretary"]["lerato_munche"]
        assert [e.get("model") for e in entries] == ["phi4", "qwen2.5_14b", None]
        assert entries[-1] == {"candidate": "LERATO MUNCHE",
                               "summary": "Meets the typing requirement.",
                               "type": agg.ESSENTIAL_TYPE}

    def test_the_marker_is_the_one_judge_py_looks_for(self):
        import judge

        assert agg.ESSENTIAL_TYPE == judge.ESSENTIAL_TYPE

    def test_a_failed_check_is_kept_with_its_error(self, tmp_path):
        # judge.py decides to leave a failed check out of the prompt; the
        # combined file keeps the record so that decision can be seen.
        files = [_council(tmp_path, "phi4"),
                 _essential(tmp_path, entries=[{"candidate": "LERATO MUNCHE", "summary": "",
                                                "error": "CV PARSE ERROR: boom"}])]

        entries = agg.aggregate(files)[0]["secretary"]["lerato_munche"]

        assert entries[-1]["error"] == "CV PARSE ERROR: boom"

    def test_a_check_with_no_model_evaluations_is_not_written_on_its_own(self, tmp_path, capsys):
        files = [_council(tmp_path, "phi4"),
                 _essential(tmp_path, entries=[{"candidate": "SOMEONE ELSE", "summary": "x"}])]

        combined, stats = agg.aggregate(files)

        assert "someone_else" not in combined["secretary"]
        assert stats["essential_unmatched"] == [("secretary", "someone_else")]
        assert "no council evaluations" in capsys.readouterr().err

    def test_a_second_check_for_the_same_candidate_keeps_the_first(self, tmp_path, capsys):
        files = [_council(tmp_path, "phi4"),
                 _essential(tmp_path, entries=[{"candidate": "LERATO MUNCHE", "summary": "first"},
                                               {"candidate": "LERATO MUNCHE", "summary": "second"}])]

        entries = agg.aggregate(files)[0]["secretary"]["lerato_munche"]

        checks = [e for e in entries if e.get("type") == agg.ESSENTIAL_TYPE]
        assert [c["summary"] for c in checks] == ["first"]
        assert "keeping the first" in capsys.readouterr().err

    def test_a_check_joins_only_its_own_job(self, tmp_path):
        files = [_council(tmp_path, "phi4", job="secretary"),
                 _council(tmp_path, "phi4", job="welder"),
                 _essential(tmp_path, job="welder")]

        combined, _ = agg.aggregate(files)

        assert not any(e.get("type") for e in combined["secretary"]["lerato_munche"])
        assert combined["welder"]["lerato_munche"][-1]["type"] == agg.ESSENTIAL_TYPE

    def test_drop_scores_strips_model_scores(self, tmp_path):
        combined, _ = agg.aggregate([_council(tmp_path, "phi4")], drop_scores=True)

        assert "score" not in combined["secretary"]["lerato_munche"][0]

    def test_a_file_not_matching_the_naming_convention_is_skipped(self, tmp_path):
        stray = _write(tmp_path / "notes.json", [])

        combined, stats = agg.aggregate([stray, _council(tmp_path, "phi4")])

        assert stats["files_skipped"] == 1
        assert list(combined["secretary"]) == ["lerato_munche"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def _layout(self, tmp_path):
        """results/ as Evaluate_CV.py writes it: model files at the top,
        essential-requirements files in their own subfolder."""
        results = tmp_path / "results"
        essential_dir = results / "essential_requirements"
        essential_dir.mkdir(parents=True)
        _council(results, "phi4")
        _council(results, "qwen2.5_14b")
        _essential(essential_dir)
        return results, essential_dir, tmp_path / "combined"

    def _run(self, monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["aggregate_results.py", *argv])
        agg.main()

    def test_essential_dir_results_land_in_the_combined_file(self, tmp_path, monkeypatch, capsys):
        results, essential_dir, outdir = self._layout(tmp_path)

        self._run(monkeypatch, "--results-dir", str(results),
                  "--essential-dir", str(essential_dir), "--outdir", str(outdir))

        written = json.loads((outdir / "secretary" / "lerato_munche.json").read_text(encoding="utf-8"))
        assert [e.get("model") for e in written] == ["phi4", "qwen2.5_14b", None]
        assert written[-1]["type"] == agg.ESSENTIAL_TYPE
        assert "essential check for 1" in capsys.readouterr().out

    def test_without_essential_dir_the_combined_file_is_models_only(self, tmp_path, monkeypatch):
        # --results-dir does not descend into subfolders, so the council-only
        # condition is still one flag away.
        results, _, outdir = self._layout(tmp_path)

        self._run(monkeypatch, "--results-dir", str(results), "--outdir", str(outdir))

        written = json.loads((outdir / "secretary" / "lerato_munche.json").read_text(encoding="utf-8"))
        assert [e.get("model") for e in written] == ["phi4", "qwen2.5_14b"]
