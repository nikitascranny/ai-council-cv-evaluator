"""Candidate mode's seam onto rewriter v2.

v2 exists now. These pin the contract the UI depends on: two results back,
untouched, and an honest failure when the rewriter cannot be reached.
"""

import pytest

from ui.pipeline import candidate


def _result(**kw):
    from types import SimpleNamespace
    base = dict(candidate="Jane Doe", job_title="Secretary", agent_model="phi4",
                original_text="orig", rewritten_text="new", used_fallback=False,
                rejected=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_backend_gets_the_cv_text_the_title_and_the_name():
    seen = {}

    def fake(cv_text, job_title, candidate_name):
        seen["args"] = (cv_text, job_title, candidate_name)
        return [_result()]

    candidate.generate("cv body", "Secretary", "Jane Doe", backend=fake)
    assert seen["args"] == ("cv body", "Secretary", "Jane Doe")


def test_the_call_fits_the_real_rewriters_signature():
    """v2 dropped job_text from rewrite_cv and this seam went on passing it,
    so every real rewrite died with a TypeError that no fake could show."""
    import inspect
    import rewriter_v2

    seen = {}

    def spy(*args):
        seen["args"] = args
        return [_result()]

    candidate.generate("cv", "Secretary", "Jane Doe", backend=spy)
    inspect.signature(rewriter_v2.rewrite_cv).bind(*seen["args"])


def test_both_agents_results_come_back_untouched():
    """The page and the .docx both key off the guard flags on each result,
    so this seam must not flatten or summarise them."""
    a, b = _result(agent_model="phi4"), _result(agent_model="qwen2.5:14b",
                                                used_fallback=True)
    got = candidate.generate("cv", "Secretary", backend=lambda *a_: [a, b])
    assert got == [a, b]
    assert got[1].used_fallback is True


def test_no_rewrites_is_an_error_not_a_silent_success():
    with pytest.raises(candidate.NotAvailable):
        candidate.generate("cv", "Secretary", backend=lambda *a: [])


def test_an_unreachable_rewriter_names_v2_so_the_ui_can_say_what_is_missing(monkeypatch):
    def boom():
        raise candidate.NotAvailable("rewriter v2 could not be loaded (no module)")
    monkeypatch.setattr(candidate, "_load_backend", boom)
    with pytest.raises(candidate.NotAvailable, match="rewriter v2"):
        candidate.generate("cv", "Secretary")


def test_the_module_backend_is_used_when_none_is_passed(monkeypatch):
    monkeypatch.setattr(candidate, "BACKEND", lambda *a: [_result()])
    assert len(candidate.generate("cv", "Secretary")) == 1


def test_only_text_a_model_wrote_counts_as_a_rewrite():
    """A fallback and a failed agent both hold the candidate's original CV."""
    assert candidate.is_rewrite(_result())
    assert not candidate.is_rewrite(_result(used_fallback=True))
    assert not candidate.is_rewrite(_result(error="timeout"))


# --- what the guard's verdict means in plain words --------------------------


def test_a_dropped_number_is_not_reported_as_an_invention():
    """The two are opposites. Telling someone their rewrite "added
    information" when the guard's complaint was that it dropped half their
    numbers points them at the wrong problem entirely - and truncation makes
    dropped the common case, not the rare one."""
    said = candidate.describe_rejection(["number dropped from the CV: 2015"])
    assert "left out" in said
    assert "added" not in said


def test_an_invented_fact_is_reported_as_one():
    said = candidate.describe_rejection(["phd"])
    assert "added information" in said
    assert "left out" not in said


def test_a_borrowed_job_title_is_named_as_its_own_failure():
    said = candidate.describe_rejection(
        ["job title claimed but not held: Web Dev Manager"])
    assert "job title" in said


def test_several_kinds_are_all_reported():
    said = candidate.describe_rejection(
        ["phd", "number dropped from the CV: 4",
         "job title claimed but not held: Web Dev Manager"])
    assert "added information" in said and "left out" in said
    assert "job title" in said


def test_no_reasons_does_not_invent_a_cause():
    """A fallback with an empty reasons list must not assert why."""
    said = candidate.describe_rejection([])
    assert "could not confirm" in said


def test_classification_does_not_repeat_a_kind():
    kinds = candidate.classify_reasons(
        ["number dropped from the CV: 4", "number dropped from the CV: 7"])
    assert kinds == [candidate.DROPPED]


# --- the rewrite judge's feedback -------------------------------------------


def test_feedback_hands_the_judge_each_version_with_its_guard_flags():
    seen = {}

    def fake(job_text, rows, job_title, candidate_name):
        seen["call"] = (job_text, job_title, candidate_name)
        seen["rows"] = rows
        return {"recommended_rewrite": "phi4"}, None

    got = candidate.feedback(
        [_result(agent_model="phi4", rewritten_text="mine"),
         _result(agent_model="qwen2.5:14b", used_fallback=True, rejected=True,
                 reasons=["phd"])],
        "job body", "Secretary", "Jane Doe", backend=fake)
    assert got == ({"recommended_rewrite": "phi4"}, None)
    assert seen["call"] == ("job body", "Secretary", "Jane Doe")
    assert [r["agent_model"] for r in seen["rows"]] == ["phi4", "qwen2.5:14b"]
    assert seen["rows"][0]["text"] == "mine"
    assert seen["rows"][1]["used_fallback"] is True
    assert seen["rows"][1]["reasons"] == ["phd"]


def test_a_failed_agent_is_not_offered_to_the_judge():
    """Its text is the original CV with no guard flags set - the judge would
    read it as a clean rewrite that changed nothing, and could recommend it."""
    seen = {}

    def fake(job_text, rows, *rest):
        seen["rows"] = rows
        return {}, None

    candidate.feedback([_result(), _result(agent_model="qwen2.5:14b",
                                           error="timeout")],
                       "job", "Secretary", backend=fake)
    assert [r["agent_model"] for r in seen["rows"]] == ["phi4"]


def test_nothing_to_judge_is_an_error_not_a_call():
    def fake(*a):
        raise AssertionError("the judge must not be asked about no rewrites")

    parsed, error = candidate.feedback([_result(error="timeout")], "job",
                                       "Secretary", backend=fake)
    assert parsed is None and error


def test_an_unloadable_judge_is_reported_rather_than_raised(monkeypatch):
    def boom():
        raise candidate.NotAvailable("the rewrite judge could not be loaded")
    monkeypatch.setattr(candidate, "_load_feedback_backend", boom)
    parsed, error = candidate.feedback([_result()], "job", "Secretary")
    assert parsed is None and "rewrite judge" in error


def test_the_real_judge_accepts_the_versions_as_this_seam_builds_them(monkeypatch):
    """The judge was written to read its rows back off disk. This seam builds
    them in memory, so its prompt builder must take them as they come."""
    import rewriter_v2_judge
    seen = {}

    def fake_call(prompt, names, *args, **kwargs):
        seen["prompt"], seen["names"] = prompt, names
        return {"recommended_rewrite": "phi4"}, None

    monkeypatch.setattr(rewriter_v2_judge, "call_judge_with_retries", fake_call)
    parsed, error = candidate.feedback(
        [_result(agent_model="phi4", rewritten_text="A tailored CV."),
         _result(agent_model="qwen2.5:14b", used_fallback=True, reasons=["phd"])],
        "Must type 60wpm.", "Secretary", "Jane Doe")
    assert error is None and parsed == {"recommended_rewrite": "phi4"}
    assert seen["names"] == ["phi4", "qwen2.5:14b"]
    assert "A tailored CV." in seen["prompt"]
    assert "Must type 60wpm." in seen["prompt"]


def test_an_unreachable_ollama_does_not_exit_the_web_app(monkeypatch):
    """The judge exits the process when Ollama is down. On a worker thread
    that ends the run with no done event, and the page waits forever."""
    import rewriter_v2_judge

    def exits(*a, **k):
        raise SystemExit("err: could not get a response from ollama")

    monkeypatch.setattr(rewriter_v2_judge, "call_judge_with_retries", exits)
    parsed, error = candidate.feedback([_result()], "job", "Secretary")
    assert parsed is None and "could not get a response" in error
