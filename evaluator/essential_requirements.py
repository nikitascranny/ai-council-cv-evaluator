"""
essential_requirements.py

Essential-requirements check for a single candidate CV against a job
description, using a single 7B/8B model.

This module is a LIBRARY, not a script: it exposes
evaluate_essential_requirement() for Evaluate_CV.py to call once per
candidate, inside the same loop that runs the two evaluator models. It has
no CONFIG section, no folder scanning, and no main() of its own -
Evaluate_CV.py owns the job/CV loop and folder locations, which guarantees
this check always runs against the exact same CVs and job descriptions as
the evaluators, never a different set.

FOR NOW: the whole job description text is passed in as-is (no separate
step to break it into discrete essential requirements yet). This version
asks the model for a single qualitative summary rather than a strict
per-requirement yes/no - deliberately built with some leeway, so a CV
doesn't get marked down just for phrasing something differently to the job
listing, or for having closely related rather than identical experience.
"""

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR / "common"))

from ollama_utils import call_ollama_json

DEFAULT_MODEL = "qwen2.5:7b"  # narrow summary check - swap for your preferred 7B/8B


def build_essential_requirement_prompt(job_text: str, cv_sections_text: str) -> str:
    return f"""You are assessing whether a candidate's CV meets the essential
requirements listed in a job description. Use reasonable leeway: a CV does
not need to use the exact same wording as the job listing, and closely
related or clearly transferable experience can count towards a requirement.
Do not invent skills or experience that are not evidenced in the CV and do
not assume the candidate meets a requirement just because it would be
typical for someone applying to this kind of role.

JOB DESCRIPTION:
{job_text}

CANDIDATE CV (relevant sections):
{cv_sections_text}

Write a short summary (2-4 sentences) of how well the candidate meets the
job's essential requirements overall: which ones are clearly met, which
are a reasonable/close match and which are clearly not met or unclear from
the CV.

Respond with ONLY a JSON object in exactly this format, no other text
before or after it. Do not repeat the job description or CV text back -
just answer:

{{
  "summary": "<your summary>"
}}"""


def evaluate_essential_requirement(job_text: str, cv_sections_text: str, candidate_name: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Ask the essential-requirements model for a leeway-aware summary of one
    candidate's fit against one job's essential requirements.

    Always returns a dict with "candidate" and "summary" keys; "error" is
    present only when the request or JSON parsing failed.
    """
    prompt = build_essential_requirement_prompt(job_text, cv_sections_text)
    parsed, error = call_ollama_json(model, prompt, num_predict=1024)

    if error:
        print(f"  [!] Essential-requirements check failed for {candidate_name}: {error}")
        parsed = {"summary": "", "error": error}

    parsed["candidate"] = candidate_name
    return parsed


def to_judge_payload(essential_requirements_results):
    """
    Pass through just candidate + summary for a future judge step. The
    judge only gets the job description and the evaluator models' output,
    never the CV, so this is already judge-ready as-is.
    """
    return [
        {"candidate": r["candidate"], "summary": r.get("summary", "")}
        for r in essential_requirements_results
    ]
