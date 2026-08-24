"""
essential_requirements.py
This is just a testing script for learning purposes
Runs an essential-requirements check for every candidate CV against every
job listing, using a single 7B/8B model.

FOR NOW: the whole job description PDF is parsed and passed in as-is (no
separate step to break it into discrete essential requirements yet).
"""

import sys
import re
import json
from pathlib import Path

import requests
import pdfplumber

sys.path.insert(0, "pipeline")
import parsed_cv

# ---------------- CONFIG ----------------
JOB_DESCRIPTIONS_FOLDER = "job descriptions"
CVS_FOLDER = "cvs"
RESULTS_FOLDER = "results/essential_requirements"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"  # swap for your preferred 7B/8B


def build_essential_requirement_prompt(requirement_text, cv_sections_text):
    return f"""You are assessing whether a candidate's CV meets the essential
requirements listed in a job description. Use reasonable leeway: a CV does
not need to use the exact same wording as the job listing, and closely
related or clearly transferable experience can count towards a requirement.
Do not invent skills or experience that are not evidenced in the CV and do
not assume the candidate meets a requirement just because it would be
typical for someone applying to this kind of role.

JOB DESCRIPTION:
{requirement_text}

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


def strip_think_tags(text):
    """DeepSeek-R1 (and similar reasoning models) wrap thinking in <think>...</think>.
    Remove that so we're left with just the final output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text):
    """Pull the first {...} JSON object out of the model's reply,
    in case it added extra commentary around it."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(match.group(0))


def extract_text_from_pdf(pdf_path):
    """Pull all text out of a job description PDF (same approach as
    Evaluate_CVs2.py, kept independent here so this file can run standalone)."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def get_cv_sections_text(cv_path):
    """Parse the CV and pull only the sections relevant to essential-
    requirements checking - experience and education."""
    cv = parsed_cv.parse(str(cv_path))
    return cv.sections_named("experience", "education")


def evaluate_essential_requirement(requirement_text, cv_sections_text, candidate_name, model=MODEL):
    """Send one (candidate, job description) pair to the model and get back
    a short leeway-aware summary of essential-requirements fit."""
    prompt = build_essential_requirement_prompt(requirement_text, cv_sections_text)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 1024},
                # Unload immediately after use, same convention as Evaluate_CV.py
                "keep_alive": 0,
            },
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"  [!] Ollama rejected the request for {candidate_name}:")
        print(f"      {response.text}")
        return {
            "candidate": candidate_name,
            "summary": "",
            "error": f"REQUEST ERROR: {e}",
        }

    raw_reply = response.json()["message"]["content"]
    cleaned = strip_think_tags(raw_reply)

    try:
        parsed = extract_json(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  [!] Failed to parse JSON for {candidate_name}: {e}")
        parsed = {
            "summary": "",
            "error": "PARSE ERROR",
        }

    parsed["candidate"] = candidate_name
    return parsed


def to_judge_payload(job_results):
    """Pass through just candidate + summary for the judge. The judge only
    gets the job description and the two evaluator models' output, never
    the CV, so this is already judge-ready as-is."""
    return [
        {"candidate": r["candidate"], "summary": r.get("summary", "")}
        for r in job_results
    ]


def save_results_to_json(results, job_description_name):
    """Save the results for one job."""
    output_dir = Path(RESULTS_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_job = re.sub(r"[^\w\-]+", "_", job_description_name)
    filename = output_dir / f"essential_requirements__{safe_job}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return filename


def main():
    job_folder = Path(JOB_DESCRIPTIONS_FOLDER)
    job_files = sorted(job_folder.glob("*.pdf"))

    if not job_files:
        print(f"No PDFs found in '{JOB_DESCRIPTIONS_FOLDER}/'. Check the folder path.")
        return

    cv_folder = Path(CVS_FOLDER)
    cv_files = sorted(cv_folder.glob("*.pdf"))

    if not cv_files:
        print(f"No PDFs found in '{CVS_FOLDER}/'. Check the folder path.")
        return

    for job_pdf in job_files:
        print(f"\n{'#' * 60}")
        print(f" ESSENTIAL REQUIREMENTS: {job_pdf.name}")
        print(f"{'#' * 60}\n")

        try:
            job_text = extract_text_from_pdf(job_pdf)
        except Exception as e:
            print(f"  [!] Failed to extract text from {job_pdf.name}: {e}")
            continue
        print(f"Loaded job description ({len(job_text)} characters)\n")

        job_results = []

        for cv_path in cv_files:
            candidate_name = cv_path.stem
            print(f"Checking {candidate_name}...")

            try:
                cv_sections_text = get_cv_sections_text(cv_path)
            except Exception as e:
                print(f"  [!] Failed to parse {candidate_name}: {e}")
                job_results.append({
                    "candidate": candidate_name,
                    "summary": "",
                    "error": f"CV PARSE ERROR: {e}",
                })
                continue

            result = evaluate_essential_requirement(job_text, cv_sections_text, candidate_name)
            print(f"    {result.get('summary', '')[:100]}")

            job_results.append(result)

        output_json = save_results_to_json(job_results, job_description_name=job_pdf.stem)
        print(f"\nResults saved to {output_json}\n")


if __name__ == "__main__":
    main()
