import sys
import requests
import pdfplumber
import json
import re
from pathlib import Path

# Make the new CV parser importable, then load it
sys.path.insert(0, "pipeline")
import parsed_cv

JOB_DESCRIPTIONS_FOLDER = "job descriptions"  # folder of job listing PDFs, same dir as this script
CVS_FOLDER = "cvs"                            # folder of candidate CV PDFs, same dir as this script
RESULTS_FOLDER = "results"                    # where per job/model JSON files get written

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS = [
    "qwen2.5:14b",
    "phi4"
]


def extract_text_from_pdf(pdf_path):
    """Pull all text out of a job description PDF."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_cv(cv_path):
    """Use the new parsed_cv parser to pull full text out of a candidate CV PDF."""
    cv = parsed_cv.parse(str(cv_path))
    return cv.full_text()


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


def evaluate_cv(job_text, cv_text, candidate_name, model):
    """Send one CV + job description to the specified model and get back a structured score."""
    prompt = f"""You are an expert recruiter evaluating a candidate for a job.

    JOB DESCRIPTION:
    {job_text}

    CANDIDATE CV:
    {cv_text}
    Evaluate how well this candidate fits the job description based on the following criteria:
    required skills and evidence of required skills,
    relevant work experience,
    up-skilling potential.
    Respond with ONLY a JSON
    object in exactly this format, no other text before or after it:

{{
  "score": <integer 1-10>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "summary": "one sentence overall justification"
}}"""

    print(f"  (CV text length: {len(cv_text)} characters)")

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": 1024  # enough room for reasoning + full JSON
                },
                # Unload the model immediately after this request instead of
                # keeping it resident for Ollama's default 5-minute keep-alive.
                "keep_alive": 0,
            },
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Print Ollama's actual error message so we know WHY it was rejected
        print(f"  [!] Ollama rejected the request for {candidate_name}:")
        print(f"      {response.text}")
        return {
            "candidate": candidate_name,
            "score": None,
            "strengths": [],
            "weaknesses": [],
            "summary": f"REQUEST ERROR: {e}",
        }

    raw_reply = response.json()["message"]["content"]

    cleaned = strip_think_tags(raw_reply)

    try:
        parsed = extract_json(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  [!] Failed to parse JSON for {candidate_name}: {e}")
        parsed = {"score": None, "strengths": [], "weaknesses": [], "summary": "PARSE ERROR"}

    parsed["candidate"] = candidate_name
    return parsed


def save_results_to_json(results, model_name, job_description_name):
    """Save the results list to a JSON file named after the model + job description."""
    output_dir = Path(RESULTS_FOLDER)
    output_dir.mkdir(exist_ok=True)

    safe_model = model_name.replace(":", "_")
    safe_job = re.sub(r"[^\w\-]+", "_", job_description_name)

    filename = output_dir / f"{safe_model}__{safe_job}.json"

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
        print(f" EVALUATING JOB DESCRIPTION: {job_pdf.name}")
        print(f"{'#' * 60}\n")

        try:
            job_text = extract_text_from_pdf(job_pdf)
        except Exception as e:
            print(f"  [!] Failed to extract text from {job_pdf.name}: {e}")
            continue
        print(f"Loaded job description ({len(job_text)} characters)\n")

        for model in MODELS:
            print(f"\n{'=' * 60}")
            print(f" RUNNING EVALUATIONS FOR MODEL: {model}")
            print(f"{'=' * 60}\n")

            results = []
            for cv_path in cv_files:
                candidate_name = cv_path.stem
                print(f"Evaluating ({model}): {candidate_name}...")

                try:
                    cv_text = extract_text_from_cv(cv_path)
                except Exception as e:
                    print(f"  [!] Failed to extract text from {candidate_name}: {e}")
                    results.append({
                        "candidate": candidate_name,
                        "score": None,
                        "strengths": [],
                        "weaknesses": [],
                        "summary": f"PDF EXTRACTION ERROR: {e}",
                    })
                    continue

                result = evaluate_cv(job_text, cv_text, candidate_name, model)
                results.append(result)
                print(f"  Score: {result.get('score')}\n")

            results.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))

            print(f"=== FINAL RANKING ({model}) ===")
            for i, r in enumerate(results, start=1):
                print(f"{i}. {r['candidate']} — score {r['score']}")
                print(f"   {r.get('summary', '')}")

            output_json = save_results_to_json(results, model, job_description_name=job_pdf.stem)
            print(f"\nResults saved to {output_json}\n")


if __name__ == "__main__":
    main()