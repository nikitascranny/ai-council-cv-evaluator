"""
Evaluate_CV.py

For every job description PDF, scores every candidate CV using each
evaluator model (the "council"), and optionally also runs the
essential-requirements check alongside them.

Usage:
    python Evaluate_CV.py                              # council scoring only
    python Evaluate_CV.py --essential-requirements      # council + essential-requirements check
    python Evaluate_CV.py -e                            # same, short form
    python Evaluate_CV.py -e --cvs data/cvs/Batch2 --jobs data/job-listings \
        --results results/evaluation/batch2              # other folders

run_evaluation.py (next to this file) runs this, aggregate_results.py and
judge.py in sequence over one CV folder and one job listing folder.

Folder layout (repo root, three levels above this file at
evaluator/pipeline/Evaluate_CV.py):
    data/job-listings/   - job listing PDFs
    data/cvs/            - candidate CV PDFs, in per-batch subfolders
                            (Batch1/, batch2/, batch3/, batch4/, ...)
    common/              - document_utils.py, ollama_utils.py, results_io.py
    parser/              - cv_pdf_parser.py, the one canonical CV PDF parser,
                            and job_req_parser.py for the essential check's
                            job-listing side
    results/             - created automatically for output JSON

essential_requirements.py is a file used only by this script.
"""

import argparse
import sys
from pathlib import Path


# where common/ (shared utilities), parser/ (job_req_parser) and data/ (CVs,
# job listings) live.
_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR / "common"))
sys.path.insert(0, str(_ROOT_DIR / "parser"))

from document_utils import extract_text_from_pdf, load_cv
from ollama_utils import call_ollama_json
from results_io import safe_filename_part, save_json
import essential_requirements  # sibling file, still in this same folder - no path change needed
import job_req_parser


JOB_DESCRIPTIONS_FOLDER = _ROOT_DIR / "data" / "job-listings"
# data/cvs/ holds one subfolder per batch (Batch1, batch2, batch3, batch4...) -
# see rglob("*.pdf") in main(), which descends into all of them rather than
# assuming a fixed set or a fixed casing of batch folder names.
CVS_FOLDER = _ROOT_DIR / "data" / "cvs" / "Batch1"
RESULTS_FOLDER = _ROOT_DIR / "results"
ESSENTIAL_REQUIREMENTS_FOLDER = RESULTS_FOLDER / "essential_requirements"

EVALUATOR_MODELS = ["qwen2.5:14b", "phi4"] # The models used for scoring CVs against job descriptions
ESSENTIAL_REQUIREMENTS_MODEL = "qwen2.5:7b"


def evaluate_cv(job_text: str, cv_text: str, candidate_name: str, model: str) -> dict:
    """
    Send one CV + job description to an evaluator model and get back a
    structured score.

    Note: This exact prompt was used to obtain our results shown in
    findings.docx, and it is the prompt we intend to use going forward for
    all evaluations. Do not alter this function. Instead, create a new
    function if you want to experiment with different prompts or
    evaluation criteria.
    """
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

    # num_predict=1024: enough room for reasoning plus the full JSON reply
    parsed, error = call_ollama_json(model, prompt, num_predict=1024)

    if error:
        print(f"  [!] Evaluation failed for {candidate_name} ({model}): {error}") # returns an error message if the model fails to respond with valid JSON
        parsed = {"score": None, "strengths": [], "weaknesses": [], "summary": error} # This ensures the runs continue even if one candidate fails to be evaluated

    parsed["candidate"] = candidate_name
    return parsed


def parse_args() -> argparse.Namespace: # Parse command-line arguments for the script
    parser = argparse.ArgumentParser(
        description="Run the CV-EVAL council, with an optional essential-requirements check."
    )
    parser.add_argument( # To enable the essential req check
        "-e", "--essential-requirements",
        action="store_true",
        help="Also run the essential-requirements check for each candidate, "
             "using ESSENTIAL_REQUIREMENTS_MODEL.",
    )
    parser.add_argument(
        "--cvs", type=Path, default=CVS_FOLDER,
        help="Folder of candidate CV PDFs, searched recursively (default: data/cvs/Batch1).",
    )
    parser.add_argument(
        "--jobs", type=Path, default=JOB_DESCRIPTIONS_FOLDER,
        help="Folder of job listing PDFs (default: data/job-listings).",
    )
    parser.add_argument(
        "--results", type=Path, default=RESULTS_FOLDER,
        help="Folder to write result JSON into (default: results/). Essential-requirements "
             "results go into its essential_requirements/ subfolder.",
    )
    return parser.parse_args()


def main():
    """
    Main function to evaluate CVs against job descriptions using specified models.
    It processes each job description and evaluates all candidate CVs, saving the results in JSON format"""
    args = parse_args() # Retrieve command-line arguments

    print(f"Evaluator models: {', '.join(EVALUATOR_MODELS)}")
    if args.essential_requirements: # Checks if the essential req model is in use
        print(f"Essential-requirements check: ON ({ESSENTIAL_REQUIREMENTS_MODEL})")
    else:
        print("Essential-requirements check: OFF (pass --essential-requirements to enable)")

    results_folder = args.results
    essential_requirements_folder = results_folder / ESSENTIAL_REQUIREMENTS_FOLDER.name

    job_files = sorted(args.jobs.glob("*.pdf")) # Retrieve all job description PDFs from folder
    if not job_files:
        print(f"No PDFs found in '{args.jobs}'. Check the folder path.")
        return

    # rglob descends into every batch subfolder (Batch1/, batch2/, etc.)
    # regardless of naming/casing, rather than assuming a fixed batch list.
    cv_files = sorted(args.cvs.rglob("*.pdf"))
    if not cv_files:
        print(f"No PDFs found under '{args.cvs}'. Check the folder path.")
        return

    for job_pdf in job_files: # Loop through each job description PDF
        print(f"\n{'#' * 60}")
        print(f" EVALUATING JOB DESCRIPTION: {job_pdf.name}")
        print(f"{'#' * 60}\n")

        try:
            job_text = extract_text_from_pdf(job_pdf) # Extract text from the job description PDF
        except Exception as e:
            print(f"  [!] Failed to extract text from {job_pdf.name}: {e}")
            continue
        print(f"Loaded job description ({len(job_text)} characters)\n")

        # Scoped once per job listing, not per candidate - the essential
        # check reads only the requirements section, falling back to the
        # whole listing when job_req_parser finds no requirements heading
        # (same fallback essential_stage uses in the UI pipeline).
        essential_job_text = job_text
        if args.essential_requirements:
            essential_job_text = (
                job_req_parser.extract_requirements_section(job_text) or job_text
            )
            if essential_job_text is not job_text:
                print(f"Scoped to essential requirements ({len(essential_job_text)} "
                      f"of {len(job_text)} characters)\n")

        # Creates a dictionary to hold results for each evaluator model and a list for essential requirements results
        results_by_model = {model: [] for model in EVALUATOR_MODELS}
        essential_requirements_results = []

        for cv_path in cv_files:
            candidate_name = cv_path.stem
            print(f"Processing {candidate_name}...")

            # Parse the CV PDF exactly once per candidate. Both evaluator
            # models read cv.full_text() and the essential req check reads cv.sections_named(...)
            # both views from the same parsed object to avoid re-parsing
            try:
                cv = load_cv(cv_path)
            except Exception as e: # If fails, log the error and continue
                print(f"  [!] Failed to parse {candidate_name}: {e}")
                error_result = {
                    "candidate": candidate_name,
                    "score": None,
                    "strengths": [],
                    "weaknesses": [],
                    "summary": f"PDF EXTRACTION ERROR: {e}",
                }
                for model in EVALUATOR_MODELS: # Append the error result to each model's results list
                    results_by_model[model].append(error_result)
                if args.essential_requirements:
                    essential_requirements_results.append({
                        "candidate": candidate_name,
                        "summary": "",
                        "error": f"CV PARSE ERROR: {e}",
                    })
                continue

            cv_text = cv.full_text()

            for model in EVALUATOR_MODELS:
                print(f"  Evaluating with {model}...")
                result = evaluate_cv(job_text, cv_text, candidate_name, model) # Store the evaluation result for the candidate and model
                results_by_model[model].append(result) # appends it to the list of results for that model
                print(f"    Score: {result.get('score')}")

            if args.essential_requirements: # if essential req check is selected
                print(f"  Checking essential requirements ({ESSENTIAL_REQUIREMENTS_MODEL})...")
                cv_sections_text = cv.sections_named("experience", "education", "certifications")
                er_result = essential_requirements.evaluate_essential_requirement(
                    essential_job_text, cv_sections_text, candidate_name,
                    model=ESSENTIAL_REQUIREMENTS_MODEL,
                )
                essential_requirements_results.append(er_result)
                print(f"    {er_result.get('summary', '')[:100]}")

        for model, results in results_by_model.items(): # One JSON per evaluator model, ranked ascending by score
            results.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0))) # sneaky lambda use

            print(f"\n=== FINAL RANKING ({model}) ===")
            for i, r in enumerate(results, start=1): # Prints the final ranking of candidates for each model
                print(f"{i}. {r['candidate']} — score {r['score']}") # shows their score and summary
                print(f"   {r.get('summary', '')}")

            filename = f"{model.replace(':', '_')}__{safe_filename_part(job_pdf.stem)}.json"
            output_path = save_json(results, results_folder, filename)
            print(f"Results saved to {output_path}\n")

        # One JSON for the essential req check, if ran
        if args.essential_requirements:
            filename = f"essential_requirements__{safe_filename_part(job_pdf.stem)}.json"
            output_path = save_json(
                essential_requirements_results, essential_requirements_folder, filename
            )
            print(f"Essential-requirements results saved to {output_path}\n")


if __name__ == "__main__":
    main()
