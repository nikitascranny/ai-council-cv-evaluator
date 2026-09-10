"""
run_bias_experiment.py

Independent-evaluation bias probe: batch2 CVs (each filename already labels
seniority x nationality, e.g. `middle_chinese.pdf`, `senior_american.pdf`)
scored one model at a time - no council aggregation, no essential-requirements
pass - so each model's evaluation of the same CV can be compared directly
against the others.

Reuses evaluate_cv() from Evaluate_CV.py unchanged, so scores stay comparable
with the rest of the pipeline's council runs (same prompt, same schema).

Models span two label groups, picked to avoid one company's RLHF choices
dominating either side (e.g. not two Gemma checkpoints against one Qwen):
  - Chinese labs:  qwen2.5:14b (Alibaba), qwen3:14b (Alibaba), deepseek-r1:14b (DeepSeek)
  - American labs: phi4 (Microsoft), gemma3:12b (Google), llama3.1:8b (Meta)

Note: qwen3 and deepseek-r1 are reasoning models (chain-of-thought before the JSON
answer); the others aren't. This is a second axis that can be mistaken for a
nationality effect, and must be kept separate when reading results.

Usage:
    python run_bias_experiment.py                    # full run: 6 models x 9 jobs x 9 CVs
    python run_bias_experiment.py --dry-run           # counts only, no Ollama calls
    python run_bias_experiment.py --models phi4 qwen2.5:14b

Safe to re-run: an existing <model>__job_listing_<slug>.json is skipped, so an
interrupted Colab session can just be restarted.
"""

import argparse
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR / "common"))

from document_utils import extract_text_from_pdf, load_cv
from results_io import safe_filename_part, save_json
from Evaluate_CV import evaluate_cv  # sibling file, still in evaluator/ - no path change needed

JOB_DESCRIPTIONS_FOLDER = _ROOT_DIR / "data" / "job-listings"
CVS_FOLDER = _ROOT_DIR / "data" / "cvs" / "batch2"
OUTPUT_FOLDER = _ROOT_DIR / "data" / "bias-experiment" / "batch2" / "raw-results"

BIAS_MODELS = [
    "qwen2.5:14b", "qwen3:14b", "deepseek-r1:14b",
    "phi4", "gemma3:12b", "llama3.1:8b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent (single-model) evaluations of batch2 CVs, for model-origin bias comparison."
    )
    parser.add_argument("--models", nargs="+", default=BIAS_MODELS,
                         help="Ollama model tags to evaluate independently.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the planned (model, job, CV) count and exit without calling Ollama.")
    return parser.parse_args()


def main():
    args = parse_args()

    job_files = sorted(JOB_DESCRIPTIONS_FOLDER.glob("*.pdf"))
    if not job_files:
        print(f"No PDFs found in '{JOB_DESCRIPTIONS_FOLDER}'. Check the folder path.")
        return

    cv_files = sorted(CVS_FOLDER.glob("*.pdf"))
    if not cv_files:
        print(f"No PDFs found under '{CVS_FOLDER}'. Check the folder path.")
        return

    total = len(args.models) * len(job_files) * len(cv_files)
    print(f"{len(args.models)} models x {len(job_files)} jobs x {len(cv_files)} CVs = {total} evaluations")
    if args.dry_run:
        return

    for model in args.models:  # model-major: each model loaded once, not once per job
        print(f"\n{'#' * 60}\n MODEL: {model}\n{'#' * 60}")

        for job_pdf in job_files:
            filename = f"{model.replace(':', '_')}__{safe_filename_part(job_pdf.stem)}.json"
            output_path = OUTPUT_FOLDER / filename
            if output_path.exists():
                print(f"  SKIP {job_pdf.stem} (already have {filename})")
                continue

            try:
                job_text = extract_text_from_pdf(job_pdf)
            except Exception as e:
                print(f"  [!] Failed to extract text from {job_pdf.name}: {e}")
                continue
            print(f"  Job: {job_pdf.name} ({len(job_text)} characters)")

            results = []
            for cv_path in cv_files:
                candidate_name = cv_path.stem
                try:
                    cv = load_cv(cv_path)
                    cv_text = cv.full_text()
                except Exception as e:
                    print(f"    [!] Failed to parse {candidate_name}: {e}")
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
                print(f"    {candidate_name}: score={result.get('score')}")

            results.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
            saved_path = save_json(results, OUTPUT_FOLDER, filename)
            print(f"  -> saved {saved_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
