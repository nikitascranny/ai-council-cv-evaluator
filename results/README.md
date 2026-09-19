# Results

## Batch evaluation (`evaluator/run_evaluation.py`)

Each run also adds a record to `<out>/run.json`: when it ran,
the models and options, and each stage's status, counts and timing.

Before any model call it checks that Ollama is running with every model
pulled, that job listings are named `job_listing_<slug>.pdf`, and that no two CVs share a candidate name.
