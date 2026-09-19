# Results

Output from the experiments run during development, kept for inspection and
analysis. None of it is produced by the web app.

```
results/
  evaluation/
    batches-1-to-9/          the main evaluation runs (run_evaluation.py), one folder per CV batch
    cloud-batch2/            Batch2 on the cloud models, council only
    bias-experiment-batch2/  single-model evaluations, for the bias experiment
    old/                     batches 1-4 before run_evaluation.py existed
  rewrite/
    rewriter_v2/             rewriter_v2's rewrites of Batch1 and Batch3
    rewriter_v2_judge/       those rewrites judged by three different judge models
```

## Requirements

Everything in the root README's **Prerequisites** (Ollama with qwen2.5:14b,
phi4, deepseek-r1:8b and qwen2.5:7b pulled, and the committed `data/`), plus:

| To reproduce | Also needs |
|---|---|
| `evaluation/cloud-batch2/` | `OLLAMA_API_KEY` set in the terminal or in `.env` at the repo root, with access to the cloud models gemma4:31b, gpt-oss:120b and nemotron-3-super. The essential-requirements check still runs locally (qwen2.5:7b). |
| `evaluation/bias-experiment-batch2/` | deepseek-r1:14b, gemma3:12b, llama3.1:8b and qwen3:14b pulled, and `run_bias_experiment.py`, which is not on `main` |
| `rewrite/rewriter_v2_judge/` | gemma3:12b, llama3.1:8b and qwen3:14b pulled |

### Timings

From each run's `run.json`, in minutes. The `batches-1-to-9` runs were on
Google Colab; `cloud-batch2` was on Ollama Cloud, apart from the local
essential-requirements check.

| Run | Pairs | Evaluate | Judge 1-10, with-essential | Judge 1-10, council-only | Judge 0-10, with-essential | Judge 0-10, council-only |
|---|---|---|---|---|---|---|
| `Batch1` | 90 | 44 | 33 | 29 | 30 | 31 |
| `Batch2` | 81 | 38 | 30 | 27 | 30 | 29 |
| `Batch3` | 54 | 24 | 18 | 17 | 19 | 17 |
| `Batch4` | 18 | 8 | 6 | 6 | 10 | 6 |
| `batch6_samename_A` | 9 | 4 | 3 | 3 | 4 | 3 |
| `batch6_samename_B` | 9 | 4 | 5 | 3 | 7 | 3 |
| `batch7_maybefamous` | 36 | 16 | 12 | 12 | 12 | 13 |
| `batch8_specialcases` | 117 | 55 | 41 | 43 | 41 | 23 (63 pairs)* |
| `batch9_political` | 18 | 8 | 7 | 6 | 6 | 6 |
| `cloud-batch2` | 81 | 72 | - | 16 | - | - |

\* The other 54 pairs were judged in an earlier attempt that `run.json` does
not record.

The aggregate stage takes under a minute. Locally, judging takes about 20
seconds per pair. Across all nine batches the evaluate stage took about
3.4 hours, and the 1-10 judging about 2.5 hours per condition. The bias
experiment and the rewrite-judge comparison were not timed.

## Evaluation

### Batch evaluation (`evaluator/run_evaluation.py`)

Each run also adds a record to `<out>/run.json`: when it ran,
the models and options, and each stage's status, counts and timing.

Before any model call it checks that Ollama is running with every model
pulled, that job listings are named `job_listing_<slug>.pdf`, and that no two CVs share a candidate name.

A run folder holds:

```
evaluations/                  Evaluate_CV.py: one file per council model per job listing,
  essential_requirements/       plus the essential-requirements check per job listing
combined/<job>/<candidate>.json   aggregate_results.py: one file per (job, candidate) pair
judged/with-essential/<job>/<candidate>.json   judge.py, shown the council and the check
judged/council-only/<job>/<candidate>.json     judge.py, shown the council only
judged/<condition>/failures.log   pairs with no valid final_score (empty when all succeeded)
run.json, run.log
```

### `evaluation/batches-1-to-9/`

The main results: every CV batch in `data/cvs/`, one run folder each (there is
no Batch5 - that batch was never provided).

| Folder | CVs (`--cvs`) | Pairs judged |
|---|---|---|
| `Batch1` | `data/cvs/Batch1` | 90 |
| `Batch2` | `data/cvs/Batch2` | 81 |
| `Batch3` | `data/cvs/Batch3` | 54 |
| `Batch4` | `data/cvs/Batch4` | 18 |
| `batch6_samename_A` | `data/cvs/batch6_samename/A` | 9 |
| `batch6_samename_B` | `data/cvs/batch6_samename/B` | 9 |
| `batch7_maybefamous` | `data/cvs/batch7_maybefamous` | 36 |
| `batch8_specialcases` | `data/cvs/batch8_specialcases` | 117 |
| `batch9_political` | `data/cvs/batch9_political` | 18 |

All against the nine listings in `data/job-listings/`, with the local models:
council qwen2.5:14b and phi4, essential-requirements check qwen2.5:7b, judge
deepseek-r1:8b. Every pair has a verdict in both conditions. The runs were
done on Google Colab (13-14 Sep 2026), so `out` in `run.json` is a Google
Drive path.

Each folder has two sets of verdicts:

- `judged_0-10/` - the first judging, with the judge prompt's earlier 0-10 scale.
- `judged/` - the same evaluations re-judged with the current 1-10 prompt.
  This is the set to use.

In the order `run.json` records them, each batch was run as:

```
# 1. evaluate, aggregate and judge with the check (0-10), then judge on the council only (0-10)
python evaluator/run_evaluation.py --cvs data/cvs/<batch> --out <folder>
python evaluator/run_evaluation.py --out <folder> --from judge --no-essential

# 2. with the 0-10 verdicts moved to judged_0-10/, re-judge both conditions with the 1-10 prompt
python evaluator/run_evaluation.py --out <folder> --from judge
python evaluator/run_evaluation.py --out <folder> --from judge --no-essential
```

`run.json` also holds a few runs that did no judging: resumes that found every
pair already judged, and one attempt per condition, just before the re-judge,
that stopped at the pre-flight check (exit code 1).

### `evaluation/cloud-batch2/`

Batch2 on the chosen cloud models - council gemma4:31b and gpt-oss:120b,
judge nemotron-3-super - judged on the council only. A test run, for
comparison with the local results. 81 pairs, no failures. Recorded in
`run.json` as the equivalent of:

```
python evaluator/run_evaluation.py --cvs data/cvs/Batch2 --cloud --no-essential --out results/evaluation/cloud
```

(`run.json` records `out` as `results/evaluation/cloud`; the folder was renamed
afterwards.) The essential-requirements check still ran in the evaluate stage,
so its results are in `evaluations/essential_requirements/`, but the judge was
not shown them.

### `evaluation/bias-experiment-batch2/`

Checks whether models from Western (US) and Chinese labs are biased, using
Batch2, whose CVs vary seniority (beginner to senior) against nationality
(American, Chinese, South African). Each model evaluated every Batch2 CV
against all nine job listings on its own, with the same prompt as
`Evaluate_CV.py`; there is no aggregation or judging.

`raw-results/<model>__job_listing_<slug>.json`, for six models:
deepseek-r1:14b, gemma3:12b, llama3.1:8b, phi4, qwen2.5:14b, qwen3:14b.

Produced with `run_bias_experiment.py`, which is not on `main`: it runs one
evaluation with one model, using the same `evaluate_cv` method in `Evaluate_CV.py`.

### `evaluation/old/`

Batches 1-4, from before `run_evaluation.py` existed, and in its own layout.
Kept because it was used to justify adding the essential-requirements check.
Verdicts are on the old 0-10 scale.

- `eval_models/batch<N>/raw-results/` - the council's evaluations
  (`phi4-qwen2.5_14b/`) and the essential-requirements check
  (`essential_requirements/`).
- `eval_judge/batch<N>/2-indep-models-only/` - judged on the council only.
- `eval_judge/batch<N>/2-indep-models-plus-ess-req/` - judged with the check.
  Each has a `failures.log`.

## Rewriting

### `rewrite/rewriter_v2/`

`rewriter/rewriter_v2.py`'s rewrites, with the local agents phi4 and
qwen2.5:14b, for two job listings (Computer Repair Technician, Startup
Fullstack Engineer):

- `Batch1/` - the ten Batch1 CVs.
- `Batch3/` - the three Batch3 CVs, which are deliberately poor, to see how
  the rewriter improves a bad CV.

Only these two batches were run: the rewriter needed less examination than the
evaluator. Layout: `<batch>/<job_title>/<candidate>/metadata.csv` plus one
`<agent_model>.txt` per agent.

### `rewrite/rewriter_v2_judge/`

`rewriter/rewriter_v2_judge.py`'s feedback on those rewrites, from three judge
models, to compare how different models reply:
`gemma3_12b/`, `llama3.1_8b/` and `qwen3_14b/`, each with
`Batch1/` and `Batch3/` (26 `feedback.json` files per model).

The judge used in the end is deepseek-r1:8b (`rewriter_v2_judge.py`'s
default), the same model as the evaluation judge, for consistency; its
feedback is not stored here.

Each model was run per batch, for example:

```
python rewriter/rewriter_v2_judge.py --model gemma3:12b \
    --results-dir results/rewrite/rewriter_v2/Batch1 \
    --outdir results/rewrite/rewriter_v2_judge/gemma3_12b/Batch1
```
