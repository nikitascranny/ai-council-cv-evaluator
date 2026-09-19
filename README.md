# AI CV Evaluator

A local web app that evaluates a candidate's CV against a job listing using a
small council of local LLMs (via Ollama), plus an independent judge model that
reads the council's summaries and gives a final verdict. It also runs a
phrasing-sensitivity probe: the CV is paraphrased and re-scored, and a fact
guard flags any rewrite that invents details the original CV did not contain.

The app is a single Python process (`app.py`): the web server uses only the
standard library, serving plain HTML with a small amount of JavaScript for
live progress (SSE), and there is no build step. The components it drives need
the packages in `requirements.txt` (PDF parsing, the calls to Ollama, and the
Word export).

## Running it

```
pip install -r requirements.txt
python app.py            # serves http://localhost:8000 and opens it
python app.py --port 9000
python app.py --no-browser
```

Requires Python 3.12 or later. Stop the server with Ctrl-C.

## Modes

The app has two modes, linked from the home page:

- **Reviewer** — pick a job listing and one or more CVs, run the evaluation,
  and read the report: parsed CV structure, each council model's score and
  summary, the essential-requirements summary, the judge's final score, a
  facts-as-stated panel (numbers, organisations and qualifications pulled
  from the CV text, not inferred), and the phrasing-sensitivity delta from
  the rewrite probe. With more than one CV selected, Reviewer mode also ranks
  the candidates by the judge's final score.

  The essential-requirements check is ticked by default, and the judge reads
  its summary alongside the council's; untick it for a council-only score.
  The report and the saved-runs list say which the judge used.

- **Candidate** — pick one CV and a target job, then choose what to do:
  - **Rewrite** — rewriter v2's two models each tailor the CV to the job
    title; download either version as a Word document. The rewrite runs on
    this computer by default, or on Ollama Cloud's larger models if chosen,
    which needs an Ollama Cloud API key (see Prerequisites). The CV text is
    then sent to ollama.com; a version that fails there, including when the
    key is rejected, is rewritten locally instead, and the results say so.
    The rewrite judge
    (`rewriter_v2_judge.py`) then recommends which version to send, and
    suggests a career roadmap and alternative careers.
  - **Evaluate** — the council scores the CV against the listing, the
    essential-requirements model summarises which must-haves it meets, and
    the judge gives an overall verdict from both.
  - **Evaluate and rewrite** — both, scoring the original first.

  Any real rewrite (not a fact-guard fallback) can then be sent back through
  the council with **Re-evaluate this version**, and its new scores are shown
  beside the original CV's.

## Layout

One folder per component, matching the rest of the repo:

| | |
|---|---|
| `parser/` | PDF extraction — structure-aware CV parsing, and finding a job listing's requirements section |
| `evaluator/` | the council of evaluator models, the essential-requirements check and the judge; also includes `aggregate_results.py` and `run_evaluation.py`, which run them as a batch |
| `rewriter/` | rewriter v1 (the phrasing-sensitivity probe), rewriter v2 (Candidate mode's two rewrites) and `rewriter_v2_judge.py` (which one to send) |
| `common/` | shared helpers: Ollama calls, PDF text, the fact guard, writing results |
| `ui/` | this web app — `pages.py`, `report.py`, `facts.py` for what a reviewer sees, and `pipeline/` for the orchestration that drives the components above in order and reports progress |
| `app.py` | the server entry point |
| `data/` | `job-listings/` and `cvs/`, both committed |
| `results/` | output of the experiments run during development (see [results/README.md](results/README.md)) |
| `tests/` | the test suite |

`ui/pipeline/` imports nothing from the rendering modules, so the runners are
testable without a server and without a model.

## Prerequisites

- **Ollama** running locally on the default port (`http://localhost:11434`),
  with these models pulled:
  - `qwen2.5:14b` and `phi4` — the evaluator council
  - `deepseek-r1:8b` — the judge
  - `qwen2.5:7b` — the essential-requirements check
  - `qwen2.5:14b` is also the default rewriter model for the phrasing probe,
    and with `phi4` one of rewriter v2's two local models

  If a model is missing, the run reports it as best it can rather than
  reporting a false average.

- **An Ollama Cloud API key**, required only for the cloud options —
  Candidate mode's cloud rewrite and batch evaluation's `--cloud`. See
  [Setting up the Ollama Cloud API key](#setting-up-the-ollama-cloud-api-key)
  below. Without it, everything else still runs locally.

- **The corpora.** Both are committed, so a fresh clone can run the app with
  no local setup: job listings under `data/job-listings/` (9), and 48 CVs
  under `data/cvs/` in batch folders `Batch1` to `Batch4`,
  `batch6_samename` (`A/` and `B/`), `batch7_maybefamous`,
  `batch8_specialcases` and `batch9_political`. Candidates are labelled by
  batch in the Reviewer form, because `Batch3` holds `bad_` and
  `professional_` versions of the same three people.

## Setting up the Ollama Cloud API key

Only needed for the cloud options. Without a key, the "Ollama Cloud" choice in
Candidate mode is shown greyed out with a note saying a key is missing, and
`run_evaluation.py --cloud` stops at its pre-flight check before any model is
called.

**1. Create a key.** Sign in at [ollama.com](https://ollama.com), open
[Settings → Keys](https://ollama.com/settings/keys) and create a key. Copy it.
How much you can use is set by your Ollama account's plan.

**2. Give it to the app, in one of two ways.**

*Option A — a `.env` file (recommended; set once, kept between sessions).*
Create a file named `.env` in the repo root, next to `app.py`, containing one
line:

```
OLLAMA_API_KEY=your-key-here
```

The app and the command-line scripts read this file when they start. Quotes
around the key and an `export ` prefix are both accepted, and blank lines and
lines starting with `#` are ignored. `.env` is listed in `.gitignore`, so the
key is never committed — do not put it anywhere else in the repo.

*Option B — the terminal (this terminal session only).* Set it in the same
terminal you then start the app or script from:

```
# Windows PowerShell
$env:OLLAMA_API_KEY = "your-key-here"

# Windows Command Prompt
set OLLAMA_API_KEY=your-key-here

# macOS / Linux
export OLLAMA_API_KEY="your-key-here"
```

If the key is set in both places, the terminal's value is used.

**3. Restart the app** (Ctrl-C, then `python app.py`). The key is read at
start-up, so an app that was already running will not see it.

**4. Check it worked.** In Candidate mode, the "Ollama Cloud" choice can now be
selected. For a batch run, `run_evaluation.py --cloud` gets past the pre-flight
check.

The app only checks that a key is set, not that ollama.com accepts it. If the
key is wrong or the plan's usage limit is reached, a cloud rewrite falls back
to this computer and the results say why. In a `--cloud` batch run, the
council's evaluations are recorded without a score, and each judge call fails
with a message to check `OLLAMA_API_KEY` and is listed in that condition's
`failures.log`.

## Tests

```
python -m pytest -q
```

Every stage that would otherwise need Ollama is tested against an injected
fake (`ui/pipeline/deps.py`'s `Deps`), so the suite needs no model and no network
access. A handful of tests do import the team's real modules (the parser,
evaluator, judge and essential-requirements code) to check the import wiring,
but do not call a model.

### The test against the real Ollama Cloud

One test, `tests/unit-tests/test_cloud_rewrite.py`, calls the real Ollama Cloud
and so spends account usage. It is marked `cloud`, and `pytest.ini` leaves it
out of a normal run - pytest reports it as "1 deselected". It sends a made-up
CV, written inside the test, through rewriter v2's cloud agents
(gpt-oss:120b and gemma4:31b), and passes if both rewrite it on the cloud
without falling back to their local partner; if one does fall back, the
failure message shows the cloud's error.

To run it:

1. Set the API key **in the terminal** (see
   [Setting up the Ollama Cloud API key](#setting-up-the-ollama-cloud-api-key),
   option B). pytest does not read `.env`, so a key kept only there is not
   seen, and the test is skipped with "OLLAMA_API_KEY is not set".
2. In the same terminal, select the `cloud` marker:

   ```
   python -m pytest -m cloud -v -s
   ```

   `-m cloud` replaces the default selection in `pytest.ini`; `-s` shows the
   token counts the test prints for each model.

## Batch evaluation (developer tooling, not part of the web app)

To evaluate a folder of CVs against a folder of job listings without the web
app, run the pipeline script:

```
python evaluator/run_evaluation.py --cvs data/cvs/Batch1 --out results/evaluation/my-run
```

It runs three steps in order:

1. `Evaluate_CV.py` scores every CV against every job listing with each council
   model, and runs the essential-requirements check.
2. `aggregate_results.py` combines those into one file per job and candidate.
3. `judge.py` gives each of those a final verdict, with the
   essential-requirements summary in its own section of the prompt.

`--no-essential` judges on the council's summaries alone. Running again on the
same `--out` with `--from judge --no-essential` gives both conditions from one
evaluation, in `judged/with-essential/` and `judged/council-only/`. A rerun
skips pairs that already have a valid verdict. `--help` lists every option.

`--cloud` runs the council (gemma4:31b and gpt-oss:120b) and the judge
(nemotron-3-super) on Ollama Cloud instead, and needs `OLLAMA_API_KEY`; the
essential-requirements check still runs locally. This was used for one test,
Batch2 judged on the council only, to compare the cloud models with the local
ones:

```
python evaluator/run_evaluation.py --cvs data/cvs/Batch2 --cloud --no-essential --out results/evaluation/cloud-batch2
```

What each folder in `results/` is, how it was produced, what it needs and how
long it took is in [results/README.md](results/README.md).
