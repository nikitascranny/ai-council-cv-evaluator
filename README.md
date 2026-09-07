# AI CV Evaluator

A local web app that evaluates a candidate's CV against a job listing using a
small council of local LLMs (via Ollama), plus an independent judge model that
reads the council's summaries and gives a final verdict. It also runs a
phrasing-sensitivity probe: the CV is paraphrased and re-scored, and a fact
guard flags any rewrite that invents details the original CV did not contain.

The app is a single Python process (`app.py`), standard library only, serving
plain HTML with a small amount of JavaScript for live progress (SSE). There is
no build step and no external Python dependencies for the app itself.

## Running it

```
python app.py            # serves http://localhost:8000 and opens it
python app.py --port 9000
python app.py --no-browser
```

Requires Python 3.13. Stop the server with Ctrl-C.

## Modes

The app has two modes, linked from the home page:

- **Reviewer** — pick a job listing and one or more CVs, run the evaluation,
  and read the report: parsed CV structure, each council model's score and
  summary, the judge's final score, a facts-as-stated panel (numbers,
  organisations and qualifications pulled from the CV text, not inferred),
  and the phrasing-sensitivity delta from the rewrite probe. With more than
  one CV selected, Reviewer mode also ranks the candidates by the judge's
  final score.

- **Candidate** — intended to take one CV and a target job and produce an
  improved version of that CV. It depends on "rewriter v2", a teammate's
  component that generates the rewritten document; that component is not
  built yet, so Candidate mode currently reports plainly that it is not
  available rather than faking a result (see `ui/pipeline/candidate.py`).

## Layout

One folder per component, matching the rest of the repo:

| | |
|---|---|
| `parser/` | PDF extraction — structure-aware CV parsing |
| `evaluator/` | the council of evaluator models, the essential-requirements check and the judge |
| `rewriter/` | rewriter v1 (the phrasing-sensitivity probe); v2 lands here |
| `ui/` | this web app — `pages.py`, `report.py`, `facts.py` for what a reviewer sees, and `pipeline/` for the orchestration that drives the components above in order and reports progress |
| `app.py` | the server entry point |
| `data/` | `job-listings/` and `cvs/`, both committed |

`ui/pipeline/` imports nothing from the rendering modules, so the runners are
testable without a server and without a model.

## Prerequisites

- **Ollama** running locally on the default port (`http://localhost:11434`),
  with these models pulled:
  - `qwen2.5:14b` and `phi4` — the evaluator council
  - `deepseek-r1:8b` — the judge
  - `qwen2.5:7b` — the essential-requirements check, and the lighter model
    set used when "fast mode" is selected
  - `qwen2.5:14b` is also the default rewriter model for the phrasing probe

  If a model is missing, the run reports it as best it can rather than
  reporting a false average.

- **The corpora.** Both are committed, so a fresh clone can run the app with
  no local setup: job listings under `data/job-listings/` (9), and CVs under
  `data/cvs/` in batch folders `batch1` to `batch4` (27). Candidates are
  labelled by batch in the Reviewer form, because `batch3` holds `bad_` and
  `professional_` versions of the same three people.

## Tests

```
python -m pytest -q
```

Every stage that would otherwise need Ollama is tested against an injected
fake (`ui/pipeline/deps.py`'s `Deps`), so the suite needs no model and no network
access. A handful of tests do import the team's real modules to check the
import wiring (`evaluator/` — the parser, evaluator, judge and essential-
requirements code) but do not call a model.
