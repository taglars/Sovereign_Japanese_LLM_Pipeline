# Code for the master's thesis

*Towards Sovereign LLM Pipelines: Moving a Japanese Enterprise Sales Workflow to
Self-Hosted Open-Weight LLMs.* Lars Ostertag, University of Mannheim, 2026.

This repository holds the code of the thesis that can leave the company whose
pipeline it studies. Appendix A of the thesis describes all of the code, including
the parts withheld here, and how each experiment was run.

## What is here

| Folder | Content |
|---|---|
| `harness/` | The evaluation harness of Experiments 2 to 4: the judge client and rubric (`jv/rubric.py`), the store of judgements, the agreement statistics, the human ratings, the analysis of Experiment 3 and its robustness checks, the feedback requests and the codebook (`jv/`); the generation code of Experiment 3 (`gen/`); the batch jobs for the GPU cluster (`cluster/`); the settings (`config/`); the tests (`tests/`). |
| `thesis-build/` | The scripts that turn the results into the numbers, tables and figures of the thesis: `extract.py` (data), `macros.py` and `results_macros.py` (one LaTeX macro per number), `figures.py`, `tables.py`, and `constants.py` for the node prices. |

## What is withheld, and why

The thesis was written under a confidentiality agreement with the company. The
following stay with the author and are not in this repository:

- **The Approach 3 pipeline,** the self-hosted pipeline built inside the company
  (orchestration, worker, Azure Batch set-up, Databricks notebooks). Appendix A of
  the thesis describes its structure, tests and deployment.
- **The company's prompts and texts,** which Experiment 3 copied byte for byte
  (originally `harness/gen/vendor/` and `harness/gen/vendor_company/`), and their
  Japanese translation table (`harness/gen/translations/`).
- **The generator of the synthetic customers** (`harness/gen/customers.py`), whose
  product catalogue was reconstructed from those prompts.
- **The morphological keigo measures,** part of the pipeline's code, which
  `jv/checks.py`, `gen/metrics.py` and `thesis-build/scripts/tables.py` import.
  Appendix B of the thesis lists their word lists and rules.
- **Every text and score that came from the company or names its customers:** the
  twelve blind scripts and the rating form built from them, the returned rating
  forms, the GPT-5.6 judge's scores, and all raw judgements and feedback
  (`harness/data/`, `harness/out/`).
- **The inputs of `thesis-build/scripts/extract.py`:** the production telemetry, the
  exported runs of Experiment 1 and the project's status review.
- **The names replaced in the example of Appendix D** (manufacturer, models, model
  codes, branded systems), which `thesis-build/scripts/tables.py` reads from
  `thesis-build/scripts/example_names.json`.

## Changes against the code that ran

Only what was needed to remove company and personal details:

- brand, product and feature names replaced by marked placeholders
  (`BRAND_NAMES` in `gen/metrics.py`, one line each in `gen/review.py` and
  `gen/templates.py`);
- the product-family keys renamed to `light`, `electric` and `heavy`, as the thesis
  calls them; the withheld data uses the company's names;
- the raters' names replaced by `rater1` to `rater5` (`RATER_ORDER` in
  `thesis-build/scripts/results_macros.py`);
- the paths to the withheld inputs anonymised (`thesis-build/scripts/extract.py`);
- the harness folder renamed from `judge-validation` to `harness`;
- the generation test removed, since it checks the withheld prompts byte for byte,
  and the rating-workbook test made to skip when the form is absent;
- comments and docstrings shortened to what explains the code.

## What runs here

```
cd harness
make test PY=python3
```

runs seven test suites on a laptop, without a GPU and without the withheld
material; the rating-workbook suite skips. Everything else, generating and judging
scripts on the cluster and rebuilding the thesis numbers, needs the withheld
inputs. The targets are listed at the top of `harness/Makefile` and in Appendix A.

Python 3.11 or later; packages in `harness/requirements.txt` and
`thesis-build/requirements.txt`.
