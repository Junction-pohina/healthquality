# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Junction hackathon project for the **Equity in Cancer Care** challenge, building a diagnosis model on the DDXPlus synthetic patient dataset. Background material (the Finnish National Cancer Strategy 2026–2035, its roadmap, a link collection, the challenge slides) is in `Materials/` as PDFs, mostly in Finnish. `Materials/` and `data/` are git-ignored and must never be committed (they hold patient data).

## Commands

Python 3.10+ in a local venv at `.venv`:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab notebooks/
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/sampling.ipynb --output /tmp/out.ipynb   # run a notebook headless
```

There is no test runner or linter yet (`tests/` is empty), and the training entry point `src/train.py` is still an empty stub. `requirements.txt` holds pinned top-level packages only; add new ones by hand with an exact version.

## Data

`data/` must contain the DDXPlus release files:
- `release_{train,validate,test}_patients`: CSVs **without a file extension** (~1.03M / 132k / 135k rows). Columns: `AGE, SEX, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE, DIFFERENTIAL_DIAGNOSIS`.
- `release_conditions.json`: 49 pathologies, keyed by name, with ICD-10 code, severity and the evidence codes each can present.
- `release_evidences.json`: 223 questions keyed by code (`E_91`). `data_type` is `B` (binary), `C` (categorical) or `M` (multi-choice).

Evidence encoding in `EVIDENCES`: a bare code (`E_91`) means "yes" to a binary question; `E_54_@_V_112` is question `E_54` with value `V_112` (look up the meaning in `value_meaning` of the evidences JSON); `E_56_@_6` is a numeric value. Multi-choice questions appear once per chosen value. Some value meanings are just `N`/`Y`.

## Code layout

- `src/data.py`: the loading layer everything else should use. `load_all()` returns a dict of DataFrames (`train`, `validate`, `test`, `conditions`, `evidences`); `load_patients(split, nrows=...)` loads one split. List columns are parsed with fast string splitting rather than `ast.literal_eval` (relies on no condition name containing a quote). Full load takes ~20 s and ~0.5 GB.
- `src/features.py`, `src/models.py`, `src/train.py`, `src/evaluate.py`: empty stubs. Planned roles: multi-hot evidence features, model definitions, training entry point, validation/test metrics.
- `notebooks/`: exploration only; move reusable code into `src/`. Notebooks find the repo root themselves and add it to `sys.path`, so `from src.data import ...` works whether opened from the root or `notebooks/`. `sampling.ipynb` saves samples to `data/samples/` as pickles (keeps list and category columns).

## Gotchas

- pandas 3: `groupby(...).apply` drops the grouping column from each group; prefer `groupby(...).sample(...)` / `.head(n)` on shuffled data.
- Classes are very unbalanced (URTI ~64k vs Bronchiolitis 261 in train); use stratified sampling for subsets.

## Workflow

Branch off `main` as `your-name/short-description` and open a PR against `main`; don't commit straight to `main`. Keep the README up to date when setup steps change. Secrets go in `.env` (git-ignored), with a placeholder added to `.env.example`.
