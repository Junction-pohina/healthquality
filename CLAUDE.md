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
.venv/bin/python -m src.features                        # encode all splits -> data/features/ (needed before training)
.venv/bin/python -m src.train                           # logistic regression on all train rows (~20 s), scored on validate
.venv/bin/python -m src.train --train-rows 50000        # quick run on a random subset
.venv/bin/python -m src.train --no-demographics         # same model without SEX/AGE tokens
```

Each training run writes to `runs/<timestamp>_<name>/` (git-ignored): `metrics.json`, `per_class.csv`, `subgroups.csv`, `probs_validate.npy`, the pickled model, and `_adjusted` variants of the metrics after post-hoc logit adjustment. There is no test runner or linter yet (`tests/` is empty). `requirements.txt` holds pinned top-level packages only; add new ones by hand with an exact version.

## Data

`data/` must contain the DDXPlus release files:
- `release_{train,validate,test}_patients`: CSVs **without a file extension** (~1.03M / 132k / 135k rows). Columns: `AGE, SEX, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE, DIFFERENTIAL_DIAGNOSIS`.
- `release_conditions.json`: 49 pathologies, keyed by name, with ICD-10 code, severity and the evidence codes each can present.
- `release_evidences.json`: 223 questions keyed by code (`E_91`). `data_type` is `B` (binary), `C` (categorical) or `M` (multi-choice).

Evidence encoding in `EVIDENCES`: a bare code (`E_91`) means "yes" to a binary question; `E_54_@_V_112` is question `E_54` with value `V_112` (look up the meaning in `value_meaning` of the evidences JSON); `E_56_@_6` is a numeric value. Multi-choice questions appear once per chosen value. Some value meanings are just `N`/`Y`.

## Code layout

- `src/data.py`: the loading layer everything else should use. `load_all()` returns a dict of DataFrames (`train`, `validate`, `test`, `conditions`, `evidences`); `load_patients(split, nrows=...)` loads one split. List columns are parsed with fast string splitting rather than `ast.literal_eval` (relies on no condition name containing a quote). Full load takes ~20 s and ~0.5 GB.
- `src/features.py`: streams each patient CSV in chunks and encodes every patient as a sorted set of token ids (evidence answers, INITIAL_EVIDENCE, demographic tokens `DEM_SEX_@_M` / `DEM_AGE_@_40-49`), saved CSR-style in `data/features/{split}.npz` with labels `y` and the differential as soft targets `diff`. `vocab.json` holds token and label order plus `demographic_ids`.
- `src/models.py`: `to_csr()` turns an encoded split into a scipy matrix (optionally zeroing the demographic columns), `make_logreg()` is the baseline, `logit_adjust()` divides out the training prior.
- `src/evaluate.py`: `evaluate(probs, y, labels, groups)` returns headline metrics (top-1/3/5, macro-F1, balanced accuracy, log-loss, ECE), a per-pathology table, and metrics per subgroup. `subgroups()` derives sex and age band from the demographic tokens, and severity and cancer/non-cancer from the true label. DDXPlus's differential metrics (DDR/DDP/DDF1) are not implemented yet.
- `src/train.py`: CLI that trains, scores on validate (never test) and saves the run.
- `notebooks/`: exploration only; move reusable code into `src/`. Notebooks find the repo root themselves and add it to `sys.path`, so `from src.data import ...` works whether opened from the root or `notebooks/`. `sampling.ipynb` saves samples to `data/samples/` as pickles (keeps list and category columns).

## Gotchas

- pandas 3: `groupby(...).apply` drops the grouping column from each group; prefer `groupby(...).sample(...)` / `.head(n)` on shuffled data.
- Classes are very unbalanced (URTI ~64k vs Bronchiolitis 261 in train); use stratified sampling for subsets.

## Workflow

Branch off `main` as `your-name/short-description` and open a PR against `main`; don't commit straight to `main`. Keep the README up to date when setup steps change. Secrets go in `.env` (git-ignored), with a placeholder added to `.env.example`.




## What the model does
Input: AGE, SEX, INITIAL_EVIDENCE, and the EVIDENCES list, turned into a sparse feature vector.
Output: a probability over the 49 pathologies.
Labels: PATHOLOGY is the hard label. DIFFERENTIAL_DIAGNOSIS gives a probability distribution per patient, which I’d use as a soft training target. It is not a feature, because it comes from the generator and using it at inference would leak the answer.
Features (src/features.py)
Evidence vocabulary. Build one index over the evidence tokens from release_evidences.json. Binary questions (208 of 223) contribute one token each, E_91. Categorical and multi-choice questions contribute one token per value, E_54_@_V_112. Numeric values like E_56_@_6 should be binned into ordinal tokens.
Multi-hot encoding. Each patient becomes a sparse row. A missing evidence means “not reported,” not “denied.” That is a limit of the dataset that the model cannot fix, so note it in the write-up.
Demographics. Add AGE as bins plus a scaled continuous value, and SEX as one-hot. Keep the flags separate so the equity analysis can drop them (see below).
INITIAL_EVIDENCE. Add it as its own one-hot feature. It is a strong signal.
Chunked featurization. The train file is about 670 MB. Parsing all of it into Python lists at once is heavy, so build the sparse matrix in chunks and save it as .npz under data/features/, which is git-ignored.
Models, in order
Logistic regression (sklearn, sparse input). This is a fast baseline and gives reference numbers.
MLP (PyTorch, EmbeddingBag or sparse linear first layer). Roughly 1,000 → 512 → 256 → 49 with dropout and softmax. This is the main model: about a million training rows is enough for it.
Optional ensemble of a few MLP seeds, and possibly gradient boosting later. Pinned packages in requirements.txt would need exact versions added by hand, and torch would need adding.
Training (src/train.py)
Loss: cross-entropy on the hard label, mixed with KL divergence to DIFFERENTIAL_DIAGNOSIS as a soft target. Weight the mix with a single parameter α and tune it on validation.
Imbalance. Classes range from about 64k (URTI) to 261 (Bronchiolitis) in train. Prefer class-weighted loss or post-hoc logit adjustment over oversampling. Oversampling distorts calibration. train_balanced_seed42.pkl is useful for experiments, but evaluate on the natural distribution too, because real prevalence matters.
Splits. Select models on validate. Touch test once, at the end.
Seeds and artifacts. Fix seeds, save checkpoints and the vocabulary together, and log metrics per run.
Evaluation (src/evaluate.py)
Accuracy: top-1, top-3, and top-5. Check the DDXPlus paper for its exact differential metrics before reporting numbers.
Imbalance-aware: macro-F1 and per-class recall. Overall accuracy will look good even if rare diseases fail.
Calibration: log-loss and expected calibration error, since a triage tool needs honest probabilities.
Equity analysis

The challenge is about equity, so the evaluation should be built around it:

Report every metric by SEX, by age band, and by pathology group.
Train with and without demographic features and compare the gaps.
Check calibration within each subgroup, not just overall.

Two caveats belong in any write-up. DDXPlus is synthetic, so subgroup results say nothing direct about real patients. Also, only 2 of the 49 conditions are cancers: Pulmonary neoplasm (c34) and Pancreatic neoplasm (c25). The rest are general conditions such as pneumonia, GERD, and anaphylaxis. The “Equity in Cancer Care” framing therefore needs a deliberate choice about what the model is for.

## Suggested sequence
Vocabulary and chunked feature matrices.
Logistic regression baseline with the evaluation harness.
MLP with soft-label distillation.
Subgroup and calibration analysis.
One test-set run.

## Decisions so far

Training is done on all 49 conditions and both demographics, SEX and AGE are trained.


