# Project

Junction hackathon project for the **Equity in Cancer Care** challenge. Background material (the Finnish National Cancer Strategy 2026–2035, its implementation roadmap, a link collection on cancer strategies and data, and the challenge slides) is in [`Materials/`](Materials/).

## Getting started

### Prerequisites

- Git
- Python 3.10 or newer

### Clone the repository

```bash
git clone https://github.com/Junction-pohina/healthquality.git
cd healthquality
```

### Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

Install dependencies once the project has a dependency file:

```bash
pip install -r requirements.txt
```

If you add a new package, add it to `requirements.txt` in the same commit (`pip freeze > requirements.txt`, or add the line by hand).

### Environment variables

Keep secrets such as API keys in a `.env` file at the repository root. `.env` is listed in `.gitignore`, so it is never committed. When you add a new variable, also add it with a placeholder value to `.env.example` so others know it exists.

### Run the Web Application (Patient Triage UI + Municipal Dashboard)

A pre-trained PyTorch `MLP (α=0.5)` inference checkpoint (`models/mlp_a0.5/`) is bundled in the repository so you can run and edit the web app immediately on any machine without downloading the DDXPlus CSVs first:

```bash
.venv/bin/python -m uvicorn src.api:app --reload --port 8000
```

Then open http://127.0.0.1:8000 in your browser.
- If you train a newer checkpoint into `runs/` (`python -m src.train --model mlp --alpha 0.5`), `src/api.py` automatically loads the latest `runs/` checkpoint first and falls back to `models/mlp_a0.5/`.

## Development workflow

1. Pull the latest `main` before you start:
   ```bash
   git checkout main
   git pull
   ```
2. Create a branch for your work:
   ```bash
   git checkout -b your-name/short-description
   ```
3. Make small, focused commits with clear messages.
4. Push your branch and open a pull request against `main`:
   ```bash
   git push -u origin your-name/short-description
   ```
5. Ask a teammate to review, then merge. Delete the branch after it is merged.

Avoid committing straight to `main`, so that teammates working in parallel don't run into conflicts.

## Repository layout

```
.
├── data/                    # DDXPlus CSVs, JSONs, local SQLite DB (git-ignored)
├── Materials/               # Challenge background documents (PDFs, git-ignored)
├── models/
│   └── mlp_a0.5/            # Bundled PyTorch MLP checkpoint + vocab for instant UI startup
├── notebooks/               # Exploratory & equity analysis notebooks
├── src/                     # Project code
│   ├── __init__.py
│   ├── api.py               # FastAPI server, PyTorch MLP inference & SQLite municipal stats
│   ├── clinical_catalog.py  # DDXPlus evidence codes, Finnish UI mappings & 49 specialties
│   ├── data.py              # Load patient CSVs, parse EVIDENCES, build labels
│   ├── features.py          # Convert evidences into multi-hot feature vectors
│   ├── models.py            # Model definitions (LogReg + PyTorch EmbeddingBag MLP)
│   ├── train.py             # Training entry point
│   ├── evaluate.py          # Metrics on validation and test sets
│   ├── analysis.py          # Subgroup CIs, gaps, demographics ablation, reliability
│   └── final_eval.py        # One-time held-out test evaluation
├── web/
│   └── index.html           # Stitch-styled SPA frontend (Patient Survey + Municipal Table)
├── tests/                   # Unit tests
├── .env.example             # Template for environment variables
├── .gitignore
├── requirements.txt         # Python dependencies
└── README.md
```

Put dataset files in `data/`. Keep notebooks for exploration; move reusable code into `src/`.

Update this section as code is added.

## Guidelines

- Do not commit secrets, credentials, or personal or patient data.
- Large data files belong outside the repo, or should be added to `.gitignore`.
- Keep the README up to date when setup steps change.
