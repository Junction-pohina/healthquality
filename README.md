# Project

Questionnaire demo: https://claude.ai/artifact/GA4au2LQJtXveucreqJH5P

Diagnosis model test results: https://claude.ai/artifact/YEQbzb4gRiYgSKx8GwYJBJ 


## Problem & Solution
Early cancer symptoms often mimic minor illnesses, delaying care and straining primary health systems. We built a diagnostic tool to safely lower the threshold for seeking help and route patients to the right care immediately.

## Core Features (FI, SV, EN)
* **AI Assessment:** A neural network ranks 49 conditions based on a short adaptive questionnaire, returning the top 3 probabilities.
* **Cancer-Aware Routing:** Built-in rules prioritize symptoms lasting >3 weeks and 7 specific red flags.
* **Actionable Output:** Gives patients an urgency tier, a recommended specialty, and a plain-language rationale.
* **Clinician Dashboard:** Provides anonymous, regional reporting for wellbeing-counties.

## Data & Model
* Trained on **DDXPlus** (1.3 million synthetic patients) achieving 99.75% test accuracy. 
* Uses probability to handle the uncertainty of early symptoms.
* *Note:* Currently a proof of concept relying on synthetic data. It requires real-world data integration for clinical reliability.

## Path to Implementation & Next Steps
1. **Dynamic Scaling:** Implement Bayesian updating so the questionnaire adapts in real-time, minimizing unnecessary questions.
2. **Public Integration:** Connect securely with existing health portals (e.g., OmaKanta, Maisa).
3. **Real-World Validation:** Pretrain on Nordic health data, then track early users from assessment to final diagnosis. 
4. **Continuous Learning:** Replace synthetic training labels with confirmed patient outcomes to continuously improve accuracy.

## Sources
* *Kansallinen syöpästrategia 2026–2035* (Goal 3)
* THL, Ohjaus 21/2016: *Hoitoonpääsyn seuranta syövän hoidossa*
* *Nordic systematic review*, Acta Oncologica 2022;61(11):1317–1331.
* *DDXPlus synthetic dataset*: https://doi.org/10.6084/m9.figshare.22687585


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
