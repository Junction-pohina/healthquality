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
├── Materials/     # Challenge background documents (PDFs)
├── .gitignore
└── README.md
```

Update this section as code is added.

## Guidelines

- Do not commit secrets, credentials, or personal or patient data.
- Large data files belong outside the repo, or should be added to `.gitignore`.
- Keep the README up to date when setup steps change.
