"""One-time evaluation on the held-out test split (step 5 of the model plan).

The models are fixed before this runs: the final model is the MLP with alpha = 0.5 trained
with demographics (see CLAUDE.md, Decisions). Logistic regression and the alpha = 0 MLP are
scored in the same run as reference points only; nothing is selected on test.

Safeguards:
  - Every checkpoint is first reloaded and must reproduce its saved validate predictions,
    so a loading bug cannot silently produce test numbers for a different model.
  - Test results are written once to runs/test/. Running again refuses unless --force,
    because repeated test runs turn the test set into a second validation set.

Usage:
    python -m src.final_eval --check-only   # reload checkpoints, compare on validate; test untouched
    python -m src.final_eval                # the one test run
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import analysis
from src.data import DATA_DIR
from src.evaluate import evaluate, format_summary, subgroups
from src.features import EncodedSplit, load_split
from src.models import MLP, SparseRows, drop_tokens, predict_proba_mlp, to_csr

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "runs" / "test"
FINAL = "mlp α=0.5"
REFERENCES = ("logreg", "mlp α=0")
VALIDATE_TOLERANCE = 1e-4

Predictor = Callable[[EncodedSplit], np.ndarray]


def load_predictor(run: analysis.Run, vocab: dict) -> Predictor:
    """Rebuild a saved model and return enc -> probs over all labels."""
    n_features, n_labels = len(vocab["tokens"]), len(vocab["labels"])
    drop = tuple(vocab["demographic_ids"]) if run.config["no_demographics"] else ()

    if run.config["model"] == "logreg":
        with open(run.path / "model.pkl", "rb") as f:
            model = pickle.load(f)

        def predict(enc: EncodedSplit) -> np.ndarray:
            probs = np.zeros((enc.n_rows, n_labels), dtype=np.float32)
            probs[:, model.classes_] = model.predict_proba(to_csr(enc, n_features, drop))
            return probs
        return predict

    ckpt = torch.load(run.path / "model.pt", map_location="cpu")
    if tuple(ckpt["drop_ids"]) != drop:
        raise ValueError(f"{run.path.name}: checkpoint drop_ids do not match its config")
    model = MLP(ckpt["n_features"], ckpt["n_labels"], tuple(ckpt["hidden"]), ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])

    def predict(enc: EncodedSplit) -> np.ndarray:
        return predict_proba_mlp(model, SparseRows(*drop_tokens(enc, drop), device=torch.device("cpu")))
    return predict


def check_against_validate(runs: dict[str, analysis.Run], predictors: dict[str, Predictor],
                           val: EncodedSplit) -> dict[str, float]:
    diffs = {}
    for label, run in runs.items():
        diffs[label] = float(np.abs(predictors[label](val) - run.probs).max())
        status = "ok" if diffs[label] <= VALIDATE_TOLERANCE else "MISMATCH"
        print(f"  {label:<12} max |Δp| on validate = {diffs[label]:.2e}  {status}")
    return diffs


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> Path | None:
    p = argparse.ArgumentParser(description="One-time test-set evaluation of the fixed models.")
    p.add_argument("--features-dir", type=Path, default=DATA_DIR / "features")
    p.add_argument("--check-only", action="store_true", help="verify checkpoints on validate; skip test")
    p.add_argument("--force", action="store_true", help="run again although a test result exists")
    p.add_argument("--n-boot", type=int, default=200)
    args = p.parse_args(argv)

    previous = sorted(TEST_DIR.glob("*/results.json")) if TEST_DIR.exists() else []
    if previous and not args.check_only and not args.force:
        raise SystemExit(f"test already evaluated: {previous[-1].parent.relative_to(ROOT)}\n"
                         "Report those numbers. Use --force only for a deliberate, documented re-run.")

    vocab = json.loads((args.features_dir / "vocab.json").read_text(encoding="utf-8"))
    labels = vocab["labels"]
    available = {r.label: r for r in analysis.latest_runs(ROOT / "runs")}
    missing = [m for m in (FINAL, *REFERENCES) if m not in available]
    if missing:
        raise SystemExit(f"missing trained runs for {missing}; train them with python -m src.train")
    runs = {m: available[m] for m in (FINAL, *REFERENCES)}
    predictors = {m: load_predictor(r, vocab) for m, r in runs.items()}

    print("checking reloaded checkpoints against saved validate predictions")
    val = load_split("validate", args.features_dir)
    diffs = check_against_validate(runs, predictors, val)
    if any(d > VALIDATE_TOLERANCE for d in diffs.values()):
        raise SystemExit("a reloaded model does not reproduce its validate predictions; test not run")
    if args.check_only:
        print("check passed; test split not touched")
        return None

    test = load_split("test", args.features_dir)
    y_test = test.y.astype(np.int64)
    groups = subgroups(test, vocab)
    out = TEST_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)

    results = {"final_model": FINAL, "git_commit": _git_commit(), "test_rows": int(test.n_rows),
               "runs": {m: str(r.path.relative_to(ROOT)) for m, r in runs.items()}, "models": {}}
    print(f"\ntest split: {test.n_rows:,} patients (evaluated once)")
    for label, predict in predictors.items():
        probs = predict(test)
        res = evaluate(probs, y_test, labels, groups, test.diff)
        val_summary = json.loads((runs[label].path / "metrics.json").read_text())["validate"]
        results["models"][label] = {"test": res["summary"], "validate": val_summary}
        safe = label.replace(" ", "_").replace("α=", "a")
        np.save(out / f"probs_test_{safe}.npy", probs)
        res["per_class"].to_csv(out / f"per_class_{safe}.csv")
        stats = analysis.row_stats(probs, y_test, test.diff)
        ci = analysis.subgroup_ci(stats, groups, args.n_boot)
        ci.to_csv(out / f"subgroup_ci_{safe}.csv")
        analysis.gap_table(ci).to_csv(out / f"gaps_{safe}.csv", index=False)
        tag = "final" if label == FINAL else "reference"
        print(f"  {label:<12} ({tag})  {format_summary(res['summary'])}")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    comparison = pd.DataFrame({
        (m, split): pd.Series(results["models"][m][split]) for m in runs for split in ("validate", "test")
    }).T[["top1_acc", "top3_acc", "macro_f1", "log_loss", "ece", "diff_kl"]]
    comparison.to_csv(out / "validate_vs_test.csv")
    print(f"\n{comparison.round(4).to_string()}")
    print(f"\nsaved to {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    main()
