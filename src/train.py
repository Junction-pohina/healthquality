"""Train a model on the encoded features and evaluate it on the validation split.

Model selection happens on ``validate``; the test split is not touched here.

Usage:
    python -m src.train                                   # logistic regression, all train rows
    python -m src.train --train-rows 100000 --C 0.5       # quick experiment on a random subset
    python -m src.train --no-demographics                 # equity comparison: drop SEX/AGE tokens

Each run writes to runs/<timestamp>_<name>/ (git-ignored):
    config.json, model.pkl, vocab.json, metrics.json,
    per_class.csv, subgroups.csv, probs_validate.npy
and the same files with an "_adjusted" suffix for the logit-adjusted predictions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data import DATA_DIR
from src.evaluate import evaluate, format_summary, subgroups
from src.features import load_split
from src.models import logit_adjust, make_logreg, to_csr

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def _save_eval(result: dict, run_dir: Path, suffix: str) -> None:
    result["per_class"].to_csv(run_dir / f"per_class{suffix}.csv")
    result["subgroups"].to_csv(run_dir / f"subgroups{suffix}.csv")


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Train and validate a DDXPlus model.")
    p.add_argument("--model", choices=["logreg"], default="logreg")
    p.add_argument("--features-dir", type=Path, default=DATA_DIR / "features")
    p.add_argument("--train-rows", type=int, default=None, help="random subset of train (default: all)")
    p.add_argument("--no-demographics", action="store_true", help="zero out SEX and AGE tokens")
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 regularisation strength")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--class-weight", choices=["balanced"], default=None)
    p.add_argument("--tau", type=float, default=1.0, help="strength of post-hoc logit adjustment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None, help="run name (default: derived from options)")
    args = p.parse_args(argv)

    vocab = json.loads((args.features_dir / "vocab.json").read_text(encoding="utf-8"))
    labels, n_features = vocab["labels"], len(vocab["tokens"])
    drop = tuple(vocab["demographic_ids"]) if args.no_demographics else ()

    name = args.name or "_".join(filter(None, [
        args.model, f"C{args.C:g}", "nodemo" if drop else None, args.class_weight,
        f"n{args.train_rows}" if args.train_rows else None]))
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    shutil.copy(args.features_dir / "vocab.json", run_dir / "vocab.json")

    t0 = time.time()
    train = load_split("train", args.features_dir)
    X_train, y_train = to_csr(train, n_features, drop), train.y.astype(np.int64)
    if args.train_rows and args.train_rows < len(y_train):
        idx = np.sort(np.random.default_rng(args.seed).choice(len(y_train), args.train_rows, replace=False))
        X_train, y_train = X_train[idx], y_train[idx]
    val = load_split("validate", args.features_dir)
    X_val, y_val = to_csr(val, n_features, drop), val.y.astype(np.int64)
    groups = subgroups(val, vocab)
    print(f"loaded train {X_train.shape}, validate {X_val.shape} in {time.time() - t0:.0f}s")

    t0 = time.time()
    model = make_logreg(C=args.C, max_iter=args.max_iter, class_weight=args.class_weight, seed=args.seed)
    model.fit(X_train, y_train)
    fit_s = time.time() - t0
    n_iter = int(np.max(model.n_iter_))
    print(f"fit in {fit_s:.0f}s, {n_iter} iterations"
          + ("  (hit max-iter; consider raising it)" if n_iter >= args.max_iter else ""))
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # model.classes_ may skip labels absent from a small subset; map back to all labels
    probs = np.zeros((len(y_val), len(labels)), dtype=np.float32)
    probs[:, model.classes_] = model.predict_proba(X_val)
    np.save(run_dir / "probs_validate.npy", probs)

    prior = np.bincount(y_train, minlength=len(labels)) / len(y_train)
    results = {
        "": evaluate(probs, y_val, labels, groups),
        "_adjusted": evaluate(logit_adjust(probs, prior, args.tau), y_val, labels, groups),
    }
    for suffix, res in results.items():
        _save_eval(res, run_dir, suffix)
    (run_dir / "metrics.json").write_text(json.dumps({
        "fit_seconds": round(fit_s, 1), "n_iter": n_iter, "train_rows": int(len(y_train)),
        "validate": results[""]["summary"],
        "validate_adjusted": results["_adjusted"]["summary"],
    }, indent=2))

    print(f"validate           {format_summary(results['']['summary'])}")
    print(f"validate adjusted  {format_summary(results['_adjusted']['summary'])}")
    print(f"saved to {run_dir.relative_to(ROOT)}")
    return run_dir


if __name__ == "__main__":
    main()
