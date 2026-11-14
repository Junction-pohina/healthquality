"""Train a model on the encoded features and evaluate it on the validation split.

Model selection happens on ``validate``; the test split is not touched here.

Usage:
    python -m src.train                                   # logistic regression, all train rows
    python -m src.train --train-rows 100000 --C 0.5       # quick experiment on a random subset
    python -m src.train --no-demographics                 # equity comparison: drop SEX/AGE tokens
    python -m src.train --model mlp --alpha 0.5           # MLP, half hard label / half differential
    python -m src.train --model mlp --alpha 0 --epochs 3  # MLP on hard labels only, short run

Each run writes to runs/<timestamp>_<name>/ (git-ignored):
    config.json, model.pkl (logreg) or model.pt (mlp), vocab.json, metrics.json,
    per_class.csv, subgroups.csv, probs_validate.npy
and the same CSVs with an "_adjusted" suffix for the logit-adjusted predictions.
MLP runs also write history.json with per-epoch training and validation numbers.
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
import torch

from src.data import DATA_DIR
from src.evaluate import evaluate, format_summary, subgroups
from src.features import EncodedSplit, load_split
from src.models import (MLP, SparseRows, distillation_loss, drop_tokens, logit_adjust, make_logreg,
                        predict_proba_mlp, to_csr)

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def _subset(enc: EncodedSplit, idx: np.ndarray) -> EncodedSplit:
    """Rows ``idx`` (sorted) of an encoded split."""
    starts, ends = enc.indptr[idx], enc.indptr[idx + 1]
    lengths = ends - starts
    flat = np.repeat(starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths) + np.arange(lengths.sum())
    indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return EncodedSplit(indptr, enc.indices[flat], enc.y[idx], enc.diff[idx], enc.stats)


def _objective(probs: np.ndarray, y: np.ndarray, diff: np.ndarray, alpha: float) -> float:
    """The training loss evaluated on numpy probabilities (used for early stopping)."""
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    hard = -log_p[np.arange(len(y)), y].mean()
    soft = -(diff * log_p).sum(axis=1).mean()
    return float((1 - alpha) * hard + alpha * soft)


# ---------------------------------------------------------------- model-specific fitting

def fit_logreg(args, train: EncodedSplit, val: EncodedSplit, n_features: int, n_labels: int,
               drop: tuple[int, ...], run_dir: Path) -> tuple[np.ndarray, dict]:
    X_train, X_val = to_csr(train, n_features, drop), to_csr(val, n_features, drop)
    model = make_logreg(C=args.C, max_iter=args.max_iter, class_weight=args.class_weight, seed=args.seed)
    model.fit(X_train, train.y.astype(np.int64))
    n_iter = int(np.max(model.n_iter_))
    if n_iter >= args.max_iter:
        print("  hit max-iter; consider raising --max-iter")
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    # model.classes_ may skip labels absent from a small subset; map back to all labels
    probs = np.zeros((val.n_rows, n_labels), dtype=np.float32)
    probs[:, model.classes_] = model.predict_proba(X_val)
    return probs, {"n_iter": n_iter}


def fit_mlp(args, train: EncodedSplit, val: EncodedSplit, n_features: int, n_labels: int,
            drop: tuple[int, ...], run_dir: Path) -> tuple[np.ndarray, dict]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_rows = SparseRows(*drop_tokens(train, drop), device=device)
    val_rows = SparseRows(*drop_tokens(val, drop), device=device)
    y = torch.as_tensor(train.y.astype(np.int64), device=device)
    soft = torch.as_tensor(train.diff, device=device)
    y_val = val.y.astype(np.int64)

    class_weight = None
    if args.class_weight == "balanced":
        counts = np.bincount(train.y, minlength=n_labels).astype(np.float32)
        w = np.where(counts > 0, len(train.y) / (n_labels * np.maximum(counts, 1)), 0.0)
        class_weight = torch.as_tensor(w, dtype=torch.float32, device=device)

    model = MLP(n_features, n_labels, tuple(args.hidden), args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = -(-train_rows.n_rows // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    history, best, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(train_rows.n_rows, generator=gen).to(device)
        for start in range(0, train_rows.n_rows, args.batch_size):
            ids = perm[start:start + args.batch_size]
            logits = model(*train_rows.batch(ids))
            loss = distillation_loss(logits, y[ids], soft[ids], args.alpha, class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(ids)
        probs = predict_proba_mlp(model, val_rows)
        val_obj = _objective(probs, y_val, val.diff, args.alpha)
        top1 = float((probs.argmax(1) == y_val).mean())
        history.append({"epoch": epoch, "train_loss": total / train_rows.n_rows, "val_objective": val_obj,
                        "val_top1": top1, "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {epoch:2d}  train_loss={history[-1]['train_loss']:.4f}  "
              f"val_objective={val_obj:.4f}  val_top1={top1:.4f}  ({history[-1]['seconds']}s)")
        if val_obj < best - 1e-5:
            best, bad_epochs = val_obj, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop: no improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "n_features": n_features, "n_labels": n_labels,
                "hidden": list(args.hidden), "dropout": args.dropout, "drop_ids": list(drop)},
               run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    best_epoch = min(history, key=lambda h: h["val_objective"])["epoch"]
    return predict_proba_mlp(model, val_rows), {"epochs_run": len(history), "best_epoch": best_epoch}


# ---------------------------------------------------------------- entry point

def _default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Train and validate a DDXPlus model.")
    p.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    p.add_argument("--features-dir", type=Path, default=DATA_DIR / "features")
    p.add_argument("--train-rows", type=int, default=None, help="random subset of train (default: all)")
    p.add_argument("--no-demographics", action="store_true", help="drop SEX and AGE tokens")
    p.add_argument("--class-weight", choices=["balanced"], default=None)
    p.add_argument("--tau", type=float, default=1.0, help="strength of post-hoc logit adjustment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None, help="run name (default: derived from options)")
    lr = p.add_argument_group("logreg")
    lr.add_argument("--C", type=float, default=1.0, help="inverse L2 regularisation strength")
    lr.add_argument("--max-iter", type=int, default=300)
    mlp = p.add_argument_group("mlp")
    mlp.add_argument("--alpha", type=float, default=0.5,
                     help="weight of the differential (soft) target; 0 = hard labels only")
    mlp.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    mlp.add_argument("--dropout", type=float, default=0.2)
    mlp.add_argument("--epochs", type=int, default=10)
    mlp.add_argument("--batch-size", type=int, default=1024)
    mlp.add_argument("--lr", type=float, default=2e-3, help="peak learning rate (one-cycle schedule)")
    mlp.add_argument("--weight-decay", type=float, default=1e-4)
    mlp.add_argument("--patience", type=int, default=3, help="early-stopping patience in epochs")
    mlp.add_argument("--device", default=_default_device(), help="cpu, mps or cuda")
    args = p.parse_args(argv)

    vocab = json.loads((args.features_dir / "vocab.json").read_text(encoding="utf-8"))
    labels, n_features = vocab["labels"], len(vocab["tokens"])
    drop = tuple(vocab["demographic_ids"]) if args.no_demographics else ()

    model_tag = f"C{args.C:g}" if args.model == "logreg" else f"a{args.alpha:g}"
    name = args.name or "_".join(filter(None, [
        args.model, model_tag, "nodemo" if drop else None, args.class_weight,
        f"n{args.train_rows}" if args.train_rows else None]))
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    shutil.copy(args.features_dir / "vocab.json", run_dir / "vocab.json")

    train = load_split("train", args.features_dir)
    if args.train_rows and args.train_rows < train.n_rows:
        idx = np.sort(np.random.default_rng(args.seed).choice(train.n_rows, args.train_rows, replace=False))
        train = _subset(train, idx)
    val = load_split("validate", args.features_dir)
    print(f"{args.model}: train {train.n_rows:,} rows, validate {val.n_rows:,} rows"
          + (f", device {args.device}" if args.model == "mlp" else ""))

    t0 = time.time()
    fit = fit_logreg if args.model == "logreg" else fit_mlp
    probs, fit_info = fit(args, train, val, n_features, len(labels), drop, run_dir)
    fit_s = time.time() - t0
    print(f"fit in {fit_s:.0f}s")
    np.save(run_dir / "probs_validate.npy", probs)

    y_val = val.y.astype(np.int64)
    groups = subgroups(val, vocab)
    prior = np.bincount(train.y, minlength=len(labels)) / train.n_rows
    results = {
        "": evaluate(probs, y_val, labels, groups, val.diff),
        "_adjusted": evaluate(logit_adjust(probs, prior, args.tau), y_val, labels, groups, val.diff),
    }
    for suffix, res in results.items():
        res["per_class"].to_csv(run_dir / f"per_class{suffix}.csv")
        res["subgroups"].to_csv(run_dir / f"subgroups{suffix}.csv")
    (run_dir / "metrics.json").write_text(json.dumps({
        "fit_seconds": round(fit_s, 1), "train_rows": int(train.n_rows), **fit_info,
        "validate": results[""]["summary"],
        "validate_adjusted": results["_adjusted"]["summary"],
    }, indent=2))

    print(f"validate           {format_summary(results['']['summary'])}")
    print(f"validate adjusted  {format_summary(results['_adjusted']['summary'])}")
    print(f"saved to {run_dir.relative_to(ROOT)}")
    return run_dir


if __name__ == "__main__":
    main()
