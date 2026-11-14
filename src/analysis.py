"""Equity analysis on saved validation predictions (step 4 of the model plan).

Works from the runs/ directories written by src.train, so nothing is retrained here:

  - subgroup metrics with bootstrap 95% confidence intervals
  - gaps between the best and worst subgroup per metric, with whether they exceed noise
  - with vs. without demographic features: paired bootstrap of the difference per subgroup
  - reliability (calibration) curves per subgroup

Subgroups: sex, age band (from the demographic tokens), severity and cancer/non-cancer
(from the true pathology). Only the validate split is used; test stays untouched.

Usage:
    python -m src.analysis                  # latest run of each model config -> runs/analysis/
    python -m src.analysis --n-boot 500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR
from src.evaluate import ECE_BINS, subgroups
from src.features import load_split

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
GROUP_COLUMNS = ("sex", "age_band", "severity", "cancer")
METRICS = ("top1_acc", "macro_f1", "log_loss", "ece", "diff_kl")


# ---------------------------------------------------------------- runs

@dataclass
class Run:
    path: Path
    config: dict
    probs: np.ndarray

    @property
    def label(self) -> str:
        c = self.config
        name = "logreg" if c["model"] == "logreg" else f"mlp α={c['alpha']:g}"
        return name + (" (no demo)" if c["no_demographics"] else "")


def list_runs(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """One row per finished run (has probs_validate.npy), newest first."""
    rows = []
    for d in sorted(Path(runs_dir).glob("*/"), reverse=True):
        if not (d / "probs_validate.npy").exists() or not (d / "config.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        rows.append({"path": d, "model": c["model"], "alpha": c.get("alpha") if c["model"] == "mlp" else None,
                     "no_demographics": c["no_demographics"], "train_rows": c.get("train_rows"),
                     "class_weight": c.get("class_weight")})
    return pd.DataFrame(rows)


def latest_runs(runs_dir: Path = RUNS_DIR) -> list[Run]:
    """Newest full-data, unweighted run of each (model, alpha, no_demographics) config."""
    df = list_runs(runs_dir)
    df = df[df.train_rows.isna() & df.class_weight.isna()]
    keep = df.drop_duplicates(["model", "alpha", "no_demographics"], keep="first")
    keep = keep.sort_values(["model", "alpha", "no_demographics"], na_position="first")
    return [load_run(p) for p in keep.path]


def load_run(path: Path) -> Run:
    path = Path(path)
    return Run(path, json.loads((path / "config.json").read_text()), np.load(path / "probs_validate.npy"))


# ---------------------------------------------------------------- per-row ingredients

@dataclass
class RowStats:
    """Everything the metrics need, per validation row, so bootstraps are cheap."""
    y: np.ndarray
    pred: np.ndarray
    conf: np.ndarray
    nll: np.ndarray
    kl: np.ndarray
    n_labels: int


def row_stats(probs: np.ndarray, y: np.ndarray, diff: np.ndarray, eps: float = 1e-12) -> RowStats:
    p = np.clip(probs, eps, 1.0)
    d = diff.astype(np.float64)
    kl = np.where(d > 0, d * (np.log(np.clip(d, eps, None)) - np.log(p)), 0.0).sum(axis=1)
    return RowStats(y=y, pred=probs.argmax(axis=1), conf=probs.max(axis=1),
                    nll=-np.log(p[np.arange(len(y)), y]), kl=kl, n_labels=probs.shape[1])


def metrics_from_rows(s: RowStats, idx: np.ndarray) -> dict[str, float]:
    y, pred, conf = s.y[idx], s.pred[idx], s.conf[idx]
    correct = pred == y
    support = np.bincount(y, minlength=s.n_labels)
    predicted = np.bincount(pred, minlength=s.n_labels)
    tp = np.bincount(y[correct], minlength=s.n_labels)
    present = support > 0
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    bins = np.minimum((conf * ECE_BINS).astype(int), ECE_BINS - 1)
    n_bin = np.bincount(bins, minlength=ECE_BINS)
    acc_bin = np.bincount(bins, weights=correct, minlength=ECE_BINS)
    conf_bin = np.bincount(bins, weights=conf, minlength=ECE_BINS)
    ece = np.abs(acc_bin - conf_bin).sum() / len(idx)
    return {"top1_acc": correct.mean(), "macro_f1": f1[present].mean(), "log_loss": s.nll[idx].mean(),
            "ece": ece, "diff_kl": s.kl[idx].mean()}


# ---------------------------------------------------------------- bootstrap tables

def _group_indices(groups: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    out = [("all", "all", np.arange(len(groups)))]
    for col in GROUP_COLUMNS:
        for value, idx in groups.groupby(col, sort=True).indices.items():
            out.append((col, str(value), np.asarray(idx)))
    return out


def subgroup_ci(s: RowStats, groups: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Point estimate and percentile 95% CI per subgroup and metric."""
    rng = np.random.default_rng(seed)
    rows = []
    for col, value, idx in _group_indices(groups):
        point = metrics_from_rows(s, idx)
        boots = pd.DataFrame([metrics_from_rows(s, rng.choice(idx, len(idx))) for _ in range(n_boot)])
        row = {"group": col, "value": value, "n": len(idx)}
        for m in METRICS:
            row[m] = point[m]
            row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(boots[m], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows).set_index(["group", "value"])


def gap_table(ci: pd.DataFrame) -> pd.DataFrame:
    """Best-to-worst spread per group and metric. ``exceeds_noise`` is True when the worst
    subgroup's CI does not overlap the best one's (a conservative test)."""
    rows = []
    higher_is_better = {"top1_acc": True, "macro_f1": True, "log_loss": False, "ece": False, "diff_kl": False}
    for col in GROUP_COLUMNS:
        sub = ci.loc[col]
        for m in METRICS:
            best = sub[m].idxmax() if higher_is_better[m] else sub[m].idxmin()
            worst = sub[m].idxmin() if higher_is_better[m] else sub[m].idxmax()
            if higher_is_better[m]:
                separated = sub.loc[worst, f"{m}_hi"] < sub.loc[best, f"{m}_lo"]
            else:
                separated = sub.loc[worst, f"{m}_lo"] > sub.loc[best, f"{m}_hi"]
            rows.append({"group": col, "metric": m, "best": best, "worst": worst,
                         "gap": abs(sub.loc[best, m] - sub.loc[worst, m]), "exceeds_noise": bool(separated)})
    return pd.DataFrame(rows)


def ablation(with_demo: RowStats, without_demo: RowStats, groups: pd.DataFrame,
             n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """(without - with) per subgroup and metric, with a paired bootstrap 95% CI: the same
    resampled patients are scored by both models, so shared noise cancels."""
    rng = np.random.default_rng(seed)
    rows = []
    for col, value, idx in _group_indices(groups):
        a, b = metrics_from_rows(with_demo, idx), metrics_from_rows(without_demo, idx)
        deltas = []
        for _ in range(n_boot):
            r = rng.choice(idx, len(idx))
            ma, mb = metrics_from_rows(with_demo, r), metrics_from_rows(without_demo, r)
            deltas.append({m: mb[m] - ma[m] for m in METRICS})
        deltas = pd.DataFrame(deltas)
        row = {"group": col, "value": value, "n": len(idx)}
        for m in METRICS:
            row[f"{m}_delta"] = b[m] - a[m]
            row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(deltas[m], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows).set_index(["group", "value"])


def reliability(s: RowStats, groups: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Mean confidence vs. accuracy per confidence bin, per subgroup (long format)."""
    rows = []
    for col, value, idx in _group_indices(groups):
        conf, correct = s.conf[idx], s.pred[idx] == s.y[idx]
        bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
        for b in np.unique(bins):
            m = bins == b
            rows.append({"group": col, "value": value, "bin": int(b), "n": int(m.sum()),
                         "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Subgroup, ablation and calibration analysis of saved runs.")
    p.add_argument("--features-dir", type=Path, default=DATA_DIR / "features")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    vocab = json.loads((args.features_dir / "vocab.json").read_text(encoding="utf-8"))
    val = load_split("validate", args.features_dir)
    y = val.y.astype(np.int64)
    groups = subgroups(val, vocab)
    runs = latest_runs()
    if not runs:
        raise SystemExit("no finished runs in runs/; train some with python -m src.train first")

    out = RUNS_DIR / "analysis" / f"{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    stats = {r.label: row_stats(r.probs, y, val.diff) for r in runs}
    (out / "runs.json").write_text(json.dumps({r.label: str(r.path.relative_to(ROOT)) for r in runs}, indent=2))

    cis, gaps, rel = [], [], []
    for label, s in stats.items():
        ci = subgroup_ci(s, groups, args.n_boot, args.seed)
        cis.append(ci.assign(run=label))
        gaps.append(gap_table(ci).assign(run=label))
        rel.append(reliability(s, groups).assign(run=label))
        print(f"{label}: {sum(g.exceeds_noise for g in gap_table(ci).itertuples())} gaps exceed noise")
    pd.concat(cis).to_csv(out / "subgroup_ci.csv")
    pd.concat(gaps).to_csv(out / "gaps.csv", index=False)
    pd.concat(rel).to_csv(out / "reliability.csv", index=False)

    abl = []
    for label in stats:
        if "(no demo)" in label or f"{label} (no demo)" not in stats:
            continue
        abl.append(ablation(stats[label], stats[f"{label} (no demo)"], groups, args.n_boot, args.seed)
                   .assign(run=label))
    if abl:
        pd.concat(abl).to_csv(out / "ablation.csv")
    print(f"saved to {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    main()
