"""Evaluation harness: accuracy, imbalance-aware and calibration metrics, overall and per subgroup.

Every model is scored from a probability matrix ``probs`` of shape (n, n_labels) and the
true label ids ``y``. Metrics:

  - top-1 / top-3 / top-5 accuracy
  - macro-F1 and balanced accuracy (mean per-class recall), so rare pathologies count
  - log-loss and expected calibration error (top-label, equal-width bins)
  - diff_kl: KL(differential || model), how closely the model reproduces the dataset's
    differential diagnosis distribution (only when the differential is passed in)

Subgroups (equity analysis): SEX and age band come from the demographic tokens of the
encoded split; severity and cancer/non-cancer come from the true pathology.

The DDXPlus paper also reports differential-diagnosis metrics (DDR/DDP/DDF1); those are
not implemented here yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR
from src.features import SEP, EncodedSplit

TOP_KS = (1, 3, 5)
ECE_BINS = 15
CANCER_LABELS = ("Pulmonary neoplasm", "Pancreatic neoplasm")


# ---------------------------------------------------------------- core metrics

def top_k_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    true_score = probs[np.arange(len(y)), y]
    # rank of the true label = number of labels scored strictly higher
    rank = (probs > true_score[:, None]).sum(axis=1)
    return float((rank < k).mean())


def log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0)).mean())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Top-label ECE: |accuracy - confidence| per confidence bin, weighted by bin size."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def differential_kl(probs: np.ndarray, diff: np.ndarray, eps: float = 1e-12) -> float:
    """Mean KL(diff || probs) over patients; 0 when the model matches the differential exactly."""
    d = diff.astype(np.float64)
    terms = np.where(d > 0, d * (np.log(np.clip(d, eps, None)) - np.log(np.clip(probs, eps, 1.0))), 0.0)
    return float(terms.sum(axis=1).mean())


def per_class_table(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    n_labels = len(labels)
    pred = probs.argmax(axis=1)
    support = np.bincount(y, minlength=n_labels)
    predicted = np.bincount(pred, minlength=n_labels)
    tp = np.bincount(y[pred == y], minlength=n_labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    return pd.DataFrame({"support": support, "predicted": predicted, "recall": recall,
                         "precision": precision, "f1": f1}, index=pd.Index(labels, name="pathology"))


def summary_metrics(probs: np.ndarray, y: np.ndarray, labels: list[str],
                    diff: np.ndarray | None = None) -> dict[str, float]:
    """Headline metrics. Macro scores average over pathologies present in ``y``."""
    table = per_class_table(probs, y, labels)
    present = table[table.support > 0]
    out = {"n": int(len(y))}
    for k in TOP_KS:
        out[f"top{k}_acc"] = top_k_accuracy(probs, y, k)
    out["macro_f1"] = float(present.f1.mean())
    out["balanced_acc"] = float(present.recall.mean())
    out["log_loss"] = log_loss(probs, y)
    out["ece"] = expected_calibration_error(probs, y)
    if diff is not None:
        out["diff_kl"] = differential_kl(probs, diff)
    return out


# ---------------------------------------------------------------- subgroups

def subgroups(enc: EncodedSplit, vocab: dict, conditions_path: Path = DATA_DIR / "release_conditions.json"
              ) -> pd.DataFrame:
    """One row per patient with the groups used in the equity analysis.

    ``vocab`` is the dict saved as vocab.json (tokens, demographic_ids, labels).
    """
    tokens = vocab["tokens"]
    labels = vocab["labels"]
    row_of = np.repeat(np.arange(enc.n_rows), np.diff(enc.indptr))
    groups = pd.DataFrame(index=np.arange(enc.n_rows))
    for i in vocab["demographic_ids"]:
        kind, value = tokens[i].split(SEP)          # e.g. DEM_SEX, M
        col = {"DEM_SEX": "sex", "DEM_AGE": "age_band"}[kind]
        rows = row_of[enc.indices == i]
        groups.loc[rows, col] = value

    raw = json.loads(Path(conditions_path).read_text(encoding="utf-8"))
    severity = np.array([raw[name]["severity"] for name in labels])
    groups["severity"] = severity[enc.y]
    groups["cancer"] = np.isin(np.array(labels)[enc.y], CANCER_LABELS)
    return groups


def subgroup_table(probs: np.ndarray, y: np.ndarray, labels: list[str], groups: pd.DataFrame,
                   diff: np.ndarray | None = None) -> pd.DataFrame:
    """Summary metrics for every value of every group column."""
    rows = []
    for col in groups.columns:
        for value, idx in groups.groupby(col, sort=True).indices.items():
            rows.append({"group": col, "value": str(value),
                         **summary_metrics(probs[idx], y[idx], labels,
                                           None if diff is None else diff[idx])})
    return pd.DataFrame(rows).set_index(["group", "value"])


# ---------------------------------------------------------------- one call for everything

def evaluate(probs: np.ndarray, y: np.ndarray, labels: list[str],
             groups: pd.DataFrame | None = None, diff: np.ndarray | None = None) -> dict:
    """Returns {"summary": dict, "per_class": DataFrame, "subgroups": DataFrame | None}."""
    if probs.shape != (len(y), len(labels)):
        raise ValueError(f"probs shape {probs.shape} does not match ({len(y)}, {len(labels)})")
    return {
        "summary": summary_metrics(probs, y, labels, diff),
        "per_class": per_class_table(probs, y, labels),
        "subgroups": subgroup_table(probs, y, labels, groups, diff) if groups is not None else None,
    }


def format_summary(summary: dict[str, float]) -> str:
    parts = [f"n={summary['n']:,}"]
    parts += [f"top{k}={summary[f'top{k}_acc']:.4f}" for k in TOP_KS]
    parts += [f"{k}={summary[k]:.4f}" for k in ("macro_f1", "balanced_acc", "log_loss", "ece", "diff_kl")
              if k in summary]
    return "  ".join(parts)
