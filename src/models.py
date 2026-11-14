"""Model definitions and the sparse-matrix glue they share.

Baseline: multinomial logistic regression (scikit-learn, lbfgs) on the multi-hot token
matrix from src.features. The MLP (step 3 of the plan) will live here too.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from src.features import EncodedSplit


def to_csr(enc: EncodedSplit, n_features: int, drop_ids: tuple[int, ...] = ()) -> sp.csr_matrix:
    """Multi-hot float32 CSR matrix. ``drop_ids`` columns (e.g. demographics) are zeroed out,
    keeping the column count fixed so models with and without them share one vocabulary."""
    data = np.ones(len(enc.indices), dtype=np.float32)
    X = sp.csr_matrix((data, enc.indices, enc.indptr), shape=(enc.n_rows, n_features))
    if drop_ids:
        keep = np.ones(n_features, dtype=np.float32)
        keep[list(drop_ids)] = 0.0
        X = (X @ sp.diags(keep)).tocsr()
        X.eliminate_zeros()
    return X


def make_logreg(C: float = 1.0, max_iter: int = 300, class_weight: str | None = None,
                seed: int = 0) -> LogisticRegression:
    """Multinomial logistic regression; lbfgs handles sparse input and 49 classes directly."""
    return LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight,
                              solver="lbfgs", tol=1e-4, random_state=seed)


def logit_adjust(probs: np.ndarray, prior: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Post-hoc logit adjustment: divide out the training prior (to the power tau) and
    renormalise. Trades some top-1 accuracy on the natural distribution for better recall
    on rare pathologies, without retraining."""
    logits = np.log(np.clip(probs, 1e-12, 1.0)) - tau * np.log(np.clip(prior, 1e-12, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)
