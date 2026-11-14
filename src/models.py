"""Model definitions and the sparse-matrix glue they share.

Baseline: multinomial logistic regression (scikit-learn, lbfgs) on the multi-hot token
matrix from src.features.

Main model: an MLP (PyTorch) whose first layer is an EmbeddingBag over the token ids, which
equals a linear layer on the multi-hot vector without ever building the dense matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

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


# ---------------------------------------------------------------- MLP

def drop_tokens(enc: EncodedSplit, drop_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """CSR (indptr, indices) with the given token ids removed from every row."""
    if not drop_ids:
        return enc.indptr, enc.indices
    keep = ~np.isin(enc.indices, drop_ids)
    row_of = np.repeat(np.arange(enc.n_rows), np.diff(enc.indptr))
    lengths = np.bincount(row_of[keep], minlength=enc.n_rows)
    indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return indptr, enc.indices[keep]


class SparseRows:
    """Token-id rows held as torch tensors, sliced into EmbeddingBag batches."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, device: torch.device):
        self.indptr = torch.as_tensor(indptr, dtype=torch.long, device=device)
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=device)
        self.n_rows = len(indptr) - 1

    def batch(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(flat token ids, offsets) for the given row ids, in that order."""
        starts, ends = self.indptr[rows], self.indptr[rows + 1]
        lengths = ends - starts
        offsets = torch.cumsum(lengths, 0) - lengths
        # position j of the batch's flat array belongs to row r and reads indices[starts[r] + (j - offsets[r])]
        row_of = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), lengths)
        flat = starts[row_of] + torch.arange(int(lengths.sum()), device=rows.device) - offsets[row_of]
        return self.indices[flat], offsets


class MLP(nn.Module):
    """tokens -> EmbeddingBag(sum) -> [ReLU, dropout, Linear]* -> logits over the pathologies."""

    def __init__(self, n_features: int, n_labels: int, hidden: tuple[int, ...] = (512, 256),
                 dropout: float = 0.2):
        super().__init__()
        self.embed = nn.EmbeddingBag(n_features, hidden[0], mode="sum")
        self.bias = nn.Parameter(torch.zeros(hidden[0]))
        layers: list[nn.Module] = []
        widths = list(hidden) + [n_labels]
        for i in range(len(hidden)):
            layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(widths[i], widths[i + 1])]
        self.head = nn.Sequential(*layers)
        nn.init.normal_(self.embed.weight, std=n_features ** -0.5 * 2)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(tokens, offsets) + self.bias)


def distillation_loss(logits: torch.Tensor, y: torch.Tensor, soft: torch.Tensor, alpha: float,
                      class_weight: torch.Tensor | None = None) -> torch.Tensor:
    """(1 - alpha) * cross-entropy on the hard label + alpha * cross-entropy to the soft target.

    Cross-entropy to the soft target differs from KL(soft || model) only by the target's
    entropy, a constant, so the gradients are the same as for the KL term."""
    log_p = torch.log_softmax(logits, dim=1)
    hard = nn.functional.nll_loss(log_p, y, weight=class_weight)
    if alpha == 0:
        return hard
    soft_ce = -(soft * log_p).sum(dim=1).mean()
    return (1 - alpha) * hard + alpha * soft_ce


@torch.no_grad()
def predict_proba_mlp(model: MLP, rows: SparseRows, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    device = rows.indptr.device
    out = []
    for start in range(0, rows.n_rows, batch_size):
        ids = torch.arange(start, min(start + batch_size, rows.n_rows), device=device)
        out.append(torch.softmax(model(*rows.batch(ids)), dim=1).float().cpu().numpy())
    return np.concatenate(out)
