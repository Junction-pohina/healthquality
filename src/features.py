"""Encode DDXPlus patients as sparse rows of token ids (step 1 of the model plan).

Every patient becomes a sorted set of token ids, stored CSR-style as two numpy
arrays (indptr, indices), so no scipy dependency is needed. Tokens are:

  - evidence answers from release_evidences.json: binary "E_91", value "E_56_@_6"
  - demographic tokens "DEM_SEX_@_M" and "DEM_AGE_@_40-49"

Demographic ids are listed in Vocab.demographic_ids, so a model can be trained
with or without them (the equity comparison).

Labels are indices into the sorted condition names. The differential diagnosis
probabilities are kept as a dense (n, n_labels) float32 matrix for use as soft
targets only. They must never be used as input features (leakage).

Usage:
    python -m src.features                                  # all splits -> data/features/
    python -m src.features --split validate --nrows 50000   # preview -> data/features_preview/
"""

from __future__ import annotations

import argparse
import json
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, SPLITS, _parse_differential, _parse_str_list

SEP = "_@_"
SEXES = ("M", "F")
AGE_EDGES = (0, 18, 30, 40, 50, 60, 70, 80, 200)  # right-open bins; the last is open-ended
PATIENT_DTYPES = {"AGE": "int16", "SEX": "category", "PATHOLOGY": "category",
                  "INITIAL_EVIDENCE": "category"}


def _age_band_names() -> tuple[str, ...]:
    names = []
    for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:]):
        names.append(f"{lo}+" if hi == AGE_EDGES[-1] else f"{lo}-{hi - 1}")
    return tuple(names)


AGE_BANDS = _age_band_names()


def demographic_tokens() -> list[str]:
    return ([f"DEM_SEX{SEP}{s}" for s in SEXES]
            + [f"DEM_AGE{SEP}{b}" for b in AGE_BANDS])


@dataclass(frozen=True)
class Vocab:
    tokens: tuple[str, ...]        # id -> token
    index: dict[str, int]          # token -> id
    demographic_ids: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.tokens)


def build_vocab(evidences_path: Path = DATA_DIR / "release_evidences.json") -> Vocab:
    """One token per binary question, one per (question, value) for categorical/multi-choice."""
    raw = json.loads(Path(evidences_path).read_text(encoding="utf-8"))
    tokens: list[str] = []
    for q in raw.values():
        name = q["name"]
        if q["data_type"] == "B":
            tokens.append(name)
        else:
            values = q["possible-values"] or list(q["value_meaning"])
            tokens.extend(f"{name}{SEP}{v}" for v in values)
    demo = demographic_tokens()
    tokens.extend(demo)
    if len(set(tokens)) != len(tokens):
        dupes = sorted({t for t in tokens if tokens.count(t) > 1})[:5]
        raise ValueError(f"duplicate tokens in vocabulary, e.g. {dupes}")
    index = {t: i for i, t in enumerate(tokens)}
    return Vocab(tuple(tokens), index, tuple(index[t] for t in demo))


def build_label_index(conditions_path: Path = DATA_DIR / "release_conditions.json") -> dict[str, int]:
    raw = json.loads(Path(conditions_path).read_text(encoding="utf-8"))
    names = sorted(v["condition_name"] for v in raw.values())
    return {name: i for i, name in enumerate(names)}


@dataclass
class EncodedSplit:
    indptr: np.ndarray   # int64, (n + 1,): row i owns indices[indptr[i]:indptr[i + 1]]
    indices: np.ndarray  # int32, token ids, sorted within each row
    y: np.ndarray        # int16, pathology id; -1 if the label is unknown
    diff: np.ndarray     # float32, (n, n_labels): differential probabilities (targets only)
    stats: dict[str, int]

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1


def encode_frame(df: pd.DataFrame, vocab: Vocab, label_index: dict[str, int]) -> EncodedSplit:
    """Encode a parsed frame (EVIDENCES as lists, DIFFERENTIAL_DIAGNOSIS as pairs)."""
    index = vocab.index
    n = len(df)
    stats = {"unknown_evidence": 0, "unknown_initial": 0, "unknown_demographic": 0,
             "unknown_pathology": 0, "unknown_differential": 0}

    ages = pd.to_numeric(df["AGE"]).to_numpy()
    bands = np.digitize(ages, AGE_EDGES[1:-1])
    sex_tokens = [f"DEM_SEX{SEP}{s}" for s in df["SEX"].astype(str)]
    age_tokens = [f"DEM_AGE{SEP}{AGE_BANDS[b]}" for b in bands]
    initials = df["INITIAL_EVIDENCE"].astype(str).tolist()
    evidences = df["EVIDENCES"].tolist()

    lengths = np.empty(n, dtype=np.int64)
    flat = array("i")
    for i in range(n):
        ids: set[int] = set()
        for tok in evidences[i]:
            j = index.get(tok)
            if j is None:
                stats["unknown_evidence"] += 1
            else:
                ids.add(j)
        j = index.get(initials[i])
        if j is None:
            stats["unknown_initial"] += 1
        else:
            ids.add(j)
        for tok in (sex_tokens[i], age_tokens[i]):
            j = index.get(tok)
            if j is None:
                stats["unknown_demographic"] += 1
            else:
                ids.add(j)
        row = sorted(ids)
        lengths[i] = len(row)
        flat.extend(row)

    y = np.array([label_index.get(p, -1) for p in df["PATHOLOGY"].astype(str)], dtype=np.int16)
    stats["unknown_pathology"] = int((y < 0).sum())

    diff = np.zeros((n, len(label_index)), dtype=np.float32)
    for i, pairs in enumerate(df["DIFFERENTIAL_DIAGNOSIS"]):
        for name, prob in pairs:
            j = label_index.get(name)
            if j is None:
                stats["unknown_differential"] += 1
            else:
                diff[i, j] = prob

    indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return EncodedSplit(indptr, np.frombuffer(flat, dtype=np.int32).copy(), y, diff, stats)


def _concat(parts: list[EncodedSplit]) -> EncodedSplit:
    if not parts:
        raise ValueError("no rows to encode")
    indptrs = [parts[0].indptr]
    offset = parts[0].indptr[-1]
    for p in parts[1:]:
        indptrs.append(p.indptr[1:] + offset)
        offset += p.indptr[-1]
    stats: dict[str, int] = {}
    for p in parts:
        for k, v in p.stats.items():
            stats[k] = stats.get(k, 0) + v
    return EncodedSplit(
        np.concatenate(indptrs),
        np.concatenate([p.indices for p in parts]),
        np.concatenate([p.y for p in parts]),
        np.concatenate([p.diff for p in parts]),
        stats,
    )


def encode_split(split: str, data_dir: Path = DATA_DIR, vocab: Vocab | None = None,
                 label_index: dict[str, int] | None = None, chunksize: int = 200_000,
                 nrows: int | None = None) -> EncodedSplit:
    """Stream one patient CSV in chunks so the full 670 MB train file never sits in memory as lists."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data_dir = Path(data_dir)
    if vocab is None:
        vocab = build_vocab(data_dir / "release_evidences.json")
    if label_index is None:
        label_index = build_label_index(data_dir / "release_conditions.json")
    reader = pd.read_csv(data_dir / f"release_{split}_patients", chunksize=chunksize,
                         nrows=nrows, dtype=PATIENT_DTYPES)
    parts = []
    for chunk in reader:
        chunk["EVIDENCES"] = _parse_str_list(chunk["EVIDENCES"])
        chunk["DIFFERENTIAL_DIAGNOSIS"] = _parse_differential(chunk["DIFFERENTIAL_DIAGNOSIS"])
        parts.append(encode_frame(chunk, vocab, label_index))
    return _concat(parts)


def save_vocab(vocab: Vocab, label_index: dict[str, int], out_dir: Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = sorted(label_index, key=label_index.get)
    (out / "vocab.json").write_text(json.dumps({
        "tokens": list(vocab.tokens),
        "demographic_ids": list(vocab.demographic_ids),
        "labels": labels,
    }, indent=1), encoding="utf-8")


def save_split(enc: EncodedSplit, split: str, out_dir: Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{split}.npz", indptr=enc.indptr, indices=enc.indices,
                        y=enc.y, diff=enc.diff)
    (out / f"{split}_stats.json").write_text(json.dumps(enc.stats, indent=2), encoding="utf-8")


def load_split(split: str, out_dir: Path) -> EncodedSplit:
    out = Path(out_dir)
    stats = json.loads((out / f"{split}_stats.json").read_text(encoding="utf-8"))
    with np.load(out / f"{split}.npz") as z:
        return EncodedSplit(z["indptr"], z["indices"], z["y"], z["diff"], stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Encode DDXPlus patients as sparse feature rows.")
    parser.add_argument("--split", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--nrows", type=int, default=None, help="preview only; writes to features_preview/")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args(argv)

    out = args.out or (args.data_dir / ("features_preview" if args.nrows else "features"))
    vocab = build_vocab(args.data_dir / "release_evidences.json")
    labels = build_label_index(args.data_dir / "release_conditions.json")
    save_vocab(vocab, labels, out)
    for split in args.split:
        enc = encode_split(split, args.data_dir, vocab, labels, args.chunksize, args.nrows)
        save_split(enc, split, out)
        print(f"{split}: {enc.n_rows} rows, {len(enc.indices)} tokens, stats={enc.stats}")


if __name__ == "__main__":
    main()
