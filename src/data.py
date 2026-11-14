"""Load the DDXPlus dataset into pandas DataFrames.

Files expected in ``data/``:
    release_train_patients, release_validate_patients, release_test_patients  (CSV, no extension)
    release_conditions.json, release_evidences.json

Example:
    from src.data import load_all
    d = load_all()                 # dict of DataFrames
    d["train"].head()
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS = ("train", "validate", "test")


def _parse_str_list(s: pd.Series) -> pd.Series:
    """Parse "['E_1', 'E_2_@_V_3']" into a Python list of strings (much faster than ast)."""
    inner = s.str.slice(2, -2)
    return inner.str.split("', '").where(inner != "", other=pd.Series([[]] * len(s), index=s.index))


def _parse_differential(s: pd.Series) -> pd.Series:
    """Parse "[['GERD', 0.17], ...]" into a list of (pathology, probability) tuples."""
    return s.map(lambda x: [tuple(pair) for pair in json.loads(x.replace("'", '"'))])


def load_patients(split: str, data_dir: Path = DATA_DIR, parse: bool = True,
                  nrows: int | None = None) -> pd.DataFrame:
    """Load one patient split ("train", "validate" or "test").

    With ``parse=True`` the EVIDENCES column becomes a list of evidence strings and
    DIFFERENTIAL_DIAGNOSIS a list of (pathology, probability) tuples.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    df = pd.read_csv(
        Path(data_dir) / f"release_{split}_patients",
        nrows=nrows,
        dtype={"AGE": "int16", "SEX": "category", "PATHOLOGY": "category",
               "INITIAL_EVIDENCE": "category"},
    )
    if parse:
        df["EVIDENCES"] = _parse_str_list(df["EVIDENCES"])
        df["DIFFERENTIAL_DIAGNOSIS"] = _parse_differential(df["DIFFERENTIAL_DIAGNOSIS"])
    return df


def load_conditions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per pathology (49), with its symptom and antecedent evidence codes as lists."""
    raw = json.loads((Path(data_dir) / "release_conditions.json").read_text(encoding="utf-8"))
    rows = [
        {
            "condition_name": v["condition_name"],
            "cond_name_eng": v["cond-name-eng"],
            "cond_name_fr": v["cond-name-fr"],
            "icd10_id": v["icd10-id"],
            "severity": v["severity"],
            "symptoms": list(v["symptoms"]),
            "antecedents": list(v["antecedents"]),
        }
        for v in raw.values()
    ]
    return pd.DataFrame(rows).set_index("condition_name")


def load_evidences(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per evidence/question (223). data_type: B=binary, C=categorical, M=multi-choice."""
    raw = json.loads((Path(data_dir) / "release_evidences.json").read_text(encoding="utf-8"))
    rows = [
        {
            "name": v["name"],
            "code_question": v["code_question"],
            "question_en": v["question_en"],
            "question_fr": v["question_fr"],
            "is_antecedent": v["is_antecedent"],
            "data_type": v["data_type"],
            "default_value": v["default_value"],
            "possible_values": v["possible-values"],
            "value_meaning_en": {k: m["en"] for k, m in v["value_meaning"].items()},
        }
        for v in raw.values()
    ]
    return pd.DataFrame(rows).set_index("name")


def load_all(data_dir: Path = DATA_DIR, parse: bool = True,
             nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """Load every file: keys "train", "validate", "test", "conditions", "evidences"."""
    out = {split: load_patients(split, data_dir, parse=parse, nrows=nrows) for split in SPLITS}
    out["conditions"] = load_conditions(data_dir)
    out["evidences"] = load_evidences(data_dir)
    return out
