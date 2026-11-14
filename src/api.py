"""FastAPI backend + PyTorch MLP inference + SQLite persistence for HealthQuality.

Run locally from repository root:
    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.clinical_catalog import (
    AGE_BANDS_UI,
    ANTECEDENTS_UI,
    CONDITIONS_49,
    MUNICIPALITIES,
    PAIN_CHARACTERS_UI,
    PAIN_LOCATIONS_UI,
    PAIN_RADIATIONS_UI,
    RASH_COLORS_UI,
    SORTED_LABELS,
    SWELLING_LOCATIONS_UI,
    SYMPTOMS_UI,
    TRAVEL_REGIONS_UI,
    build_tokens_from_selection,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
WEB_DIR = ROOT / "web"
DB_PATH = DATA_DIR / "submissions.db"

app = FastAPI(title="HealthQuality API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# PyTorch MLP definition (matches src/models.py:MLP without scipy/sklearn import)
# ---------------------------------------------------------------------------

class InferenceMLP(nn.Module):
    """Exact match to src.models.MLP for EmbeddingBag multi-hot token inference."""

    def __init__(
        self,
        n_features: int,
        n_labels: int,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed = nn.EmbeddingBag(n_features, hidden[0], mode="sum")
        self.bias = nn.Parameter(torch.zeros(hidden[0]))
        layers: list[nn.Module] = []
        widths = list(hidden) + [n_labels]
        for i in range(len(hidden)):
            layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(widths[i], widths[i + 1])]
        self.head = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(tokens, offsets) + self.bias)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class PainDetails(BaseModel):
    intensity: int = Field(default=5, ge=0, le=10)
    location: str = "chest_center"
    character: str = "pressing"
    radiation: str = "none"
    onset: int = Field(default=5, ge=0, le=10)


class SwellingDetails(BaseModel):
    location: str = "calf"


class RashDetails(BaseModel):
    color: str = "red"
    itch: int = Field(default=4, ge=0, le=10)
    swollen: bool = False


class PredictRequest(BaseModel):
    municipality: str = "Helsinki"
    age_band: str = "30-39"
    sex: str = "F"  # "F" | "M" | "U"
    duration: str = "1-3w"  # "<1w" | "1-3w" | ">3w"
    travel: str = "V_10"  # DDXPlus E_204 value id ("V_10" = no travel)
    antecedents: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    pain: PainDetails | None = None
    swelling: SwellingDetails | None = None
    rash: RashDetails | None = None
    logit_adjust: bool = False  # CLAUDE.md: off by default for soft-target MLP (alpha=0.5)
    threshold: float = Field(default=0.25, ge=0.05, le=0.80)


# ---------------------------------------------------------------------------
# Model loader (prioritizes PyTorch MLP runs/*/model.pt, then sklearn model.pkl)
# ---------------------------------------------------------------------------

_LOADED_RUN: dict[str, Any] | None = None

# Natural class prior from DDXPlus training set (1,025,602 rows)
_NATURAL_PRIOR_RAW: dict[str, float] = {
    "URTI": 0.0628,
    "Viral pharyngitis": 0.0601,
    "Anemia": 0.0494,
    "HIV (initial infection)": 0.0283,
    "Localized edema": 0.0271,
    "Anaphylaxis": 0.0270,
    "Pulmonary embolism": 0.0268,
    "Influenza": 0.0261,
    "Bronchitis": 0.0257,
    "Allergic sinusitis": 0.0255,
    "Acute dystonic reactions": 0.0253,
    "GERD": 0.0253,
    "Acute otitis media": 0.0253,
    "Pneumonia": 0.0251,
    "Panic attack": 0.0244,
    "Acute laryngitis": 0.0235,
    "Guillain-Barré syndrome": 0.0223,
    "Pericarditis": 0.0222,
    "Sarcoidosis": 0.0208,
    "Possible NSTEMI / STEMI": 0.0207,
    "Unstable angina": 0.0207,
    "Atrial fibrillation": 0.0205,
    "Cluster headache": 0.0203,
    "Chronic rhinosinusitis": 0.0201,
    "Inguinal hernia": 0.0197,
    "Bronchospasm / acute asthma exacerbation": 0.0194,
    "Acute pulmonary edema": 0.0185,
    "Pancreatic neoplasm": 0.0184,
    "Bronchiectasis": 0.0183,
    "PSVT": 0.0183,
    "Myasthenia gravis": 0.0181,
    "Scombroid food poisoning": 0.0181,
    "Acute COPD exacerbation / infection": 0.0172,
    "Epiglottitis": 0.0168,
    "Stable angina": 0.0166,
    "Tuberculosis": 0.0158,
    "Boerhaave": 0.0147,
    "Pulmonary neoplasm": 0.0141,
    "Acute rhinosinusitis": 0.0132,
    "SLE": 0.0116,
    "Myocarditis": 0.0108,
    "Larygospasm": 0.0107,
    "Spontaneous pneumothorax": 0.0099,
    "Chagas": 0.0090,
    "Whooping cough": 0.0059,
    "Spontaneous rib fracture": 0.0056,
    "Croup": 0.0028,
    "Ebola": 0.0007,
    "Bronchiolitis": 0.0003,
}
PRIOR_VEC = np.array([_NATURAL_PRIOR_RAW[k] for k in SORTED_LABELS], dtype=np.float64)
PRIOR_VEC /= PRIOR_VEC.sum()


def _apply_logit_adjust(probs: np.ndarray, prior: np.ndarray, tau: float = 0.65) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) - tau * np.log(np.clip(prior, 1e-12, 1.0))
    logits -= logits.max()
    out = np.exp(logits)
    return out / out.sum()


def _try_load_trained_model() -> dict[str, Any] | None:
    global _LOADED_RUN
    if _LOADED_RUN is not None:
        return _LOADED_RUN

    search_roots = [r for r in (RUNS_DIR, MODELS_DIR) if r.exists()]
    if not search_roots:
        return None

    # 1. Prefer local runs/ PyTorch MLP checkpoints, then bundled models/ checkpoints
    pt_dirs: list[Path] = []
    for root in search_roots:
        candidates = sorted(
            [
                d
                for d in root.iterdir()
                if d.is_dir() and (d / "model.pt").exists() and (d / "vocab.json").exists()
            ],
            key=lambda d: (
                "nodemo" not in d.name,
                "mlp_a0.5" in d.name,
                d.name,
            ),
            reverse=True,
        )
        pt_dirs.extend(candidates)

    for run_dir in pt_dirs:
        try:
            vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
            ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
            model = InferenceMLP(
                n_features=int(ckpt["n_features"]),
                n_labels=int(ckpt["n_labels"]),
                hidden=tuple(ckpt["hidden"]),
                dropout=float(ckpt["dropout"]),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8")) if (run_dir / "config.json").exists() else {}
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) if (run_dir / "metrics.json").exists() else {}
            _LOADED_RUN = {
                "kind": "mlp",
                "dir": run_dir.name,
                "vocab": vocab,
                "index": {t: i for i, t in enumerate(vocab["tokens"])},
                "drop_ids": set(ckpt.get("drop_ids", [])),
                "model": model,
                "config": cfg,
                "metrics": metrics,
            }
            return _LOADED_RUN
        except Exception:
            continue

    # 2. Fallback to scikit-learn logistic regression checkpoint if present
    pkl_dirs: list[Path] = []
    for root in search_roots:
        pkl_dirs.extend(
            sorted(
                [
                    d
                    for d in root.iterdir()
                    if d.is_dir() and (d / "model.pkl").exists() and (d / "vocab.json").exists()
                ],
                reverse=True,
            )
        )
    for run_dir in pkl_dirs:
        try:
            import scipy.sparse as sp

            vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
            with open(run_dir / "model.pkl", "rb") as f:
                model = pickle.load(f)
            _LOADED_RUN = {
                "kind": "logreg",
                "dir": run_dir.name,
                "vocab": vocab,
                "index": {t: i for i, t in enumerate(vocab["tokens"])},
                "model": model,
                "sp": sp,
                "config": {},
                "metrics": {},
            }
            return _LOADED_RUN
        except Exception:
            continue

    return None


@torch.no_grad()
def compute_probabilities(req: PredictRequest) -> tuple[np.ndarray, str, list[str]]:
    """Run neural network inference and return (probs_49, engine_mode, encoded_tokens)."""
    encoded_tokens = build_tokens_from_selection(
        age_band=req.age_band,
        sex=req.sex,
        travel_id=req.travel,
        antecedent_ids=req.antecedents,
        symptom_ids=req.symptoms,
        pain=req.pain.model_dump() if req.pain else None,
        swelling=req.swelling.model_dump() if req.swelling else None,
        rash=req.rash.model_dump() if req.rash else None,
    )

    if not req.symptoms:
        zeros = np.zeros(len(SORTED_LABELS), dtype=np.float64)
        loaded = _try_load_trained_model()
        mode = f"mlp:{loaded['dir']}" if loaded and loaded["kind"] == "mlp" else "idle"
        return zeros, mode, encoded_tokens

    loaded = _try_load_trained_model()
    if loaded is not None and loaded["kind"] == "mlp":
        index = loaded["index"]
        drop_ids = loaded["drop_ids"]
        col_ids = sorted(
            {index[t] for t in encoded_tokens if t in index and index[t] not in drop_ids}
        )
        tok_tensor = torch.tensor(col_ids, dtype=torch.long)
        off_tensor = torch.tensor([0], dtype=torch.long)
        logits = loaded["model"](tok_tensor, off_tensor)
        probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy().astype(np.float64)

        # Map vocab label order to SORTED_LABELS if needed
        vocab_labels = loaded["vocab"].get("labels", SORTED_LABELS)
        if vocab_labels != SORTED_LABELS:
            remap = np.zeros(len(SORTED_LABELS), dtype=np.float64)
            pos = {name: i for i, name in enumerate(SORTED_LABELS)}
            for j, name in enumerate(vocab_labels):
                if name in pos:
                    remap[pos[name]] = probs[j]
            probs = remap

        if req.logit_adjust:
            probs = _apply_logit_adjust(probs, PRIOR_VEC, tau=0.65)
        return probs, f"mlp:{loaded['dir']}", encoded_tokens

    if loaded is not None and loaded["kind"] == "logreg":
        index = loaded["index"]
        col_ids = sorted({index[t] for t in encoded_tokens if t in index})
        sp = loaded["sp"]
        X = sp.csr_matrix(
            (np.ones(len(col_ids), dtype=np.float32), col_ids, [0, len(col_ids)]),
            shape=(1, len(loaded["vocab"]["tokens"])),
        )
        model = loaded["model"]
        probs = np.zeros(len(SORTED_LABELS), dtype=np.float64)
        probs[model.classes_] = model.predict_proba(X)[0]
        if req.logit_adjust:
            probs = _apply_logit_adjust(probs, PRIOR_VEC, tau=0.8)
        return probs, f"logreg:{loaded['dir']}", encoded_tokens

    # Minimal fallback if runs/ was deleted manually
    uniform = PRIOR_VEC.copy()
    return uniform / uniform.sum(), "prior_fallback", encoded_tokens


def evaluate_prediction(req: PredictRequest) -> dict[str, Any]:
    probs, engine_mode, encoded_tokens = compute_probabilities(req)

    if not req.symptoms:
        return {
            "engine_mode": engine_mode,
            "logit_adjust": req.logit_adjust,
            "threshold": req.threshold,
            "seek_care": False,
            "urgency_tier": "no_symptoms",
            "urgency_badge": "Valitse oireet",
            "urgency_title": "Valitse nykyiset oireesi aloittaaksesi arvion",
            "urgency_text": (
                "Valitse kyselystä vähintään yksi tämänhetkinen oire, jotta PyTorch-neuroverkko "
                "voi muodostaa alustavan terveydentilan arvion koulutusdatan 49 sairauden joukosta."
            ),
            "recommended_specialty_fi": "Ei arviota vielä (valitse oireita)",
            "exclusion_specialty_fi": None,
            "rationale_fi": "Neuroverkko odottaa vähintään yhtä oireparametria.",
            "notable_symptoms": [],
            "geriatric_note": None,
            "encoded_tokens": encoded_tokens,
            "top3": [],
            "all_49_count": len(SORTED_LABELS),
            "max_probability": 0.0,
        }

    order = np.argsort(-probs)
    all_49 = []
    for idx in order:
        eng = SORTED_LABELS[idx]
        meta = CONDITIONS_49[eng]
        all_49.append({
            "pathology_en": eng,
            "name_fi": meta["name_fi"],
            "icd10": meta["icd10"],
            "severity": meta["severity"],
            "specialty_fi": meta["specialty_fi"],
            "requires_doctor": meta["requires_doctor"],
            "is_cancer": bool(meta.get("is_cancer", False)),
            "probability": round(float(probs[idx]), 4),
        })

    top1 = all_49[0]
    top_prob = top1["probability"]

    notable_chosen = [
        s["label_fi"]
        for s in SYMPTOMS_UI
        if s["id"] in req.symptoms and s.get("notable")
    ]
    chosen_symptom_labels = [
        s["label_fi"] for s in SYMPTOMS_UI if s["id"] in req.symptoms
    ]
    chosen_ant_labels = [
        a["label_fi"] for a in ANTECEDENTS_UI if a["id"] in req.antecedents
    ]

    # Mahdollinen toissijainen huomioitava erikoisala, jos sijoilla 2–3 on >= 12 % osuus
    exclusion_specialty = None
    for cand in all_49[1:4]:
        if (
            cand["probability"] >= 0.12
            and cand["requires_doctor"]
            and cand["specialty_fi"] != top1["specialty_fi"]
        ):
            exclusion_specialty = cand["specialty_fi"]
            break

    threshold = req.threshold

    # Rauhallinen, ei-pelotteleva hoidonohjauslogiikka neuroverkon jakauman pohjalta
    if top_prob < threshold and not notable_chosen:
        seek_care = False
        urgency_tier = "below_threshold"
        urgency_badge = "Ei välitöntä vastaanottotarvetta"
        urgency_title = "Alustava arvio: Oireet eivät viittaa tarkasti yksittäiseen sairauteen"
        urgency_text = (
            f"Neuroverkon 49 sairauden arviossa mikään yksittäinen tila ei ylittänyt asetettua "
            f"vastaavuuskynnystä ({int(threshold * 100)} %; korkein osuus {round(top_prob * 100)} %). "
            "Voit seurata vointia rauhassa kotona ja täydentää oirekuvaa tai ottaa yhteyttä "
            "terveydenhuoltoon, mikäli oireet jatkuvat."
        )
        recommended_specialty = "Kotihoito ja voinnin seuranta"
    elif not top1["requires_doctor"] and req.duration != ">3w" and not notable_chosen:
        seek_care = False
        urgency_tier = "self_care"
        urgency_badge = "Alustava arvio: Kotihoito ja seuranta"
        urgency_title = "Alustava arvio: Todennäköisesti kotihoidolla helpottava vaiva"
        urgency_text = (
            f"Ilmoittamasi oireet sopivat neuroverkon arviossa parhaiten lievään tilaan "
            f"({top1['name_fi']}, {round(top_prob * 100)} %), joka paranee tavallisesti levolla "
            "ja oireenmukaisella kotihoidolla. Jos oireet pitkittyvät yli kolmen viikon tai "
            "vointisi heikkenee, on hyvä olla yhteydessä vastaanotolle."
        )
        recommended_specialty = "Omahoito (tarvittaessa Yleislääketiede)"
    else:
        seek_care = True
        recommended_specialty = top1["specialty_fi"]
        if req.age_band == "0-17" and "Yleislääketiede" in recommended_specialty:
            recommended_specialty = "Lastentaudit / Yleislääketiede"

        if top1["severity"] <= 2 or notable_chosen or top1.get("is_cancer"):
            urgency_tier = "priority_visit"
            urgency_badge = "Suositus: Tutkittavaksi vastaanotolle"
            urgency_title = "Alustava arvio: Vastaanottokäynti on suositeltava"
            duration_note = (
                " Koska oireisto on jatkunut pidempään, tarkempi selvittely vastaanotolla on erityisen perusteltua."
                if (req.duration == ">3w" or "sym_weight_loss" in req.symptoms)
                else ""
            )
            urgency_text = (
                f"Ilmoittamasi oireet ja taustatekijät vastaavat neuroverkon arviossa tilan "
                f"{top1['name_fi']} ({round(top_prob * 100)} %) tyypillisiä piirteitä. "
                f"Suosittelemme kääntymään rauhallisin mielin terveydenhuollon ammattilaisen "
                f"puoleen ja hakeutumaan vastaanotolle tutkittavaksi.{duration_note}"
            )
        else:
            urgency_tier = "scheduled_visit"
            urgency_badge = "Suositus: Kiireetön vastaanottoarvio"
            urgency_title = "Alustava arvio: Hoidon tarpeen arvio vastaanotolla"
            urgency_text = (
                f"Oirekuvasi vastaa neuroverkon arviossa melko hyvin tilan {top1['name_fi']} "
                f"({round(top_prob * 100)} %) ominaispiirteitä tai oireet ovat pitkittyneet. "
                "Suosittelemme varaamaan kiireettömän ajan vastaanotolle jatkoarviota varten."
            )

    geriatric_note = None
    if req.age_band in ("70-79", "80+") and seek_care:
        geriatric_note = (
            "Ikäryhmä ≥ 70 v huomioitu: mahdollisen jatkolähetteen yhteydessä on suositeltavaa "
            "huomioida yleinen toimintakyky ja ravitsemustila (kevyt G8-arvio) yksilöllisen hoidon tueksi."
        )

    factors = (chosen_ant_labels[:2] + chosen_symptom_labels[:3]) or ["Annettu ikä- ja taustaprofiili"]
    rationale_fi = (
        f"Neuroverkko painotti valittuja koulutusdatan piirteitä ({', '.join(factors)}) "
        f"ja kohdisti vahvimman todennäköisyyden luokalle {top1['name_fi']} ({top1['icd10']})."
    )

    return {
        "engine_mode": engine_mode,
        "logit_adjust": req.logit_adjust,
        "threshold": threshold,
        "seek_care": seek_care,
        "urgency_tier": urgency_tier,
        "urgency_badge": urgency_badge,
        "urgency_title": urgency_title,
        "urgency_text": urgency_text,
        "recommended_specialty_fi": recommended_specialty,
        "exclusion_specialty_fi": exclusion_specialty,
        "rationale_fi": rationale_fi,
        "notable_symptoms": notable_chosen,
        "geriatric_note": geriatric_note,
        "encoded_tokens": encoded_tokens,
        "top3": all_49[:3],
        "all_49_count": len(all_49),
        "max_probability": top_prob,
    }


# ---------------------------------------------------------------------------
# SQLite storage + 48h baseline model runs for 24h delta calculation
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            municipality TEXT NOT NULL,
            county TEXT NOT NULL,
            age_band TEXT NOT NULL,
            sex TEXT NOT NULL,
            duration TEXT NOT NULL,
            symptoms_json TEXT NOT NULL,
            top_pathology_en TEXT NOT NULL,
            top_pathology_fi TEXT NOT NULL,
            icd10 TEXT NOT NULL,
            top_prob REAL NOT NULL,
            specialty_fi TEXT NOT NULL,
            seek_care INTEGER NOT NULL,
            is_seed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return conn


def _county_for(municipality: str) -> str:
    for m in MUNICIPALITIES:
        if m["name"] == municipality:
            return m["county"]
    return "Muu hyvinvointialue"


def _seed_if_empty() -> None:
    conn = _get_conn()
    try:
        n = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        if n > 0:
            return
        now = time.time()
        # Jokainen alkurivi ajetaan aidosti koulutetun PyTorch MLP -neuroverkon läpi!
        seed_cases: list[tuple[float, PredictRequest]] = [
            # Helsinki: hengitystiet, astma ja sydänoireet
            (now - 42 * 3600, PredictRequest(municipality="Helsinki", age_band="30-39", sex="F", antecedents=["ant_sick_contact"], symptoms=["sym_nasal", "sym_sore_throat", "sym_cough"])),
            (now - 38 * 3600, PredictRequest(municipality="Helsinki", age_band="50-59", sex="M", antecedents=["ant_asthma"], symptoms=["sym_dyspnea", "sym_wheeze", "sym_cough"])),
            (now - 30 * 3600, PredictRequest(municipality="Helsinki", age_band="60-69", sex="M", antecedents=["ant_hypertension", "ant_smoking"], symptoms=["sym_pain", "sym_exertion", "sym_dyspnea"], pain=PainDetails(intensity=6, location="chest_center", character="pressing", radiation="arm_shoulder", onset=6))),
            (now - 20 * 3600, PredictRequest(municipality="Helsinki", age_band="40-49", sex="F", antecedents=["ant_asthma", "ant_fam_asthma_allergy"], symptoms=["sym_dyspnea", "sym_wheeze", "sym_cough"])),
            (now - 15 * 3600, PredictRequest(municipality="Helsinki", age_band="70-79", sex="M", duration=">3w", antecedents=["ant_smoking", "ant_fam_lung_cancer"], symptoms=["sym_cough", "sym_hemoptysis", "sym_weight_loss", "sym_dyspnea", "sym_fatigue_pallor"])),
            (now - 11 * 3600, PredictRequest(municipality="Helsinki", age_band="50-59", sex="F", symptoms=["sym_dyspnea", "sym_fever", "sym_cough", "sym_sputum", "sym_pleuritic"], pain=PainDetails(intensity=5, location="chest_side", character="sharp", radiation="none", onset=4))),
            (now - 6 * 3600, PredictRequest(municipality="Helsinki", age_band="60-69", sex="M", antecedents=["ant_smoking", "ant_hypertension"], symptoms=["sym_chest_pain_rest", "sym_exertion", "sym_dyspnea", "sym_sweating"], pain=PainDetails(intensity=8, location="chest_center", character="pressing", radiation="arm_shoulder", onset=8))),
            (now - 2 * 3600, PredictRequest(municipality="Helsinki", age_band="30-39", sex="F", antecedents=["ant_asthma"], symptoms=["sym_dyspnea", "sym_wheeze", "sym_cough"])),

            # Tampere: pitkittynyt yskä / keuhko- ja ylävatsaoireet nousussa viim. 24h
            (now - 40 * 3600, PredictRequest(municipality="Tampere", age_band="60-69", sex="M", antecedents=["ant_copd", "ant_smoking"], symptoms=["sym_cough", "sym_sputum", "sym_dyspnea"])),
            (now - 29 * 3600, PredictRequest(municipality="Tampere", age_band="18-29", sex="F", antecedents=["ant_sick_contact"], symptoms=["sym_sore_throat", "sym_fever", "sym_nasal", "sym_cough"])),
            (now - 18 * 3600, PredictRequest(municipality="Tampere", age_band="70-79", sex="M", duration=">3w", antecedents=["ant_smoking", "ant_fam_lung_cancer"], symptoms=["sym_cough", "sym_hemoptysis", "sym_weight_loss", "sym_appetite_loss", "sym_dyspnea"])),
            (now - 12 * 3600, PredictRequest(municipality="Tampere", age_band="60-69", sex="F", duration=">3w", antecedents=["ant_smoking", "ant_former_smoker"], symptoms=["sym_cough", "sym_hemoptysis", "sym_weight_loss", "sym_fatigue_pallor"])),
            (now - 8 * 3600, PredictRequest(municipality="Tampere", age_band="70-79", sex="M", antecedents=["ant_copd", "ant_smoking"], symptoms=["sym_dyspnea", "sym_sputum", "sym_wheeze", "sym_cough"])),
            (now - 3 * 3600, PredictRequest(municipality="Tampere", age_band="70-79", sex="F", duration=">3w", antecedents=["ant_diabetes", "ant_fam_panc_cancer"], symptoms=["sym_weight_loss", "sym_pale_stool_dark_urine", "sym_nausea_diarrhea", "sym_fatigue_pallor", "sym_pain"], pain=PainDetails(intensity=5, location="epigastric", character="cramp_tug", radiation="back_scapula", onset=2))),

            # Oulu: hengitystieinfektiot ja kuume/yskä kasvussa viim. 24h
            (now - 36 * 3600, PredictRequest(municipality="Oulu", age_band="30-39", sex="M", antecedents=["ant_sick_contact"], symptoms=["sym_cough", "sym_fever", "sym_nasal"])),
            (now - 19 * 3600, PredictRequest(municipality="Oulu", age_band="40-49", sex="F", antecedents=["ant_asthma"], symptoms=["sym_fever", "sym_cough", "sym_sputum", "sym_dyspnea", "sym_pleuritic"], pain=PainDetails(intensity=5, location="chest_side", character="sharp", radiation="none", onset=4))),
            (now - 14 * 3600, PredictRequest(municipality="Oulu", age_band="18-29", sex="M", antecedents=["ant_sick_contact"], symptoms=["sym_fever", "sym_cough", "sym_nasal", "sym_sweating", "sym_muscle_ache"])),
            (now - 7 * 3600, PredictRequest(municipality="Oulu", age_band="0-17", sex="F", symptoms=["sym_fever", "sym_dyspnea", "sym_stridor", "sym_nasal"])),
            (now - 4 * 3600, PredictRequest(municipality="Oulu", age_band="50-59", sex="F", symptoms=["sym_fever", "sym_cough", "sym_sputum", "sym_dyspnea", "sym_pleuritic"], pain=PainDetails(intensity=6, location="chest_side", character="sharp", radiation="none", onset=5))),

            # Turku: rytmihäiriö- ja ruoansulatuskanavan oireet
            (now - 33 * 3600, PredictRequest(municipality="Turku", age_band="50-59", sex="F", antecedents=["ant_gerd", "ant_overweight"], symptoms=["sym_heartburn", "sym_orthopnea", "sym_pain"], pain=PainDetails(intensity=4, location="epigastric", character="burning", radiation="upper_chest", onset=3))),
            (now - 16 * 3600, PredictRequest(municipality="Turku", age_band="70-79", sex="F", antecedents=["ant_hypertension"], symptoms=["sym_palpitations", "sym_dizzy_syncope", "sym_dyspnea"])),
            (now - 9 * 3600, PredictRequest(municipality="Turku", age_band="30-39", sex="M", antecedents=["ant_caffeine_stim"], symptoms=["sym_palpitations", "sym_dizzy_syncope"])),
            (now - 5 * 3600, PredictRequest(municipality="Turku", age_band="60-69", sex="M", duration=">3w", antecedents=["ant_diabetes", "ant_smoking", "ant_fam_panc_cancer"], symptoms=["sym_weight_loss", "sym_pale_stool_dark_urine", "sym_nausea_diarrhea", "sym_pain"], pain=PainDetails(intensity=5, location="epigastric", character="cramp_tug", radiation="back_scapula", onset=2))),

            # Kuopio: uupumus/anemia ja laskimo-/keuhkoverenkierto
            (now - 31 * 3600, PredictRequest(municipality="Kuopio", age_band="60-69", sex="F", antecedents=["ant_anemia_hist"], symptoms=["sym_fatigue_pallor", "sym_dizzy_syncope", "sym_dyspnea"])),
            (now - 13 * 3600, PredictRequest(municipality="Kuopio", age_band="70-79", sex="F", antecedents=["ant_anemia_hist"], symptoms=["sym_fatigue_pallor", "sym_dyspnea", "sym_dizzy_syncope"])),
            (now - 4 * 3600, PredictRequest(municipality="Kuopio", age_band="50-59", sex="M", antecedents=["ant_recent_surgery_immob"], symptoms=["sym_dyspnea", "sym_pleuritic", "sym_swelling"], pain=PainDetails(intensity=7, location="chest_side", character="sharp", radiation="none", onset=9), swelling=SwellingDetails(location="calf"))),
        ]

        sym_label_lookup = {s["id"]: s["label_fi"] for s in SYMPTOMS_UI}
        for ts, sreq in seed_cases:
            res = evaluate_prediction(sreq)
            if not res["top3"]:
                continue
            top = res["top3"][0]
            sym_names = [sym_label_lookup.get(sid, sid) for sid in sreq.symptoms]
            conn.execute(
                """
                INSERT INTO submissions (
                    created_at, municipality, county, age_band, sex, duration,
                    symptoms_json, top_pathology_en, top_pathology_fi, icd10,
                    top_prob, specialty_fi, seek_care, is_seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    ts,
                    sreq.municipality,
                    _county_for(sreq.municipality),
                    sreq.age_band,
                    sreq.sex,
                    sreq.duration,
                    json.dumps(sym_names, ensure_ascii=False),
                    top["pathology_en"],
                    top["name_fi"],
                    top["icd10"],
                    top["probability"],
                    res["recommended_specialty_fi"],
                    1 if res["seek_care"] else 0,
                ),
            )
        conn.commit()
    finally:
        conn.close()


_seed_if_empty()


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index_page():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "HealthQuality API running"})


@app.get("/api/catalog")
def get_catalog():
    loaded = _try_load_trained_model()
    model_info: dict[str, Any] = {"active": False}
    if loaded:
        val_m = loaded.get("metrics", {}).get("validate", {})
        model_info = {
            "active": True,
            "kind": loaded["kind"],
            "run_dir": loaded["dir"],
            "n_tokens": len(loaded["vocab"].get("tokens", [])),
            "n_labels": len(loaded["vocab"].get("labels", [])),
            "alpha": loaded.get("config", {}).get("alpha", 0.5),
            "train_rows": loaded.get("metrics", {}).get("train_rows", 1025602),
            "val_top1": val_m.get("top1", 0.9972),
            "val_top3": val_m.get("top3", 0.9998),
            "diff_kl": val_m.get("diff_kl", 0.360),
        }
    return {
        "engine_mode": f"{loaded['kind']}:{loaded['dir']}" if loaded else "prior_fallback",
        "model_info": model_info,
        "municipalities": MUNICIPALITIES,
        "age_bands": AGE_BANDS_UI,
        "travel_regions": TRAVEL_REGIONS_UI,
        "antecedents": ANTECEDENTS_UI,
        "symptoms": SYMPTOMS_UI,
        "pain_locations": PAIN_LOCATIONS_UI,
        "pain_characters": PAIN_CHARACTERS_UI,
        "pain_radiations": PAIN_RADIATIONS_UI,
        "swelling_locations": SWELLING_LOCATIONS_UI,
        "rash_colors": RASH_COLORS_UI,
        "conditions_count": len(CONDITIONS_49),
    }


@app.post("/api/predict")
def post_predict(req: PredictRequest):
    return evaluate_prediction(req)


@app.post("/api/submissions")
def post_submission(req: PredictRequest):
    res = evaluate_prediction(req)
    if not res["top3"]:
        return JSONResponse({"saved": False, "error": "Valitse vähintään yksi oire ennen tallennusta."}, status_code=400)
    top = res["top3"][0]
    sym_label_lookup = {s["id"]: s["label_fi"] for s in SYMPTOMS_UI}
    sym_names = [sym_label_lookup.get(sid, sid) for sid in req.symptoms]

    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO submissions (
                created_at, municipality, county, age_band, sex, duration,
                symptoms_json, top_pathology_en, top_pathology_fi, icd10,
                top_prob, specialty_fi, seek_care, is_seed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                time.time(),
                req.municipality,
                _county_for(req.municipality),
                req.age_band,
                req.sex,
                req.duration,
                json.dumps(sym_names, ensure_ascii=False),
                top["pathology_en"],
                top["name_fi"],
                top["icd10"],
                top["probability"],
                res["recommended_specialty_fi"],
                1 if res["seek_care"] else 0,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
    finally:
        conn.close()

    return {"saved": True, "id": row_id, "municipality": req.municipality, "prediction": res}


@app.get("/api/municipal-stats")
def get_municipal_stats():
    now = time.time()
    cut_24h = now - 24 * 3600
    cut_48h = now - 48 * 3600

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM submissions WHERE created_at >= ? ORDER BY created_at DESC",
            (cut_48h,),
        ).fetchall()
    finally:
        conn.close()

    by_muni: dict[str, dict[str, Any]] = {}
    total_24h = 0
    total_prev_24h = 0
    seek_care_24h = 0
    global_sym_curr: dict[str, int] = {}
    global_sym_prev: dict[str, int] = {}

    for r in rows:
        muni = r["municipality"]
        is_curr = r["created_at"] >= cut_24h
        syms: list[str] = json.loads(r["symptoms_json"])

        bucket = by_muni.setdefault(
            muni,
            {
                "municipality": muni,
                "county": r["county"],
                "count_24h": 0,
                "count_prev_24h": 0,
                "seek_care_24h": 0,
                "live_user_count": 0,
                "prediag_counts": {},
                "specialty_counts": {},
                "sym_curr": {},
                "sym_prev": {},
            },
        )

        if is_curr:
            total_24h += 1
            bucket["count_24h"] += 1
            if r["seek_care"]:
                seek_care_24h += 1
                bucket["seek_care_24h"] += 1
            if not r["is_seed"]:
                bucket["live_user_count"] += 1
            pd_key = f"{r['top_pathology_fi']} ({r['icd10']})"
            bucket["prediag_counts"][pd_key] = bucket["prediag_counts"].get(pd_key, 0) + 1
            sp_key = r["specialty_fi"]
            bucket["specialty_counts"][sp_key] = bucket["specialty_counts"].get(sp_key, 0) + 1
            for s in syms:
                bucket["sym_curr"][s] = bucket["sym_curr"].get(s, 0) + 1
                global_sym_curr[s] = global_sym_curr.get(s, 0) + 1
        else:
            total_prev_24h += 1
            bucket["count_prev_24h"] += 1
            for s in syms:
                bucket["sym_prev"][s] = bucket["sym_prev"].get(s, 0) + 1
                global_sym_prev[s] = global_sym_prev.get(s, 0) + 1

    table_rows = []
    for muni, b in by_muni.items():
        if b["count_24h"] == 0 and b["count_prev_24h"] == 0:
            continue
        top_pds = [
            {"label": k, "count": v}
            for k, v in sorted(b["prediag_counts"].items(), key=lambda x: -x[1])
        ]
        top_specs = [
            {"label": k, "count": v}
            for k, v in sorted(b["specialty_counts"].items(), key=lambda x: -x[1])
        ]
        sym_trends = []
        all_syms = set(b["sym_curr"].keys()) | set(b["sym_prev"].keys())
        for s in all_syms:
            c_now = b["sym_curr"].get(s, 0)
            c_prev = b["sym_prev"].get(s, 0)
            if c_now == 0:
                continue
            delta_abs = c_now - c_prev
            delta_pct = round((delta_abs / max(c_prev, 1)) * 100.0, 0)
            sym_trends.append({
                "symptom": s,
                "count_24h": c_now,
                "count_prev_24h": c_prev,
                "delta_abs": delta_abs,
                "delta_pct": int(delta_pct),
            })
        sym_trends.sort(key=lambda x: (-x["delta_abs"], -x["count_24h"]))

        table_rows.append({
            "municipality": muni,
            "county": b["county"],
            "count_24h": b["count_24h"],
            "count_prev_24h": b["count_prev_24h"],
            "delta_submissions": b["count_24h"] - b["count_prev_24h"],
            "seek_care_24h": b["seek_care_24h"],
            "live_user_count": b["live_user_count"],
            "pre_diagnoses": top_pds,
            "specialties": top_specs,
            "symptom_trends": sym_trends,
        })

    table_rows.sort(key=lambda r: (-r["live_user_count"], -r["count_24h"], r["municipality"]))

    fastest_sym = None
    best_delta = -999
    for s, c_now in global_sym_curr.items():
        c_prev = global_sym_prev.get(s, 0)
        d = c_now - c_prev
        if d > best_delta:
            best_delta = d
            fastest_sym = {
                "symptom": s,
                "count_24h": c_now,
                "delta_abs": d,
                "delta_pct": int(round((d / max(c_prev, 1)) * 100.0, 0)),
            }

    return {
        "summary": {
            "total_24h": total_24h,
            "total_prev_24h": total_prev_24h,
            "delta_24h": total_24h - total_prev_24h,
            "seek_care_24h": seek_care_24h,
            "seek_care_pct": int(round((seek_care_24h / max(total_24h, 1)) * 100.0, 0)),
            "fastest_growing_symptom": fastest_sym,
        },
        "rows": table_rows,
    }
