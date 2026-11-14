"""FastAPI backend for HealthQuality: preliminary automated evaluation & municipal dashboard.

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
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from torch import nn

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

SUPPORTED_LANGS = ("fi", "sv", "en")

app = FastAPI(title="HealthQuality API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def _norm_lang(lang: str | None) -> str:
    if lang and lang.lower() in SUPPORTED_LANGS:
        return lang.lower()
    return "fi"


# ---------------------------------------------------------------------------
# Internal model architecture
# ---------------------------------------------------------------------------

class InferenceMLP(nn.Module):
    """Matches src.models.MLP for multi-hot sparse token inference."""

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
    lang: str = "fi"  # "fi" | "sv" | "en"
    municipality: str = "Helsinki"
    age_band: str = "30-39"
    sex: str = "F"  # "F" | "M" | "U"
    duration: str = "1-3w"  # "<1w" | "1-3w" | ">3w"
    travel: str = "V_10"
    antecedents: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    pain: PainDetails | None = None
    swelling: SwellingDetails | None = None
    rash: RashDetails | None = None
    logit_adjust: bool = False
    threshold: float = Field(default=0.25, ge=0.05, le=0.85)


# ---------------------------------------------------------------------------
# Model loader (local runs/ first, then bundled models/)
# ---------------------------------------------------------------------------

_LOADED_RUN: dict[str, Any] | None = None

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
            _LOADED_RUN = {
                "kind": "mlp",
                "dir": run_dir.name,
                "vocab": vocab,
                "index": {t: i for i, t in enumerate(vocab["tokens"])},
                "drop_ids": set(ckpt.get("drop_ids", [])),
                "model": model,
            }
            return _LOADED_RUN
        except Exception:
            continue

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
            }
            return _LOADED_RUN
        except Exception:
            continue

    return None


@torch.no_grad()
def compute_probabilities(req: PredictRequest) -> np.ndarray:
    """Compute probability distribution across the 49 clinical classes."""
    if not req.symptoms:
        return np.zeros(len(SORTED_LABELS), dtype=np.float64)

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

    loaded = _try_load_trained_model()
    if loaded is not None and loaded["kind"] == "mlp":
        index = loaded["index"]
        drop = loaded["drop_ids"]
        col_ids = sorted({index[t] for t in encoded_tokens if t in index and index[t] not in drop})
        tok_tensor = torch.tensor(col_ids, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        logits = loaded["model"](tok_tensor, offsets)[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        labels = loaded["vocab"]["labels"]
        if labels != SORTED_LABELS:
            remap = {lbl: i for i, lbl in enumerate(labels)}
            probs = np.array([probs[remap[k]] for k in SORTED_LABELS], dtype=np.float64)
        if req.logit_adjust:
            probs = _apply_logit_adjust(probs, PRIOR_VEC, tau=0.65)
        return probs

    if loaded is not None and loaded["kind"] == "logreg":
        index = loaded["index"]
        sp = loaded["sp"]
        col_ids = [index[t] for t in encoded_tokens if t in index]
        X = sp.csr_matrix(
            (np.ones(len(col_ids), dtype=np.float32), ([0] * len(col_ids), col_ids)),
            shape=(1, len(index)),
        )
        probs = loaded["model"].predict_proba(X)[0].astype(np.float64)
        if req.logit_adjust:
            probs = _apply_logit_adjust(probs, PRIOR_VEC, tau=0.65)
        return probs

    probs = PRIOR_VEC.copy()
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# Multilingual clinical guidance builder (FI / SV / EN)
# ---------------------------------------------------------------------------

_COPY: dict[str, dict[str, str]] = {
    "fi": {
        "no_sym_badge": "Vointi hyvä • Ei valittuja oireita",
        "no_sym_title": "Hoidon automaattinen ensiarvio: Ei hoidon tarvetta tällä hetkellä",
        "no_sym_text_short": "Et ole valinnut nykyisiä oireita, joten alustavan arvion perusteella välitöntä hoidon tarvetta ei ole.",
        "no_sym_text_long": "Et ole valinnut nykyisiä oireita. Jos sinulla kuitenkin on jo viikkoja jatkunut vaiva, valitse oire listalta tai hakeudu tarkastukseen, mikäli oireilu jatkuu vielä.",
        "no_sym_spec": "Ei hoidon tarvetta (voinnin normaali seuranta)",
        "no_sym_rat": "Valitse yksi tai useampi oire saadaksesi oirekuvaan perustuvan hoidon automaattisen ensiarvion.",

        "below_badge": "Ei välitöntä vastaanottotarvetta",
        "below_badge_long": "Matala vastaavuus • Huomioi pitkittynyt kesto",
        "below_title": "Hoidon automaattinen ensiarvio: Oirekuva jää vastaavuuskynnyksen alle",
        "below_text": "Mikään yksittäinen arvioitava tila ei ylittänyt asetettua vastaavuuskynnystä ({thr} %; korkein vastaavuus {top} % tilalle {cond}).",
        "below_tail_lt1w": " Oireilu on kestänyt alle viikon, joten voit seurata vointia rauhassa kotona ja ottaa yhteyttä terveydenhuoltoon, jos oireet voimistuvat.",
        "below_tail_1_3w": " Koska oireilu on kuitenkin kestänyt jo 1–3 viikkoa, suositellaan menemään tarkastukseen vastaanotolle, mikäli oireet jatkuvat vielä.",
        "below_tail_gt3w": " Koska oireilu on jatkunut jo yli 3 viikkoa, suositellaan hyvälläkin ensiarviolla menemään tarkastukseen vastaanotolle, mikäli oireet jatkuvat vielä.",
        "below_spec": "Kotihoito ja voinnin seuranta",
        "below_spec_long": "Voinnin seuranta (tarvittaessa Yleislääketiede)",

        "self_badge": "Hoidon automaattinen ensiarvio: Kotihoito ja seuranta",
        "self_title": "Hoidon automaattinen ensiarvio: Todennäköisesti kotihoidolla helpottava vaiva",
        "self_text_lt1w": "Ilmoittamasi oireet sopivat parhaiten lievään tilaan ({cond}, {top} %), joka helpottaa tavallisesti levolla ja oireenmukaisella kotihoidolla muutamassa päivässä.",
        "self_text_1_3w": "Ilmoittamasi oireet sopivat ensisijaisesti lievään tilaan ({cond}, {top} %). Koska oireilu on kestänyt jo 1–3 viikkoa, suositellaan menemään tarkastukseen vastaanotolle, mikäli oireet jatkuvat vielä.",
        "self_spec": "Omahoito (tarvittaessa Yleislääketiede)",

        "prolonged_self_badge": "Suositus: Tarkastus pitkittyneen oireilun vuoksi",
        "prolonged_self_title": "Hoidon automaattinen ensiarvio: Pitkittynyt oireilu (> 3 viikkoa)",
        "prolonged_self_text": "Oirekuva vastaa ensisijaisesti tilaa {cond} ({top} %), joka on tyypillisesti lievä. Koska oireilu on kuitenkin jatkunut yli kolme viikkoa, suositellaan menemään tarkastukseen vastaanotolle, mikäli oireet jatkuvat vielä.",

        "sched_badge": "Suositus: Kiireetön vastaanottoarvio",
        "sched_title": "Hoidon automaattinen ensiarvio: Tarkastuskäynti vastaanotolla on suositeltava",
        "sched_text_lt1w": "Ilmoittamasi oireet vastaavat melko hyvin tilan {cond} ({top} %) ominaispiirteitä. Suosittelemme varaamaan kiireettömän ajan vastaanotolle jatkoarviota varten, mikäli oireet eivät ala helpottaa.",
        "sched_text_1_3w": "Ilmoittamasi oireet vastaavat tilan {cond} ({top} %) ominaispiirteitä ja ovat kestäneet 1–3 viikkoa. Suosittelemme varaamaan ajan vastaanotolle tarkastukseen.",
        "sched_text_gt3w": "Ilmoittamasi oireet vastaavat tilan {cond} ({top} %) ominaispiirteitä ja ovat jatkuneet yli 3 viikkoa. Pitkittyneen keston vuoksi suositellaan hakeutumaan vastaanotolle tutkittavaksi.",

        "prio_badge": "Suositus: Tutkittavaksi vastaanotolle",
        "prio_title": "Hoidon automaattinen ensiarvio: Vastaanottokäynti on suositeltava",
        "prio_text_lt1w": "Ilmoittamasi oireet ja esitiedot vastaavat tilan {cond} ({top} %) tyypillisiä piirteitä. Suosittelemme kääntymään rauhallisin mielin terveydenhuollon ammattilaisen puoleen tarkempaa arviota varten.",
        "prio_text_1_3w": "Ilmoittamasi oireet ja esitiedot vastaavat tilan {cond} ({top} %) tyypillisiä piirteitä ja oireisto on jatkunut 1–3 viikkoa. Suosittelemme varaamaan vastaanottoajan tutkittavaksi lähiaikoina.",
        "prio_text_gt3w": "Ilmoittamasi oireet ja esitiedot vastaavat tilan {cond} ({top} %) tyypillisiä piirteitä ja oireisto on jatkunut yli 3 viikkoa. Pitkittyneen oireilun vuoksi tarkempi selvittely vastaanotolla on erityisen suositeltavaa.",

        "dur_note_1_3w_calm": "Kestohuomio (1–3 viikkoa): Vaikka ensiarvio on rauhallinen, oireilu on kestänyt jo yli viikon. Mikäli oireet jatkuvat vielä, suositellaan menemään tarkastukseen.",
        "dur_note_gt3w_calm": "Kestohuomio (> 3 viikkoa): Pitkittyneessä yli kolmen viikon oireilussa suositellaan hyvälläkin arviolla menemään tarkastukseen vastaanotolle, mikäli vaiva jatkuu vielä.",
        "dur_note_gt3w_visit": "Pitkittynyt kesto (> 3 viikkoa): Yli kolme viikkoa kestäneet oireet on suositeltavaa selvittää vastaanotolla ilman turhaa viivettä.",
        "reassurance": "Rauhallinen jatko-ohje: Hoidon automaattinen ensiarvio ei ole lääketieteellinen diagnoosi, vaan apuväline hoidon tarpeen arviointiin. Ota tämä yhteenveto mukaan vastaanotolle.",
        "geriatric": "Ikäryhmä ≥ 70 v huomioitu: mahdollisen jatkolähetteen yhteydessä suositellaan yleisen toimintakyvyn ja ravitsemustilan kevyttä kartoitusta (G8-arvio) yksilöllisen hoidon tueksi.",
        "pediatric_spec": "Lastentaudit / Yleislääketiede",
        "general_spec": "Yleislääketiede",
        "rationale": "Hoidon automaattinen ensiarvio huomioi ilmoitetut tekijät ({factors}) ja arvioi vahvimmaksi vastaavuudeksi tilan {cond} (ICD-10: {icd10}).",
        "default_factor": "Ikä- ja esitietoprofiili",
    },
    "sv": {
        "no_sym_badge": "God hälsa • Inga valda symtom",
        "no_sym_title": "Automatiserad första vårdbedömning: Inget aktuellt vårdbehov",
        "no_sym_text_short": "Du har inte valt några pågående symtom, så bedömningen visar inget omedelbart vårdbehov.",
        "no_sym_text_long": "Du har inte valt några pågående symtom. Om du ändå har haft besvär i flera veckor, välj symtomet i listan eller uppsök vården för kontroll om besvären fortsätter.",
        "no_sym_spec": "Inget vårdbehov (normal egenuppföljning)",
        "no_sym_rat": "Välj ett eller flera symtom för att få en automatiserad första vårdbedömning.",

        "below_badge": "Inget omedelbart behov av mottagningsbesök",
        "below_badge_long": "Låg överensstämmelse • Beakta lång varaktighet",
        "below_title": "Automatiserad första vårdbedömning: Symtombilden ligger under tröskelvärdet",
        "below_text": "Inget enskilt tillstånd överskred det valda tröskelvärdet ({thr} %; högsta överensstämmelse {top} % för {cond}).",
        "below_tail_lt1w": " Symtomen har pågått i mindre än en vecka, så du kan lugnt följa måendet hemma och kontakta vården om symtomen tilltar.",
        "below_tail_1_3w": " Eftersom symtomen har pågått i 1–3 veckor rekommenderas ändå en kontroll på mottagning om besvären fortsätter.",
        "below_tail_gt3w": " Eftersom symtomen har pågått i över 3 veckor rekommenderas även vid en lugn bedömning att gå på kontroll om besvären fortsätter.",
        "below_spec": "Egenvård och uppföljning hemma",
        "below_spec_long": "Uppföljning (vid behov Allmänmedicin)",

        "self_badge": "Automatiserad första vårdbedömning: Egenvård och uppföljning",
        "self_title": "Automatiserad första vårdbedömning: Sannolikt lindrigt besvär som läker med egenvård",
        "self_text_lt1w": "Dina symtom stämmer bäst överens med ett lindrigt tillstånd ({cond}, {top} %), som vanligen lindras med vila och symtomatisk egenvård.",
        "self_text_1_3w": "Dina symtom pekar främst på ett lindrigt tillstånd ({cond}, {top} %). Eftersom besvären har pågått i 1–3 veckor rekommenderas en vårdkontroll om symtomen fortsätter.",
        "self_spec": "Egenvård (vid behov Allmänmedicin)",

        "prolonged_self_badge": "Rekommendation: Kontroll vid långvariga symtom",
        "prolonged_self_title": "Automatiserad första vårdbedömning: Långvariga symtom (> 3 veckor)",
        "prolonged_self_text": "Symtombilden motsvarar närmast {cond} ({top} %), som vanligen är lindrigt. Eftersom symtomen har pågått i mer än tre veckor rekommenderas en kontroll på mottagning om besvären fortsätter.",

        "sched_badge": "Rekommendation: Icke-brådskande mottagningsbedömning",
        "sched_title": "Automatiserad första vårdbedömning: Kontrollbesök rekommenderas",
        "sched_text_lt1w": "Dina symtom stämmer väl överens med dragen för {cond} ({top} %). Vi rekommenderar att boka en icke-brådskande tid på mottagning om besvären inte avtar.",
        "sched_text_1_3w": "Dina symtom motsvarar dragen för {cond} ({top} %) och har varat i 1–3 veckor. Vi rekommenderar att du bokar tid för kontroll på mottagning.",
        "sched_text_gt3w": "Dina symtom motsvarar dragen för {cond} ({top} %) och har pågått i över 3 veckor. På grund av varaktigheten rekommenderas undersökning på mottagning.",

        "prio_badge": "Rekommendation: Undersökning på mottagning",
        "prio_title": "Automatiserad första vårdbedömning: Mottagningsbesök rekommenderas",
        "prio_text_lt1w": "Dina symtom och bakgrundsuppgifter överensstämmer med typiska drag för {cond} ({top} %). Vi rekommenderar att du i lugn och ro kontaktar hälso- och sjukvården för närmare bedömning.",
        "prio_text_1_3w": "Dina symtom och bakgrundsuppgifter motsvarar {cond} ({top} %) och har pågått i 1–3 veckor. Vi rekommenderar att du bokar mottagningstid för undersökning inom kort.",
        "prio_text_gt3w": "Dina symtom och bakgrundsuppgifter motsvarar {cond} ({top} %) och har pågått i över 3 veckor. Vid långvariga besvär är fortsatt utredning på mottagning särskilt motiverad.",

        "dur_note_1_3w_calm": "Varaktighet (1–3 veckor): Även om bedömningen är lugn har symtomen pågått i över en vecka. Om besvären fortsätter rekommenderas en kontroll.",
        "dur_note_gt3w_calm": "Varaktighet (> 3 veckor): Vid symtom som varat över tre veckor rekommenderas även vid god bedömning att uppsöka mottagning för kontroll om besvären fortsätter.",
        "dur_note_gt3w_visit": "Lång varaktighet (> 3 veckor): Symtom som pågått i över tre veckor bör utredas på mottagning utan onödig fördröjning.",
        "reassurance": "Lugn vägledning: Den automatiserade första vårdbedömningen är inte en medicinsk diagnos, utan ett stöd för att bedöma vårdbehovet. Ta gärna med denna sammanfattning till mottagningen.",
        "geriatric": "Åldersgrupp ≥ 70 år beaktad: vid eventuell remiss rekommenderas en lätt kartläggning av funktionsförmåga och näringstillstånd (G8-bedömning) som stöd för individuell vårdplanering.",
        "pediatric_spec": "Pediatrik / Allmänmedicin",
        "general_spec": "Allmänmedicin",
        "rationale": "Den automatiserade första vårdbedömningen vägde in angivna uppgifter ({factors}) och fann starkast överensstämmelse med {cond} (ICD-10: {icd10}).",
        "default_factor": "Ålders- och bakgrundsprofil",
    },
    "en": {
        "no_sym_badge": "Feeling well • No active symptoms",
        "no_sym_title": "Preliminary automated evaluation: No current need for care",
        "no_sym_text_short": "You have not selected any current symptoms, so this preliminary evaluation indicates no immediate need for medical care.",
        "no_sym_text_long": "You have not selected any active symptoms. However, if you have had a lingering concern for weeks, select the symptom from the list or schedule a check-up if it continues.",
        "no_sym_spec": "No care needed (routine self-monitoring)",
        "no_sym_rat": "Select one or more symptoms to generate a symptom-based preliminary automated evaluation.",

        "below_badge": "No immediate clinic visit needed",
        "below_badge_long": "Below match threshold • Note prolonged duration",
        "below_title": "Preliminary automated evaluation: Symptom profile is below the match threshold",
        "below_text": "No single evaluated condition exceeded your selected match threshold ({thr}%; highest match {top}% for {cond}).",
        "below_tail_lt1w": " Since symptoms have lasted less than a week, you can calmly monitor how you feel at home and contact healthcare if symptoms worsen.",
        "below_tail_1_3w": " However, since symptoms have lasted 1–3 weeks, we recommend scheduling a check-up if they continue.",
        "below_tail_gt3w": " Because symptoms have persisted for over 3 weeks, even with a reassuring evaluation we recommend going for a medical check-up if they continue.",
        "below_spec": "Self-care and home monitoring",
        "below_spec_long": "Home monitoring (General Practice if persistent)",

        "self_badge": "Preliminary automated evaluation: Self-care & monitoring",
        "self_title": "Preliminary automated evaluation: Likely mild condition manageable at home",
        "self_text_lt1w": "Your reported symptoms align best with a mild condition ({cond}, {top}%), which typically improves with rest and supportive self-care within a few days.",
        "self_text_1_3w": "Your symptoms primarily fit a mild condition ({cond}, {top}%). Since they have lasted 1–3 weeks, we recommend going for a check-up if symptoms continue.",
        "self_spec": "Self-care (General Practice if needed)",

        "prolonged_self_badge": "Recommendation: Check-up for prolonged symptoms",
        "prolonged_self_title": "Preliminary automated evaluation: Prolonged symptoms (> 3 weeks)",
        "prolonged_self_text": "Your symptom pattern aligns closest with {cond} ({top}%), which is typically mild. Because symptoms have lasted over three weeks, we recommend visiting a clinic for a check-up if they continue.",

        "sched_badge": "Recommendation: Non-urgent clinic evaluation",
        "sched_title": "Preliminary automated evaluation: A clinic check-up is recommended",
        "sched_text_lt1w": "Your reported symptoms match the typical profile of {cond} ({top}%). We recommend scheduling a non-urgent appointment if symptoms do not begin to ease.",
        "sched_text_1_3w": "Your symptoms match the profile of {cond} ({top}%) and have lasted 1–3 weeks. We recommend booking an appointment for a medical check-up.",
        "sched_text_gt3w": "Your symptoms match {cond} ({top}%) and have persisted for over 3 weeks. Given the prolonged duration, an in-person clinical evaluation is recommended.",

        "prio_badge": "Recommendation: Clinical evaluation advised",
        "prio_title": "Preliminary automated evaluation: Visiting a clinic for examination is advised",
        "prio_text_lt1w": "Your symptoms and background align with characteristic features of {cond} ({top}%). We calmly recommend contacting a healthcare professional for a clinical assessment.",
        "prio_text_1_3w": "Your symptoms and background align with {cond} ({top}%) and have lasted 1–3 weeks. We recommend scheduling a clinic examination in the near future.",
        "prio_text_gt3w": "Your symptoms and background align with {cond} ({top}%) and have persisted for over 3 weeks. Due to the prolonged duration, a timely diagnostic check-up is strongly advised.",

        "dur_note_1_3w_calm": "Duration note (1–3 weeks): Even though the initial evaluation is reassuring, your symptoms have lasted over a week. If they continue, going for a check-up is recommended.",
        "dur_note_gt3w_calm": "Duration note (> 3 weeks): When symptoms have lasted over three weeks, even with a good initial evaluation, a medical check-up is recommended if they continue.",
        "dur_note_gt3w_visit": "Prolonged duration (> 3 weeks): Symptoms lasting longer than three weeks warrant clinical evaluation without unnecessary delay.",
        "reassurance": "Calm next steps: A preliminary automated evaluation is not a medical diagnosis, but an aid to assess your need for care. Bring this summary to your appointment.",
        "geriatric": "Age group ≥ 70 noted: alongside any specialist referral, a brief functional and nutritional check (G8 screening) is recommended to support individualized care.",
        "pediatric_spec": "Pediatrics / General Practice",
        "general_spec": "General Practice",
        "rationale": "The preliminary automated evaluation weighed your reported factors ({factors}) and identified {cond} (ICD-10: {icd10}) as the closest match.",
        "default_factor": "Age and background profile",
    },
}


def evaluate_prediction(req: PredictRequest) -> dict[str, Any]:
    lang = _norm_lang(req.lang)
    c = _COPY[lang]
    name_key = f"name_{lang}"
    spec_key = f"specialty_{lang}"
    label_key = f"label_{lang}"

    probs = compute_probabilities(req)
    threshold = float(req.threshold)

    if not req.symptoms:
        long_dur = req.duration in ("1-3w", ">3w")
        dur_note = None
        if req.duration == "1-3w":
            dur_note = c["dur_note_1_3w_calm"]
        elif req.duration == ">3w":
            dur_note = c["dur_note_gt3w_calm"]
        return {
            "lang": lang,
            "threshold": threshold,
            "duration": req.duration,
            "seek_care": False,
            "urgency_tier": "no_symptoms",
            "urgency_badge": c["no_sym_badge"],
            "urgency_title": c["no_sym_title"],
            "urgency_text": c["no_sym_text_long"] if long_dur else c["no_sym_text_short"],
            "duration_note": dur_note,
            "reassurance_note": None,
            "geriatric_note": None,
            "recommended_specialty": c["no_sym_spec"],
            "exclusion_specialty": None,
            "rationale": c["no_sym_rat"],
            "top3": [],
            "max_probability": 0.0,
        }

    order = np.argsort(-probs)
    all_49: list[dict[str, Any]] = []
    for idx in order:
        eng = SORTED_LABELS[idx]
        meta = CONDITIONS_49[eng]
        p = round(float(probs[idx]), 4)
        all_49.append({
            "pathology_en": eng,
            "name": meta[name_key],
            "name_fi": meta["name_fi"],
            "icd10": meta["icd10"],
            "severity": meta["severity"],
            "specialty": meta[spec_key],
            "specialty_fi": meta["specialty_fi"],
            "requires_doctor": meta["requires_doctor"],
            "is_cancer": bool(meta.get("is_cancer", False)),
            "probability": p,
            "meets_threshold": p >= threshold,
        })

    top1 = all_49[0]
    top_prob = top1["probability"]
    top_pct = round(top_prob * 100)
    thr_pct = round(threshold * 100)

    notable_chosen = [
        s[label_key]
        for s in SYMPTOMS_UI
        if s["id"] in req.symptoms and s.get("notable")
    ]
    chosen_symptom_labels = [
        s[label_key] for s in SYMPTOMS_UI if s["id"] in req.symptoms
    ]
    chosen_ant_labels = [
        a[label_key] for a in ANTECEDENTS_UI if a["id"] in req.antecedents
    ]

    exclusion_specialty = None
    for cand in all_49[1:4]:
        if (
            cand["probability"] >= max(0.12, threshold * 0.5)
            and cand["requires_doctor"]
            and cand["specialty"] != top1["specialty"]
        ):
            exclusion_specialty = cand["specialty"]
            break

    duration_note: str | None = None

    # 1. Threshold check: directly governed by `threshold` so adjusting threshold always takes effect
    if top_prob < threshold:
        seek_care = False
        urgency_tier = "below_threshold"
        urgency_title = c["below_title"]
        base_msg = c["below_text"].format(thr=thr_pct, top=top_pct, cond=top1["name"])
        if req.duration == "<1w":
            urgency_badge = c["below_badge"]
            urgency_text = base_msg + c["below_tail_lt1w"]
            recommended_specialty = c["below_spec"]
        elif req.duration == "1-3w":
            urgency_badge = c["below_badge"]
            urgency_text = base_msg + c["below_tail_1_3w"]
            duration_note = c["dur_note_1_3w_calm"]
            recommended_specialty = c["below_spec_long"]
        else:  # ">3w"
            urgency_badge = c["below_badge_long"]
            urgency_text = base_msg + c["below_tail_gt3w"]
            duration_note = c["dur_note_gt3w_calm"]
            recommended_specialty = c["below_spec_long"]

    # 2. Mild / self-care condition above threshold: duration controls escalation cleanly
    elif not top1["requires_doctor"] and not notable_chosen:
        if req.duration == "<1w":
            seek_care = False
            urgency_tier = "self_care"
            urgency_badge = c["self_badge"]
            urgency_title = c["self_title"]
            urgency_text = c["self_text_lt1w"].format(cond=top1["name"], top=top_pct)
            recommended_specialty = c["self_spec"]
        elif req.duration == "1-3w":
            seek_care = False
            urgency_tier = "self_care"
            urgency_badge = c["self_badge"]
            urgency_title = c["self_title"]
            urgency_text = c["self_text_1_3w"].format(cond=top1["name"], top=top_pct)
            duration_note = c["dur_note_1_3w_calm"]
            recommended_specialty = c["self_spec"]
        else:  # ">3w" -> prolonged mild symptoms escalate to non-urgent check-up recommendation
            seek_care = True
            urgency_tier = "scheduled_visit"
            urgency_badge = c["prolonged_self_badge"]
            urgency_title = c["prolonged_self_title"]
            urgency_text = c["prolonged_self_text"].format(cond=top1["name"], top=top_pct)
            duration_note = c["dur_note_gt3w_calm"]
            recommended_specialty = c["general_spec"]

    # 3. Clinic evaluation recommended (above threshold & requires doctor or notable symptom)
    else:
        seek_care = True
        recommended_specialty = top1["specialty"]
        if req.age_band == "0-17" and (
            "Yleislääketiede" in top1["specialty_fi"]
        ):
            recommended_specialty = c["pediatric_spec"]

        is_priority = (
            top1["severity"] <= 2
            or bool(notable_chosen)
            or top1.get("is_cancer", False)
            or req.duration == ">3w"
        )
        if is_priority:
            urgency_tier = "priority_visit"
            urgency_badge = c["prio_badge"]
            urgency_title = c["prio_title"]
            if req.duration == "<1w":
                urgency_text = c["prio_text_lt1w"].format(cond=top1["name"], top=top_pct)
            elif req.duration == "1-3w":
                urgency_text = c["prio_text_1_3w"].format(cond=top1["name"], top=top_pct)
            else:
                urgency_text = c["prio_text_gt3w"].format(cond=top1["name"], top=top_pct)
                duration_note = c["dur_note_gt3w_visit"]
        else:
            urgency_tier = "scheduled_visit"
            urgency_badge = c["sched_badge"]
            urgency_title = c["sched_title"]
            if req.duration == "<1w":
                urgency_text = c["sched_text_lt1w"].format(cond=top1["name"], top=top_pct)
            elif req.duration == "1-3w":
                urgency_text = c["sched_text_1_3w"].format(cond=top1["name"], top=top_pct)
            else:
                urgency_text = c["sched_text_gt3w"].format(cond=top1["name"], top=top_pct)
                duration_note = c["dur_note_gt3w_visit"]

    geriatric_note = None
    if req.age_band in ("70-79", "80+") and seek_care:
        geriatric_note = c["geriatric"]

    factors = (chosen_ant_labels[:2] + chosen_symptom_labels[:3]) or [c["default_factor"]]
    rationale = c["rationale"].format(
        factors=", ".join(factors),
        cond=top1["name"],
        icd10=top1["icd10"],
    )

    return {
        "lang": lang,
        "threshold": threshold,
        "duration": req.duration,
        "seek_care": seek_care,
        "urgency_tier": urgency_tier,
        "urgency_badge": urgency_badge,
        "urgency_title": urgency_title,
        "urgency_text": urgency_text,
        "duration_note": duration_note,
        "reassurance_note": c["reassurance"] if seek_care else None,
        "geriatric_note": geriatric_note,
        "recommended_specialty": recommended_specialty,
        "exclusion_specialty": exclusion_specialty,
        "rationale": rationale,
        "top3": all_49[:3],
        "max_probability": top_prob,
    }


# ---------------------------------------------------------------------------
# Regional 24h / 48h persistence & multilingual aggregation
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
            symptom_ids_json TEXT NOT NULL DEFAULT '[]',
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
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(submissions)").fetchall()}
    if "symptom_ids_json" not in cols:
        conn.execute("ALTER TABLE submissions ADD COLUMN symptom_ids_json TEXT NOT NULL DEFAULT '[]'")
        conn.execute("DELETE FROM submissions WHERE is_seed = 1")
        conn.commit()
    return conn


def _muni_meta(municipality: str) -> dict[str, str]:
    for m in MUNICIPALITIES:
        if m["name"] == municipality:
            return m
    return {
        "name": municipality,
        "name_sv": municipality,
        "county_fi": "Muu hyvinvointialue",
        "county_sv": "Annat välfärdsområde",
        "county_en": "Other wellbeing county",
    }


def _seed_if_empty() -> None:
    conn = _get_conn()
    try:
        n = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        if n > 0:
            return
        now = time.time()
        seed_cases: list[tuple[float, PredictRequest]] = [
            # Helsinki: hengitystiet, astma ja keuhkojen jatkoselvittely
            (now - 42 * 3600, PredictRequest(municipality="Helsinki", age_band="30-39", sex="F", duration="<1w", antecedents=["ant_sick_contact"], symptoms=["sym_nasal", "sym_sore_throat", "sym_cough"])),
            (now - 34 * 3600, PredictRequest(municipality="Helsinki", age_band="40-49", sex="M", duration="1-3w", antecedents=["ant_asthma"], symptoms=["sym_cough", "sym_dyspnea", "sym_wheeze"])),
            (now - 20 * 3600, PredictRequest(municipality="Helsinki", age_band="20-29" if False else "18-29", sex="F", duration="<1w", antecedents=["ant_sick_contact"], symptoms=["sym_nasal", "sym_sore_throat", "sym_cough", "sym_fever"])),
            (now - 16 * 3600, PredictRequest(municipality="Helsinki", age_band="30-39", sex="M", duration="<1w", antecedents=["ant_sick_contact"], symptoms=["sym_cough", "sym_fever", "sym_muscle_ache"])),
            (now - 11 * 3600, PredictRequest(municipality="Helsinki", age_band="60-69", sex="F", duration=">3w", antecedents=["ant_former_smoker", "ant_fam_lung_cancer"], symptoms=["sym_cough", "sym_weight_loss", "sym_appetite_loss", "sym_fatigue_pallor", "sym_pain"], pain=PainDetails(intensity=4, onset=2, location="chest_side", character="dull", radiation="none"))),
            (now - 5 * 3600, PredictRequest(municipality="Helsinki", age_band="30-39", sex="F", duration="1-3w", antecedents=["ant_asthma", "ant_fam_asthma_allergy"], symptoms=["sym_cough", "sym_dyspnea", "sym_wheeze"])),
            (now - 2 * 3600, PredictRequest(municipality="Helsinki", age_band="50-59", sex="M", duration="1-3w", antecedents=["ant_hypertension", "ant_smoking"], symptoms=["sym_pain", "sym_exertion", "sym_dyspnea"], pain=PainDetails(intensity=6, onset=6, location="chest_center", character="pressing", radiation="arm_shoulder"))),

            # Tampere: keuhkoselvittelyt & pitkittynyt yskä
            (now - 39 * 3600, PredictRequest(municipality="Tampere", age_band="50-59", sex="M", duration="1-3w", antecedents=["ant_smoking"], symptoms=["sym_cough", "sym_sputum"])),
            (now - 19 * 3600, PredictRequest(municipality="Tampere", age_band="70-79", sex="M", duration=">3w", antecedents=["ant_smoking", "ant_fam_lung_cancer"], symptoms=["sym_cough", "sym_hemoptysis", "sym_weight_loss", "sym_appetite_loss", "sym_dyspnea", "sym_fatigue_pallor"])),
            (now - 13 * 3600, PredictRequest(municipality="Tampere", age_band="60-69", sex="F", duration=">3w", antecedents=["ant_copd", "ant_smoking"], symptoms=["sym_cough", "sym_sputum", "sym_dyspnea", "sym_wheeze"])),
            (now - 7 * 3600, PredictRequest(municipality="Tampere", age_band="50-59", sex="F", duration=">3w", antecedents=["ant_fam_lung_cancer", "ant_secondhand_smoke"], symptoms=["sym_cough", "sym_hemoptysis", "sym_weight_loss", "sym_appetite_loss", "sym_pain"], pain=PainDetails(intensity=4, onset=2, location="chest_center", character="dull", radiation="back_scapula"))),
            (now - 3 * 3600, PredictRequest(municipality="Tampere", age_band="40-49", sex="M", duration="<1w", antecedents=["ant_sick_contact"], symptoms=["sym_cough", "sym_fever", "sym_sore_throat"])),

            # Turku: ylävatsa, painonlasku ja gastroenterologia
            (now - 36 * 3600, PredictRequest(municipality="Turku", age_band="40-49", sex="F", duration="<1w", antecedents=["ant_gerd", "ant_overweight"], symptoms=["sym_heartburn"])),
            (now - 18 * 3600, PredictRequest(municipality="Turku", age_band="60-69", sex="M", duration=">3w", antecedents=["ant_diabetes", "ant_smoking", "ant_fam_panc_cancer"], symptoms=["sym_weight_loss", "sym_pale_stool_dark_urine", "sym_nausea_diarrhea", "sym_fatigue_pallor", "sym_pain"], pain=PainDetails(intensity=5, onset=2, location="epigastric", character="cramp_tug", radiation="back_scapula"))),
            (now - 9 * 3600, PredictRequest(municipality="Turku", age_band="70-79", sex="F", duration=">3w", antecedents=["ant_pancreatitis", "ant_fam_panc_cancer"], symptoms=["sym_weight_loss", "sym_appetite_loss", "sym_pale_stool_dark_urine", "sym_rash", "sym_pain"], pain=PainDetails(intensity=6, onset=2, location="epigastric", character="dull", radiation="back_scapula"), rash=RashDetails(color="yellow", itch=6, swollen=False))),
            (now - 4 * 3600, PredictRequest(municipality="Turku", age_band="50-59", sex="M", duration="1-3w", antecedents=["ant_gerd"], symptoms=["sym_heartburn", "sym_pain"], pain=PainDetails(intensity=4, onset=4, location="chest_lower", character="burning", radiation="upper_chest"))),

            # Oulu: infektiot & korva/hengitystiet
            (now - 30 * 3600, PredictRequest(municipality="Oulu", age_band="18-29", sex="F", duration="<1w", antecedents=["ant_allergy"], symptoms=["sym_nasal", "sym_itchy_nose_eyes"])),
            (now - 15 * 3600, PredictRequest(municipality="Oulu", age_band="0-17", sex="M", duration="<1w", antecedents=["ant_recent_viral"], symptoms=["sym_fever", "sym_cough", "sym_pain"], pain=PainDetails(intensity=6, onset=6, location="ear", character="throbbing", radiation="none"))),
            (now - 10 * 3600, PredictRequest(municipality="Oulu", age_band="30-39", sex="F", duration="<1w", antecedents=["ant_sick_contact"], symptoms=["sym_fever", "sym_muscle_ache", "sym_cough", "sym_sore_throat"])),
            (now - 4 * 3600, PredictRequest(municipality="Oulu", age_band="40-49", sex="M", duration="1-3w", antecedents=["ant_sick_contact"], symptoms=["sym_fever", "sym_cough", "sym_sputum", "sym_dyspnea"])),

            # Espoo
            (now - 28 * 3600, PredictRequest(municipality="Espoo", age_band="30-39", sex="F", duration="<1w", antecedents=["ant_allergy"], symptoms=["sym_nasal", "sym_itchy_nose_eyes"])),
            (now - 12 * 3600, PredictRequest(municipality="Espoo", age_band="40-49", sex="F", duration="1-3w", antecedents=["ant_anemia_hist"], symptoms=["sym_fatigue_pallor", "sym_dizzy_syncope", "sym_dyspnea"])),
            (now - 6 * 3600, PredictRequest(municipality="Espoo", age_band="70-79", sex="F", duration=">3w", antecedents=["ant_anemia_hist"], symptoms=["sym_fatigue_pallor", "sym_weight_loss", "sym_dyspnea", "sym_dizzy_syncope"])),

            # Kuopio
            (now - 31 * 3600, PredictRequest(municipality="Kuopio", age_band="60-69", sex="M", duration="1-3w", antecedents=["ant_cardiac", "ant_hypertension"], symptoms=["sym_dyspnea", "sym_orthopnea", "sym_swelling"], swelling=SwellingDetails(location="ankles"))),
            (now - 14 * 3600, PredictRequest(municipality="Kuopio", age_band="70-79", sex="F", duration="1-3w", antecedents=["ant_hypertension"], symptoms=["sym_palpitations", "sym_dyspnea", "sym_dizzy_syncope"])),
            (now - 3 * 3600, PredictRequest(municipality="Kuopio", age_band="60-69", sex="M", duration="1-3w", antecedents=["ant_cardiac"], symptoms=["sym_palpitations", "sym_pain", "sym_exertion"], pain=PainDetails(intensity=6, onset=6, location="chest_center", character="pressing", radiation="jaw_neck"))),

            # Jyväskylä & Rovaniemi
            (now - 27 * 3600, PredictRequest(municipality="Jyväskylä", age_band="18-29", sex="F", duration="<1w", antecedents=["ant_anxiety", "ant_caffeine_stim"], symptoms=["sym_palpitations", "sym_dyspnea", "sym_sweating"])),
            (now - 8 * 3600, PredictRequest(municipality="Jyväskylä", age_band="30-39", sex="M", duration="1-3w", antecedents=[], symptoms=["sym_pain"], pain=PainDetails(intensity=8, onset=8, location="head_sinus", character="stabbing" if False else "sharp", radiation="none"))),
            (now - 6 * 3600, PredictRequest(municipality="Rovaniemi", age_band="50-59", sex="F", duration="1-3w", antecedents=["ant_recent_surgery_immob"], symptoms=["sym_dyspnea", "sym_pleuritic", "sym_swelling"], pain=PainDetails(intensity=6, onset=8, location="chest_side", character="sharp", radiation="none"), swelling=SwellingDetails(location="calf"))),
        ]

        sym_label_fi = {s["id"]: s["label_fi"] for s in SYMPTOMS_UI}
        rows_to_insert = []
        for ts, req in seed_cases:
            res = evaluate_prediction(req)
            if not res["top3"]:
                continue
            top = res["top3"][0]
            sym_names_fi = [sym_label_fi.get(sid, sid) for sid in req.symptoms]
            m_meta = _muni_meta(req.municipality)
            rows_to_insert.append((
                ts,
                req.municipality,
                m_meta["county_fi"],
                req.age_band,
                req.sex,
                req.duration,
                json.dumps(sym_names_fi, ensure_ascii=False),
                json.dumps(req.symptoms, ensure_ascii=False),
                top["pathology_en"],
                top["name_fi"],
                top["icd10"],
                top["probability"],
                res["recommended_specialty"],
                1 if res["seek_care"] else 0,
                1,
            ))
        conn.executemany(
            """
            INSERT INTO submissions (
                created_at, municipality, county, age_band, sex, duration,
                symptoms_json, symptom_ids_json, top_pathology_en, top_pathology_fi, icd10,
                top_prob, specialty_fi, seek_care, is_seed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def _on_startup() -> None:
    _try_load_trained_model()
    _seed_if_empty()


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "ok", "message": "HealthQuality API running"})


@app.get("/api/catalog")
def get_catalog():
    return {
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
        err_by_lang = {
            "fi": "Valitse vähintään yksi oire ennen tallennusta.",
            "sv": "Välj minst ett symtom innan du skickar in.",
            "en": "Please select at least one symptom before submitting.",
        }
        return JSONResponse(
            {"saved": False, "error": err_by_lang[_norm_lang(req.lang)]},
            status_code=400,
        )
    top = res["top3"][0]
    sym_label_fi = {s["id"]: s["label_fi"] for s in SYMPTOMS_UI}
    sym_names_fi = [sym_label_fi.get(sid, sid) for sid in req.symptoms]
    m_meta = _muni_meta(req.municipality)

    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO submissions (
                created_at, municipality, county, age_band, sex, duration,
                symptoms_json, symptom_ids_json, top_pathology_en, top_pathology_fi, icd10,
                top_prob, specialty_fi, seek_care, is_seed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                time.time(),
                req.municipality,
                m_meta["county_fi"],
                req.age_band,
                req.sex,
                req.duration,
                json.dumps(sym_names_fi, ensure_ascii=False),
                json.dumps(req.symptoms, ensure_ascii=False),
                top["pathology_en"],
                top["name_fi"],
                top["icd10"],
                top["probability"],
                top["specialty_fi"],
                1 if res["seek_care"] else 0,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
    finally:
        conn.close()

    return {"saved": True, "id": row_id, "municipality": req.municipality, "prediction": res}


@app.get("/api/municipal-stats")
def get_municipal_stats(lang: str = Query(default="fi")):
    lang = _norm_lang(lang)
    name_key = f"name_{lang}"
    spec_key = f"specialty_{lang}"
    sym_key = f"label_{lang}"
    county_key = f"county_{lang}"

    sym_by_id = {s["id"]: s[sym_key] for s in SYMPTOMS_UI}
    sym_fi_to_lang = {s["label_fi"]: s[sym_key] for s in SYMPTOMS_UI}

    now = time.time()
    cut_24h = now - 24 * 3600
    cut_48h = now - 48 * 3600

    _seed_if_empty()
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM submissions WHERE created_at >= ? ORDER BY created_at DESC",
            (cut_48h,),
        ).fetchall()
    finally:
        conn.close()

    by_muni: dict[str, dict[str, Any]] = {}
    global_sym_curr: dict[str, int] = {}
    global_sym_prev: dict[str, int] = {}
    total_24h = 0
    total_prev_24h = 0
    seek_care_24h = 0

    for r in rows:
        muni = r["municipality"]
        m_meta = _muni_meta(muni)
        display_muni = m_meta["name_sv"] if lang == "sv" else m_meta["name"]
        display_county = m_meta[county_key]

        is_curr = r["created_at"] >= cut_24h
        b = by_muni.setdefault(
            muni,
            {
                "municipality": muni,
                "display_municipality": display_muni,
                "county": display_county,
                "count_24h": 0,
                "count_prev_24h": 0,
                "seek_care_24h": 0,
                "eval_counts": {},
                "specialty_counts": {},
                "sym_curr": {},
                "sym_prev": {},
                "live_user_count": 0,
            },
        )

        # Resolve symptom labels in requested language
        sym_ids = json.loads(r["symptom_ids_json"] or "[]")
        if sym_ids:
            syms = [sym_by_id.get(sid, sid) for sid in sym_ids]
        else:
            raw_fi = json.loads(r["symptoms_json"] or "[]")
            syms = [sym_fi_to_lang.get(lbl, lbl) for lbl in raw_fi]

        # Resolve condition & specialty in requested language
        eng = r["top_pathology_en"]
        cond_meta = CONDITIONS_49.get(eng)
        eval_label = cond_meta[name_key] if cond_meta else r["top_pathology_fi"]
        spec_label = cond_meta[spec_key] if cond_meta else r["specialty_fi"]

        if is_curr:
            total_24h += 1
            b["count_24h"] += 1
            if r["seek_care"]:
                seek_care_24h += 1
                b["seek_care_24h"] += 1
            if not r["is_seed"]:
                b["live_user_count"] += 1
            b["eval_counts"][eval_label] = b["eval_counts"].get(eval_label, 0) + 1
            b["specialty_counts"][spec_label] = b["specialty_counts"].get(spec_label, 0) + 1
            for s in syms:
                b["sym_curr"][s] = b["sym_curr"].get(s, 0) + 1
                global_sym_curr[s] = global_sym_curr.get(s, 0) + 1
        else:
            total_prev_24h += 1
            b["count_prev_24h"] += 1
            for s in syms:
                b["sym_prev"][s] = b["sym_prev"].get(s, 0) + 1
                global_sym_prev[s] = global_sym_prev.get(s, 0) + 1

    table_rows = []
    for muni, b in by_muni.items():
        if b["count_24h"] == 0 and b["count_prev_24h"] == 0:
            continue
        top_evals = [
            {"label": k, "count": v}
            for k, v in sorted(b["eval_counts"].items(), key=lambda x: -x[1])
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
            "display_municipality": b["display_municipality"],
            "county": b["county"],
            "count_24h": b["count_24h"],
            "count_prev_24h": b["count_prev_24h"],
            "delta_submissions": b["count_24h"] - b["count_prev_24h"],
            "seek_care_24h": b["seek_care_24h"],
            "live_user_count": b["live_user_count"],
            "evaluations": top_evals,
            "specialties": top_specs,
            "symptom_trends": sym_trends,
        })

    table_rows.sort(key=lambda r: (-r["live_user_count"], -r["count_24h"], r["display_municipality"]))

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
        "lang": lang,
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
