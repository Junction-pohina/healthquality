"""Build a standalone HTML demo of the final model on test-set patients.

For a handful of test patients it shows who they are, what they reported, the true
pathology, and the model's likeliest pathologies with their probabilities, next to the
dataset's own differential diagnosis. A toggle switches to the calibrated alpha = 0 model.

It reads the saved test predictions from runs/test/ (written once by src.final_eval), so
the test set is not re-evaluated. The page embeds patient rows from DDXPlus, so it is
written under data/ (git-ignored) by default.

Usage:
    python -m src.demo                       # -> data/demo/test_demo.html
    python -m src.demo --random 8 --seed 1   # more or different randomly chosen patients
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, load_conditions, load_evidences, load_patients

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "runs" / "test"
TOP_K = 5
MODELS = {"final": ("probs_test_mlp_a0.5.npy", "Final model (MLP, α = 0.5)"),
          "calibrated": ("probs_test_mlp_a0.npy", "Calibrated model (MLP, α = 0)")}


# ---------------------------------------------------------------- patient details

def decode_evidences(items: list[str], evidences: pd.DataFrame) -> list[dict[str, str]]:
    """Evidence codes -> [{question, answer}], one entry per question (multi-choice answers joined)."""
    answers: dict[str, list[str]] = {}
    for item in items:
        code, _, value = item.partition("_@_")
        ev = evidences.loc[code]
        if not value:
            answer = "yes"
        else:
            answer = ev.value_meaning_en.get(value, value)
            answer = {"N": "no", "Y": "yes"}.get(answer, answer)
        answers.setdefault(ev.question_en, []).append(str(answer))
    return [{"question": q, "answer": ", ".join(a), "antecedent": False} for q, a in answers.items()]


def pick_patients(y: np.ndarray, probs: np.ndarray, labels: list[str], n_random: int,
                  rng: np.random.Generator) -> list[tuple[int, str]]:
    """A mix that shows typical behaviour and the known weak spots, each with a reason."""
    pred = probs.argmax(axis=1)
    label_id = {name: i for i, name in enumerate(labels)}
    picks: list[tuple[int, str]] = []
    used: set[int] = set()

    def take(candidates: np.ndarray, reason: str) -> None:
        candidates = np.setdiff1d(candidates, list(used))
        if len(candidates):
            i = int(rng.choice(candidates))
            picks.append((i, reason))
            used.add(i)

    correct = np.where(pred == y)[0]
    for _ in range(n_random):
        take(correct, "Randomly chosen, correctly diagnosed")
    take(np.where((y == label_id["Pulmonary neoplasm"]) & (pred == y))[0], "Cancer: pulmonary neoplasm")
    take(np.where((y == label_id["Pancreatic neoplasm"]) & (pred == y))[0], "Cancer: pancreatic neoplasm")
    take(np.where((y == label_id["Acute rhinosinusitis"]) & (pred == label_id["Chronic rhinosinusitis"]))[0],
         "Most common error: acute taken for chronic rhinosinusitis")
    take(np.where((y == label_id["Unstable angina"]) & (pred == label_id["Stable angina"]))[0],
         "Second most common error: unstable taken for stable angina")
    # the least confident correct prediction: the model hesitated but got it right
    conf = probs.max(axis=1)
    hesitant = correct[np.argsort(conf[correct])[:50]]
    take(hesitant, "Correct, but among the model's least confident cases")
    return picks


def build_data(args) -> dict:
    run = sorted(p for p in TEST_DIR.glob("*/") if (p / "results.json").exists())
    if not run:
        raise SystemExit("no test results in runs/test/; run python -m src.final_eval first")
    run = run[-1]
    vocab = json.loads((args.features_dir / "vocab.json").read_text(encoding="utf-8"))
    labels = vocab["labels"]
    probs = {k: np.load(run / f) for k, (f, _) in MODELS.items()}

    patients = load_patients("test", args.data_dir)
    evidences = load_evidences(args.data_dir)
    conditions = load_conditions(args.data_dir)
    y = patients.PATHOLOGY.astype(str).map({n: i for i, n in enumerate(labels)}).to_numpy()
    if len(y) != len(probs["final"]):
        raise SystemExit("test predictions and test patients have different lengths")

    rng = np.random.default_rng(args.seed)
    picks = pick_patients(y, probs["final"], labels, args.random, rng)

    out = []
    for i, reason in picks:
        row = patients.iloc[i]
        diff = dict(row.DIFFERENTIAL_DIAGNOSIS)
        answers = decode_evidences(row.EVIDENCES, evidences)
        antecedent_q = set(evidences.loc[evidences.is_antecedent, "question_en"])
        for a in answers:
            a["antecedent"] = a["question"] in antecedent_q
        predictions = {}
        for key, p in probs.items():
            top = np.argsort(-p[i])[:TOP_K]
            predictions[key] = [{"pathology": labels[j], "p": round(float(p[i, j]), 4),
                                 "differential": round(float(diff.get(labels[j], 0.0)), 4)} for j in top]
        out.append({
            "id": int(i), "reason": reason, "age": int(row.AGE), "sex": str(row.SEX),
            "first_complaint": str(evidences.loc[str(row.INITIAL_EVIDENCE), "question_en"]),
            "truth": str(row.PATHOLOGY), "truth_icd10": str(conditions.loc[str(row.PATHOLOGY), "icd10_id"]).upper(),
            "truth_severity": int(conditions.loc[str(row.PATHOLOGY), "severity"]),
            "answers": answers, "predictions": predictions,
            "differential_top": sorted(diff.items(), key=lambda kv: -kv[1])[:TOP_K],
        })
    results = json.loads((run / "results.json").read_text())
    return {"patients": out, "models": {k: v[1] for k, v in MODELS.items()}, "top_k": TOP_K,
            "test_rows": results["test_rows"],
            "test_top1": {k: results["models"][m]["test"]["top1_acc"]
                          for k, m in (("final", "mlp α=0.5"), ("calibrated", "mlp α=0"))}}


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnosis Model Demo</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --series-final: #eb6834; --series-calibrated: #2a78d6; --series-diff: #b9b8b2;
  --good: #0a7d32; --good-bg: #e3f4e8; --bad: #b42318; --bad-bg: #fbe9e7;
  --accent-bg: #eef4fc;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2d2b;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --series-final: #d95926; --series-calibrated: #3987e5; --series-diff: #5a5955;
    --good: #5fd08a; --good-bg: #16301f; --bad: #f2867a; --bad-bg: #3a1c19;
    --accent-bg: #1c2633;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2d2b;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --series-final: #d95926; --series-calibrated: #3987e5; --series-diff: #5a5955;
  --good: #5fd08a; --good-bg: #16301f; --bad: #f2867a; --bad-bg: #3a1c19;
  --accent-bg: #1c2633;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 28px 16px 8px; max-width: 1180px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 6px; letter-spacing: -0.01em; }
.lede { color: var(--text-secondary); margin: 0; max-width: 760px; }
.notice { margin: 14px 0 0; font-size: 13px; color: var(--text-secondary); background: var(--surface-1);
          border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; max-width: 760px; }
main { max-width: 1180px; margin: 0 auto; padding: 16px; display: grid; gap: 16px;
       grid-template-columns: 320px minmax(0, 1fr); align-items: start; }
@media (max-width: 860px) { main { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; }
.list { padding: 6px; display: flex; flex-direction: column; gap: 2px; }
.list button { all: unset; cursor: pointer; display: block; padding: 10px 12px; border-radius: 8px; }
.list button:hover { background: var(--surface-0); }
.list button[aria-selected="true"] { background: var(--accent-bg); }
.list button:focus-visible { outline: 2px solid var(--series-calibrated); outline-offset: -2px; }
.li-top { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; font-size: 14px; }
.li-sub { color: var(--text-muted); font-size: 12.5px; }
.mark { font-size: 12px; font-weight: 600; white-space: nowrap; }
.mark.ok { color: var(--good); } .mark.miss { color: var(--bad); }
.detail { padding: 20px; }
.why { font-size: 12.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin: 0 0 4px; }
.who { font-size: 20px; font-weight: 650; margin: 0; }
.facts { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 10px 0 18px; color: var(--text-secondary); font-size: 14px; }
.facts b { color: var(--text-primary); font-weight: 600; }
.verdict { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
           font-size: 13px; font-weight: 600; }
.verdict.ok { background: var(--good-bg); color: var(--good); }
.verdict.miss { background: var(--bad-bg); color: var(--bad); }
.section-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin: 0 0 10px; }
h2 { font-size: 15px; margin: 0; }
.toggle { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.toggle button { all: unset; cursor: pointer; padding: 5px 10px; font-size: 13px; color: var(--text-secondary); }
.toggle button[aria-pressed="true"] { background: var(--accent-bg); color: var(--text-primary); font-weight: 600; }
.toggle button:focus-visible { outline: 2px solid var(--series-calibrated); outline-offset: -2px; }
.legend { display: flex; gap: 16px; font-size: 12.5px; color: var(--text-secondary); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 8px; border-radius: 2px; display: inline-block; }
.rows { display: flex; flex-direction: column; gap: 12px; }
.row { display: grid; grid-template-columns: minmax(130px, 220px) minmax(0, 1fr); gap: 4px 14px; align-items: center; }
@media (max-width: 520px) { .row { grid-template-columns: minmax(0, 1fr); } }
.pname { font-size: 14px; display: flex; align-items: center; gap: 6px; min-width: 0; }
.pname span.n { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.truth-tag { font-size: 11px; font-weight: 700; color: var(--good); border: 1px solid currentColor;
             border-radius: 4px; padding: 0 4px; white-space: nowrap; }
.bars { display: flex; flex-direction: column; gap: 2px; }
.bar { display: flex; align-items: center; gap: 8px; }
.track { flex: 1; height: 10px; position: relative; }
.fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 4px 4px 0; min-width: 2px; }
.bar.diff .track { height: 6px; }
.val { font-variant-numeric: tabular-nums; font-size: 12.5px; color: var(--text-secondary); width: 44px; text-align: right; }
.bar.model .val { color: var(--text-primary); font-weight: 600; }
.note { font-size: 13px; color: var(--text-secondary); margin: 12px 0 0; }
details { margin-top: 22px; border-top: 1px solid var(--border); padding-top: 14px; }
summary { cursor: pointer; font-weight: 600; font-size: 15px; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13.5px; }
td { padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
td:first-child { color: var(--text-secondary); width: 62%; }
.kind { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }
footer { max-width: 1180px; margin: 0 auto; padding: 0 16px 32px; font-size: 12.5px; color: var(--text-muted); }
</style>
</head>
<body>
<header>
  <h1>What does the model think each patient has?</h1>
  <p class="lede">Patients from the held-out DDXPlus test set. For each one, the bars show the
  __TOPK__ pathologies the model rates most likely and how likely it thinks each is, next to the
  dataset's own differential diagnosis.</p>
  <p class="notice">Synthetic patients (DDXPlus), not real people. A research demo, not medical advice.
  On all __N__ test patients the final model's top guess is right __ACC__ of the time.</p>
</header>
<main>
  <nav class="panel list" id="list" aria-label="Patients"></nav>
  <section class="panel detail" id="detail" aria-live="polite"></section>
</main>
<footer>Probabilities from the saved test-set predictions of each model. The differential is the
probability distribution DDXPlus ships with each patient; the final model was trained to blend it with
the true diagnosis, so its probabilities read as a ranked differential rather than as its chance of being right.</footer>
<script>
const DATA = __DATA__;
const MODEL_COLOR = { final: "var(--series-final)", calibrated: "var(--series-calibrated)" };
let current = 0, model = "final";
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = p => (p >= 0.995 ? (p * 100).toFixed(1) : p < 0.001 && p > 0 ? "<0.1" : (p * 100).toFixed(1)) + "%";
const sexWord = s => s === "F" ? "Female" : s === "M" ? "Male" : s;

function renderList() {
  const list = document.getElementById("list");
  list.innerHTML = DATA.patients.map((p, i) => {
    const top = p.predictions.final[0].pathology, ok = top === p.truth;
    return `<button role="tab" aria-selected="${i === current}" data-i="${i}">
      <div class="li-top"><span>${sexWord(p.sex)}, ${p.age}</span>
        <span class="mark ${ok ? "ok" : "miss"}">${ok ? "✓ correct" : "✗ missed"}</span></div>
      <div class="li-sub">${esc(p.truth)}</div></button>`;
  }).join("");
  list.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    current = +b.dataset.i; renderList(); renderDetail();
    if (window.matchMedia("(max-width: 860px)").matches)
      document.getElementById("detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }));
}

function renderDetail() {
  const p = DATA.patients[current], preds = p.predictions[model];
  const ok = p.predictions[model][0].pathology === p.truth;
  const rows = preds.map(r => `
    <div class="row">
      <div class="pname"><span class="n" title="${esc(r.pathology)}">${esc(r.pathology)}</span>
        ${r.pathology === p.truth ? '<span class="truth-tag">TRUE</span>' : ""}</div>
      <div class="bars">
        <div class="bar model" title="${esc(DATA.models[model])}: ${pct(r.p)}">
          <div class="track"><div class="fill" style="width:${(r.p * 100).toFixed(2)}%;background:${MODEL_COLOR[model]}"></div></div>
          <span class="val">${pct(r.p)}</span></div>
        <div class="bar diff" title="Dataset differential: ${pct(r.differential)}">
          <div class="track"><div class="fill" style="width:${(r.differential * 100).toFixed(2)}%;background:var(--series-diff)"></div></div>
          <span class="val">${r.differential > 0 ? pct(r.differential) : "–"}</span></div>
      </div>
    </div>`).join("");
  const truthRank = preds.findIndex(r => r.pathology === p.truth);
  const missNote = ok ? "" : `<p class="note">The true pathology, ${esc(p.truth)}, is ${truthRank >= 0
      ? `the model's number ${truthRank + 1} pick` : `outside the model's top ${DATA.top_k}`}.</p>`;
  const answers = p.answers.filter(a => !a.antecedent), history = p.answers.filter(a => a.antecedent);
  const table = (title, items) => items.length ? `<tr><td colspan="2" class="kind">${title}</td></tr>` +
      items.map(a => `<tr><td>${esc(a.question)}</td><td>${esc(a.answer)}</td></tr>`).join("") : "";
  document.getElementById("detail").innerHTML = `
    <p class="why">${esc(p.reason)}</p>
    <p class="who">${sexWord(p.sex)}, ${p.age} years</p>
    <div class="facts">
      <span>First complaint: <b>${esc(p.first_complaint)}</b></span>
      <span>True pathology: <b>${esc(p.truth)}</b> (ICD-10 ${esc(p.truth_icd10)}, severity ${p.truth_severity})</span>
      <span class="verdict ${ok ? "ok" : "miss"}">${ok ? "✓ Top guess correct" : "✗ Top guess wrong"}</span>
    </div>
    <div class="section-head">
      <h2>Likeliest pathologies</h2>
      <div class="toggle" role="group" aria-label="Model">
        <button aria-pressed="${model === "final"}" data-m="final">Final (α = 0.5)</button>
        <button aria-pressed="${model === "calibrated"}" data-m="calibrated">Calibrated (α = 0)</button>
      </div>
    </div>
    <div class="legend">
      <span><i class="sw" style="background:${MODEL_COLOR[model]}"></i>${esc(DATA.models[model])}</span>
      <span><i class="sw" style="background:var(--series-diff)"></i>Dataset differential</span>
    </div>
    <div class="rows">${rows}</div>
    ${missNote}
    <details>
      <summary>What the patient reported (${p.answers.length} answers)</summary>
      <table>${table("Symptoms", answers)}${table("History and risk factors", history)}</table>
    </details>`;
  document.querySelectorAll(".toggle button").forEach(b => b.addEventListener("click", () => {
    model = b.dataset.m; renderList(); renderDetail();
  }));
}
renderList(); renderDetail();
</script>
</body>
</html>
"""


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (PAGE.replace("__DATA__", payload)
                .replace("__TOPK__", str(data["top_k"]))
                .replace("__N__", f"{data['test_rows']:,}")
                .replace("__ACC__", html.escape(f"{data['test_top1']['final'] * 100:.2f}%")))


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Standalone HTML demo of test-set predictions.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--features-dir", type=Path, default=DATA_DIR / "features")
    p.add_argument("--random", type=int, default=6, help="number of randomly chosen correct cases")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=DATA_DIR / "demo" / "test_demo.html")
    args = p.parse_args(argv)

    data = build_data(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data), encoding="utf-8")
    print(f"{len(data['patients'])} patients -> {args.out}")
    return args.out


if __name__ == "__main__":
    main()
