# 🔍 Support Integrity Auditor (SIA)

An evidence-grounded, self-supervised auditor that detects **Priority Mismatch** in
CRM support tickets — cases where a ticket's objective characteristics (text,
channel, resolution time) conflict with its human-assigned priority.

There are **no ground-truth mismatch labels** in the data. SIA bootstraps its own
supervision signal from raw tickets, trains a fine-tuned classifier on those
pseudo-labels, and produces a **hallucination-free Evidence Dossier** for every
flagged ticket.

---

## 1. Architecture

```
                ┌─────────────────────────────────────────────┐
                │      Customer Support Tickets (20,000)       │
                │  subject · description · priority · channel  │
                │  category · resolution_time · satisfaction   │
                └──────────────────────┬──────────────────────┘
                                       │
        ┌──────────────────────────────────────────────────────┐
        │  STAGE 1 — PSEUDO-LABEL GENERATION (self-supervised)   │
        │                                                        │
        │   Signal A: Rule-based NLP severity (keywords +        │
        │             negation)                                  │
        │   Signal B: Resolution-time severity (quartile proxy)  │
        │                                                        │
        │   fused = round(0.6·ruleA + 0.4·resB)                  │
        │   mismatch_label = 1 if fused disagrees with assigned  │
        │   mismatch_type  = Hidden Crisis | False Alarm         │
        └──────────────────────┬─────────────────────────────────┘
                               │  pseudo_labeled.csv
        ┌──────────────────────────────────────────────────────┐
        │  STAGE 2 — FINE-TUNED CLASSIFIER                       │
        │                                                        │
        │   microsoft/deberta-v3-small (fine-tuned, 2 classes)   │
        │   Input = text fields + structured metadata            │
        │   (channel, category, resolution-bin, satisfaction-bin)│
        │   Imbalance: weighted CrossEntropy + WeightedSampler   │
        │   Threshold tuned on validation macro-F1               │
        └──────────────────────┬─────────────────────────────────┘
                               │  models/deberta_final/
        ┌──────────────────────────────────────────────────────┐
        │  STAGE 3 — EVIDENCE DOSSIER (grounded, schema-locked)  │
        │   Every feature_evidence item names its source field.  │
        │   No fabricated claims → no hallucination.             │
        └──────────────────────┬─────────────────────────────────┘
                               │
                   ┌───────────┴───────────┐
                   │   Streamlit Web App    │
                   │  single · batch · dash │
                   └────────────────────────┘
```

**Single source of truth:** all severity / text / binning / dossier logic lives in
[`sia_core.py`](sia_core.py). `train_pipeline.py`, `predict.py`, and `app.py` import
from it, so training and inference are computed **identically** — no drift.

---

## 2. Stage 1 — Pseudo-label strategy & fusion justification

We fuse **two independent signals** (the spec requires ≥ 2):

| Signal | Source field | How severity is inferred |
|---|---|---|
| **Rule-based NLP** | `Ticket_Subject` + `Ticket_Description` | Critical / High / Low keyword banks + negation handling → severity 1–4 |
| **Resolution time** | `Resolution_Time_Hours` | Quartile bucket (≤Q25→1 … >Q75→4) as a severity proxy |

**Fusion:** `inferred = round(0.6 · rule + 0.4 · resolution)`, clamped to 1–4.
A ticket is labelled **mismatch** when the fused inference disagrees with the
assigned priority by a strong margin or when both signals agree on the direction
of disagreement. The mismatch is typed:

- **Hidden Crisis** — inferred severity **>** assigned (under-prioritised).
- **False Alarm** — inferred severity **<** assigned (over-prioritised).

**Why 0.6 / 0.4 (rule-weighted)?** The ablation below shows the rule signal aligns
far more strongly with the final pseudo-labels than resolution time does. In this
dataset, Critical tickets are actually resolved *fastest* (mean ≈ 12h vs ≈ 45h for
Low), so raw resolution time is a noisy, partly *inverted* severity cue. We
therefore use it as a **secondary corroborating signal**, not a primary one — it
breaks ties and catches slow-burning issues without dominating the label.

### Ablation — each signal's individual contribution

Run reproduced in `notebook.ipynb` (Cell 13). κ = Cohen's kappa vs. the final
fused pseudo-labels.

| Signal combination | % flagged mismatch | κ vs. final labels |
|---|---:|---:|
| **Rule only** | 40.1% | **0.6559** |
| Resolution only | 79.5% | 0.0889 |
| Rule + Res (0.6 / 0.4) — **chosen** | 59.9% | 0.3579 |
| Rule + Res (0.5 / 0.5) | 65.9% | 0.2890 |

**Pseudo-Label Signal Agreement** (Cohen's κ between the two raw signals):
≈ **−0.04** — the signals are near-independent, which is *expected* given the
inverted resolution-time relationship and confirms they contribute complementary
(not redundant) information.

---

## 3. Stage 2 — Classifier

- **Model:** `microsoft/deberta-v3-small`, fine-tuned with a fresh 2-class head
  (not a frozen zero-shot pipeline).
- **Inputs:** text fields **plus** structured metadata, joined as
  `subject [SEP] description [SEP] channel [SEP] category [SEP] resolution_bin [SEP] satisfaction_bin`.
- **Class imbalance** (≈ 25% mismatch) is handled with **weighted CrossEntropy
  loss** *and* a **WeightedRandomSampler**.
- **Threshold** is tuned on the validation set for best macro-F1 and saved to
  `feature_config.json`.

### Results (held-out test split, 2,000 tickets)

| Metric | Result | Threshold | Status |
|---|---:|---|:--:|
| Binary Accuracy | **0.8450** | ≥ 0.83 | ✅ |
| Macro F1 | **0.8131** | ≥ 0.82 | ⚠️ |
| Recall (Correct) | **0.8342** | ≥ 0.78 | ✅ |
| Recall (Mismatch) | **0.8780** | ≥ 0.78 | ✅ |

> The decision threshold (0.80) and resolution-time quartiles are persisted in
> `models/deberta_final/feature_config.json` so inference reproduces the exact
> bins seen during training.

---

## 4. Stage 3 — Evidence Dossier

Generated for **every flagged ticket** with this schema:

```json
{
  "ticket_id": "TKT-0001",
  "assigned_priority": "Low",
  "inferred_severity": "High",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": "+2",
  "feature_evidence": [
    { "signal": "keyword",         "value": "cannot access", "weight": "0.60", "field": "Ticket_Subject + Ticket_Description" },
    { "signal": "resolution_time", "value": "90.0h",         "weight": "0.40", "field": "Resolution_Time_Hours", "interpretation": "3.0x median (above median 30h)" },
    { "signal": "ticket_channel",  "value": "Email",                           "field": "Ticket_Channel" }
  ],
  "constraint_analysis": "<grounded 2-3 sentence explanation>",
  "confidence": "0.91"
}
```

**Anti-hallucination guarantee:** every `feature_evidence` item carries a `field`
pointing at the exact input column it came from. When a signal does not fire the
value is reported honestly as `"none"` — no keyword, weight, or claim is ever
invented.

---

## 5. Repository layout

```
.
├── notebook.ipynb            Full reproducible pipeline (EDA → pseudo-label → train → eval)
├── train_pipeline.py         Standalone Stage 1 + Stage 2 training script
├── predict.py                Inference: CSV in → predictions CSV + dossiers out
├── app.py                    Streamlit web app (single · batch · dashboard)
├── sia_core.py               Shared logic — single source of truth
├── fix_checkpoint.py         One-time checkpoint key migration (see §6 note)
├── requirements.txt          Pinned dependencies
├── data/
│   └── adversarial_tickets.csv   10 hand-crafted tickets to fool keyword systems (bonus)
└── assets/
    ├── training_curves.png
    └── eda/                  EDA figures
```

---

## 6. How to run

```bash
# 1. install
pip install -r requirements.txt

# 2. train (Stage 1 + Stage 2) — needs the Kaggle CSV
python train_pipeline.py --data data/customer_support_tickets.csv --output models/deberta_final

# 3. batch inference + dossiers
python predict.py --input data/adversarial_tickets.csv --model models/deberta_final \
                  --output predictions.csv --dossiers outputs/dossiers

# 4. web app
streamlit run app.py
```

> **Model artifact:** `models/deberta_final/` (the fine-tuned weights +
> `feature_config.json`) must be present for `predict.py` and `app.py`. Train
> locally, or download the folder produced by `notebook.ipynb` on Kaggle.
>
> **⚠️ transformers version:** the checkpoint must run on `transformers==4.46.3`
> (the version it was trained with). Newer 5.x releases compute DeBERTa-v3
> differently and produce flat ~0.5 output. The pinned `requirements.txt` already
> handles this. A checkpoint saved by an older transformers may use the legacy
> `LayerNorm.gamma/beta` parameter names — run `python fix_checkpoint.py --model
> models/deberta_final` once to migrate them to `.weight/.bias`.

**Dataset:** [Customer Support Tickets — CRM Dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data)
(20,000 rows). Key columns used: `Ticket_Subject`, `Ticket_Description`,
`Priority_Level`, `Ticket_Channel`, `Issue_Category`, `Resolution_Time_Hours`,
`Satisfaction_Score`.

---

## 7. Web app features

- **Single Ticket** — form input → binary judgment + full Evidence Dossier (downloadable).
- **Batch Upload** — CSV upload → per-ticket predictions table + downloadable CSV.
- **Dashboard** — mismatch-type distribution, assigned-priority distribution, a
  **severity-delta heatmap across categories × channels**, and the top flagged tickets.

---

## 8. Evaluation metrics covered

Binary accuracy · Macro-F1 · per-class recall (both classes) · Pseudo-Label Signal
Agreement (Cohen's κ between the two signals) — all reported above and reproducible
from `notebook.ipynb`.
