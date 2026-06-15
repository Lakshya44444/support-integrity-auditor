#!/usr/bin/env python3
"""
sia_core.py — Single source of truth for the Support Integrity Auditor.

Every script (train_pipeline.py, predict.py, app.py) imports the SAME
constants and functions from here, so the "inferred severity", the text
fed to the model, the resolution-time bins, and the dossier are computed
identically everywhere. This is what makes the three deliverables
consistent (no drift between training and inference).
"""

import os
import json

# Priority encoding
PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
NUM_TO_LABEL = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]

# Keyword signals (rule-based NLP)
CRITICAL_PHRASES = [
    "system down", "outage", "data loss", "security breach", "data breach",
    "production down", "cannot access", "complete failure", "total failure",
    "breach", "hacked", "stolen", "phishing", "fraud", "unauthorized",
    "account compromised", "ransomware",
]
HIGH_PHRASES = [
    "urgent", "broken", "failed", "crash", "not working", "corrupted",
    "missing data", "blocked", "locked out", "payment failed",
    "charged twice", "login failed", "cannot login", "2fa issues",
    "screen freezes", "data not syncing",
]
LOW_PHRASES = [
    "how to", "question", "wondering", "inquiry", "feature request",
    "suggestion", "typo", "color", "font", "alignment", "cosmetic",
    "hours of operation", "office location", "faq", "where is", "password reset",
]
NEGATION_WORDS = ["not", "doesn't", "don't", "can't", "never", "doesnt", "dont", "cant"]

# Fusion weights (rule-based NLP vs resolution-time). Justified in README.
W_RULE = 0.6
W_RES = 0.4

# Fallback resolution-time quartiles, used ONLY when feature_config.json
# is missing. The real values are saved at training time — always ship the
# config so inference reproduces the training bins exactly.
DEFAULT_RES_QUARTILES = {"q25": 10.0, "q50": 30.0, "q75": 62.0, "median": 30.0}
DEFAULT_THRESHOLD = 0.80


def to_num(x):
    """Priority label -> int (defaults to Medium=2 for unknown values)."""
    return PRIORITY_MAP.get(str(x).strip(), 2)


# SIGNAL 1 — Rule-based NLP severity
def rule_score(subject, description, assigned_num):
    """Infer severity (1-4) from text keywords + negation."""
    t = (str(subject) + " " + str(description)).lower()
    score = assigned_num
    if any(p in t for p in CRITICAL_PHRASES):
        score = 4
    elif any(p in t for p in HIGH_PHRASES):
        score = 3
    elif any(p in t for p in LOW_PHRASES):
        score = 1
    if any(n in t.split() for n in NEGATION_WORDS):
        score = max(1, score - 1)
    return max(1, min(4, score))


# SIGNAL 2 — Resolution-time severity (quartile proxy)
def res_score(res_time, quartiles):
    """Map resolution hours to a 1-4 severity using training quartiles."""
    if res_time is None:
        res_time = quartiles["median"]
    if res_time <= quartiles["q25"]:
        return 1
    if res_time <= quartiles["q50"]:
        return 2
    if res_time <= quartiles["q75"]:
        return 3
    return 4


def res_bin_label(res_time, quartiles):
    """Categorical bin token used inside the model input text."""
    return {1: "res_low", 2: "res_medium", 3: "res_high", 4: "res_critical"}[
        res_score(res_time, quartiles)
    ]


def sat_bin_label(score):
    """Satisfaction bin token. Matches pd.cut(bins=[0,2,3,4,5]) from training."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 3.0
    if s <= 2:
        return "very_dissatisfied"
    if s <= 3:
        return "dissatisfied"
    if s <= 4:
        return "neutral"
    return "satisfied"


# FUSION — combined "true" severity (independent of assigned label)
def fused_severity(subject, description, assigned_num, res_time, quartiles):
    """Fuse the two signals into a single inferred severity (1-4)."""
    r = rule_score(subject, description, assigned_num)
    v = res_score(res_time, quartiles)
    return max(1, min(4, round(W_RULE * r + W_RES * v)))


def decide_mismatch_type(delta, rule_sc, assigned_num, found_keyword):
    """Schema requires exactly 'Hidden Crisis' or 'False Alarm'."""
    if delta > 0:
        return "Hidden Crisis"
    if delta < 0:
        return "False Alarm"
    # delta == 0 but the model still flagged it: break the tie by the text signal
    if rule_sc > assigned_num or found_keyword:
        return "Hidden Crisis"
    return "False Alarm"


# MODEL INPUT TEXT — must be IDENTICAL to training
def build_text(subject, description, channel, category, res_bin, sat_bin):
    """The exact [SEP]-joined string the DeBERTa model was trained on."""
    return (
        f"{subject} [SEP] {description} [SEP] {channel} [SEP] "
        f"{category} [SEP] {res_bin} [SEP] {sat_bin}"
    )


# CONFIG IO
def save_feature_config(model_dir, quartiles, threshold, model_name):
    os.makedirs(model_dir, exist_ok=True)
    cfg = {
        "res_quartiles": quartiles,
        "threshold": float(threshold),
        "model_name": model_name,
        "fusion_weights": {"rule": W_RULE, "resolution_time": W_RES},
    }
    with open(os.path.join(model_dir, "feature_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # keep the legacy threshold file too (older readers expect it)
    with open(os.path.join(model_dir, "best_threshold.json"), "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    return cfg


def load_feature_config(model_dir):
    """Load quartiles + threshold from a local dir OR a Hugging Face repo id;
    fall back to defaults with a warning."""
    path = os.path.join(model_dir, "feature_config.json")

    # Not a local folder? Treat model_dir as a HF repo id and pull the config.
    if not os.path.isdir(model_dir):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=model_dir, filename="feature_config.json")
        except Exception as e:
            print(f"[sia_core] Could not fetch feature_config.json from HF '{model_dir}': {e}")

    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        return cfg["res_quartiles"], float(cfg.get("threshold", DEFAULT_THRESHOLD))
    # fallback
    thresh = DEFAULT_THRESHOLD
    tpath = os.path.join(model_dir, "best_threshold.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            thresh = float(json.load(f)["threshold"])
    print(
        "[sia_core] WARNING: feature_config.json not found in "
        f"'{model_dir}'. Using default resolution quartiles "
        f"{DEFAULT_RES_QUARTILES} — inference bins may not match training. "
        "Re-run the notebook's config-save cell and ship feature_config.json."
    )
    return dict(DEFAULT_RES_QUARTILES), thresh


# EVIDENCE DOSSIER — every field is traceable to an input column
def build_dossier(ticket_id, subject, description, channel,
                  assigned, inferred, confidence, res_time,
                  quartiles, mismatch_type):
    """Build a hallucination-free dossier. Each evidence item names the
    exact source field, and a value of 'none' is reported honestly when a
    signal did not fire (no fabricated keywords or weights)."""
    text = (str(subject) + " " + str(description)).lower()
    found = [p for p in CRITICAL_PHRASES + HIGH_PHRASES if p in text]
    delta = to_num(inferred) - to_num(assigned)
    median = quartiles["median"]
    ratio = (res_time / median) if median else 0.0

    evidence = [
        {
            "signal": "keyword",
            "value": found[0] if found else "none",
            "matched_count": len(found),
            "weight": f"{W_RULE:.2f}",
            "field": "Ticket_Subject + Ticket_Description",
        },
        {
            "signal": "resolution_time",
            "value": f"{res_time:.1f}h",
            "interpretation": (
                f"{ratio:.1f}x median ({'above' if res_time > median else 'below'} "
                f"median {median:.0f}h)"
            ),
            "weight": f"{W_RES:.2f}",
            "field": "Resolution_Time_Hours",
        },
        {
            "signal": "ticket_channel",
            "value": str(channel),
            "field": "Ticket_Channel",
        },
    ]

    analysis = (
        f"Ticket assigned '{assigned}' but fused signals infer '{inferred}' "
        f"(severity_delta={delta:+d}). Resolution took {res_time:.0f}h "
        f"({'above' if res_time > median else 'below'} the {median:.0f}h median). "
        f"Keyword evidence: {found[0] if found else 'none'}. "
        f"Classified as '{mismatch_type}'."
    )

    return {
        "ticket_id": str(ticket_id),
        "assigned_priority": str(assigned),
        "inferred_severity": str(inferred),
        "mismatch_type": mismatch_type,
        "severity_delta": f"{delta:+d}",
        "feature_evidence": evidence,
        "constraint_analysis": analysis,
        "confidence": str(round(float(confidence), 4)),
    }
