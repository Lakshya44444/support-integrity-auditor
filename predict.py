#!/usr/bin/env python3
"""
SIA Inference Script
Usage: python predict.py --input test_tickets.csv --model models/deberta_final --output predictions.csv --dossiers outputs/dossiers
"""

import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Constants ─────────────────────────────────────────────────
PRIORITY_MAP = {"Low":1, "Medium":2, "High":3, "Critical":4}
NUM_TO_LABEL = {1:"Low", 2:"Medium", 3:"High", 4:"Critical"}

CRITICAL_PHRASES = [
    "system down","outage","data loss","security breach","data breach",
    "production down","cannot access","complete failure","total failure",
    "breach","hacked","stolen","phishing","fraud","unauthorized","ransomware"
]
HIGH_PHRASES = [
    "urgent","broken","failed","crash","not working","corrupted",
    "missing data","blocked","locked out","payment failed",
    "charged twice","login failed","cannot login","2fa issues",
    "screen freezes","data not syncing"
]
LOW_PHRASES = [
    "how to","question","wondering","inquiry","feature request",
    "suggestion","typo","color","font","alignment","cosmetic",
    "hours of operation","office location","faq","where is","password reset"
]

def to_num(x):
    return PRIORITY_MAP.get(str(x).strip(), 2)

def get_inferred_severity(subject, description, assigned_priority):
    text       = (subject + " " + description).lower()
    rule_score = to_num(assigned_priority)
    if any(p in text for p in CRITICAL_PHRASES):   rule_score = 4
    elif any(p in text for p in HIGH_PHRASES):      rule_score = 3
    elif any(p in text for p in LOW_PHRASES):       rule_score = 1
    if any(n in text.split() for n in ["not","doesnt","dont","cant","never"]):
        rule_score = max(1, rule_score - 1)
    return NUM_TO_LABEL[max(1, min(4, rule_score))]

# ══════════════════════════════════════════════════════════════
# DOSSIER GENERATION
# ══════════════════════════════════════════════════════════════

def generate_dossier(row, confidence, time_median):
    text     = str(row["Ticket_Subject"]) + " " + str(row["Ticket_Description"])
    res_time = float(row["Resolution_Time_Hours"]) if pd.notna(row["Resolution_Time_Hours"]) else 0.0
    found    = [p for p in CRITICAL_PHRASES + HIGH_PHRASES if p in text.lower()]
    delta    = int(row["severity_delta"])

    return {
        "ticket_id":         str(row["Ticket_ID"]),
        "assigned_priority": str(row["Priority_Level"]),
        "inferred_severity": str(row["inferred_severity"]),
        "mismatch_type":     str(row["mismatch_type"]),
        "severity_delta":    f"{delta:+d}",
        "feature_evidence":  [
            {
                "signal": "keyword",
                "value":  found[0] if found else "none",
                "weight": "0.60",
                "field":  "Ticket_Subject + Ticket_Description"
            },
            {
                "signal":         "resolution_time",
                "value":          f"{res_time:.1f}h",
                "interpretation": f"{res_time/time_median:.1f}x median ({'above' if res_time > time_median else 'below'})",
                "field":          "Resolution_Time_Hours"
            },
            {
                "signal": "ticket_channel",
                "value":  str(row["Ticket_Channel"]),
                "weight": "0.10",
                "field":  "Ticket_Channel"
            }
        ],
        "constraint_analysis": (
            f"Ticket '{row['Ticket_Subject']}' assigned {row['Priority_Level']} "
            f"but signals infer {row['inferred_severity']} (delta={delta:+d}). "
            f"Resolution took {res_time:.0f}h "
            f"({'above' if res_time > time_median else 'below'} median of {time_median:.0f}h). "
            f"Classified as '{row['mismatch_type']}'."
        ),
        "confidence": str(round(confidence, 4))
    }

# ══════════════════════════════════════════════════════════════
# MAIN PREDICT
# ══════════════════════════════════════════════════════════════

def predict(args):
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} tickets")

    # Load model
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model)
    model     = model.float().to(device)
    model.eval()
    print(f"Model loaded | Device: {device}")

    # Load threshold
    thresh_path = os.path.join(args.model, "best_threshold.json")
    threshold   = 0.75
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            threshold = json.load(f)["threshold"]
    print(f"Threshold: {threshold}")

    # Prepare texts
    res_bins = pd.qcut(
        df["Resolution_Time_Hours"].fillna(0), q=4,
        labels=["res_low","res_medium","res_high","res_critical"],
        duplicates="drop"
    ).astype(str)

    sat_bin = pd.cut(
        df["Satisfaction_Score"].fillna(3),
        bins=[0,2,3,4,5],
        labels=["very_dissatisfied","dissatisfied","neutral","satisfied"]
    ).astype(str)

    texts = (
        df["Ticket_Subject"].fillna("")     + " [SEP] " +
        df["Ticket_Description"].fillna("") + " [SEP] " +
        df["Ticket_Channel"].fillna("")     + " [SEP] " +
        df["Issue_Category"].fillna("")     + " [SEP] " +
        res_bins                            + " [SEP] " +
        sat_bin
    ).tolist()

    # Batch inference
    all_probs  = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, padding=True,
            max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            out   = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device)
            )
            probs = torch.softmax(out.logits.float(), dim=-1)
            all_probs.extend(probs[:,1].cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)}")

    all_probs = np.array(all_probs)
    preds     = (all_probs >= threshold).astype(int)

    # Add predictions
    df["mismatch_predicted"] = preds
    df["confidence"]         = all_probs.round(4)
    df["inferred_severity"]  = [
        get_inferred_severity(
            str(row["Ticket_Subject"]),
            str(row["Ticket_Description"]),
            str(row["Priority_Level"])
        )
        for _, row in df.iterrows()
    ]
    df["severity_delta"] = df.apply(
        lambda row: to_num(row["inferred_severity"]) - to_num(row["Priority_Level"]),
        axis=1
    )
    df["mismatch_type"] = df.apply(
        lambda row: (
            "Hidden Crisis" if (row["mismatch_predicted"]==1 and row["severity_delta"] > 0)
            else "False Alarm" if (row["mismatch_predicted"]==1 and row["severity_delta"] < 0)
            else "Correct"
        ),
        axis=1
    )

    # Save predictions
    df.to_csv(args.output, index=False)
    print(f"\n✓ Predictions saved → {args.output}")

    # Generate dossiers
    os.makedirs(args.dossiers, exist_ok=True)
    time_median = df["Resolution_Time_Hours"].median()
    dossiers    = []

    for i, (_, row) in enumerate(df.iterrows()):
        if row["mismatch_predicted"] != 1:
            continue
        dossier = generate_dossier(row, float(all_probs[i]), time_median)
        dossiers.append(dossier)
        with open(f"{args.dossiers}/{row['Ticket_ID']}.json", "w") as f:
            json.dump(dossier, f, indent=2)

    print(f"✓ Dossiers saved: {len(dossiers)} → {args.dossiers}/")

    # Summary
    print("\n" + "="*45)
    print("PREDICTION SUMMARY")
    print("="*45)
    print(f"Total tickets  : {len(df)}")
    print(f"Mismatches     : {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"Hidden Crisis  : {(df['mismatch_type']=='Hidden Crisis').sum()}")
    print(f"False Alarms   : {(df['mismatch_type']=='False Alarm').sum()}")
    print(f"Correct        : {(df['mismatch_type']=='Correct').sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    parser.add_argument("--input",    type=str, required=True,                  help="Input CSV path")
    parser.add_argument("--model",    type=str, default="models/deberta_final", help="Model directory")
    parser.add_argument("--output",   type=str, default="predictions.csv",      help="Output CSV path")
    parser.add_argument("--dossiers", type=str, default="outputs/dossiers",     help="Dossiers directory")
    args = parser.parse_args()
    predict(args)
