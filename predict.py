#!/usr/bin/env python3
"""
SIA Inference Script — Stage 2 (classify) + Stage 3 (dossier)
Usage:
  python predict.py --input test_tickets.csv --model models/deberta_final \
                    --output predictions.csv --dossiers outputs/dossiers

All severity/text/bin/dossier logic comes from sia_core, so predictions
and dossiers are computed exactly as in training.
"""

import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sia_core as core


def build_texts(df, quartiles):
    texts = []
    for _, row in df.iterrows():
        rt = row.get("Resolution_Time_Hours", quartiles["median"])
        rt = quartiles["median"] if pd.isna(rt) else float(rt)
        texts.append(core.build_text(
            str(row.get("Ticket_Subject", "")),
            str(row.get("Ticket_Description", "")),
            str(row.get("Ticket_Channel", "")),
            str(row.get("Issue_Category", "")),
            core.res_bin_label(rt, quartiles),
            core.sat_bin_label(row.get("Satisfaction_Score", 3)),
        ))
    return texts


def predict(args):
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} tickets")

    quartiles, threshold = core.load_feature_config(args.model)
    print(f"Resolution quartiles: {quartiles} | Threshold: {threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).float().to(device)
    model.eval()
    print(f"Model loaded | Device: {device}")

    texts = build_texts(df, quartiles)

    all_probs = []
    bs = 32
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], truncation=True, padding=True,
                        max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            probs = torch.softmax(out.logits.float(), dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy())
        if (i // bs + 1) % 10 == 0:
            print(f"  Processed {min(i + bs, len(texts))}/{len(texts)}")

    all_probs = np.array(all_probs)
    preds = (all_probs >= threshold).astype(int)

    # Per-ticket inferred severity (fused, identical to training)
    inferred, deltas, mtypes = [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        assigned_num = core.to_num(row.get("Priority_Level", "Medium"))
        rt = row.get("Resolution_Time_Hours", quartiles["median"])
        rt = quartiles["median"] if pd.isna(rt) else float(rt)
        fused = core.fused_severity(row.get("Ticket_Subject", ""),
                                    row.get("Ticket_Description", ""),
                                    assigned_num, rt, quartiles)
        delta = fused - assigned_num
        inferred.append(core.NUM_TO_LABEL[fused])
        deltas.append(delta)
        if preds[i] == 1:
            r = core.rule_score(row.get("Ticket_Subject", ""),
                                row.get("Ticket_Description", ""), assigned_num)
            found = any(p in (str(row.get("Ticket_Subject", "")) + " " +
                              str(row.get("Ticket_Description", ""))).lower()
                        for p in core.CRITICAL_PHRASES + core.HIGH_PHRASES)
            mtypes.append(core.decide_mismatch_type(delta, r, assigned_num, found))
        else:
            mtypes.append("Correct")

    df["mismatch_predicted"] = preds
    df["confidence"] = all_probs.round(4)
    df["inferred_severity"] = inferred
    df["severity_delta"] = deltas
    df["mismatch_type"] = mtypes

    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved -> {args.output}")

    # Stage 3 — dossiers for flagged tickets only
    os.makedirs(args.dossiers, exist_ok=True)
    dossiers = []
    for i, (_, row) in enumerate(df.iterrows()):
        if row["mismatch_predicted"] != 1:
            continue
        rt = row.get("Resolution_Time_Hours", quartiles["median"])
        rt = quartiles["median"] if pd.isna(rt) else float(rt)
        dossier = core.build_dossier(
            row.get("Ticket_ID", f"row_{i}"),
            row.get("Ticket_Subject", ""), row.get("Ticket_Description", ""),
            row.get("Ticket_Channel", ""), row.get("Priority_Level", ""),
            row["inferred_severity"], float(all_probs[i]), rt,
            quartiles, row["mismatch_type"],
        )
        dossiers.append(dossier)
        with open(f"{args.dossiers}/{row.get('Ticket_ID', f'row_{i}')}.json", "w") as f:
            json.dump(dossier, f, indent=2)

    print(f"Dossiers saved: {len(dossiers)} -> {args.dossiers}/")

    print("\n" + "=" * 45)
    print("PREDICTION SUMMARY")
    print("=" * 45)
    print(f"Total tickets  : {len(df)}")
    print(f"Mismatches     : {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"Hidden Crisis  : {(df['mismatch_type']=='Hidden Crisis').sum()}")
    print(f"False Alarms   : {(df['mismatch_type']=='False Alarm').sum()}")
    print(f"Correct        : {(df['mismatch_type']=='Correct').sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--model", type=str, default="models/deberta_final", help="Model directory")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--dossiers", type=str, default="outputs/dossiers", help="Dossiers directory")
    args = parser.parse_args()
    predict(args)
