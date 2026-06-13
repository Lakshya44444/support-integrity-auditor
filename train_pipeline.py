#!/usr/bin/env python3
"""
SIA Training Pipeline — Stage 1 (Pseudo-labeling) + Stage 2 (Classifier)
Usage: python train_pipeline.py --data data/customer_support_tickets.csv --output models/deberta_final
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ══════════════════════════════════════════════════════════════
# STAGE 1 — PSEUDO LABEL GENERATION
# ══════════════════════════════════════════════════════════════

def generate_pseudo_labels(df):
    logger.info("[Stage 1] Generating pseudo labels...")

    times         = df["Resolution_Time_Hours"].fillna(df["Resolution_Time_Hours"].median())
    q25, q50, q75 = times.quantile([0.25, 0.50, 0.75])
    time_median   = times.median()

    mismatch_labels, mismatch_types  = [], []
    severity_deltas, inferred_labels = [], []
    rule_scores_list, res_scores_list = [], []

    for _, row in df.iterrows():
        text     = str(row["Ticket_Subject"]) + " " + str(row["Ticket_Description"])
        t        = text.lower()
        assigned = to_num(row["Priority_Level"])

        # Signal 1: Rule-based NLP
        rule_score = assigned
        if any(p in t for p in CRITICAL_PHRASES):   rule_score = 4
        elif any(p in t for p in HIGH_PHRASES):      rule_score = 3
        elif any(p in t for p in LOW_PHRASES):       rule_score = 1
        if any(n in t.split() for n in ["not","doesnt","dont","cant","never"]):
            rule_score = max(1, rule_score - 1)

        # Signal 2: Resolution time
        rh = row["Resolution_Time_Hours"] if pd.notna(row["Resolution_Time_Hours"]) else time_median
        if rh <= q25:   res_score = 1
        elif rh <= q50: res_score = 2
        elif rh <= q75: res_score = 3
        else:           res_score = 4

        rule_scores_list.append(rule_score)
        res_scores_list.append(res_score)

        # Fusion
        rule_gap    = rule_score - assigned
        res_gap     = res_score  - assigned
        same_dir    = rule_gap * res_gap > 0
        rule_strong = abs(rule_gap) >= 2
        res_strong  = abs(res_gap)  >= 2
        both_agree  = abs(rule_gap) >= 1 and abs(res_gap) >= 1 and same_dir
        is_mismatch = rule_strong or (res_strong and abs(rule_gap) >= 1 and same_dir) or both_agree

        if is_mismatch:
            fused = max(1, min(4, round(0.6*rule_score + 0.4*res_score)))
            delta = fused - assigned
            mismatch_labels.append(1)
            mismatch_types.append("Hidden Crisis" if delta > 0 else "False Alarm")
            severity_deltas.append(delta)
            inferred_labels.append(NUM_TO_LABEL[fused])
        else:
            mismatch_labels.append(0)
            mismatch_types.append("Correct")
            severity_deltas.append(0)
            inferred_labels.append(NUM_TO_LABEL[assigned])

    df_out = df.copy()
    df_out["inferred_severity"] = inferred_labels
    df_out["severity_delta"]    = severity_deltas
    df_out["mismatch_label"]    = mismatch_labels
    df_out["mismatch_type"]     = mismatch_types
    df_out["rule_label"]        = [NUM_TO_LABEL[r] for r in rule_scores_list]
    df_out["res_label"]         = [NUM_TO_LABEL[r] for r in res_scores_list]
    df_out["emb_label"]         = [NUM_TO_LABEL[r] for r in rule_scores_list]

    n, total = sum(mismatch_labels), len(mismatch_labels)
    logger.info(f"Mismatch: {n} ({n/total:.1%}) | Correct: {total-n} ({(total-n)/total:.1%})")

    kappa = cohen_kappa_score(rule_scores_list, res_scores_list)
    logger.info(f"Cohen kappa (Rule vs Resolution): {kappa:.4f}")

    return df_out

# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════

class TicketDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.labels = df["mismatch_label"].values

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

        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ══════════════════════════════════════════════════════════════
# STAGE 2 — TRAINING
# ══════════════════════════════════════════════════════════════

def train(args):
    # Load data
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} tickets")

    # Stage 1
    pseudo_df = generate_pseudo_labels(df)
    os.makedirs("data", exist_ok=True)
    pseudo_df.to_csv("data/pseudo_labeled.csv", index=False)
    logger.info("Pseudo labels saved → data/pseudo_labeled.csv")

    # Split
    train_df, temp_df = train_test_split(
        pseudo_df, test_size=0.2,
        stratify=pseudo_df["mismatch_label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df["mismatch_label"], random_state=42
    )
    logger.info(f"Train:{len(train_df)} | Val:{len(val_df)} | Test:{len(test_df)}")

    # Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds  = TicketDataset(train_df, tokenizer)
    val_ds    = TicketDataset(val_df,   tokenizer)
    test_ds   = TicketDataset(test_df,  tokenizer)

    counts  = np.bincount(train_df["mismatch_label"].values)
    weights = 1.0 / counts[train_df["mismatch_label"].values]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model = model.float().to(device)

    counts        = np.bincount(train_df["mismatch_label"].values)
    class_weights = torch.tensor(
        [1.0, counts[0]/counts[1] * 0.9], dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_threshold, patience = 0.0, 0.5, 0
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits.float(), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        vprobs, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                out   = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )
                probs = torch.softmax(out.logits.float(), dim=-1)
                vprobs.extend(probs[:,1].cpu().numpy())
                vl.extend(batch["labels"].numpy())

        best_val_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.30, 0.85, 0.05):
            preds_t = (np.array(vprobs) >= thresh).astype(int)
            f1_t    = f1_score(vl, preds_t, average="macro", zero_division=0)
            if f1_t > best_val_f1:
                best_val_f1 = f1_t
                best_thresh = thresh

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss:{total_loss/len(train_loader):.4f} | "
            f"Val F1:{best_val_f1:.4f} | Thresh:{best_thresh:.2f}"
        )

        if best_val_f1 > best_f1:
            best_f1        = best_val_f1
            best_threshold = best_thresh
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            with open(f"{args.output}/best_threshold.json", "w") as f:
                json.dump({"threshold": float(best_threshold)}, f)
            logger.info(f"  ✓ Best saved (F1:{best_f1:.4f}, Thresh:{best_threshold:.2f})")
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                logger.info("Early stopping!")
                break

    # Test
    model.eval()
    tprobs, tl = [], []
    with torch.no_grad():
        for batch in test_loader:
            out   = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            probs = torch.softmax(out.logits.float(), dim=-1)
            tprobs.extend(probs[:,1].cpu().numpy())
            tl.extend(batch["labels"].numpy())

    tprobs = np.array(tprobs)
    tl     = np.array(tl)
    tp     = (tprobs >= best_threshold).astype(int)

    acc    = accuracy_score(tl, tp)
    f1     = f1_score(tl, tp, average="macro")
    recall = recall_score(tl, tp, average=None)

    logger.info("\n" + "="*55)
    logger.info("FINAL RESULTS")
    logger.info("="*55)
    logger.info(f"Accuracy : {acc:.4f}  ({'✓' if acc>=0.83 else '✗'} Target≥0.83)")
    logger.info(f"Macro F1 : {f1:.4f}  ({'✓' if f1>=0.82 else '✗'} Target≥0.82)")
    logger.info(f"Recall[0]: {recall[0]:.4f}  ({'✓' if recall[0]>=0.78 else '✗'} Target≥0.78)")
    logger.info(f"Recall[1]: {recall[1]:.4f}  ({'✓' if recall[1]>=0.78 else '✗'} Target≥0.78)")
    logger.info("\n" + classification_report(tl, tp, target_names=["Correct","Mismatch"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Training Pipeline")
    parser.add_argument("--data",       type=str,   default="data/customer_support_tickets.csv")
    parser.add_argument("--output",     type=str,   default="models/deberta_final")
    parser.add_argument("--model_name", type=str,   default="microsoft/deberta-v3-small")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
