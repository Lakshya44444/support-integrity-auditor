#!/usr/bin/env python3
"""
SIA Streamlit Dashboard
Run: streamlit run app.py
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sia_core as core

st.set_page_config(page_title="Support Integrity Auditor", page_icon="🔍", layout="wide")

MODEL_PATH = "models/deberta_final"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).float().eval()
    quartiles, threshold = core.load_feature_config(MODEL_PATH)
    return tokenizer, model, threshold, quartiles


def predict_ticket(tokenizer, model, subject, description, channel,
                   category, res_time, satisfaction, quartiles):
    text = core.build_text(
        subject, description, channel, category,
        core.res_bin_label(res_time, quartiles),
        core.sat_bin_label(satisfaction),
    )
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        probs = torch.softmax(out.logits.float(), dim=-1)
    return float(probs[0][1])


# ── UI ────────────────────────────────────────────────────────
st.title("🔍 Support Integrity Auditor (SIA)")
st.markdown("*Detects priority mismatches in CRM support tickets*")
st.divider()

tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket", "📦 Batch Upload", "📊 Dashboard"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE TICKET
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("Analyze Single Ticket")

    col1, col2 = st.columns(2)
    with col1:
        ticket_id = st.text_input("Ticket ID", value="TKT-0001")
        subject = st.text_input("Subject", value="Login failed - cannot access account")
        description = st.text_area(
            "Description",
            value="I have been unable to login for 2 hours. Getting error 403.",
            height=120,
        )
        assigned = st.selectbox("Assigned Priority", core.PRIORITY_ORDER, index=0)

    with col2:
        channel = st.selectbox("Channel", ["Email", "Chat", "Phone", "Web Form", "Social Media"])
        category = st.selectbox("Category", ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
        res_time = st.slider("Resolution Time (hours)", 1, 200, 48)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)

    if st.button("🔍 Analyze Ticket", type="primary", use_container_width=True):
        try:
            tokenizer, model, threshold, quartiles = load_model()
            confidence = predict_ticket(tokenizer, model, subject, description,
                                        channel, category, res_time, satisfaction, quartiles)
            is_mismatch = confidence >= threshold

            assigned_num = core.to_num(assigned)
            fused = core.fused_severity(subject, description, assigned_num, res_time, quartiles)
            inferred = core.NUM_TO_LABEL[fused]
            delta = fused - assigned_num
            r = core.rule_score(subject, description, assigned_num)
            found = any(p in (subject + " " + description).lower()
                        for p in core.CRITICAL_PHRASES + core.HIGH_PHRASES)
            mtype = (core.decide_mismatch_type(delta, r, assigned_num, found)
                     if is_mismatch else "Correct")

            st.divider()
            if is_mismatch:
                icon = "🔴" if mtype == "Hidden Crisis" else "🟡"
                st.error(f"{icon} **MISMATCH DETECTED — {mtype}**")
            else:
                st.success("✅ **Priority Correctly Assigned**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Assigned Priority", assigned)
            c2.metric("Inferred Severity", inferred)
            c3.metric("Severity Delta", f"{delta:+d}")
            c4.metric("Confidence", f"{confidence:.1%}")

            if is_mismatch:
                dossier = core.build_dossier(
                    ticket_id, subject, description, channel,
                    assigned, inferred, confidence, float(res_time), quartiles, mtype,
                )
                st.subheader("📋 Evidence Dossier")
                st.json(dossier)
                st.download_button(
                    "⬇ Download Dossier JSON",
                    data=json.dumps(dossier, indent=2),
                    file_name=f"{ticket_id}_dossier.json",
                    mime="application/json",
                )
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure the model is available at models/deberta_final/")

# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("Batch CSV Analysis")
    st.info("Upload a CSV with columns: Ticket_ID, Ticket_Subject, Ticket_Description, "
            "Priority_Level, Ticket_Channel, Issue_Category, Resolution_Time_Hours, Satisfaction_Score")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"✓ Loaded **{len(df)}** tickets")
        st.dataframe(df.head(3), use_container_width=True)

        if st.button("🔍 Analyze All Tickets", type="primary", use_container_width=True):
            try:
                tokenizer, model, threshold, quartiles = load_model()
                results = []
                progress = st.progress(0, text="Analyzing...")

                for i, (_, row) in enumerate(df.iterrows()):
                    subj = str(row.get("Ticket_Subject", ""))
                    desc = str(row.get("Ticket_Description", ""))
                    chan = str(row.get("Ticket_Channel", "Email"))
                    cat = str(row.get("Issue_Category", "General Inquiry"))
                    rt = float(row.get("Resolution_Time_Hours", quartiles["median"]))
                    sat = float(row.get("Satisfaction_Score", 3))
                    assigned = str(row.get("Priority_Level", "Medium"))
                    assigned_num = core.to_num(assigned)

                    conf = predict_ticket(tokenizer, model, subj, desc, chan, cat, rt, sat, quartiles)
                    is_m = conf >= threshold
                    fused = core.fused_severity(subj, desc, assigned_num, rt, quartiles)
                    delta = fused - assigned_num
                    r = core.rule_score(subj, desc, assigned_num)
                    found = any(p in (subj + " " + desc).lower()
                                for p in core.CRITICAL_PHRASES + core.HIGH_PHRASES)
                    mtype = core.decide_mismatch_type(delta, r, assigned_num, found) if is_m else "Correct"

                    results.append({
                        "Ticket_ID": row.get("Ticket_ID", f"row_{i}"),
                        "Priority_Level": assigned,
                        "inferred_severity": core.NUM_TO_LABEL[fused],
                        "severity_delta": delta,
                        "Mismatch": "Yes" if is_m else "No",
                        "mismatch_type": mtype,
                        "Issue_Category": cat,
                        "Ticket_Channel": chan,
                        "confidence": round(conf, 4),
                    })
                    progress.progress((i + 1) / len(df), text=f"Analyzing {i+1}/{len(df)}...")

                results_df = pd.DataFrame(results)
                n_mismatch = results_df["Mismatch"].eq("Yes").sum()
                st.success(f"✓ Done! **{n_mismatch}** mismatches found out of {len(df)} tickets")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(df))
                c2.metric("Hidden Crisis", (results_df["mismatch_type"] == "Hidden Crisis").sum())
                c3.metric("False Alarms", (results_df["mismatch_type"] == "False Alarm").sum())

                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "⬇ Download Predictions CSV",
                    results_df.to_csv(index=False),
                    "predictions.csv", "text/csv", use_container_width=True,
                )
                st.caption("Tip: download this CSV and upload it in the Dashboard tab.")
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("Priority Mismatch Dashboard")
    st.info("Upload a predictions CSV (output from the Batch tab or predict.py)")

    dash_file = st.file_uploader("Upload Predictions CSV", type=["csv"], key="dashboard")
    if not dash_file:
        st.markdown("""
        ### How to use:
        1. Go to **Batch Upload**, upload your tickets CSV, click Analyze.
        2. Download the predictions CSV.
        3. Upload that CSV here.
        """)
    else:
        df = pd.read_csv(dash_file)
        type_col = "mismatch_type" if "mismatch_type" in df.columns else None
        asgn_col = "Priority_Level" if "Priority_Level" in df.columns else (
            "Assigned" if "Assigned" in df.columns else None)

        if not type_col:
            st.warning("CSV needs a 'mismatch_type' column. Use output from the Batch tab or predict.py.")
        else:
            total = len(df)
            n_mis = df[type_col].ne("Correct").sum()
            n_hc = (df[type_col] == "Hidden Crisis").sum()
            n_fa = (df[type_col] == "False Alarm").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets", total)
            c2.metric("Mismatches", int(n_mis), f"{n_mis/total*100:.1f}%")
            c3.metric("Hidden Crises", int(n_hc))
            c4.metric("False Alarms", int(n_fa))
            st.divider()

            # Row 1 — distributions
            col1, col2 = st.columns(2)
            with col1:
                counts = df[type_col].value_counts()
                fig = px.pie(values=counts.values, names=counts.index,
                             title="Mismatch Type Distribution", hole=0.4,
                             color=counts.index,
                             color_discrete_map={"Correct": "#2ecc71",
                                                 "Hidden Crisis": "#e74c3c",
                                                 "False Alarm": "#f39c12"})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if asgn_col:
                    counts2 = df[asgn_col].value_counts()
                    fig2 = px.bar(x=counts2.index, y=counts2.values,
                                  title="Assigned Priority Distribution",
                                  labels={"x": "Priority", "y": "Count"},
                                  color=counts2.index,
                                  color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12",
                                                      "High": "#e67e22", "Critical": "#e74c3c"})
                    st.plotly_chart(fig2, use_container_width=True)

            # Row 2 — REQUIRED: severity-delta heatmap across categories x channels
            cat_col = "Issue_Category" if "Issue_Category" in df.columns else None
            chan_col = "Ticket_Channel" if "Ticket_Channel" in df.columns else None
            if cat_col and chan_col and "severity_delta" in df.columns:
                st.subheader("Severity Delta Heatmap — Category × Channel")
                df["severity_delta"] = pd.to_numeric(df["severity_delta"], errors="coerce")
                pivot = df.pivot_table(index=cat_col, columns=chan_col,
                                       values="severity_delta", aggfunc="mean")
                fig3 = px.imshow(pivot, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 color_continuous_midpoint=0, aspect="auto",
                                 title="Mean Severity Delta (inferred − assigned). "
                                       "Red = under-prioritised (Hidden Crisis), Blue = over-prioritised (False Alarm)",
                                 labels={"x": "Channel", "y": "Category", "color": "Δ"})
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("Severity-delta heatmap needs Issue_Category, Ticket_Channel "
                           "and severity_delta columns.")

            # Row 3 — top flagged tickets
            st.subheader("Top Flagged Tickets")
            conf_col = "confidence" if "confidence" in df.columns else None
            flagged = df[df[type_col] != "Correct"]
            if conf_col:
                flagged = flagged.sort_values(conf_col, ascending=False)
            st.dataframe(flagged.head(10), use_container_width=True)
