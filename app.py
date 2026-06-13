#!/usr/bin/env python3
"""
SIA Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🔍",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────
PRIORITY_MAP = {"Low":1, "Medium":2, "High":3, "Critical":4}
NUM_TO_LABEL = {1:"Low", 2:"Medium", 3:"High", 4:"Critical"}
MODEL_PATH   = "models/deberta_final"

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

def to_num(x):
    return PRIORITY_MAP.get(str(x).strip(), 2)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model     = model.float().eval()
    threshold = 0.75
    thresh_path = os.path.join(MODEL_PATH, "best_threshold.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            threshold = json.load(f)["threshold"]
    return tokenizer, model, threshold

# ── Helper Functions ──────────────────────────────────────────
def get_inferred_severity(subject, description, assigned):
    text = (subject + " " + description).lower()
    score = to_num(assigned)
    if any(p in text for p in CRITICAL_PHRASES):  score = 4
    elif any(p in text for p in HIGH_PHRASES):     score = 3
    return NUM_TO_LABEL[max(1, min(4, score))]

def predict_ticket(tokenizer, model, threshold,
                   subject, description, channel,
                   category, res_time, satisfaction):
    res_label = (
        "res_low"      if res_time < 20  else
        "res_medium"   if res_time < 50  else
        "res_high"     if res_time < 100 else
        "res_critical"
    )
    sat_label = (
        "very_dissatisfied" if satisfaction <= 2 else
        "dissatisfied"      if satisfaction == 3 else
        "neutral"           if satisfaction == 4 else
        "satisfied"
    )
    text = (f"{subject} [SEP] {description} [SEP] "
            f"{channel} [SEP] {category} [SEP] "
            f"{res_label} [SEP] {sat_label}")
    enc  = tokenizer(text, truncation=True, padding=True,
                     max_length=256, return_tensors="pt")
    with torch.no_grad():
        out   = model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"])
        probs = torch.softmax(out.logits.float(), dim=-1)
    return float(probs[0][1])

def build_dossier(ticket_id, subject, description, channel,
                  assigned, inferred, confidence, res_time, res_median=50):
    text  = (subject + " " + description).lower()
    found = [p for p in CRITICAL_PHRASES + HIGH_PHRASES if p in text]
    delta = to_num(inferred) - to_num(assigned)
    mtype = "Hidden Crisis" if delta > 0 else "False Alarm"
    return {
        "ticket_id":          ticket_id,
        "assigned_priority":  assigned,
        "inferred_severity":  inferred,
        "mismatch_type":      mtype,
        "severity_delta":     f"{delta:+d}",
        "feature_evidence": [
            {
                "signal": "keyword",
                "value":  found[0] if found else "none",
                "weight": "0.60",
                "field":  "Ticket_Subject + Ticket_Description"
            },
            {
                "signal":         "resolution_time",
                "value":          f"{res_time:.1f}h",
                "interpretation": f"{res_time/res_median:.1f}x median",
                "field":          "Resolution_Time_Hours"
            },
            {
                "signal": "ticket_channel",
                "value":  channel,
                "weight": "0.10",
                "field":  "Ticket_Channel"
            }
        ],
        "constraint_analysis": (
            f"Ticket assigned {assigned} but signals infer {inferred} "
            f"(delta={delta:+d}). Resolution took {res_time:.0f}h. "
            f"Classified as {mtype}."
        ),
        "confidence": str(round(confidence, 4))
    }

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
        subject   = st.text_input("Subject",
                        value="Login failed - cannot access account")
        description = st.text_area("Description",
                        value="I have been unable to login for 2 hours. Getting error 403.",
                        height=120)
        assigned  = st.selectbox("Assigned Priority",
                        ["Low","Medium","High","Critical"], index=0)

    with col2:
        channel      = st.selectbox("Channel",
                            ["Email","Chat","Phone","Web Form"])
        category     = st.selectbox("Category",
                            ["Technical","Billing","Account","General Inquiry","Fraud"])
        res_time     = st.slider("Resolution Time (hours)", 1, 200, 48)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)

    if st.button("🔍 Analyze Ticket", type="primary", use_container_width=True):
        try:
            tokenizer, model, threshold = load_model()
            confidence = predict_ticket(
                tokenizer, model, threshold,
                subject, description, channel,
                category, res_time, satisfaction
            )
            is_mismatch = confidence >= threshold
            inferred    = get_inferred_severity(subject, description, assigned)
            delta       = to_num(inferred) - to_num(assigned)
            mtype       = "Hidden Crisis" if delta > 0 else "False Alarm" if delta < 0 else "Correct"

            st.divider()

            if is_mismatch:
                icon = "🔴" if mtype == "Hidden Crisis" else "🟡"
                st.error(f"{icon} **MISMATCH DETECTED — {mtype}**")
            else:
                st.success("✅ **Priority Correctly Assigned**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Assigned Priority",  assigned)
            c2.metric("Inferred Severity",  inferred)
            c3.metric("Severity Delta",     f"{delta:+d}")
            c4.metric("Confidence",         f"{confidence:.1%}")

            if is_mismatch:
                dossier = build_dossier(
                    ticket_id, subject, description, channel,
                    assigned, inferred, confidence, res_time
                )
                st.subheader("📋 Evidence Dossier")
                st.json(dossier)
                st.download_button(
                    label="⬇ Download Dossier JSON",
                    data=json.dumps(dossier, indent=2),
                    file_name=f"{ticket_id}_dossier.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure model is available at models/deberta_final/")

# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("Batch CSV Analysis")
    st.info("Upload a CSV with columns: Ticket_ID, Ticket_Subject, Ticket_Description, Priority_Level, Ticket_Channel, Issue_Category, Resolution_Time_Hours, Satisfaction_Score")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"✓ Loaded **{len(df)}** tickets")
        st.dataframe(df.head(3), use_container_width=True)

        if st.button("🔍 Analyze All Tickets", type="primary", use_container_width=True):
            try:
                tokenizer, model, threshold = load_model()
                results  = []
                progress = st.progress(0, text="Analyzing...")

                for i, (_, row) in enumerate(df.iterrows()):
                    conf = predict_ticket(
                        tokenizer, model, threshold,
                        str(row.get("Ticket_Subject","")),
                        str(row.get("Ticket_Description","")),
                        str(row.get("Ticket_Channel","Email")),
                        str(row.get("Issue_Category","General Inquiry")),
                        float(row.get("Resolution_Time_Hours", 50)),
                        float(row.get("Satisfaction_Score", 3))
                    )
                    is_m     = conf >= threshold
                    inferred = get_inferred_severity(
                        str(row.get("Ticket_Subject","")),
                        str(row.get("Ticket_Description","")),
                        str(row.get("Priority_Level","Medium"))
                    )
                    delta = to_num(inferred) - to_num(str(row.get("Priority_Level","Medium")))
                    mtype = (
                        "Hidden Crisis" if (is_m and delta > 0) else
                        "False Alarm"   if (is_m and delta < 0) else
                        "Correct"
                    )
                    results.append({
                        "Ticket_ID":   row.get("Ticket_ID",""),
                        "Assigned":    row.get("Priority_Level",""),
                        "Inferred":    inferred,
                        "Delta":       f"{delta:+d}",
                        "Mismatch":    "Yes" if is_m else "No",
                        "Type":        mtype,
                        "Confidence":  round(conf, 4)
                    })
                    progress.progress((i+1)/len(df),
                        text=f"Analyzing {i+1}/{len(df)}...")

                results_df = pd.DataFrame(results)
                n_mismatch = results_df["Mismatch"].eq("Yes").sum()

                st.success(f"✓ Done! **{n_mismatch}** mismatches found out of {len(df)} tickets")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total",         len(df))
                c2.metric("Hidden Crisis", (results_df["Type"]=="Hidden Crisis").sum())
                c3.metric("False Alarms",  (results_df["Type"]=="False Alarm").sum())

                st.dataframe(results_df, use_container_width=True)

                st.download_button(
                    "⬇ Download Predictions CSV",
                    results_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("Priority Mismatch Dashboard")
    st.info("Upload a predictions CSV (output from Batch Upload or predict.py)")

    dash_file = st.file_uploader("Upload Predictions CSV",
                                  type=["csv"], key="dashboard")

    if dash_file:
        df = pd.read_csv(dash_file)

        type_col = "Type" if "Type" in df.columns else "mismatch_type"
        asgn_col = "Assigned" if "Assigned" in df.columns else "Priority_Level"

        if type_col in df.columns:
            # Metrics
            total     = len(df)
            n_mis     = df[type_col].ne("Correct").sum()
            n_hc      = (df[type_col] == "Hidden Crisis").sum()
            n_fa      = (df[type_col] == "False Alarm").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets",  total)
            c2.metric("Mismatches",     n_mis, f"{n_mis/total*100:.1f}%")
            c3.metric("Hidden Crises",  n_hc)
            c4.metric("False Alarms",   n_fa)

            st.divider()

            # Row 1 — Pie + Bar
            col1, col2 = st.columns(2)

            with col1:
                counts = df[type_col].value_counts()
                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    title="Mismatch Type Distribution",
                    color_discrete_map={
                        "Correct":       "#2ecc71",
                        "Hidden Crisis": "#e74c3c",
                        "False Alarm":   "#f39c12"
                    },
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                counts2 = df[asgn_col].value_counts()
                fig2    = px.bar(
                    x=counts2.index,
                    y=counts2.values,
                    title="Assigned Priority Distribution",
                    labels={"x":"Priority","y":"Count"},
                    color=counts2.index,
                    color_discrete_map={
                        "Low":"#2ecc71","Medium":"#f39c12",
                        "High":"#e67e22","Critical":"#e74c3c"
                    }
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Row 2 — Heatmap
            if "Inferred" in df.columns or "inferred_severity" in df.columns:
                inf_col = "Inferred" if "Inferred" in df.columns else "inferred_severity"
                st.subheader("Severity Delta Heatmap")
                order = ["Low","Medium","High","Critical"]
                pivot = pd.crosstab(df[asgn_col], df[inf_col])
                pivot = pivot.reindex(index=order, columns=order, fill_value=0)
                fig3  = px.imshow(
                    pivot,
                    text_auto=True,
                    color_continuous_scale="RdYlGn_r",
                    title="Assigned Priority vs Inferred Severity",
                    labels={"x":"Inferred Severity","y":"Assigned Priority"}
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Row 3 — Top mismatched tickets
            st.subheader("Top Flagged Tickets")
            conf_col = "Confidence" if "Confidence" in df.columns else "confidence"
            if conf_col in df.columns:
                top = (df[df[type_col] != "Correct"]
                       .sort_values(conf_col, ascending=False)
                       .head(10))
                st.dataframe(top, use_container_width=True)
        else:
            st.warning("Upload a valid predictions CSV with mismatch type column.")
    else:
        st.markdown("""
        ### How to use Dashboard:
        1. Go to **Batch Upload** tab
        2. Upload your tickets CSV
        3. Click Analyze
        4. Download predictions CSV
        5. Come back here and upload that CSV
        """)
