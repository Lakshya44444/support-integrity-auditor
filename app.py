#!/usr/bin/env python3
"""
SIA Streamlit Dashboard
Run: streamlit run app.py
"""

import os
import json
import html

import pandas as pd
import streamlit as st
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sia_core as core

st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon=":material/policy:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Local folder by default; on Streamlit Cloud set SIA_MODEL_PATH (env var or
# secret) to your Hugging Face repo id (e.g. "lakshya234/sia-deberta-v3").
def _resolve_model_path():
    p = os.environ.get("SIA_MODEL_PATH")
    if p:
        return p
    try:
        if "SIA_MODEL_PATH" in st.secrets:
            return st.secrets["SIA_MODEL_PATH"]
    except Exception:
        pass
    return "models/deberta_final"


MODEL_PATH = _resolve_model_path()

# Plotly shared theme
PLOTLY_FONT = dict(family="Inter, sans-serif", color="#1e293b")


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


# ══════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background:
            radial-gradient(1100px 500px at 50% -10%, #e3ecfb 0%, rgba(227,236,251,0) 60%), #eef2f7; }
        #MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }
        .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1140px; }

        /* Inputs */
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {
            background:#fff !important; border:1px solid #e2e8f0 !important;
            border-radius:10px !important; color:#0f172a !important; }
        [data-baseweb="select"] > div {
            background:#fff !important; border:1px solid #e2e8f0 !important; border-radius:10px !important; }
        [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
            border-color:#2563eb !important; box-shadow:0 0 0 3px rgba(37,99,235,.12) !important; }
        label[data-testid="stWidgetLabel"] p { font-weight:600; color:#334155; font-size:.86rem; }

        /* Tabs as a segmented control */
        .stTabs [data-baseweb="tab-list"] {
            gap:6px; background:#fff; padding:6px; border-radius:12px;
            border:1px solid #e2e8f0; box-shadow:0 1px 2px rgba(15,23,42,.04); }
        .stTabs [data-baseweb="tab"] {
            border-radius:9px; padding:9px 20px; color:#475569; }
        .stTabs [data-baseweb="tab"]:hover { background:#f1f5f9; }
        .stTabs [aria-selected="true"] {
            background:linear-gradient(120deg,#2563eb,#1e3a8a) !important; color:#fff !important; }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

        /* Primary button */
        .stButton button { font-weight:700; border-radius:11px; height:3.1rem; border:none; }
        .stButton button[kind="primary"] {
            background:linear-gradient(120deg,#2563eb,#1e3a8a); color:#fff;
            box-shadow:0 10px 22px -10px rgba(37,99,235,.7); }
        .stButton button[kind="primary"]:hover { filter:brightness(1.07); }
        .stDownloadButton button { border-radius:10px; border:1px solid #cbd5e1; font-weight:600; }

        /* Bordered containers (form card) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background:#fff; border:1px solid #e6ebf3 !important; border-radius:16px;
            box-shadow:0 2px 10px rgba(15,23,42,.05); }

        /* File uploader + dataframe + expander polish */
        [data-testid="stFileUploaderDropzone"] {
            background:#fff; border:1.5px dashed #c7d2e3; border-radius:12px; }
        [data-testid="stExpander"] { border-radius:12px; border:1px solid #e2e8f0; background:#fff; }

        /* Hero */
        .sia-hero {
            background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
            border-radius: 18px; padding: 34px 40px; color: #fff;
            display: flex; align-items: center; gap: 26px;
            box-shadow: 0 18px 40px -18px rgba(37,99,235,.55);
        }
        .sia-hero .icon {
            background: rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25);
            border-radius: 16px; padding: 14px; display:flex; flex-shrink:0;
        }
        .sia-hero h1 { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing:-.5px; }
        .sia-hero p { margin: 6px 0 0; color: #cbd5e1; font-size: 1.02rem; }
        .chips { margin-top: 14px; display:flex; gap:10px; flex-wrap:wrap; }
        .chip { background: rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
                color:#e2e8f0; padding:5px 13px; border-radius:999px; font-size:.8rem; font-weight:600; }

        /* Section heading */
        .sec { font-size: 1.25rem; font-weight: 700; color:#0f172a; margin: 6px 0 2px; }
        .sec-sub { color:#64748b; font-size:.9rem; margin-bottom: 8px; }

        /* Result banner */
        .result { border-radius: 14px; padding: 20px 24px; margin: 6px 0 4px;
                  border:1px solid; display:flex; align-items:center; gap:16px; }
        .result .tag { font-size:.72rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; }
        .result .head { font-size:1.35rem; font-weight:800; margin-top:2px; }
        .result.crisis     { background:#fef2f2; border-color:#fecaca; color:#b91c1c; }
        .result.falsealarm { background:#fffbeb; border-color:#fde68a; color:#b45309; }
        .result.correct    { background:#ecfdf5; border-color:#a7f3d0; color:#047857; }
        .dot { width:14px; height:14px; border-radius:50%; flex-shrink:0; }
        .dot.crisis{background:#dc2626;} .dot.falsealarm{background:#d97706;} .dot.correct{background:#059669;}

        /* Metric cards */
        .mrow { display:grid; grid-template-columns: repeat(4,1fr); gap:14px; margin:14px 0 6px; }
        .mcard { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:16px 18px;
                 box-shadow:0 1px 2px rgba(15,23,42,.04); }
        .mcard .l { color:#64748b; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.4px; }
        .mcard .v { color:#0f172a; font-size:1.6rem; font-weight:800; margin-top:4px; }

        /* Dossier */
        .dossier { background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:22px 24px; margin-top:6px; }
        .ev { border:1px solid #e2e8f0; border-radius:12px; padding:13px 16px; margin-bottom:10px; background:#f8fafc; }
        .ev .sig { font-weight:700; color:#1e3a8a; font-size:.82rem; text-transform:uppercase; letter-spacing:.5px; }
        .ev .val { font-size:1.05rem; font-weight:700; color:#0f172a; margin:2px 0; }
        .ev .meta { color:#64748b; font-size:.84rem; }
        .ev .field { color:#2563eb; font-size:.78rem; font-weight:600; }
        .analysis { background:#f1f5f9; border-left:4px solid #2563eb; border-radius:8px;
                    padding:14px 16px; color:#334155; font-size:.95rem; line-height:1.55; margin-top:6px; }

        .stTabs [data-baseweb="tab"] { font-weight:600; font-size:1rem; }
        .stButton button { font-weight:700; border-radius:10px; height:3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


HERO_SVG = """
<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#ffffff"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  <path d="m9 12 2 2 4-4"/>
</svg>
"""


def render_hero():
    st.markdown(
        f"""
        <div class="sia-hero">
          <div class="icon">{HERO_SVG}</div>
          <div>
            <h1>Support Integrity Auditor</h1>
            <p>Semantics-driven, evidence-grounded detection of priority mismatches in CRM support tickets.</p>
            <div class="chips">
              <span class="chip">Self-supervised pseudo-labels</span>
              <span class="chip">Fine-tuned DeBERTa-v3</span>
              <span class="chip">Hallucination-free dossiers</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(pairs):
    cards = "".join(
        f'<div class="mcard"><div class="l">{html.escape(str(l))}</div>'
        f'<div class="v">{html.escape(str(v))}</div></div>'
        for l, v in pairs
    )
    st.markdown(f'<div class="mrow">{cards}</div>', unsafe_allow_html=True)


def result_banner(is_mismatch, mtype):
    if not is_mismatch:
        st.markdown(
            '<div class="result correct"><span class="dot correct"></span>'
            '<div><div class="tag">Audit result</div>'
            '<div class="head">Priority Correctly Assigned</div></div></div>',
            unsafe_allow_html=True,
        )
        return
    cls = "crisis" if mtype == "Hidden Crisis" else "falsealarm"
    st.markdown(
        f'<div class="result {cls}"><span class="dot {cls}"></span>'
        f'<div><div class="tag">Mismatch detected</div>'
        f'<div class="head">{html.escape(mtype)}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_dossier(d):
    st.markdown('<div class="sec">Evidence Dossier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Every signal is traceable to a source field in the ticket. '
        'Absent signals are reported as "none" — no fabricated evidence.</div>',
        unsafe_allow_html=True,
    )
    ev_html = ""
    for e in d["feature_evidence"]:
        extra = e.get("interpretation") or (f"weight {e['weight']}" if "weight" in e else "")
        if "weight" in e and e.get("interpretation"):
            extra = f"{e['interpretation']} &middot; weight {e['weight']}"
        ev_html += (
            f'<div class="ev"><div class="sig">{html.escape(e["signal"])}</div>'
            f'<div class="val">{html.escape(str(e["value"]))}</div>'
            f'<div class="meta">{html.escape(str(extra))}</div>'
            f'<div class="field">source: {html.escape(e["field"])}</div></div>'
        )
    st.markdown(
        f'<div class="dossier">{ev_html}'
        f'<div class="analysis">{html.escape(d["constraint_analysis"])}</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("View raw JSON dossier"):
        st.json(d)
    st.download_button(
        "Download dossier (JSON)",
        data=json.dumps(d, indent=2),
        file_name=f"{d['ticket_id']}_dossier.json",
        mime="application/json",
        icon=":material/download:",
    )


def style_fig(fig):
    fig.update_layout(
        font=PLOTLY_FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, l=10, r=10, b=10), title_font_size=15,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
inject_css()
render_hero()
st.write("")

tab1, tab2, tab3 = st.tabs(["Single Ticket", "Batch Analysis", "Dashboard"])

# ── TAB 1 — SINGLE TICKET ─────────────────────────────────────
with tab1:
    st.markdown('<div class="sec">Analyze a Single Ticket</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Enter the ticket details and run the auditor.</div>',
                unsafe_allow_html=True)

    with st.container(border=True):
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

    if st.button("Analyze Ticket", type="primary", use_container_width=True,
                 icon=":material/search:"):
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

            st.write("")
            result_banner(is_mismatch, mtype)
            metric_cards([
                ("Assigned Priority", assigned),
                ("Inferred Severity", inferred),
                ("Severity Delta", f"{delta:+d}"),
                ("Confidence", f"{confidence:.1%}"),
            ])

            if is_mismatch:
                dossier = core.build_dossier(
                    ticket_id, subject, description, channel,
                    assigned, inferred, confidence, float(res_time), quartiles, mtype,
                )
                render_dossier(dossier)
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure the model is available at the configured SIA_MODEL_PATH.")

# ── TAB 2 — BATCH ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec">Batch Ticket Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Upload a CSV with columns: Ticket_ID, Ticket_Subject, '
        'Ticket_Description, Priority_Level, Ticket_Channel, Issue_Category, '
        'Resolution_Time_Hours, Satisfaction_Score.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload tickets CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.caption(f"Loaded {len(df)} tickets")
        st.dataframe(df.head(3), use_container_width=True)

        if st.button("Analyze All Tickets", type="primary", use_container_width=True,
                     icon=":material/play_arrow:"):
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
                progress.empty()

                results_df = pd.DataFrame(results)
                n_mismatch = results_df["Mismatch"].eq("Yes").sum()
                st.success(f"Done. {n_mismatch} mismatches found out of {len(df)} tickets.")

                metric_cards([
                    ("Total Tickets", len(df)),
                    ("Mismatches", int(n_mismatch)),
                    ("Hidden Crisis", int((results_df["mismatch_type"] == "Hidden Crisis").sum())),
                    ("False Alarms", int((results_df["mismatch_type"] == "False Alarm").sum())),
                ])
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "Download Predictions CSV",
                    results_df.to_csv(index=False),
                    "predictions.csv", "text/csv", use_container_width=True,
                    icon=":material/download:",
                )
                st.caption("Tip: download this CSV and open the Dashboard tab to visualize it.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB 3 — DASHBOARD ─────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec">Priority Mismatch Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Upload a predictions CSV (from the Batch tab or predict.py).</div>',
                unsafe_allow_html=True)

    dash_file = st.file_uploader("Upload predictions CSV", type=["csv"], key="dashboard")
    if not dash_file:
        st.info("Steps: open Batch Analysis, upload your tickets CSV, click Analyze, "
                "download the predictions CSV, then upload that file here.")
    else:
        df = pd.read_csv(dash_file)
        type_col = "mismatch_type" if "mismatch_type" in df.columns else None
        asgn_col = "Priority_Level" if "Priority_Level" in df.columns else (
            "Assigned" if "Assigned" in df.columns else None)

        if not type_col:
            st.warning("CSV needs a 'mismatch_type' column. Use output from the Batch tab or predict.py.")
        else:
            total = len(df)
            n_mis = int(df[type_col].ne("Correct").sum())
            n_hc = int((df[type_col] == "Hidden Crisis").sum())
            n_fa = int((df[type_col] == "False Alarm").sum())
            metric_cards([
                ("Total Tickets", total),
                ("Mismatches", f"{n_mis}  ({n_mis/total*100:.0f}%)"),
                ("Hidden Crises", n_hc),
                ("False Alarms", n_fa),
            ])
            st.write("")

            col1, col2 = st.columns(2)
            with col1:
                counts = df[type_col].value_counts()
                fig = px.pie(values=counts.values, names=counts.index,
                             title="Mismatch Type Distribution", hole=0.45,
                             color=counts.index,
                             color_discrete_map={"Correct": "#10b981",
                                                 "Hidden Crisis": "#ef4444",
                                                 "False Alarm": "#f59e0b"})
                st.plotly_chart(style_fig(fig), use_container_width=True)
            with col2:
                if asgn_col:
                    counts2 = df[asgn_col].value_counts()
                    fig2 = px.bar(x=counts2.index, y=counts2.values,
                                  title="Assigned Priority Distribution",
                                  labels={"x": "Priority", "y": "Count"},
                                  color=counts2.index,
                                  color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b",
                                                      "High": "#f97316", "Critical": "#ef4444"})
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(style_fig(fig2), use_container_width=True)

            cat_col = "Issue_Category" if "Issue_Category" in df.columns else None
            chan_col = "Ticket_Channel" if "Ticket_Channel" in df.columns else None
            if cat_col and chan_col and "severity_delta" in df.columns:
                st.markdown('<div class="sec">Severity Delta Heatmap — Category x Channel</div>',
                            unsafe_allow_html=True)
                df["severity_delta"] = pd.to_numeric(df["severity_delta"], errors="coerce")
                pivot = df.pivot_table(index=cat_col, columns=chan_col,
                                       values="severity_delta", aggfunc="mean")
                fig3 = px.imshow(pivot, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 color_continuous_midpoint=0, aspect="auto",
                                 labels={"x": "Channel", "y": "Category", "color": "Delta"})
                st.caption("Red = under-prioritised (Hidden Crisis), Blue = over-prioritised (False Alarm)")
                st.plotly_chart(style_fig(fig3), use_container_width=True)

            st.markdown('<div class="sec">Top Flagged Tickets</div>', unsafe_allow_html=True)
            conf_col = "confidence" if "confidence" in df.columns else None
            flagged = df[df[type_col] != "Correct"]
            if conf_col:
                flagged = flagged.sort_values(conf_col, ascending=False)
            st.dataframe(flagged.head(10), use_container_width=True)

    # Model & data insights (real project images)
    st.write("")
    with st.expander("Model & Data Insights"):
        imgs = [
            ("assets/training_curves.png", "Training & validation curves"),
            ("assets/eda/eda_mismatch_dist.png", "Pseudo-label mismatch distribution"),
            ("assets/eda/eda_category_priority.png", "Priority distribution by issue category"),
            ("assets/eda/eda_resolution_time.png", "Resolution time by priority"),
        ]
        shown = [(p, c) for p, c in imgs if os.path.exists(p)]
        if not shown:
            st.caption("Insight images are available in the repository under assets/.")
        else:
            for i in range(0, len(shown), 2):
                cols = st.columns(2)
                for col, (p, c) in zip(cols, shown[i:i + 2]):
                    with col:
                        st.image(p, caption=c, use_container_width=True)
