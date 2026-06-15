"""
Support Integrity Auditor - Streamlit app.

Run locally:
    streamlit run app.py
The model directory defaults to ./models/sia_deberta (override in the sidebar
or with the SIA_MODEL_DIR environment variable). For cloud hosting, point it at
a Hugging Face Hub repo id instead of a local path.
"""
import os
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Make the src/ modules importable whether app.py sits at repo root or in src/.
HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "src"), HERE):
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

from config import normalize_columns, combined_text, ORD_TO_PRIORITY # noqa: E402
from pseudo_labeling import (rule_based_severity, fuse, derive_labels, # noqa: E402
                             DEFAULT_WEIGHTS, SEVERITY_ANCHORS)
from dossier import build_dossier # noqa: E402

DEFAULT_MODEL_DIR = os.environ.get("SIA_MODEL_DIR", "models/sia_deberta")
PRIORITIES = ["Low", "Medium", "High", "Critical"]
TYPE_COLORS = {"Consistent": "#1D9E75", "Hidden Crisis": "#E24B4A",
               "False Alarm": "#EF9F27"}

st.set_page_config(page_title="Support Integrity Auditor", page_icon="check",
                   layout="wide")
st.markdown(
    "<style>.block-container{padding-top:2rem;max-width:1200px}"
    "div[data-testid='stMetricValue']{font-size:1.6rem}</style>",
    unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tiers = sorted(SEVERITY_ANCHORS.keys())
    anchors = []
    for t in tiers:
        v = m.encode(SEVERITY_ANCHORS[t], normalize_embeddings=True)
        anchors.append(np.asarray(v).mean(axis=0))
    A = np.vstack(anchors)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return m, A, np.array(tiers)

@st.cache_resource(show_spinner="Loading classifier...")
def get_classifier(model_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return tok, model, device

def embed_sev(texts):
    m, A, tiers = get_embedder()
    emb = np.asarray(m.encode([str(x)[:2000] for x in texts],
                              normalize_embeddings=True))
    sims = emb @ A.T
    z = (sims - sims.max(axis=1, keepdims=True)) / 0.1
    w = np.exp(z)
    w = w / w.sum(axis=1, keepdims=True)
    return (w * tiers).sum(axis=1)

META_COLS = ["priority", "channel", "ticket_type", "resolution_time"]

def build_input(row):
    parts = [f"{c}: {row.get(c)}" for c in META_COLS
             if row.get(c) is not None and not (isinstance(row.get(c), float)
                                                and pd.isna(row.get(c)))]
    text = "" if pd.isna(row.get("text")) else str(row.get("text"))
    return (" | ".join(parts) + " | " + text).strip(" |")

def label_df(df):
    """Stage 1 in-memory: add evidence columns + inferred severity + mismatch."""
    df = normalize_columns(df.copy())
    df["text"] = df.apply(combined_text, axis=1)
    rp = df["text"].map(rule_based_severity)
    df["sev_rule"] = [p[0] for p in rp]
    df["rule_evidence"] = [json.dumps(p[1]) for p in rp]
    df["sev_embed"] = embed_sev(df["text"].tolist())
    df["sev_rtime"] = np.nan
    w = dict(DEFAULT_WEIGHTS)
    df["sev_fused"] = df.apply(lambda r: fuse(r, w), axis=1)
    return derive_labels(df, mismatch_threshold=2)

def classify_df(df, model_dir, max_len=256, batch_size=64):
    import torch
    tok, model, device = get_classifier(model_dir)
    texts = df.apply(build_input, axis=1).tolist()
    preds, confs = [], []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            prob = torch.softmax(model(**enc).logits, dim=-1)
            preds.extend(prob.argmax(-1).cpu().tolist())
            confs.extend(prob.max(-1).values.cpu().tolist())
    out = df.copy()
    out["pred_mismatch"] = preds
    out["pred_confidence"] = confs
    out["pred_label"] = np.where(np.array(preds) == 1, "Mismatch", "Consistent")
    return out

def show_dossier(d):
    color = TYPE_COLORS.get(d["mismatch_type"], "#888780")
    st.markdown(
        f"<span style='background:{color};color:#fff;padding:2px 10px;"
        f"border-radius:12px;font-size:0.85rem'>{d['mismatch_type']}</span>",
        unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assigned", d["assigned_priority"])
    c2.metric("Inferred", d["inferred_severity"])
    c3.metric("Severity delta", d["severity_delta"])
    c4.metric("Confidence", f"{d['confidence']:.2%}" if d["confidence"] else "-")
    st.write(d["constraint_analysis"])
    with st.expander("Feature evidence (traceable)"):
        st.json(d["feature_evidence"])
    with st.expander("Raw dossier JSON"):
        st.json(d)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Support Integrity Auditor")
st.sidebar.caption("Detects tickets whose content severity conflicts with their "
                   "assigned priority, with a traceable evidence dossier.")
model_dir = st.sidebar.text_input("Model directory / HF repo id", DEFAULT_MODEL_DIR)

st.title("Support Integrity Auditor")

tab_single, tab_batch, tab_dash = st.tabs(
    ["Audit a ticket", "Batch audit", "Dashboard"])

# -----------------------------------------------------------------------------
# Single ticket
# -----------------------------------------------------------------------------
with tab_single:
    st.subheader("Audit a single ticket")
    with st.form("single"):
        subject = st.text_input("Subject", "Login failed - cannot access account")
        description = st.text_area(
            "Description",
            "I have been locked out since this morning and cannot access any of my "
            "data. This is urgent, I have a deadline today.")
        cc = st.columns(4)
        priority = cc[0].selectbox("Assigned priority", PRIORITIES, index=0)
        channel = cc[1].selectbox("Channel",
                                  ["Email", "Chat", "Phone", "Web Form", "Social media"])
        ttype = cc[2].text_input("Issue category", "Account")
        rhours = cc[3].number_input("Resolution hours (optional)",
                                    min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Audit ticket")

    if submitted:
        row = {"ticket_id": "single-ticket", "subject": subject,
               "description": description, "priority": priority,
               "channel": channel, "ticket_type": ttype,
               "resolution_time": rhours if rhours > 0 else np.nan}
        df1 = label_df(pd.DataFrame([row]))
        df1 = classify_df(df1, model_dir)
        r = df1.iloc[0]
        verdict = r["pred_label"]
        if verdict == "Mismatch":
            st.error(f"Priority Mismatch detected ({r['mismatch_type']})")
            show_dossier(build_dossier(r, r["pred_confidence"]))
        else:
            st.success(f"Consistent - content matches the assigned "
                       f"{priority} priority ({r['pred_confidence']:.2%} confidence).")
            st.caption(f"Inferred severity: "
                       f"{ORD_TO_PRIORITY.get(int(r['inferred_ord']))}")

# -----------------------------------------------------------------------------
# Batch audit
# -----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch audit from CSV")
    st.caption("Upload a tickets CSV (subject, description, priority, channel, "
               "issue category, resolution hours). Column names are auto-mapped.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None and st.button("Run batch audit"):
        raw = pd.read_csv(up)
        with st.spinner(f"Auditing {len(raw)} tickets..."):
            res = classify_df(label_df(raw), model_dir)
        st.session_state["results"] = res
        flagged = int((res["pred_mismatch"] == 1).sum())
        st.success(f"Done. {flagged} of {len(res)} tickets flagged as mismatch.")

    if "results" in st.session_state:
        res = st.session_state["results"]
        show_cols = [c for c in ["ticket_id", "priority", "pred_label",
                                 "mismatch_type", "severity_delta", "pred_confidence"]
                     if c in res.columns]
        st.dataframe(res[show_cols], use_container_width=True, height=320)

        dossiers = [build_dossier(r, r["pred_confidence"])
                    for _, r in res[res["pred_mismatch"] == 1].iterrows()]
        d1, d2 = st.columns(2)
        d1.download_button("Download predictions CSV",
                           res[show_cols].to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
        d2.download_button("Download dossiers JSON",
                           json.dumps(dossiers, indent=2).encode(),
                           "dossiers.json", "application/json")

# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
with tab_dash:
    st.subheader("Priority Mismatch Dashboard")
    source = st.session_state.get("results")
    if source is None:
        path = st.text_input("Or load a labeled CSV for the dashboard",
                             "data/tickets_pseudolabeled.csv")
        if os.path.exists(path):
            source = pd.read_csv(path)
            if "pred_label" not in source.columns and "mismatch" in source.columns:
                source["pred_label"] = np.where(source["mismatch"] == 1,
                                                "Mismatch", "Consistent")
        else:
            st.info("Run a batch audit or provide a labeled CSV path to populate "
                    "the dashboard.")

    if source is not None:
        df = source
        total = len(df)
        flagged = int((df.get("pred_label", df.get("mismatch")) == "Mismatch").sum()) \
                  if "pred_label" in df.columns else int((df["mismatch"] == 1).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Tickets audited", f"{total:,}")
        m2.metric("Flagged mismatches", f"{flagged:,}")
        m3.metric("Mismatch rate", f"{flagged / max(total, 1):.1%}")

        col1, col2 = st.columns(2)
        with col1:
            label_col = "pred_label" if "pred_label" in df.columns else None
            if label_col:
                vc = df[label_col].value_counts().reset_index()
                vc.columns = ["label", "count"]
                fig = px.bar(vc, x="label", y="count", color="label",
                             color_discrete_map={"Consistent": "#1D9E75",
                                                 "Mismatch": "#E24B4A"},
                             title="Consistent vs Mismatch")
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "mismatch_type" in df.columns:
                mt = df[df["mismatch_type"] != "Consistent"]["mismatch_type"]
                if len(mt):
                    vc = mt.value_counts().reset_index()
                    vc.columns = ["type", "count"]
                    fig = px.pie(vc, names="type", values="count",
                                 color="type", color_discrete_map=TYPE_COLORS,
                                 title="Mismatch types")
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)

        # Top contributing signals (keyword frequency across flagged tickets).
        if "rule_evidence" in df.columns:
            flagged_mask = (df.get("pred_label") == "Mismatch") \
                           if "pred_label" in df.columns else (df.get("mismatch") == 1)
            counter = Counter()
            for ev in df.loc[flagged_mask, "rule_evidence"].dropna():
                try:
                    parsed = json.loads(ev)
                except Exception:
                    continue
                for terms in parsed.values():
                    counter.update(terms)
            if counter:
                top = pd.DataFrame(counter.most_common(12), columns=["signal", "count"])
                fig = px.bar(top.sort_values("count"), x="count", y="signal",
                             orientation="h", title="Top contributing signals (keywords)")
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

        # Severity-delta heatmap across categories and channels.
        if {"ticket_type", "channel", "severity_delta"}.issubset(df.columns):
            piv = df.pivot_table(index="ticket_type", columns="channel",
                                 values="severity_delta", aggfunc="mean")
            fig = px.imshow(piv, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", origin="lower",
                            title="Mean severity delta by category x channel")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)