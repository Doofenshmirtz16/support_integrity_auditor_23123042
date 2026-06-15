"""
Stage 1 - Self-supervised pseudo-label generation.

Infers a "true" severity for every ticket WITHOUT looking at the human-assigned
Ticket Priority, then derives a binary mismatch label by comparing the two.

Three independent signals are fused (the spec requires >= 2):
  1. rule  - rule-based NLP severity (tiered escalation lexicon + negation)
  2. rtime - resolution-time severity (quantile binning; longer => more severe)
  3. llm   - zero-shot severity from Phi-3-mini (optional, --use_llm)

Every per-ticket signal value is preserved in the output CSV so that Stage 3
can build a fully traceable, hallucination-free Evidence Dossier from real
field values rather than free-form generation.

Usage:
    python src/pseudo_labeling.py --input data/tickets.csv \
        --output data/tickets_pseudolabeled.csv --use_llm
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys

import numpy as np
import pandas as pd

from config import (
    normalize_columns, combined_text, priority_to_ord, ORD_TO_PRIORITY,
)

# -----------------------------------------------------------------------------
# Signal 1: rule-based NLP severity
# -----------------------------------------------------------------------------
# Each tier maps to a target severity value on the 0-3 scale. Terms are matched
# as whole words / phrases so "down" doesn't fire inside "download".
TIERS = {
    "critical": (3.0, [
        "data loss", "lost all", "lost everything", "cannot access", "can't access",
        "unable to access", "completely down", "system down", "service down", "outage",
        "security breach", "breach", "hacked", "fraud", "unauthorized", "charged twice",
        "double charged", "money", "legal action", "lawsuit", "urgent", "asap",
        "immediately", "emergency", "critical", "production down", "not working at all",
        "complete failure", "escalate",
    ]),
    "high": (2.3, [
        "broken", "error", "fails", "failed", "failing", "not working", "crash",
        "crashed", "crashing", "freezes", "frozen", "still not", "again", "third time",
        "repeatedly", "frustrated", "angry", "disappointed", "unacceptable", "refund",
        "cancel", "cancellation", "stuck", "blocked", "downtime",
    ]),
    "medium": (1.4, [
        "problem", "issue", "trouble", "slow", "delay", "delayed", "help", "question",
        "how do i", "how to", "not sure", "confused", "intermittent", "sometimes",
    ]),
    "low": (0.4, [
        "minor", "small", "suggestion", "feedback", "wondering", "curious",
        "no rush", "whenever you can", "low priority", "cosmetic", "typo", "nice to have",
    ]),
}

NEGATORS = {"no", "not", "never", "without", "isn't", "wasn't", "don't", "doesn't",
            "didn't", "won't", "can", "cannot resolve"}

_WORD = re.compile(r"[a-z]+")

def _matches(text: str, term: str) -> bool:
    """Whole-word/phrase containment using word boundaries."""
    return re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text) is not None

def _is_negated(text: str, term: str) -> bool:
    """True if a negator appears within 3 tokens before the term's first hit."""
    idx = text.find(term)
    if idx < 0:
        return False
    before = _WORD.findall(text[:idx])[-3:]
    return any(b in NEGATORS for b in before)

def rule_based_severity(text: str):
    """Return (severity_0_3, evidence_dict). evidence maps tier -> matched terms."""
    t = (text or "").lower()
    evidence: dict[str, list[str]] = {}
    components = []  # (tier_value, count)
    for tier, (value, terms) in TIERS.items():
        matched = []
        for term in terms:
            if _matches(t, term) and not _is_negated(t, term):
                matched.append(term)
        if matched:
            evidence[tier] = matched
            components.append((value, len(matched)))
            
    if not components:
        return 1.5, {}  # neutral when no lexical evidence
        
    # Weight tiers by sqrt(count) so many weak hits don't outweigh one strong hit.
    num = sum(v * math.sqrt(c) for v, c in components)
    den = sum(math.sqrt(c) for v, c in components)
    sev = num / den
    # A genuine critical phrase pulls the floor up (hidden-crisis sensitivity).
    if "critical" in evidence:
        sev = max(sev, 2.5)
    return float(np.clip(sev, 0.0, 3.0)), evidence

# -----------------------------------------------------------------------------
# Signal 2: resolution-time severity
# -----------------------------------------------------------------------------
def _to_hours(value) -> float:
    """Best-effort parse of a resolution-time cell into hours."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return float("nan")
    # Plain number => assume hours.
    try:
        return float(s)
    except ValueError:
        pass
    # HH:MM:SS or MM:SS
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return float("nan")
        if len(parts) == 3:
            return parts[0] + parts[1] / 60 + parts[2] / 3600
        if len(parts) == 2:
            return parts[0] + parts[1] / 60
    # pandas timedelta string (e.g. "2 days 03:00:00")
    try:
        return pd.to_timedelta(s).total_seconds() / 3600.0
    except Exception:
        return float("nan")

def resolution_time_severity(df: pd.DataFrame):
    """Vectorised: longer resolution time => higher severity (quantile bins).
    
    Returns a Series of severity values 0-3 (NaN where the time is missing).
    Assumption documented in the README: harder/severe tickets take longer.
    """
    if "resolution_time" not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    hours = df["resolution_time"].map(_to_hours)
    valid = hours.dropna()
    if valid.nunique() < 4:
        # Not enough spread to bin meaningfully.
        return pd.Series([float("nan")] * len(df), index=df.index)
    # Quantile bins -> ordinal 0..3, mapped to tier-centre severities.
    try:
        bins = pd.qcut(hours, q=4, labels=[0.4, 1.4, 2.3, 3.0], duplicates="drop")
        return bins.astype(float)
    except Exception:
        return pd.Series([float("nan")] * len(df), index=df.index)

# -----------------------------------------------------------------------------
# Signal 3: LLM zero-shot severity (optional)
# -----------------------------------------------------------------------------
_LLM_LABELS = {"low": 0.4, "medium": 1.4, "high": 2.3, "critical": 3.0}

def llm_zero_shot_severity(texts, model_name="microsoft/Phi-3-mini-4k-instruct",
                           batch_size=16, max_rows=None):
    """Zero-shot severity scoring with a small instruct model in 4-bit.
    
    Returns a list of severity floats (NaN if the model couldn't be loaded).
    """
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
    except Exception as e:  # pragma: no cover
        print(f"[llm] transformers/torch unavailable ({e}); skipping LLM signal.")
        return [float("nan")] * len(texts)

    print(f"[llm] loading {model_name} in 4-bit ...")
    try:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.float16, trust_remote_code=True)
        model.eval()
    except Exception as e:  # pragma: no cover
        print(f"[llm] could not load model ({e}). This path needs an NVIDIA/CUDA "
              f"GPU; on AMD/CPU use the embedding signal instead. Skipping LLM.")
        return [float("nan")] * len(texts)

    sys_prompt = ("You triage customer-support tickets. Classify the SEVERITY of "
                  "the underlying issue, ignoring any stated priority. Answer with "
                  "exactly one word: Low, Medium, High, or Critical.")
    out = []
    n = len(texts) if max_rows is None else min(max_rows, len(texts))
    for i in range(0, n, batch_size):
        chunk = texts[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": f"Ticket: {str(x)[:1200]}\nSeverity:"}],
            tokenize=False, add_generation_prompt=True) for x in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for j, g in enumerate(gen):
            text = tok.decode(g[enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).lower()
            sev = float("nan")
            for label, val in _LLM_LABELS.items():
                if label in text:
                    sev = val
                    break
            out.append(sev)
        print(f"[llm] {min(i + batch_size, n)}/{n}", end="\r")
    print()
    out.extend([float("nan")] * (len(texts) - n))
    return out

# -----------------------------------------------------------------------------
# Signal 4: embedding-based semantic severity (CPU-friendly)
# -----------------------------------------------------------------------------
# Short prototype phrases for each severity tier. Each ticket is scored by its
# semantic similarity to these anchors -> a label-free, embedding-based signal
# (semantic urgency grouping). Runs comfortably on CPU.
SEVERITY_ANCHORS = {
    3.0: [
        "critical emergency, system completely down, data loss, urgent escalation",
        "security breach, fraudulent charge, cannot access anything, immediate action required",
    ],
    2.3: [
        "something is broken or failing, repeated errors, app keeps crashing, customer frustrated",
        "service not working, blocked from completing a task, refund requested, persistent failure",
    ],
    1.4: [
        "general question or minor problem, slow performance, need help understanding a feature",
        "intermittent issue, asking how to do something, mild inconvenience",
    ],
    0.4: [
        "minor cosmetic issue, small suggestion or feedback, low priority, no rush, typo",
        "nice-to-have improvement, general curiosity, non-urgent comment",
    ],
}

def embedding_severity(texts, model_name="sentence-transformers/all-MiniLM-L6-v2",
                       batch_size=64, temperature=0.1):
    """Severity 0-3 from cosine similarity to per-tier prototype embeddings.
    
    Returns a list of floats (NaN if sentence-transformers is unavailable).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        print(f"[embed] sentence-transformers unavailable ({e}); skipping signal.")
        return [float("nan")] * len(texts)

    print(f"[embed] encoding with {model_name} ...")
    model = SentenceTransformer(model_name)
    tiers = sorted(SEVERITY_ANCHORS.keys())  # [0.4, 1.4, 2.3, 3.0]
    anchors = []
    for t in tiers:
        vecs = model.encode(SEVERITY_ANCHORS[t], normalize_embeddings=True)
        anchors.append(np.asarray(vecs).mean(axis=0))
    anchors = np.vstack(anchors)
    anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)

    emb = model.encode([str(x)[:2000] for x in texts], normalize_embeddings=True,
                       batch_size=batch_size, show_progress_bar=True)
    emb = np.asarray(emb)
    sims = emb @ anchors.T                      # cosine (n_tickets, 4)
    z = (sims - sims.max(axis=1, keepdims=True)) / max(temperature, 1e-6)
    w = np.exp(z)
    w = w / w.sum(axis=1, keepdims=True)        # soft tier weights
    sev = (w * np.asarray(tiers)).sum(axis=1)
    return [float(s) for s in sev]

# -----------------------------------------------------------------------------
# Fusion + label derivation
# -----------------------------------------------------------------------------
# Default fusion: the two semantic signals. rtime is computed and reported but
# excluded from fusion by default (near-chance agreement on this dataset); enable
# it with use_rtime=True. The LLM signal is added when use_llm=True.
DEFAULT_WEIGHTS = {"rule": 0.4, "embed": 0.6}

def fuse(row, weights):
    """Weighted mean over the signals present for this row -> inferred severity."""
    num, den = 0.0, 0.0
    for sig, w in weights.items():
        v = row.get(f"sev_{sig}")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            num += w * v
            den += w
    if den == 0:
        return 1.5
    return num / den

def derive_labels(df, mismatch_threshold=2):
    """Add inferred_severity, severity_delta, mismatch_type and mismatch label."""
    df = df.copy()
    df["inferred_severity_score"] = df["sev_fused"].round(3)
    df["inferred_ord"] = df["sev_fused"].round().clip(0, 3).astype(int)
    df["assigned_ord"] = df["priority"].map(priority_to_ord)
    df["severity_delta"] = df["inferred_ord"] - df["assigned_ord"]

    def label_row(d):
        if pd.isna(d):
            return 0, "Consistent"
        if d >= mismatch_threshold:
            return 1, "Hidden Crisis"    # true severity HIGHER than assigned
        if d <= -mismatch_threshold:
            return 1, "False Alarm"      # true severity LOWER than assigned
        return 0, "Consistent"

    labels = df["severity_delta"].map(label_row)
    df["mismatch"] = [a for a, _ in labels]
    df["mismatch_type"] = [b for _, b in labels]
    return df

# -----------------------------------------------------------------------------
# Ablation / signal agreement
# -----------------------------------------------------------------------------
def signal_agreement(df, weights):
    """Pairwise agreement over ALL computed signals (so excluded ones are still
    documented), and leave-one-out ablation over the signals actually fused."""
    all_signals = ["rule", "embed", "rtime", "llm"]
    available = [s for s in all_signals
                 if f"sev_{s}" in df.columns and df[f"sev_{s}"].notna().any()]
    fused = [s for s in weights
             if f"sev_{s}" in df.columns and df[f"sev_{s}"].notna().any()]
    report = {"signals_used": fused, "signals_available": available,
              "pairwise_agreement": {}, "ablation": {}}

    def to_ord(series):
        return series.round().clip(0, 3)

    for a in range(len(available)):
        for b in range(a + 1, len(available)):
            s1, s2 = available[a], available[b]
            mask = df[f"sev_{s1}"].notna() & df[f"sev_{s2}"].notna()
            if mask.sum() == 0:
                continue
            agree = (to_ord(df.loc[mask, f"sev_{s1}"]) ==
                     to_ord(df.loc[mask, f"sev_{s2}"])).mean()
            report["pairwise_agreement"][f"{s1}~{s2}"] = round(float(agree), 4)

    # Leave-one-out over the fused signals only.
    full = df["mismatch"].values
    for drop in fused:
        w = {k: v for k, v in weights.items() if k != drop}
        if not w:
            continue
        f2 = df.apply(lambda r: fuse(r, w), axis=1)
        tmp = derive_labels(df.assign(sev_fused=f2))
        agree = float((tmp["mismatch"].values == full).mean())
        report["ablation"][f"without_{drop}"] = {
            "label_agreement_with_full": round(agree, 4)
        }
    return report

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run(input_path, output_path, use_llm=False, use_embed=True, use_rtime=False,
        llm_model="microsoft/Phi-3-mini-4k-instruct",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        mismatch_threshold=2, llm_max_rows=None,
        weights=None, report_path=None):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

    df = pd.read_csv(input_path)
    df = normalize_columns(df)
    df["text"] = df.apply(combined_text, axis=1)
    print(f"Loaded {len(df)} tickets. Columns: {list(df.columns)}")

    # Signal 1: rule-based
    rule_pairs = df["text"].map(rule_based_severity)
    df["sev_rule"] = [p[0] for p in rule_pairs]
    df["rule_evidence"] = [json.dumps(p[1]) for p in rule_pairs]

    # Signal 2: resolution time (always computed for reporting; fused only if asked)
    df["sev_rtime"] = resolution_time_severity(df)
    if use_rtime:
        weights["rtime"] = 0.15

    # Signal 3: embedding-based semantic severity (primary signal)
    if use_embed:
        df["sev_embed"] = embedding_severity(df["text"].tolist(), model_name=embed_model)
    else:
        df["sev_embed"] = float("nan")
        weights.pop("embed", None)

    # Signal 4: LLM zero-shot (optional; requires an NVIDIA/CUDA GPU for 4-bit)
    if use_llm:
        df["sev_llm"] = llm_zero_shot_severity(
            df["text"].tolist(), model_name=llm_model, max_rows=llm_max_rows)
        weights["llm"] = 0.4
    else:
        df["sev_llm"] = float("nan")
        weights.pop("llm", None)  # don't reserve weight for an absent signal

    # Drop signals that are entirely missing so fusion weights renormalise cleanly.
    for s in list(weights):
        if df[f"sev_{s}"].notna().sum() == 0:
            print(f"[fuse] signal '{s}' has no values; removing from fusion.")
            weights.pop(s)
    if len(weights) < 2:
        print("WARNING: fewer than 2 usable signals. Enable --use_llm or check the "
              "resolution_time column. The spec requires fusing at least two.")

    df["sev_fused"] = df.apply(lambda r: fuse(r, weights), axis=1)
    df = derive_labels(df, mismatch_threshold=mismatch_threshold)

    # Reporting
    dist = df["mismatch"].value_counts().to_dict()
    type_dist = df["mismatch_type"].value_counts().to_dict()
    report = {
        "n_tickets": int(len(df)),
        "fusion_weights": {k: round(v, 3) for k, v in weights.items()},
        "mismatch_threshold": mismatch_threshold,
        "mismatch_distribution": {str(k): int(v) for k, v in dist.items()},
        "mismatch_type_distribution": {str(k): int(v) for k, v in type_dist.items()},
        **signal_agreement(df, weights),
    }
    print("\n=== Stage 1 report ===")
    print(json.dumps(report, indent=2))

    df.to_csv(output_path, index=False)
    print(f"\nSaved pseudo-labeled data -> {output_path}")
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {report_path}")
    return df, report

def main(argv=None):
    ap = argparse.ArgumentParser(description="SIA Stage 1 pseudo-labeling")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no_embed", action="store_true",
                    help="disable the embedding signal")
    ap.add_argument("--embed_model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--use_rtime", action="store_true",
                    help="include the resolution-time signal in the fusion")
    ap.add_argument("--use_llm", action="store_true",
                    help="enable the LLM signal (requires an NVIDIA/CUDA GPU)")
    ap.add_argument("--llm_model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--llm_max_rows", type=int, default=None)
    ap.add_argument("--mismatch_threshold", type=int, default=2)
    args = ap.parse_args(argv)
    run(args.input, args.output, use_llm=args.use_llm, use_embed=not args.no_embed,
        use_rtime=args.use_rtime, llm_model=args.llm_model, embed_model=args.embed_model,
        mismatch_threshold=args.mismatch_threshold, llm_max_rows=args.llm_max_rows,
        report_path=args.report)

if __name__ == "__main__":
    main()