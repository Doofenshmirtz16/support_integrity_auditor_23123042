"""Stage 3 - deterministic, hallucination-free Evidence Dossier builder."""
from __future__ import annotations
import json
import pandas as pd
from config import ORD_TO_PRIORITY


def _keyword_evidence(rule_evidence):
    items = []
    try:
        ev = (json.loads(rule_evidence)
              if isinstance(rule_evidence, str) else (rule_evidence or {}))
    except Exception:
        ev = {}
    for tier, terms in ev.items():
        for term in terms:
            items.append({"signal": "keyword", "value": term, "weight": tier})
    return items


def _safe_int(v):
    try:
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def build_dossier(row, confidence=None):
    inferred_ord = _safe_int(row.get("inferred_ord"))
    inferred = ORD_TO_PRIORITY.get(inferred_ord, str(inferred_ord))
    assigned = str(row.get("priority"))
    delta = _safe_int(row.get("severity_delta"))
    mtype = row.get("mismatch_type")

    evidence = []
    evidence += _keyword_evidence(row.get("rule_evidence"))
    sev_embed = row.get("sev_embed")
    if sev_embed is not None and not pd.isna(sev_embed):
        closest = ORD_TO_PRIORITY.get(round(float(sev_embed)), "")
        evidence.append({"signal": "embedding",
                         "value": round(float(sev_embed), 2),
                         "interpretation": f"text semantically closest to {closest}-severity tickets"})
    rt = row.get("resolution_time")
    if rt is not None and not pd.isna(rt):
        evidence.append({"signal": "resolution_time",
                         "value": rt,
                         "interpretation": "reported resolution hours (context only; not weighted)"})

    kws = [e["value"] for e in evidence if e["signal"] == "keyword"][:4]
    kw_str = ", ".join(f'"{k}"' for k in kws)
    embed_label = (ORD_TO_PRIORITY.get(round(float(sev_embed)))
                   if sev_embed is not None and not pd.isna(sev_embed) else None)
    direction = ("under-prioritized" if mtype == "Hidden Crisis"
                 else "over-prioritized" if mtype == "False Alarm" else "consistent")

    if delta is not None:
        s1 = (f"Assigned priority is {assigned}, but the fused signals infer "
              f"{inferred} severity (delta {delta:+d}). ")
    else:
        s1 = f"Assigned priority is {assigned}; the fused signals infer {inferred} severity. "

    bits = []
    if kws:
        bits.append(f"escalation cues {kw_str}")
    if embed_label:
        bits.append(f"an embedding profile nearest {embed_label}-severity content")
    s2 = ("The text shows " + " and ".join(bits) + ". ") if bits else ""

    if mtype in ("Hidden Crisis", "False Alarm"):
        rel = "exceeds" if mtype == "Hidden Crisis" else "falls below"
        s3 = (f"Because the inferred severity {rel} the assigned {assigned} label, "
              f"this is flagged as a {mtype} ({direction}).")
    else:
        s3 = "Inferred and assigned severities are aligned."
    analysis = (s1 + s2 + s3).strip()

    return {
        "ticket_id": row.get("ticket_id"),
        "assigned_priority": assigned,
        "inferred_severity": inferred,
        "mismatch_type": mtype,
        "severity_delta": delta,
        "feature_evidence": evidence,
        "constraint_analysis": analysis,
        "confidence": round(float(confidence), 4) if confidence is not None else None,
    }
