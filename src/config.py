"""
Shared configuration for the Support Integrity Auditor (SIA).

Keeps column-name normalization and the severity ordinal scale in one place so
that pseudo-labeling, training, inference and dossier generation all agree.
"""
from __future__ import annotations
import re
import pandas as pd

# -----------------------------------------------------------------------------
# Canonical column names used everywhere in the codebase.
# The raw Kaggle CSV uses verbose names with spaces; we map them to these.
# -----------------------------------------------------------------------------
CANON = {
    "ticket_id": "ticket_id",
    "subject": "subject",
    "description": "description",
    "priority": "priority",
    "channel": "channel",
    "product": "product",
    "ticket_type": "ticket_type",
    "resolution_time": "resolution_time",
    "email": "email",
}

# Lower-cased raw column -> canonical column. Add aliases freely; matching is
# done after lower-casing and collapsing whitespace/underscores.
COLUMN_ALIASES = {
    "ticket id": "ticket_id",
    "ticket subject": "subject",
    "ticket description": "description",
    "ticket priority": "priority",
    "ticket channel": "channel",
    "product purchased": "product",
    "ticket type": "ticket_type",
    "resolution time": "resolution_time",
    "time to resolution": "resolution_time",
    "customer email": "email",
    # enhanced_customer_support_data.csv variants
    "priority level": "priority",
    "resolution time hours": "resolution_time",
    "issue category": "ticket_type",
}

# Ordinal severity scale (single source of truth).
PRIORITY_TO_ORD = {"low": 0, "medium": 1, "high": 2, "critical": 3,
                   "urgent": 3, "normal": 1, "moderate": 1}
ORD_TO_PRIORITY = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

def _slug(name: str) -> str:
    return re.sub(r"[\s_]+", " ", str(name).strip().lower())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to canonical names; leave unknown columns untouched."""
    rename = {}
    for col in df.columns:
        key = _slug(col)
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename)

    # Guarantee the text columns exist so downstream code never KeyErrors.
    for needed in ("subject", "description"):
        if needed not in df.columns:
            df[needed] = ""
    if "ticket_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["ticket_id"] = df.index.astype(str)
    return df

def priority_to_ord(value) -> float:
    if value is None:
        return float("nan")
    return PRIORITY_TO_ORD.get(str(value).strip().lower(), float("nan"))

def combined_text(row) -> str:
    """Subject + description, used as the model's text input everywhere."""
    subj = "" if pd.isna(row.get("subject")) else str(row.get("subject"))
    desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
    return (subj + ". " + desc).strip()