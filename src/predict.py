"""SIA inference - raw tickets CSV -> predictions + Evidence Dossiers."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pseudo_labeling
from dossier import build_dossier

META_COLS = ["priority", "channel", "ticket_type", "resolution_time"]


def build_input(row) -> str:
    parts = []
    for c in META_COLS:
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            parts.append(f"{c}: {v}")
    text = "" if pd.isna(row.get("text")) else str(row.get("text"))
    return (" | ".join(parts) + " | " + text).strip(" |")


def load_model(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return tok, model, device


def classify(frame, tok, model, device, batch_size=64, max_len=256):
    texts = frame.apply(build_input, axis=1).tolist()
    preds, confs = [], []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            prob = torch.softmax(model(**enc).logits, dim=-1)
        preds.extend(prob.argmax(-1).cpu().tolist())
        confs.extend(prob.max(-1).values.cpu().tolist())
    return np.array(preds), np.array(confs)


def run(input_csv, model_dir, predictions_out="predictions.csv",
        dossiers_out="dossiers.json", use_embed=True, use_llm=False):
    labeled, _ = pseudo_labeling.run(
        input_csv, "/tmp/_sia_labeled.csv", use_embed=use_embed, use_llm=use_llm)
    tok, model, device = load_model(model_dir)
    pred, conf = classify(labeled, tok, model, device)
    labeled["pred_mismatch"] = pred
    labeled["pred_confidence"] = conf
    labeled["pred_label"] = np.where(pred == 1, "Mismatch", "Consistent")

    cols = ["ticket_id", "priority", "inferred_ord", "severity_delta",
            "mismatch_type", "pred_label", "pred_mismatch", "pred_confidence"]
    cols = [c for c in cols if c in labeled.columns]
    labeled[cols].to_csv(predictions_out, index=False)

    flagged = labeled[labeled["pred_mismatch"] == 1]
    dossiers = [build_dossier(r, c)
                for (_, r), c in zip(flagged.iterrows(), flagged["pred_confidence"])]
    with open(dossiers_out, "w") as f:
        json.dump(dossiers, f, indent=2)

    print(f"\nPredictions -> {predictions_out}  ({len(labeled)} tickets)")
    print(f"Dossiers    -> {dossiers_out}  ({len(dossiers)} flagged as mismatch)")
    return labeled, dossiers


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIA inference: CSV -> predictions + dossiers")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--predictions", default="predictions.csv")
    ap.add_argument("--dossiers", default="dossiers.json")
    ap.add_argument("--no_embed", action="store_true")
    ap.add_argument("--use_llm", action="store_true")
    args = ap.parse_args(argv)
    run(args.input, args.model_dir, args.predictions, args.dossiers,
        use_embed=not args.no_embed, use_llm=args.use_llm)


if __name__ == "__main__":
    main()
