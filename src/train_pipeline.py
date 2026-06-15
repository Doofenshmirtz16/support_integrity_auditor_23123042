"""
Stage 2 - Fine-tuned classifier.

Trains a supervised binary classifier (DeBERTa-v3-small) on the Stage-1
pseudo-labels to predict Priority Mismatch.

Input to the model = structured metadata serialized in front of the ticket text:

    "priority: High | channel: Chat | type: Technical | res_hours: 41 | <subject>. <description>"

Including the assigned priority is essential: the mismatch label is a comparison
between the content's severity and that assigned priority, so the model needs the
priority as an input to make the comparison. This also satisfies the requirement
that inputs include text fields AND at least one structured metadata feature.

Class imbalance (~65/35) is handled with a weighted cross-entropy loss.

Usage:
    python src/train_pipeline.py --input data/tickets_pseudolabeled.csv \
        --model_dir models/sia_deberta --metrics data/stage2_metrics.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset

# Structured metadata serialized into the model input (canonical column names).
META_COLS = ["priority", "channel", "ticket_type", "resolution_time"]
LABEL_NAMES = ["Consistent", "Mismatch"]

def build_input(row) -> str:
    """Serialize metadata + text into a single string for the encoder."""
    parts = []
    for c in META_COLS:
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            parts.append(f"{c}: {v}")
    meta = " | ".join(parts)
    text = "" if pd.isna(row.get("text")) else str(row.get("text"))
    joined = (meta + " | " + text).strip(" |")
    return joined

def compute_metrics(eval_pred):
    logits = getattr(eval_pred, "predictions", None)
    labels = getattr(eval_pred, "label_ids", None)
    if logits is None:
        logits, labels = eval_pred[0], eval_pred[1]
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = np.asarray(logits).argmax(axis=-1)
    rec = recall_score(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "recall_consistent": float(rec[0]),
        "recall_mismatch": float(rec[1]) if len(rec) > 1 else 0.0,
    }

class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for the imbalance."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (self.class_weights.to(logits.device)
                  if self.class_weights is not None else None)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_tokenizer(base_model):
    try:
        return AutoTokenizer.from_pretrained(base_model)
    except Exception:
        # DeBERTa-v3 occasionally needs the slow (sentencepiece) tokenizer.
        return AutoTokenizer.from_pretrained(base_model, use_fast=False)

def run(input_csv, model_dir, base_model="microsoft/deberta-v3-small",
        epochs=3, batch_size=16, max_len=256, seed=42, metrics_path=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(input_csv)
    df["input_text"] = df.apply(build_input, axis=1)
    df = df[["input_text", "mismatch"]].dropna()
    df["labels"] = df["mismatch"].astype(int)
    print(f"Dataset: {len(df)} rows | label balance:\n"
          f"  {df['labels'].value_counts().to_dict()}")

    # Stratified 70/15/15 train/val/test.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["labels"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["labels"], random_state=seed)
    print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tok = load_tokenizer(base_model)

    def tok_fn(batch):
        return tok(batch["input_text"], truncation=True, max_length=max_len)

    def to_ds(d):
        ds = Dataset.from_pandas(d[["input_text", "labels"]], preserve_index=False)
        return ds.map(tok_fn, batched=True)

    ds_train, ds_val, ds_test = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    # Inverse-frequency class weights from the training split.
    counts = train_df["labels"].value_counts().sort_index()
    w = (counts.sum() / (len(counts) * counts)).reindex([0, 1]).fillna(1.0).values
    class_weights = torch.tensor(w, dtype=torch.float)
    print(f"Class weights: {dict(zip([0, 1], w.round(3)))}")

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2,
        torch_dtype=torch.float32,  # checkpoint ships fp16; force fp32 master weights
        id2label={0: LABEL_NAMES[0], 1: LABEL_NAMES[1]},
        label2id={LABEL_NAMES[0]: 0, LABEL_NAMES[1]: 1})
    model = model.float()  # ensure every param (incl. the new head) is fp32

    args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    import inspect
    trainer_kwargs = dict(
        class_weights=class_weights,
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    # transformers renamed `tokenizer` -> `processing_class` in recent versions.
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = WeightedTrainer(**trainer_kwargs)

    trainer.train()

    # Final evaluation on the held-out test split.
    pred = trainer.predict(ds_test)
    y = pred.label_ids
    p = np.asarray(pred.predictions).argmax(axis=-1)
    rec = recall_score(y, p, average=None, zero_division=0)
    acc = accuracy_score(y, p)
    macro = f1_score(y, p, average="macro", zero_division=0)
    per_class_recall_min = float(min(rec))

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "recall_consistent": float(rec[0]),
        "recall_mismatch": float(rec[1]) if len(rec) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y, p).tolist(),
        "thresholds": {
            "accuracy_ge_0.83": bool(acc >= 0.83),
            "macro_f1_ge_0.82": bool(macro >= 0.82),
            "per_class_recall_ge_0.78": bool(per_class_recall_min >= 0.78),
        },
    }
    metrics["verified"] = all(metrics["thresholds"].values())

    print("\n=== Stage 2 held-out test metrics ===")
    print(json.dumps(metrics, indent=2))
    print("VERIFIED" if metrics["verified"]
          else "NOT yet meeting all thresholds - see notes.")

    trainer.save_model(model_dir)
    tok.save_pretrained(model_dir)
    print(f"\nSaved model -> {model_dir}")
    if metrics_path:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics -> {metrics_path}")
    return metrics

def main(argv=None):
    ap = argparse.ArgumentParser(description="SIA Stage 2 classifier training")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--base_model", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    run(args.input, args.model_dir, base_model=args.base_model, epochs=args.epochs,
        batch_size=args.batch_size, max_len=args.max_len, seed=args.seed,
        metrics_path=args.metrics)

if __name__ == "__main__":
    main()