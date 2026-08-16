"""
model_distilbert.py
===================
ARCHITECTURE 3 — Fine-tuned DistilBERT sequence classifier.

Here the whole transformer encoder is fine-tuned (not frozen) with a 4-way
classification head, using Hugging Face `Trainer`. This is the strongest of the
three twins and the one most likely to close the human/agent gap — or to reveal
that it genuinely can't, which is the interesting result.

Design choices kept deliberately Colab/Kaggle-friendly:
  - DistilBERT (66M params) instead of a 3B decoder — fits free-tier GPUs, and
    even runs (slowly) on CPU so the pipeline never hard-fails.
  - max_length capped (prompts are long; 512 tokens covers the decision-relevant
    head of the prompt: mandate + options).
  - gradient accumulation so the effective batch stays reasonable on small VRAM.

Interface matches the other models: fit(train_rows, cfg) / predict(model, rows).
"""

import numpy as np
import torch

LABELS = ["A", "B", "C", "D"]  # NONE is essentially absent in both sources
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class DistilBertModel:
    def __init__(self, model, tokenizer, max_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device


def _map_label(lbl):
    # Fold the rare NONE into a valid class bucket is wrong; instead drop it.
    return LABEL2ID.get(lbl, None)


def fit(train_rows, cfg):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    p = cfg["models"]["distilbert"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Filter to A/B/C/D (drop the negligible NONE rows for a clean 4-way task).
    rows = [r for r in train_rows if _map_label(r["label"]) is not None]
    texts = [r["prompt"] for r in rows]
    labels = [_map_label(r["label"]) for r in rows]

    tokenizer = AutoTokenizer.from_pretrained(p["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        p["model_name"],
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    enc = tokenizer(texts, truncation=True, padding=True, max_length=p["max_length"])
    ds = _TextDataset(enc, labels)

    args = TrainingArguments(
        output_dir="results/_distilbert_tmp",
        per_device_train_batch_size=p["batch_size"],
        gradient_accumulation_steps=p["grad_accum"],
        num_train_epochs=p["epochs"],
        learning_rate=p["lr"],
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        fp16=(device == "cuda"),
        dataloader_pin_memory=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    print(f"  [distilbert] fine-tuning on {len(ds)} examples ({device})...")
    trainer.train()
    model.eval()
    return DistilBertModel(model, tokenizer, p["max_length"], device)


def predict(model, test_rows):
    texts = [r["prompt"] for r in test_rows]
    preds = []
    bs = 16
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = model.tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=model.max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            logits = model.model(**enc).logits
            idx = logits.argmax(dim=1).cpu().numpy()
        preds.extend([ID2LABEL[int(j)] for j in idx])
    return preds
