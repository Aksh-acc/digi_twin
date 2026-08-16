"""
model_embed_mlp.py
==================
ARCHITECTURE 2 — Frozen sentence-transformer embeddings + a small MLP head.

Each prompt is encoded ONCE by a frozen `all-MiniLM-L6-v2` sentence encoder
(384-dim vector). We never back-prop into the encoder — only a lightweight
2-layer MLP classifier is trained on top. This is the classic "representation
learning" middle ground:

  - much stronger text understanding than TF-IDF (semantic, not lexical),
  - but far cheaper than fine-tuning a transformer (the expensive encoder is
    frozen; only ~a hundred-K head params train),
  - runs fine on CPU, faster on GPU.

Because the encoder is frozen and shared, this model is a clean probe of "how
linearly separable are agent vs human choices in a good semantic space" — which
is exactly the kind of representational-gap question a paper wants.

Interface matches the other models: fit(train_rows, cfg) / predict(model, rows).
"""

import numpy as np
import torch
import torch.nn as nn

LABELS = ["A", "B", "C", "D", "NONE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class EmbedMLPModel:
    """Bundles the frozen encoder + trained head so predict() is self-contained."""

    def __init__(self, encoder, head, device):
        self.encoder = encoder
        self.head = head
        self.device = device

    def _embed(self, texts):
        return self.encoder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64
        )


def _get_encoder(name, device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def fit(train_rows, cfg):
    p = cfg["models"]["embed_mlp"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = _get_encoder(p["encoder"], device)

    X_texts = [r["prompt"] for r in train_rows]
    y = np.array([LABEL2ID.get(r["label"], LABEL2ID["NONE"]) for r in train_rows])

    print(f"  [embed_mlp] encoding {len(X_texts)} prompts (frozen encoder)...")
    X = encoder.encode(X_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)

    in_dim = X.shape[1]
    # Only use classes that actually appear so the head isn't wasting capacity,
    # but keep a fixed 4-way (A/B/C/D) head for comparability; NONE is rare/absent.
    n_classes = len(LABELS)
    head = _MLP(in_dim, p["hidden_dim"], n_classes, p["dropout"]).to(device)

    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)

    # class weights to counter imbalance
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / (counts + 1e-6), 0.0)
    weights = torch.tensor(weights / weights.sum() * (counts > 0).sum(), dtype=torch.float32).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=p["lr"], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    bs = p["batch_size"]
    n = Xt.shape[0]
    head.train()
    for epoch in range(p["epochs"]):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            logits = head(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{p['epochs']}  loss={total/n:.4f}")

    head.eval()
    return EmbedMLPModel(encoder, head, device)


def predict(model, test_rows):
    texts = [r["prompt"] for r in test_rows]
    X = model.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    Xt = torch.tensor(X, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        logits = model.head(Xt)
        preds = logits.argmax(dim=1).cpu().numpy()
    return [ID2LABEL[int(i)] for i in preds]
