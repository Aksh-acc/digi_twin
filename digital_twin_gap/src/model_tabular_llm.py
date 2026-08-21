"""
model_tabular_llm.py
=====================
ARCHITECTURE -- Tabular LLM (attention-based deep tabular choice model).

This is the "tabular" counterpart to the text twins. The text models
(tfidf_logreg, embed_mlp, distilbert) read the raw prompt STRING. The discrete-
choice models (mnl_baseline, hier_bayes) read the engineered per-option features
but combine them LINEARLY. This model sits between them: it consumes the same
structured per-option table the choice models use, but processes it with a small
Transformer -- the architecture family that underlies LLMs -- so it can learn
NON-LINEAR interactions between attributes and cross-option comparisons
(e.g. "cheapest AND well-rated", "much cheaper than the next option") that a
linear MNL cannot express.

Why "tabular LLM": each of the 4 options is turned into a feature TOKEN; a
learned [CHOICE] query token attends over the 4 option tokens (self-attention,
exactly the LLM mechanism) and the model outputs a distribution over the 4
options. It is a compact FT-Transformer / TabTransformer-style model, trained
from scratch on the tabular features -- no text, no pretrained weights, fully
CPU-runnable in seconds, deterministic given the seed.

Input per row: (4 options x 6 engineered features) from option_features, plus a
one-hot category vector broadcast to every option token so the model can
condition on product category.

Interface matches the other architectures: fit(train_rows, cfg) / predict(model, rows).
"""

import numpy as np
import torch
import torch.nn as nn

from src import option_features as optfeat

OPTION_LABELS = optfeat.OPTION_LABELS
N_FEAT = len(optfeat.FEATURE_NAMES)          # 6
CATEGORIES = ["AIR", "COF", "EAR", "SNK"]    # one-hot conditioning
N_CAT = len(CATEGORIES)


def _cat_onehot(rows):
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    oh = np.zeros((len(rows), N_CAT), dtype=np.float32)
    for i, r in enumerate(rows):
        j = idx.get(r["category"])
        if j is not None:
            oh[i, j] = 1.0
    return oh


class _TabularChoiceTransformer(nn.Module):
    """Per-option feature tokens + a [CHOICE] query; output = logit per option."""

    def __init__(self, in_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)          # tokenize each option
        self.pos = nn.Parameter(torch.zeros(1, 4, d_model))  # option-slot embed
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # score each option token -> scalar logit
        self.score = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (B, 4, in_dim)
        h = self.proj(x) + self.pos          # (B,4,d)
        h = self.encoder(h)                  # (B,4,d) cross-option attention
        logits = self.score(h).squeeze(-1)   # (B,4)
        return logits


def _build_tensor(rows, for_train):
    """-> (X: (n,4,in_dim) float tensor, y: (n,) long tensor or None)."""
    if for_train:
        Xf, y, has_missing = optfeat.build_feature_matrix(rows)
        kept = [r for r in rows if r["label"] in OPTION_LABELS]
        keep = [not hm for hm in has_missing]
        Xf = Xf[keep]; y = y[keep]
        kept = [r for r, k in zip(kept, keep) if k]
    else:
        Xf, y, _ = optfeat.build_feature_matrix_for_predict(rows)
        kept = rows

    cat = _cat_onehot(kept)                               # (n, N_CAT)
    cat_b = np.repeat(cat[:, None, :], 4, axis=1)         # (n,4,N_CAT)
    X = np.concatenate([Xf.astype(np.float32), cat_b], axis=2)  # (n,4,6+N_CAT)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long) if for_train else None
    return Xt, yt, len(rows) - len(kept)


def fit(train_rows, cfg):
    p = cfg["models"].get("tabular_llm", {})
    seed = int(p.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt, yt, n_drop = _build_tensor(train_rows, for_train=True)
    if n_drop:
        print(f"  [tabular_llm] dropped {n_drop} row(s) (bad label/fields)")
    Xt, yt = Xt.to(device), yt.to(device)
    in_dim = Xt.shape[2]

    model = _TabularChoiceTransformer(
        in_dim=in_dim,
        d_model=int(p.get("d_model", 64)),
        n_heads=int(p.get("n_heads", 4)),
        n_layers=int(p.get("n_layers", 2)),
        dropout=float(p.get("dropout", 0.1)),
    ).to(device)

    # class weights to counter any imbalance
    counts = torch.bincount(yt, minlength=4).float()
    w = torch.where(counts > 0, counts.sum() / (counts + 1e-6), torch.zeros_like(counts))
    w = (w / w.sum() * (counts > 0).sum()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(p.get("lr", 1e-3)),
        weight_decay=float(p.get("weight_decay", 1e-4)),
    )
    epochs = int(p.get("epochs", 60))
    bs = int(p.get("batch_size", 128))
    n = Xt.shape[0]

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"    [tabular_llm] epoch {ep+1:3d}/{epochs}  loss={tot/n:.4f}")

    model.eval()
    return {"model": model, "device": device}


def predict(model, test_rows):
    m, device = model["model"], model["device"]
    Xt, _, _ = _build_tensor(test_rows, for_train=False)
    Xt = Xt.to(device)
    with torch.no_grad():
        logits = m(Xt)
        idx = logits.argmax(dim=1).cpu().numpy()
    return [OPTION_LABELS[i] for i in idx]


def predict_proba(model, test_rows):
    m, device = model["model"], model["device"]
    Xt, _, _ = _build_tensor(test_rows, for_train=False)
    Xt = Xt.to(device)
    with torch.no_grad():
        P = torch.softmax(m(Xt), dim=1).cpu().numpy()
    return P
