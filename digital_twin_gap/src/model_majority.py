"""
model_majority.py
=================
REFERENCE BASELINE -- the floor.

Always predicts the training set's most common label. This is the number every
other architecture must beat to have demonstrated anything at all, and it is
deliberately kept as a permanent row in the results table rather than quoted
once in prose: the "random baseline = 0.25" figure used elsewhere assumes a
uniform label distribution, which real data does not have. On the human
prompt-blocked split the majority class alone scores 0.3039, so a model
reporting 0.3262 is 2.2 points above the floor, not 7.6 points above chance.

Interface matches the other architectures: fit(train_rows, cfg) / predict(model, rows).
"""

from collections import Counter


def fit(train_rows, cfg=None):
    counts = Counter(r["label"] for r in train_rows)
    # Ties broken alphabetically for determinism.
    label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    share = counts[label] / len(train_rows) if train_rows else 0.0
    print(f"  [majority] training modal label = '{label}' ({100*share:.2f}% of train rows)")
    return {"label": label}


def predict(model, test_rows):
    return [model["label"]] * len(test_rows)
