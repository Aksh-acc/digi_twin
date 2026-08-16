"""
model_consensus.py
==================
REFERENCE BASELINE -- the memorisation ceiling, measured directly.

Not a model in any meaningful sense: `fit()` builds a dictionary mapping each
training prompt to the modal label chosen for it, and `predict()` is a lookup.
No features, no parameters, no optimisation.

Why it is registered as a first-class architecture: under a split that does not
block on the task, this trivial lookup scores 0.6188 on the agent test set and
0.4281 on the human test set -- 93.1% and 98.8% of what a fitted 20k-feature
TF-IDF logistic regression achieves. Any architecture that does not clearly
beat this row is not demonstrating learned behavioural structure; it is
reproducing a per-task popularity table.

Under a correctly blocked split (`split_mode: "prompt"` or `"twoway"`), no test
prompt appears in train, every lookup misses, and this model degrades to the
global majority class. **That collapse is the diagnostic**: it is the direct,
automatic evidence in every results table that the leak is closed.

Interface matches the other architectures: fit(train_rows, cfg) / predict(model, rows).
"""

from collections import Counter, defaultdict


def _modal(labels):
    """Most common label; ties broken alphabetically so results are
    deterministic across runs and platforms."""
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def fit(train_rows, cfg=None):
    by_prompt = defaultdict(list)
    for r in train_rows:
        by_prompt[r["prompt"]].append(r["label"])

    table = {prompt: _modal(labels) for prompt, labels in by_prompt.items()}
    fallback = _modal([r["label"] for r in train_rows])

    print(
        f"  [consensus] memorised {len(table)} distinct prompts from "
        f"{len(train_rows)} rows (global fallback = '{fallback}')"
    )
    return {"table": table, "fallback": fallback}


def predict(model, test_rows):
    table, fallback = model["table"], model["fallback"]
    preds = []
    n_hit = 0
    for r in test_rows:
        label = table.get(r["prompt"])
        if label is None:
            preds.append(fallback)
        else:
            preds.append(label)
            n_hit += 1
    if test_rows:
        print(
            f"  [consensus] lookup hit rate on this test set: "
            f"{n_hit}/{len(test_rows)} ({100*n_hit/len(test_rows):.2f}%)"
        )
    return preds
