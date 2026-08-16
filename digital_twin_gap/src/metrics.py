"""
metrics.py
==========
Scoring helpers shared across models. The headline number the paper cares about
is the TWIN TRANSFER GAP: how much worse a twin does when the thing it was
trained on differs from the thing it is tested on.

We report, for each (architecture, train_source, test_source) cell:
  - accuracy
  - macro-F1 (guards against a model that just predicts the majority class)
  - per-agent accuracy when the test set is agent data (so you can see whether
    the twin models some agents better than others)

The gap itself is derived in run_all.py from these cells:
  gap_on_human = acc(train=human, test=human) - acc(train=agent, test=human)
  gap_on_agent = acc(train=agent, test=agent) - acc(train=human, test=agent)
"""

from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score


def score(y_true, y_pred, test_rows=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": len(y_true),
    }
    # Per-agent breakdown when applicable.
    if test_rows is not None and all(r.get("source") == "agent" for r in test_rows):
        by_agent = defaultdict(lambda: [0, 0])  # group -> [correct, total]
        for r, yt, yp in zip(test_rows, y_true, y_pred):
            g = r["group"]
            by_agent[g][1] += 1
            if yt == yp:
                by_agent[g][0] += 1
        out["per_agent"] = {
            g: {"accuracy": c / t, "n": t} for g, (c, t) in sorted(by_agent.items())
        }
    return out
