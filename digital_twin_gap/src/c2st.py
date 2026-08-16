"""
c2st.py
=======
Classifier Two-Sample Test: the principled, standalone form of the "learn to
distinguish human from agent behavior" idea. Rather than folding a population
classifier into the twin's training objective (which would require the twin
to see cross-population labels during its own fit, and would conflate
"predicting the choice" with "predicting who's choosing" inside one loss),
this trains a SEPARATE classifier whose only job is the two-sample test:
`source in {agent, human}` from behavioral features of the chosen option.

If the two populations were behaviorally indistinguishable, a classifier
could not do better than chance (AUC = 0.5, balanced accuracy = 0.5). Held-out
AUC significantly above 0.5 is direct evidence of a real distributional
difference; how far above 0.5, and a permutation-test p-value, quantify it.

CRITICAL DESIGN CONSTRAINT: structured features only (from
`option_features.py`: the within-row z-score/rank of the CHOSEN option's
price, rating, and review count, plus category) -- never raw prompt text.
Agent prompts contain fields human prompts don't (BRAND_NAME,
BRAND_REPUTATION_LABEL, VISUAL_DESCRIPTION) and vice versa (LISTING); a
text-based classifier would hit ~100% AUC by detecting the TEMPLATE, not the
behavior, and the number would be meaningless. Restricting to the 6
within-row-normalized features of the option actually chosen isolates "given
four alternatives, what kind of option does this population tend to pick" --
genuinely behavioral, and the within-row normalization cancels absolute
configuration differences between the two prompt sets (e.g. mean price level).

Split: prompt-blocked WITHIN each source (reuses
`src.splits._prompt_blocked_split`) so the test set's tasks are unseen by the
classifier, consistent with the rest of this track's leakage correction.
Class imbalance (2254 agent rows vs 6016 human rows) is handled via
class_weight="balanced" and by reporting AUC / balanced accuracy rather than
raw accuracy.

Run:
    python -m src.c2st --config configs/config_corrected_prompt.yaml --out_dir results_c2st
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src import option_features as optfeat
from src.splits import _prompt_blocked_split, load_source

OPTION_LABELS = optfeat.OPTION_LABELS
CATEGORY_NAMES = ["AIR", "COF", "EAR", "SNK"]


def build_chosen_option_features(rows):
    """-> (X: (n, 6+4) float, source_y: (n,) int [0=agent,1=human]).
    Features: the 6 within-row-normalized features of the CHOSEN option, plus
    a one-hot of category (4 dims) -- category composition differs slightly
    between the agent and human samples, so it's included as a covariate
    rather than left as a confound."""
    label2idx = {l: i for i, l in enumerate(OPTION_LABELS)}
    cat2idx = {c: i for i, c in enumerate(CATEGORY_NAMES)}
    X, y = [], []
    for r in rows:
        if r["label"] not in label2idx:
            continue
        feats, meta = optfeat.compute_option_features(r["prompt"])
        chosen = feats[label2idx[r["label"]]]  # (6,)
        cat_onehot = np.zeros(len(CATEGORY_NAMES))
        cat_onehot[cat2idx[r["category"]]] = 1.0
        X.append(np.concatenate([chosen, cat_onehot]))
        y.append(0 if r["source"] == "agent" else 1)
    return np.array(X), np.array(y)


def _permutation_test(clf, X_test, y_test, observed_auc, n_perm=1000, seed=42):
    """Two-sided: separation could in principle show up as AUC significantly
    ABOVE 0.5 (population is separable in the expected direction) OR
    significantly BELOW 0.5 (the classifier's ranking is anti-correlated with
    the true label out of sample -- e.g. overfit on train-specific structure
    that reverses under a prompt-blocked test set). A one-sided test anchored
    at "AUC >= observed" would silently report p=1.0 for a below-chance AUC
    and read as "no effect" when the correct read is "significant effect in
    the other direction" -- so we test on distance from 0.5 instead."""
    rng = np.random.RandomState(seed)
    proba = clf.predict_proba(X_test)[:, 1]
    perm_aucs = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y_test)
        perm_aucs[i] = roc_auc_score(y_perm, proba)
    observed_dist = abs(observed_auc - 0.5)
    perm_dist = np.abs(perm_aucs - 0.5)
    p_value = float((np.sum(perm_dist >= observed_dist) + 1) / (n_perm + 1))
    return p_value, perm_aucs


def run_c2st(cfg, seed=42):
    unified = cfg["data"]["unified_dir"]
    test_size = cfg["data"]["test_size"]

    agent_rows = load_source(f"{unified}/agent.jsonl")
    human_rows = load_source(f"{unified}/human.jsonl")

    a_train, a_test = _prompt_blocked_split(agent_rows, test_size, seed)
    h_train, h_test = _prompt_blocked_split(human_rows, test_size, seed)

    X_train, y_train = build_chosen_option_features(a_train + h_train)
    X_test, y_test = build_chosen_option_features(a_test + h_test)

    print(f"  [c2st] train n={len(X_train)} (agent={sum(y_train==0)}, human={sum(y_train==1)})")
    print(f"  [c2st] test  n={len(X_test)} (agent={sum(y_test==0)}, human={sum(y_test==1)})")

    # L2-regularized logistic regression: the standard, low-variance choice for
    # a C2ST on a handful of low-dimensional features. An earlier attempt with
    # a 200-tree gradient-boosted classifier overfit badly on this data (0.77
    # in-sample AUC vs. 0.43 held-out -- it was memorizing product-specific
    # quirks that reverse on unseen products, not learning population
    # structure); logistic regression's small parameter count (11 total)
    # can't do that on ~6600 training rows.
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    proba_train = clf.predict_proba(X_train)[:, 1]

    auc = roc_auc_score(y_test, proba_test)
    auc_train = roc_auc_score(y_train, proba_train)
    bal_acc = balanced_accuracy_score(y_test, pred_test)
    p_value, perm_aucs = _permutation_test(clf, X_test, y_test, auc)
    print(f"  [c2st] train (in-sample) AUC = {auc_train:.4f}  vs. held-out AUC = {auc:.4f}"
          f"  (gap = {auc_train - auc:+.4f})")

    feature_names = optfeat.FEATURE_NAMES + [f"category_{c}" for c in CATEGORY_NAMES]
    # Logistic regression coefficients: sign + magnitude, standardized features
    # (option_features.py's z-scores/centered-ranks) so coefficients are
    # directly comparable to each other -- unlike a tree ensemble's
    # feature_importances_, the SIGN here is interpretable (positive =>
    # higher values of this feature push the prediction toward "human").
    importances = dict(zip(feature_names, clf.coef_[0].tolist()))

    result = {
        "auc": float(auc),
        "auc_train_in_sample": float(auc_train),
        "balanced_accuracy": float(bal_acc),
        "permutation_p_value": p_value,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_test_agent": int(np.sum(y_test == 0)),
        "n_test_human": int(np.sum(y_test == 1)),
        "feature_importances": importances,
        "interpretation": (
            "AUC ~0.5 => populations behaviorally indistinguishable on these features. "
            "AUC ~1.0 => near-perfect separation -- if this occurs, suspect residual "
            "template/configuration leakage rather than a genuine behavioral signal."
        ),
    }
    return result, perm_aucs, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_corrected_prompt.yaml")
    ap.add_argument("--out_dir", default="results_c2st")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    result, perm_aucs, auc = run_c2st(cfg, seed=args.seed)

    with open(os.path.join(args.out_dir, "c2st_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[c2st] AUC = {result['auc']:.4f}  balanced_accuracy = {result['balanced_accuracy']:.4f}")
    print(f"[c2st] permutation p-value = {result['permutation_p_value']:.4f}  (n_test={result['n_test']})")
    print("[c2st] logistic regression coefficients (standardized; +ve => pushes toward 'human'):")
    for name, imp in sorted(result["feature_importances"].items(), key=lambda kv: -abs(kv[1])):
        print(f"    {name:<18s} {imp:+.4f}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(perm_aucs, bins=40, color="#888888", alpha=0.7, label="permutation null (shuffled source label)")
        ax.axvline(auc, color="#C44E52", lw=2, label=f"observed AUC = {auc:.4f}")
        ax.axvline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
        ax.set_xlabel("AUC (agent vs. human classifier, chosen-option features)")
        ax.set_ylabel("count")
        ax.set_title(f"Classifier two-sample test  (p = {result['permutation_p_value']:.4f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_c2st_permutation.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[c2st] WARNING: figure generation skipped ({e})")

    print(f"\n[c2st] DONE. Wrote outputs to {args.out_dir}/")


if __name__ == "__main__":
    main()
