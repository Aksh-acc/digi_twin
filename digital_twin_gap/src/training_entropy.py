"""
training_entropy.py
====================
SEPARATE EXPERIMENT -- entropy introduced during training.

This is deliberately NOT one of the transfer-matrix architectures in
run_all.py. It is a stand-alone study of how a twin behaves when a controlled
amount of ENTROPY (label noise) is injected into its TRAINING labels, while the
test labels are left clean. It answers two questions the main results can't:

  1. How robust is each twin to noisy supervision? (a stability curve)
  2. How much of the human/agent gap is explained by the fact that the human
     TARGET is intrinsically higher-entropy? By measuring the *empirical*
     training-label entropy of each source and then matching the agent's
     training entropy UP to the human level via injection, we can ask:
     "if agent labels were as noisy as human labels, would the agent twin
     still be easier to learn?"

What "injecting entropy" means here
-----------------------------------
For a noise level eps in [0, 0.75], each training row's label is, with
probability eps, resampled uniformly at random over {A,B,C,D} (a symmetric
label-noise channel). eps = 0 is the clean baseline; eps = 0.75 makes the
training label almost uniform (maximal 4-way entropy). Test labels are never
touched, so accuracy still measures real predictive quality.

For each (source, eps) we record:
  - injected_entropy_bits : the theoretical training-label entropy after noise
  - empirical_train_entropy: measured entropy of the (noised) training labels
  - matched-cell test accuracy (train src -> test src)
  - macro-F1

Outputs (results_training_entropy/):
  training_entropy_curve.csv   one row per (model, source, eps, seed-mean)
  fig_training_entropy.png     accuracy vs injected entropy, per model & source
  training_entropy_report.md   the empirical source entropies + a short reading

Run:
  python -m src.training_entropy --config configs/config.yaml \
         --models tfidf_logreg tabular_llm mnl_baseline --seeds 3
"""

import argparse
import csv
import json
import math
import os
from collections import Counter

import numpy as np
import yaml

from src import metrics
from src.splits import build_splits

# models cheap enough to sweep many times; text-heavy distilbert excluded by
# default (opt in via --models if a GPU is available).
DEFAULT_SWEEP_MODELS = ["tfidf_logreg", "tabular_llm", "mnl_baseline", "cognitive_decay"]

MODEL_REGISTRY = {
    "tfidf_logreg": "src.model_tfidf",
    "embed_mlp": "src.model_embed_mlp",
    "distilbert": "src.model_distilbert",
    "mnl_baseline": "src.model_mnl_baseline",
    "hier_bayes": "src.model_hier_bayes",
    "tabular_llm": "src.model_tabular_llm",
    "cognitive_decay": "src.model_cognitive_decay",
}

LABELS = ["A", "B", "C", "D"]


def _import(path):
    import importlib
    return importlib.import_module(path)


def _entropy_bits(labels):
    """Shannon entropy (bits) of a label list; max = 2 for 4 classes."""
    c = Counter(labels)
    n = sum(c.values())
    if n == 0:
        return 0.0
    h = 0.0
    for v in c.values():
        pr = v / n
        h -= pr * math.log2(pr)
    return h


def _inject_noise(rows, eps, rng):
    """Return a NEW list of rows with labels resampled uniformly w.p. eps.
    Only the 'label' field is changed; a shallow copy per row keeps the
    original split data intact so different eps/seeds don't contaminate."""
    if eps <= 0:
        return rows
    out = []
    for r in rows:
        if rng.random() < eps:
            r2 = dict(r)
            r2["label"] = LABELS[rng.integers(0, 4)]
            out.append(r2)
        else:
            out.append(r)
    return out


def _theoretical_entropy(base_labels, eps):
    """Expected entropy of the label distribution after a symmetric uniform
    noise channel at rate eps: p'(k) = (1-eps) p(k) + eps/4."""
    c = Counter(base_labels)
    n = sum(c.values())
    h = 0.0
    for k in LABELS:
        p = c.get(k, 0) / n
        pp = (1 - eps) * p + eps / 4.0
        if pp > 0:
            h -= pp * math.log2(pp)
    return h


def run_sweep(cfg, model_names, eps_grid, seeds):
    splits = build_splits(cfg, verbose=False)
    rows_out = []

    # empirical clean training entropies (reported once)
    clean_entropy = {
        src: _entropy_bits([r["label"] for r in splits[src]["train"]])
        for src in ("agent", "human")
    }

    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"[skip] unknown model '{name}'")
            continue
        module = _import(MODEL_REGISTRY[name])
        for src in ("agent", "human"):
            train_full = splits[src]["train"]
            test_rows = splits[src]["test"]          # matched cell, clean labels
            y_true = [r["label"] for r in test_rows]
            base_labels = [r["label"] for r in train_full]

            for eps in eps_grid:
                accs, f1s, emp_ents = [], [], []
                for s in range(seeds):
                    rng = np.random.default_rng(1000 * s + int(eps * 100))
                    noised = _inject_noise(train_full, eps, rng)
                    emp_ents.append(_entropy_bits([r["label"] for r in noised]))
                    try:
                        model = module.fit(noised, cfg)
                        y_pred = module.predict(model, test_rows)
                        m = metrics.score(y_true, y_pred, test_rows)
                        accs.append(m["accuracy"])
                        f1s.append(m["macro_f1"])
                    except Exception as e:
                        print(f"  !! [{name}/{src}/eps={eps}/seed={s}] failed: {e}")
                if not accs:
                    continue
                rows_out.append({
                    "model": name,
                    "source": src,
                    "eps": eps,
                    "injected_entropy_bits": round(_theoretical_entropy(base_labels, eps), 4),
                    "empirical_train_entropy": round(float(np.mean(emp_ents)), 4),
                    "acc_mean": round(float(np.mean(accs)), 4),
                    "acc_std": round(float(np.std(accs)), 4),
                    "macro_f1_mean": round(float(np.mean(f1s)), 4),
                    "n_seeds": len(accs),
                })
                print(
                    f"  {name:16s} {src:5s} eps={eps:.2f} "
                    f"H_train={rows_out[-1]['empirical_train_entropy']:.3f} "
                    f"acc={rows_out[-1]['acc_mean']:.4f}"
                )
    return rows_out, clean_entropy


def _plot(rows_out, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")
        return None

    models = sorted(set(r["model"] for r in rows_out))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), squeeze=False)
    for ax, mdl in zip(axes[0], models):
        for src, style in [("agent", "o-"), ("human", "s--")]:
            pts = sorted([r for r in rows_out if r["model"] == mdl and r["source"] == src],
                         key=lambda r: r["eps"])
            if not pts:
                continue
            # x-axis = injected label-noise rate eps (the experimental knob);
            # the theoretical training entropy is near-saturated for this
            # already-uniform dataset, so eps is the readable axis.
            x = [r["eps"] for r in pts]
            y = [r["acc_mean"] for r in pts]
            yerr = [r["acc_std"] for r in pts]
            ax.errorbar(x, y, yerr=yerr, fmt=style, color="black",
                        markerfacecolor=("black" if src == "agent" else "white"),
                        label=src, capsize=2, linewidth=1)
        ax.axhline(0.25, ls=":", color="gray", lw=1, label="chance")
        ax.set_title(mdl, fontsize=11)
        ax.set_xlabel("label-noise injected into training (\u03b5)")
        ax.set_ylabel("matched-cell accuracy")
        ax.set_ylim(0.2, 0.75)
        ax.legend(fontsize=8)
    fig.suptitle("Twin accuracy vs entropy (label noise) injected into TRAINING labels", y=1.03)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_training_entropy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eps", nargs="*", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75])
    ap.add_argument("--out", default="results_training_entropy")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_names = args.models or DEFAULT_SWEEP_MODELS
    os.makedirs(args.out, exist_ok=True)

    print(f"Entropy-in-training sweep: models={model_names} "
          f"eps={args.eps} seeds={args.seeds}")
    rows_out, clean_entropy = run_sweep(cfg, model_names, args.eps, args.seeds)

    csv_path = os.path.join(args.out, "training_entropy_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    fig_path = _plot(rows_out, args.out)

    # report
    md = ["# Entropy introduced in training -- results", ""]
    md.append("Clean (eps=0) empirical training-label entropy, in bits "
              "(max = 2.0 for a uniform 4-way label):\n")
    md.append(f"- agent training labels : **{clean_entropy['agent']:.4f} bits**")
    md.append(f"- human training labels : **{clean_entropy['human']:.4f} bits**\n")
    md.append("Interpretation: as entropy is injected into the training labels, "
              "matched-cell accuracy degrades toward the 25% chance line. A twin "
              "whose accuracy collapses quickly is highly reliant on clean, "
              "low-entropy supervision; one that degrades slowly has learned more "
              "robust structure. Comparing the agent and human curves at *equal "
              "training entropy* isolates how much of the human-agent gap is due "
              "to target noise versus genuine behavioural structure.\n")
    md.append("See `training_entropy_curve.csv` for the full grid and "
              "`fig_training_entropy.png` for the curves.")
    md_path = os.path.join(args.out, "training_entropy_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\nWrote:")
    print(" ", csv_path)
    if fig_path:
        print(" ", fig_path)
    print(" ", md_path)


if __name__ == "__main__":
    main()
