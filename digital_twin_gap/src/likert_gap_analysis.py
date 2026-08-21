"""
likert_gap_analysis.py
=======================
SIDE-STUDY, decoupled from the main agent-vs-human twin study. Same
underlying question -- does a generated stand-in for a population reproduce
that population's real response distribution? -- applied to a completely
different data domain: paired REAL and SYNTHETIC consumer-behavior Likert
surveys from `Datasets_total/` (a mentor-provided dataset drop), instead of
the main study's categorical A/B/C/D product choice.

There is no train/test split and no fit() here -- the "synthetic twin" is
already a fixed, pre-generated CSV someone else produced. The analysis is
purely: do the two already-realized samples' response DISTRIBUTIONS match,
item by item and jointly?

IMPORTANT LIMITATION: no generator script for either candidate dataset pair
was found anywhere in this repo (verified by a full-repo search). So unlike
`behavioral_profiles.py`'s tau-vs-entropy check (which validates a fitted
model's parameter against an independently known ground truth), this module
cannot validate the synthetic data against a known coded model -- it can only
DESCRIBE the empirical gap. See docs/likert_side_study.md for the full
write-up and the two datasets' actual results.

Deliberately standalone (not a `run_all.py` MODEL_REGISTRY architecture):
the transfer-matrix framework hardcodes a 2x2 train/test cell structure with
no analogue here. Follows the same standalone-script convention as
gap_analysis.py / decay_analysis.py / behavioral_profiles.py instead: own
data loading, own metrics, own CSV/report/figure writing into its own
results_<name>/ directory.

Reuse from the main study, both confirmed already fully generic (imported
directly, unmodified):
  - src.gap_analysis.js_divergence / kl_divergence (generic over category count)
  - src.c2st._permutation_test (generic over feature shape)
Everything else here is new: the main study's OPTION_LABELS-based entropy/
Herfindahl helpers hardcode the A/B/C/D alphabet (two of them literally
re-read that module global instead of the input dict's own keys), so this
module defines its own generic versions instead of importing them.

Run (twice, once per dataset pair -- see docs/likert_side_study.md Sec. 5
for the exact commands used for this study's two datasets):
    python -m src.likert_gap_analysis \\
        --real_path "Datasets_total/OS_Samples/green_purchase_behavior.xlsx" \\
        --synthetic_path "Datasets_total/OS_Samples/green_purchase_behavior_400_synthetic.csv" \\
        --exclude SN CaseNo PB EC EO PI PBC SNs ATT ZPB ZEC ZEO ZPI ZPBC ZSNs ZATT EOXEC ATTXSN PBCXSN ATTXPBC Gender Age Marital_Status \\
        --scale_min 1 --scale_max 7 --dataset_name green_purchase_behavior \\
        --out_dir results_likert_green_purchase
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.c2st import _permutation_test
from src.gap_analysis import js_divergence


# -----------------------------------------------------------------------------
# Loading & schema harmonization
# -----------------------------------------------------------------------------

def _load_any(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_and_harmonize(real_path, synthetic_path, exclude, scale_min, scale_max):
    """-> (real_df, synth_df, item_cols).

    item_cols = (columns common to both files) - exclude, each further
    verified numeric-coercible and within [scale_min, scale_max] in BOTH
    frames. A column that survives `exclude` but fails this range check
    raises a clear, named error -- deliberately loud, not a silent drop, so
    an incomplete --exclude list (e.g. a forgotten ID or demographic column)
    is caught immediately rather than corrupting the item set.
    """
    real_df = _load_any(real_path)
    synth_df = _load_any(synthetic_path)

    exclude = set(exclude)
    common = sorted(set(real_df.columns) & set(synth_df.columns))
    candidates = [c for c in common if c not in exclude]

    item_cols = []
    bad = []
    for c in candidates:
        r_num = pd.to_numeric(real_df[c], errors="coerce")
        s_num = pd.to_numeric(synth_df[c], errors="coerce")
        if r_num.isna().any() or s_num.isna().any():
            bad.append((c, "not numeric-coercible in both files"))
            continue
        r_min, r_max = r_num.min(), r_num.max()
        s_min, s_max = s_num.min(), s_num.max()
        if r_min < scale_min or r_max > scale_max or s_min < scale_min or s_max > scale_max:
            bad.append(
                (c, f"out of [{scale_min},{scale_max}]: real=[{r_min},{r_max}] synth=[{s_min},{s_max}]")
            )
            continue
        item_cols.append(c)

    if bad:
        detail = "\n".join(f"    {c}: {reason}" for c, reason in bad)
        raise ValueError(
            f"{len(bad)} column(s) in the common set failed the numeric/range check "
            f"(add to --exclude if these are IDs/demographics/derived columns, or fix "
            f"--scale_min/--scale_max if the item scale is wrong):\n{detail}"
        )

    item_cols = sorted(item_cols)
    print(f"  [likert_gap_analysis] real n={len(real_df)}  synthetic n={len(synth_df)}  "
          f"harmonized item columns: {len(item_cols)}")
    return real_df, synth_df, item_cols


# -----------------------------------------------------------------------------
# Generic entropy / Herfindahl (NOT imported from gap_analysis.py -- that
# module's versions hardcode the A/B/C/D alphabet; these take an explicit
# category count instead).
# -----------------------------------------------------------------------------

def entropy_norm(shares, n_categories):
    """Normalized Shannon entropy over an arbitrary k-category share dict.
    In [0,1]; 1.0 = uniform over all k categories, 0.0 = fully deterministic."""
    ps = np.array([p for p in shares.values() if p > 0])
    if len(ps) == 0 or n_categories <= 1:
        return 0.0
    h = -(ps * np.log2(ps)).sum()
    return float(h / np.log2(n_categories))


def herfindahl(shares):
    """Sum of squared shares. In [1/k, 1.0] for a k-category scale (NOT the
    main study's fixed [0.25,1.0] -- that bound is specific to a 4-way scale)."""
    return float(sum(p ** 2 for p in shares.values()))


def item_shares(series, scale_min, scale_max):
    """Normalized value counts over the FULL FIXED range scale_min..scale_max
    (zero-filled for categories neither sample used) -- required so real and
    synthetic share vectors align positionally for JSD."""
    values = pd.to_numeric(series, errors="coerce").round().astype(int)
    counts = values.value_counts()
    n = len(values)
    return {k: counts.get(k, 0) / n for k in range(scale_min, scale_max + 1)}


# -----------------------------------------------------------------------------
# Per-item distributional comparison
# -----------------------------------------------------------------------------

def per_item_comparison(real_df, synth_df, item_cols, scale_min, scale_max):
    n_categories = scale_max - scale_min + 1
    records = []
    for item in item_cols:
        r_series = pd.to_numeric(real_df[item], errors="coerce")
        s_series = pd.to_numeric(synth_df[item], errors="coerce")
        r_shares = item_shares(r_series, scale_min, scale_max)
        s_shares = item_shares(s_series, scale_min, scale_max)

        P = np.array([[r_shares[k] for k in range(scale_min, scale_max + 1)]])
        Q = np.array([[s_shares[k] for k in range(scale_min, scale_max + 1)]])
        jsd = float(js_divergence(P, Q)[0])

        stat, pvalue = mannwhitneyu(r_series, s_series, alternative="two-sided")

        records.append(
            {
                "item": item,
                "JSD": jsd,
                "real_entropy_norm": entropy_norm(r_shares, n_categories),
                "synth_entropy_norm": entropy_norm(s_shares, n_categories),
                "real_hhi": herfindahl(r_shares),
                "synth_hhi": herfindahl(s_shares),
                "real_n": int(r_series.notna().sum()),
                "synth_n": int(s_series.notna().sum()),
                "real_mean": float(r_series.mean()),
                "synth_mean": float(s_series.mean()),
                "real_median": float(r_series.median()),
                "synth_median": float(s_series.median()),
                "mannwhitney_u": float(stat),
                "p_value": float(pvalue),
            }
        )

    df = pd.DataFrame.from_records(records)
    # Benjamini-Hochberg correction across all simultaneous per-item tests.
    df["q_value"] = false_discovery_control(df["p_value"].values, method="bh")
    df = df.sort_values("JSD", ascending=False).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Classifier two-sample test (real vs. synthetic), on the harmonized items.
# Reuses c2st.py's _permutation_test (already fully generic) but reimplements
# the short fit/eval glue locally rather than importing run_c2st/
# build_chosen_option_features, which are specific to the main study's prompt
# schema -- keeping this side-study from depending on (and risking coupling
# to future changes in) that file.
# -----------------------------------------------------------------------------

def run_c2st_likert(real_df, synth_df, item_cols, seed=42):
    X = pd.concat([real_df[item_cols], synth_df[item_cols]], axis=0).apply(
        pd.to_numeric, errors="coerce"
    ).values.astype(float)
    y = np.array([0] * len(real_df) + [1] * len(synth_df))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Unlike c2st.py's chosen-option features (already within-row z-scored/
    # ranked), raw Likert values are on a shared but unnormalized scale --
    # standardize (fit on train only) so coefficient magnitudes are
    # comparable across items.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X_train_s, y_train)

    proba_test = clf.predict_proba(X_test_s)[:, 1]
    proba_train = clf.predict_proba(X_train_s)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba_test)
    auc_train = roc_auc_score(y_train, proba_train)
    bal_acc = balanced_accuracy_score(y_test, pred_test)
    p_value, perm_aucs = _permutation_test(clf, X_test_s, y_test, auc)

    importances = dict(zip(item_cols, clf.coef_[0].tolist()))

    result = {
        "auc": float(auc),
        "auc_train_in_sample": float(auc_train),
        "balanced_accuracy": float(bal_acc),
        "permutation_p_value": p_value,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_test_real": int(np.sum(y_test == 0)),
        "n_test_synthetic": int(np.sum(y_test == 1)),
        "feature_importances": importances,
        "interpretation": (
            "AUC ~0.5 => real and synthetic response distributions indistinguishable "
            "on these items. AUC ~1.0 => near-perfect separation -- suspect a stray "
            "ID/metadata column that slipped past --exclude rather than a genuine "
            "behavioral gap."
        ),
    }
    return result, perm_aucs, auc


# -----------------------------------------------------------------------------
# Output writing
# -----------------------------------------------------------------------------

def write_outputs(item_df, c2st_result, dataset_name, real_n, synth_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    item_csv = os.path.join(out_dir, "item_comparison.csv")
    item_df.to_csv(item_csv, index=False)

    c2st_path = os.path.join(out_dir, "c2st_result.json")
    with open(c2st_path, "w") as f:
        json.dump(c2st_result, f, indent=2)

    n_sig = int((item_df["q_value"] < 0.05).sum())
    aggregate = {
        "dataset_name": dataset_name,
        "n_items": int(len(item_df)),
        "real_n": int(real_n),
        "synth_n": int(synth_n),
        "mean_JSD": float(item_df["JSD"].mean()),
        "median_JSD": float(item_df["JSD"].median()),
        "max_JSD": float(item_df["JSD"].max()),
        "n_items_q_below_0.05": n_sig,
        "frac_items_q_below_0.05": n_sig / len(item_df),
        "c2st_auc": c2st_result["auc"],
        "c2st_permutation_p_value": c2st_result["permutation_p_value"],
    }
    agg_path = os.path.join(out_dir, "aggregate_summary.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    return item_csv, c2st_path, agg_path, aggregate


def _write_report(dataset_name, item_df, c2st_result, aggregate, out_dir):
    lines = []
    lines.append(f"REAL vs SYNTHETIC LIKERT RESPONSE GAP -- {dataset_name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"real n={aggregate['real_n']}  synthetic n={aggregate['synth_n']}  "
        f"items={aggregate['n_items']}"
    )
    lines.append("")
    lines.append("Per-item distributional gap (top 5 by JSD):")
    for _, row in item_df.head(5).iterrows():
        lines.append(
            f"  {row['item']:<8s} JSD={row['JSD']:.4f}  q={row['q_value']:.4g}  "
            f"real_mean={row['real_mean']:.2f}  synth_mean={row['synth_mean']:.2f}"
        )
    lines.append("")
    lines.append(
        f"Mean JSD across all items: {aggregate['mean_JSD']:.4f} "
        f"(median {aggregate['median_JSD']:.4f}, max {aggregate['max_JSD']:.4f})"
    )
    lines.append(
        f"Items with a significant real-vs-synthetic difference after BH correction "
        f"(q<0.05): {aggregate['n_items_q_below_0.05']}/{aggregate['n_items']} "
        f"({100*aggregate['frac_items_q_below_0.05']:.1f}%)"
    )
    lines.append("")
    lines.append(
        f"Classifier two-sample test: held-out AUC = {c2st_result['auc']:.4f} "
        f"(permutation p = {c2st_result['permutation_p_value']:.4f}), "
        f"balanced accuracy = {c2st_result['balanced_accuracy']:.4f}"
    )
    lines.append("")
    lines.append("LIMITATION: no generator script for this dataset was found anywhere in")
    lines.append("this repo (verified by a full-repo search) -- these numbers describe the")
    lines.append("empirical gap only; they do not validate a known synthetic model's stated")
    lines.append("assumptions. See docs/likert_side_study.md.")

    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)


def _make_figures(real_df, synth_df, item_df, c2st_result, perm_aucs, dataset_name, scale_min, scale_max, out_dir):
    import matplotlib.pyplot as plt

    # fig 1: per-item JSD, sorted descending
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(item_df) + 1.5))
    y = np.arange(len(item_df))
    ax.barh(y, item_df["JSD"], color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(item_df["item"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Jensen-Shannon divergence (real vs. synthetic)")
    ax.set_title(f"{dataset_name}: per-item real-vs-synthetic JSD")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_item_jsd_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig 2: top-4-by-JSD item distributions, real vs synthetic, grouped bars
    top4 = item_df.head(4)
    categories = list(range(scale_min, scale_max + 1))
    xpos = np.arange(len(categories))
    width = 0.35
    fig, axes = plt.subplots(1, len(top4), figsize=(3.2 * len(top4), 3.5), sharey=True)
    if len(top4) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, top4.iterrows()):
        item = row["item"]
        r_shares = item_shares(real_df[item], scale_min, scale_max)
        s_shares = item_shares(synth_df[item], scale_min, scale_max)
        r_vals = [r_shares[k] for k in categories]
        s_vals = [s_shares[k] for k in categories]
        ax.bar(xpos - width / 2, r_vals, width, label="real", color="#4C72B0")
        ax.bar(xpos + width / 2, s_vals, width, label="synthetic", color="#DD8452")
        ax.set_xticks(xpos)
        ax.set_xticklabels(categories, fontsize=7)
        ax.set_title(f"{item}\n(JSD={row['JSD']:.3f})", fontsize=9)
    axes[0].set_ylabel("share")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{dataset_name}: top-4-by-JSD item distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_item_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig 3: C2ST permutation null (same style as c2st.py's)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(perm_aucs, bins=40, color="#888888", alpha=0.7, label="permutation null (shuffled real/synthetic label)")
    ax.axvline(c2st_result["auc"], color="#C44E52", lw=2, label=f"observed AUC = {c2st_result['auc']:.4f}")
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
    ax.set_xlabel("AUC (real vs. synthetic classifier, Likert item features)")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset_name}: classifier two-sample test (p = {c2st_result['permutation_p_value']:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_c2st_permutation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Cross-dataset-pair comparison (--compare_with)
# -----------------------------------------------------------------------------

def compare_with(this_aggregate, other_out_dir, out_dir):
    """Reads another run's aggregate_summary.json, reports both datasets'
    headline numbers side by side, and a plain-text verdict on whether the
    finding replicates in DIRECTION (not magnitude) across two independently
    chosen dataset pairs -- modeled on behavioral_profiles.py's
    validate_against_entropy precedent (read one script's artifact from
    another's output directory, gate on it existing, report a comparison)."""
    other_path = os.path.join(other_out_dir, "aggregate_summary.json")
    if not os.path.exists(other_path):
        print(f"  [likert_gap_analysis] --compare_with: {other_path} not found, skipping")
        return None

    with open(other_path) as f:
        other = json.load(f)

    both_auc_above_chance = this_aggregate["c2st_auc"] > 0.5 and other["c2st_auc"] > 0.5
    both_significant = (
        this_aggregate["c2st_permutation_p_value"] < 0.05
        and other["c2st_permutation_p_value"] < 0.05
    )
    both_majority_items_differ = (
        this_aggregate["frac_items_q_below_0.05"] > 0.5
        and other["frac_items_q_below_0.05"] > 0.5
    )

    if both_auc_above_chance and both_significant:
        verdict = (
            f"REPLICATES: both '{this_aggregate['dataset_name']}' and '{other['dataset_name']}' show "
            f"a classifier two-sample test AUC significantly above chance "
            f"(p<0.05) -- real and synthetic response distributions are "
            f"distinguishable in both independently-chosen consumer-behavior domains."
        )
    else:
        verdict = (
            f"DOES NOT CLEANLY REPLICATE: '{this_aggregate['dataset_name']}' and "
            f"'{other['dataset_name']}' disagree on whether the real-vs-synthetic gap "
            f"is statistically significant -- treat either as inconclusive on its own."
        )

    comparison = {
        "dataset_a": other["dataset_name"],
        "dataset_b": this_aggregate["dataset_name"],
        "mean_JSD_a": other["mean_JSD"],
        "mean_JSD_b": this_aggregate["mean_JSD"],
        "c2st_auc_a": other["c2st_auc"],
        "c2st_auc_b": this_aggregate["c2st_auc"],
        "c2st_p_value_a": other["c2st_permutation_p_value"],
        "c2st_p_value_b": this_aggregate["c2st_permutation_p_value"],
        "frac_items_significant_a": other["frac_items_q_below_0.05"],
        "frac_items_significant_b": this_aggregate["frac_items_q_below_0.05"],
        "both_auc_above_chance": both_auc_above_chance,
        "both_significant": both_significant,
        "both_majority_items_differ": both_majority_items_differ,
        "verdict": verdict,
    }
    path = os.path.join(out_dir, "cross_study_comparison.json")
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[likert_gap_analysis] cross-study comparison:\n  {verdict}")
    return comparison


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_path", required=True)
    ap.add_argument("--synthetic_path", required=True)
    ap.add_argument(
        "--exclude", nargs="*", default=[],
        help="Explicit column denylist (IDs, demographics, derived/composite columns). "
             "Required per-invocation, never auto-inferred -- see docs/likert_side_study.md.",
    )
    ap.add_argument("--scale_min", type=int, required=True)
    ap.add_argument("--scale_max", type=int, required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--compare_with", default=None,
        help="Path to another --out_dir from a prior run of this script, for the "
             "cross-dataset-pair replication check.",
    )
    args = ap.parse_args()

    real_df, synth_df, item_cols = load_and_harmonize(
        args.real_path, args.synthetic_path, args.exclude, args.scale_min, args.scale_max
    )

    print("[likert_gap_analysis] computing per-item distributional comparison...")
    item_df = per_item_comparison(real_df, synth_df, item_cols, args.scale_min, args.scale_max)

    print("[likert_gap_analysis] running classifier two-sample test...")
    c2st_result, perm_aucs, auc = run_c2st_likert(real_df, synth_df, item_cols, seed=args.seed)

    item_csv, c2st_path, agg_path, aggregate = write_outputs(
        item_df, c2st_result, args.dataset_name, len(real_df), len(synth_df), args.out_dir
    )

    _write_report(args.dataset_name, item_df, c2st_result, aggregate, args.out_dir)

    try:
        _make_figures(
            real_df, synth_df, item_df, c2st_result, perm_aucs,
            args.dataset_name, args.scale_min, args.scale_max, args.out_dir,
        )
    except Exception as e:
        print(f"[likert_gap_analysis] WARNING: figure generation skipped ({e})")

    if args.compare_with:
        compare_with(aggregate, args.compare_with, args.out_dir)

    print(f"\n[likert_gap_analysis] DONE. Wrote outputs to {args.out_dir}/")
    print(f"  {item_csv}\n  {c2st_path}\n  {agg_path}")


if __name__ == "__main__":
    main()
