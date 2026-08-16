"""
behavioral_profiles.py
=======================
Extracts the per-agent behavioral posteriors that `model_hier_bayes.py`
already computes but never surfaces: `beta_group` (each agent's deviation from
the population-level sensitivity to price/rating/review count) and
`log_tau_group` (each agent's decision temperature -- how concentrated vs.
noisy its choices are). These ARE the "behavioral profiles" (Grok: high
concentration, strong price sensitivity, ...) the study set out to build, with
full posterior uncertainty, sitting unused in `hier_bayes_model_agent.pkl`.

Built-in validation: independently rank the 6 agents by posterior-mean decision
temperature tau and by empirical normalized entropy (from
`results_gap_analysis/entropy_hhi_by_group.csv`, computed straight from the
raw label distributions, no model involved) and report the Spearman
correlation. These two numbers come from completely different computations --
one from MCMC posteriors over a discrete-choice likelihood, the other from
counting label frequencies -- so agreement between them is real evidence the
twin is capturing something rather than an artifact of the fitting procedure.
Grok has the lowest empirical entropy (0.9680); it should also show the
lowest tau.

Run:
    python -m src.behavioral_profiles --hier_bayes_dir results_corrected_prompt \
        --gap_dir results_gap_analysis --out_dir results_behavioral_profiles
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src import option_features as optfeat

FEATURE_NAMES = optfeat.FEATURE_NAMES  # price_z, price_rank_c, rating_z, rating_rank_c, review_log_z, review_rank_c


def _hdi(samples, prob=0.94):
    """Highest-density interval via arviz (falls back to an equal-tailed
    interval if arviz isn't available -- keeps this module usable even in a
    minimal env)."""
    try:
        import arviz as az

        lo, hi = az.hdi(np.asarray(samples), hdi_prob=prob)
        return float(lo), float(hi)
    except Exception:
        lo = np.percentile(samples, 100 * (1 - prob) / 2)
        hi = np.percentile(samples, 100 * (1 - (1 - prob) / 2))
        return float(lo), float(hi)


def load_agent_model(hier_bayes_dir):
    path = os.path.join(hier_bayes_dir, "hier_bayes_model_agent.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_feature_sensitivity_table(model, hdi_prob=0.94):
    post = model["posterior"]
    group_names = model["group_names"]
    beta_pop = post["beta_pop"]  # (S,6)
    beta_group = post["beta_group"]  # (S,6 agents,6 feats)

    records = []
    for gi, agent in enumerate(group_names):
        for ki, feat in enumerate(FEATURE_NAMES):
            dev = beta_group[:, gi, ki]
            total = beta_pop[:, ki] + dev
            dev_lo, dev_hi = _hdi(dev, hdi_prob)
            tot_lo, tot_hi = _hdi(total, hdi_prob)
            records.append(
                {
                    "agent": agent,
                    "feature": feat,
                    "deviation_mean": float(dev.mean()),
                    f"deviation_hdi{int(100*hdi_prob)}_lo": dev_lo,
                    f"deviation_hdi{int(100*hdi_prob)}_hi": dev_hi,
                    "total_sensitivity_mean": float(total.mean()),
                    f"total_hdi{int(100*hdi_prob)}_lo": tot_lo,
                    f"total_hdi{int(100*hdi_prob)}_hi": tot_hi,
                }
            )
    return pd.DataFrame.from_records(records)


def build_temperature_table(model, hdi_prob=0.94):
    post = model["posterior"]
    group_names = model["group_names"]
    log_tau_group = post["log_tau_group"]  # (S,6)
    tau = np.exp(log_tau_group)

    records = []
    for gi, agent in enumerate(group_names):
        lo, hi = _hdi(tau[:, gi], hdi_prob)
        records.append(
            {
                "agent": agent,
                "tau_mean": float(tau[:, gi].mean()),
                f"tau_hdi{int(100*hdi_prob)}_lo": lo,
                f"tau_hdi{int(100*hdi_prob)}_hi": hi,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("tau_mean")


def validate_against_entropy(tau_df, gap_dir):
    """Spearman correlation between posterior-mean tau rank and empirical
    entropy rank (independent computations -- see module docstring)."""
    from scipy.stats import spearmanr

    entropy_path = os.path.join(gap_dir, "entropy_hhi_by_group.csv")
    if not os.path.exists(entropy_path):
        print(f"  [behavioral_profiles] WARNING: {entropy_path} not found, skipping tau-vs-entropy validation")
        return None

    entropy_df = pd.read_csv(entropy_path)
    entropy_df = entropy_df[(entropy_df["category"] == "ALL") & (entropy_df["group"] != "Human (pooled)")]

    merged = tau_df.merge(entropy_df[["group", "entropy_norm"]], left_on="agent", right_on="group", how="inner")
    if len(merged) < 3:
        print("  [behavioral_profiles] WARNING: not enough matched agents to compute a correlation")
        return None

    rho, pval = spearmanr(merged["tau_mean"], merged["entropy_norm"])
    print(f"  [behavioral_profiles] Spearman(tau, empirical entropy) = {rho:.4f}  (p={pval:.4f}, n={len(merged)})")
    print("  agent ranking (low tau/entropy = more concentrated -> high):")
    for _, row in merged.sort_values("tau_mean").iterrows():
        print(f"    {row['agent']:<20s} tau_mean={row['tau_mean']:.4f}  entropy_norm={row['entropy_norm']:.4f}")
    return {"spearman_rho": float(rho), "p_value": float(pval), "n": len(merged)}


def _make_figures(sensitivity_df, tau_df, out_dir):
    import matplotlib.pyplot as plt

    # Forest plot: decision temperature by agent, sorted low->high.
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(tau_df) + 1.5))
    y = np.arange(len(tau_df))
    lo_col = [c for c in tau_df.columns if c.endswith("_lo")][0]
    hi_col = [c for c in tau_df.columns if c.endswith("_hi")][0]
    xerr = np.abs(np.vstack([tau_df["tau_mean"] - tau_df[lo_col], tau_df[hi_col] - tau_df["tau_mean"]]))
    ax.errorbar(tau_df["tau_mean"], y, xerr=xerr, fmt="o", color="#4C72B0", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(tau_df["agent"])
    ax.set_xlabel("Decision temperature τ (posterior mean, 94% HDI)")
    ax.set_title("Per-agent decision temperature — lower = more concentrated/rule-like")
    ax.axvline(1.0, ls="--", c="gray", lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tau_by_agent.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Forest plot grid: total feature sensitivity by agent, one panel per feature.
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    agents = sorted(sensitivity_df["agent"].unique())
    y = np.arange(len(agents))
    for ax, feat in zip(axes.ravel(), FEATURE_NAMES):
        sub = sensitivity_df[sensitivity_df["feature"] == feat].set_index("agent").loc[agents]
        lo_col = [c for c in sub.columns if c.startswith("total_hdi") and c.endswith("_lo")][0]
        hi_col = [c for c in sub.columns if c.startswith("total_hdi") and c.endswith("_hi")][0]
        xerr = np.abs(
            np.vstack([sub["total_sensitivity_mean"] - sub[lo_col], sub[hi_col] - sub["total_sensitivity_mean"]])
        )
        ax.errorbar(sub["total_sensitivity_mean"], y, xerr=xerr, fmt="o", color="#DD8452", capsize=2)
        ax.axvline(0, ls="--", c="gray", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(agents, fontsize=8)
        ax.set_title(feat, fontsize=10)
    fig.suptitle("Per-agent total feature sensitivity (population + agent deviation, 94% HDI)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_feature_sensitivity_by_agent.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier_bayes_dir", default="results_hier_bayes",
                     help="dir holding hier_bayes_model_agent.pkl")
    ap.add_argument("--gap_dir", default="results_gap_analysis",
                     help="dir holding entropy_hhi_by_group.csv for the tau-vs-entropy validation")
    ap.add_argument("--out_dir", default="results_behavioral_profiles")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[behavioral_profiles] loading {args.hier_bayes_dir}/hier_bayes_model_agent.pkl ...")
    model = load_agent_model(args.hier_bayes_dir)

    sensitivity_df = build_feature_sensitivity_table(model)
    sensitivity_df.to_csv(os.path.join(args.out_dir, "feature_sensitivity_by_agent.csv"), index=False)

    tau_df = build_temperature_table(model)
    tau_df.to_csv(os.path.join(args.out_dir, "decision_temperature_by_agent.csv"), index=False)

    print("\nDecision temperature by agent (posterior mean, low->high = most->least concentrated):")
    for _, row in tau_df.iterrows():
        lo_col = [c for c in tau_df.columns if c.endswith("_lo")][0]
        hi_col = [c for c in tau_df.columns if c.endswith("_hi")][0]
        print(f"  {row['agent']:<20s} tau={row['tau_mean']:.4f}  [{row[lo_col]:.4f}, {row[hi_col]:.4f}]")

    validation = validate_against_entropy(tau_df, args.gap_dir)
    if validation:
        import json

        with open(os.path.join(args.out_dir, "tau_entropy_validation.json"), "w") as f:
            json.dump(validation, f, indent=2)

    try:
        _make_figures(sensitivity_df, tau_df, args.out_dir)
    except Exception as e:
        print(f"[behavioral_profiles] WARNING: figure generation skipped ({e})")

    print(f"\n[behavioral_profiles] DONE. Wrote outputs to {args.out_dir}/")


if __name__ == "__main__":
    main()
