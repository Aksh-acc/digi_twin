"""
gap_analysis.py
================
The Human-Agent Gap Analyzer. Three tiers, none of which require re-fitting
any model:

1. EMPIRICAL entropy / Herfindahl-Hirschman concentration -- straight from
   `data/unified/{agent,human}.jsonl`, no model needed.
2. INTER-AGENT AGREEMENT -- Fleiss' kappa + pairwise agreement among the 6
   LLM agents, computed on the verified set of 334 prompts (of the 384 total)
   that all 6 agents actually answered (Fleiss' kappa requires a fixed rater
   count per item; Grok is missing 50 of the 384).
3. MODEL-BASED FLAGSHIP METRIC -- reuses the two pickled `hier_bayes`
   posteriors (agent-trained + human-trained, produced by
   `python -m src.run_all --models hier_bayes`, see model_hier_bayes.py) to
   compute each twin's posterior-predictive P(A/B/C/D|x) on the SAME rows,
   then the row-wise Jensen-Shannon divergence between them. This
   operationalizes Delta(x) = P_A(y|x) - P_H(y|x) and the flagship metric
   D_HA = JSD(P_H, P_A) using only data structures that already exist in the
   pipeline (no cross-population task ID bridge is available or needed).

Output: a NEW directory (default `results_gap_analysis/`) -- never touches
`results/`, `results_distilbert_tuned/`, `results_mnl_baseline/`, or
`results_hier_bayes/`.

Run:
    python -m src.gap_analysis --config configs/config_hier_bayes.yaml --out_dir results_gap_analysis
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from src import option_features as optfeat
from src.splits import build_splits, load_source

OPTION_LABELS = optfeat.OPTION_LABELS  # ["A","B","C","D"]


# -----------------------------------------------------------------------------
# Tier 1: empirical entropy / Herfindahl-Hirschman
# -----------------------------------------------------------------------------

def empirical_label_shares(rows):
    """rows: list of unified-schema dicts (already filtered to one group /
    category). -> {'A':share, 'B':share, ...} over A-D only; any stray non-A-D
    label is dropped with a printed count (expected: 0, verified this dataset
    has none) and shares renormalized over A-D."""
    counts = defaultdict(int)
    n_bad = 0
    for r in rows:
        if r["label"] in OPTION_LABELS:
            counts[r["label"]] += 1
        else:
            n_bad += 1
    if n_bad:
        print(f"  [gap_analysis] WARNING: {n_bad} row(s) with non-A-D label dropped from label shares")
    total = sum(counts.values())
    if total == 0:
        return {l: 0.0 for l in OPTION_LABELS}, 0
    return {l: counts.get(l, 0) / total for l in OPTION_LABELS}, total


def shannon_entropy_norm(shares):
    """Normalized Shannon entropy over a 4-way label distribution, in [0,1].
    1.0 = uniform (max entropy), 0.0 = fully deterministic. 0*log2(0) := 0."""
    ps = np.array([shares.get(l, 0.0) for l in OPTION_LABELS])
    ps = ps[ps > 0]
    if len(ps) == 0:
        return 0.0
    h = -(ps * np.log2(ps)).sum()
    return float(h / np.log2(len(OPTION_LABELS)))


def herfindahl_index(shares):
    """Sum of squared label shares, in [0.25, 1.0] for a 4-way choice.
    0.25 = uniform (least concentrated), 1.0 = fully concentrated on one option."""
    ps = np.array([shares.get(l, 0.0) for l in OPTION_LABELS])
    return float((ps ** 2).sum())


def per_group_entropy_hhi(agent_rows, human_rows, by_category=False):
    """-> DataFrame with columns: group, category, n, entropy_norm, hhi.
    `group` is one of the 6 agent identities, or the literal string "Human
    (pooled)" for the full human population (individual participants have far
    too few trials each for a meaningful per-participant entropy estimate)."""
    records = []

    def _add(group_label, rows, category):
        shares, n = empirical_label_shares(rows)
        if n == 0:
            return
        records.append(
            {
                "group": group_label,
                "category": category,
                "n": n,
                "entropy_norm": shannon_entropy_norm(shares),
                "hhi": herfindahl_index(shares),
            }
        )

    agent_groups = sorted(set(r["group"] for r in agent_rows))
    cats = ["ALL"] + (sorted(set(r["category"] for r in agent_rows)) if by_category else [])

    for g in agent_groups:
        g_rows = [r for r in agent_rows if r["group"] == g]
        _add(g, g_rows, "ALL")
        if by_category:
            for c in sorted(set(r["category"] for r in g_rows)):
                _add(g, [r for r in g_rows if r["category"] == c], c)

    _add("Human (pooled)", human_rows, "ALL")
    if by_category:
        for c in sorted(set(r["category"] for r in human_rows)):
            _add("Human (pooled)", [r for r in human_rows if r["category"] == c], c)

    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Tier 2: inter-agent agreement (Fleiss' kappa + pairwise), on the verified
# 334-of-384 prompts that all 6 agents answered.
# -----------------------------------------------------------------------------

def build_agent_rating_matrix(agent_rows):
    """-> (rating_matrix: (n_items,4) int counts, task_prompts: list[str],
    agent_names: list[str]). Groups by literal `prompt` string as the task
    key (valid within the agent population: all 6 agents answer byte-identical
    prompts), restricted to prompts with responses from ALL agents present in
    `agent_rows` (Fleiss' kappa requires a fixed rater count per item)."""
    agent_names = sorted(set(r["group"] for r in agent_rows))
    n_agents = len(agent_names)

    by_prompt = defaultdict(dict)  # prompt -> {agent_name: label}
    for r in agent_rows:
        if r["label"] not in OPTION_LABELS:
            continue
        by_prompt[r["prompt"]][r["group"]] = r["label"]

    full_prompts = [p for p, labels in by_prompt.items() if len(labels) == n_agents]
    label2idx = {l: i for i, l in enumerate(OPTION_LABELS)}
    rating_matrix = np.zeros((len(full_prompts), 4), dtype=int)
    for i, prompt in enumerate(full_prompts):
        for agent, label in by_prompt[prompt].items():
            rating_matrix[i, label2idx[label]] += 1

    return rating_matrix, full_prompts, agent_names


def fleiss_kappa(rating_matrix):
    """Standard Fleiss (1971) kappa. rating_matrix: (n_items, k_categories)
    integer counts, each row summing to the same N (raters/item)."""
    n_items, k = rating_matrix.shape
    N = rating_matrix.sum(axis=1)
    if not np.all(N == N[0]):
        raise ValueError("Fleiss' kappa requires a fixed rater count per item")
    N = int(N[0])

    P_i = (np.sum(rating_matrix ** 2, axis=1) - N) / (N * (N - 1))
    P_bar = P_i.mean()
    p_j = rating_matrix.sum(axis=0) / (n_items * N)
    P_bar_e = np.sum(p_j ** 2)

    if np.isclose(P_bar_e, 1.0):
        raise ValueError(
            "Fleiss' kappa undefined: expected agreement P_bar_e == 1.0 "
            "(one category received all votes)"
        )
    kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
    return float(kappa), n_items, N


def _kappa_interpretation(k):
    # Landis & Koch (1977) benchmark scale.
    if k < 0:
        return "poor"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


def pairwise_agreement(agent_rows):
    """-> DataFrame (agent x agent), entry (a,b) = fraction of prompts BOTH a
    and b answered where their chosen label matched. Computed per-pair (not
    forced to the 334-prompt global set) since a pair not involving Grok has
    all 384 prompts in common, while a pair involving Grok has 334."""
    by_prompt = defaultdict(dict)
    for r in agent_rows:
        if r["label"] not in OPTION_LABELS:
            continue
        by_prompt[r["prompt"]][r["group"]] = r["label"]

    agent_names = sorted(set(r["group"] for r in agent_rows))
    mat = pd.DataFrame(index=agent_names, columns=agent_names, dtype=float)
    for a in agent_names:
        for b in agent_names:
            if a == b:
                mat.loc[a, b] = 1.0
                continue
            shared = [
                (labels[a], labels[b]) for labels in by_prompt.values() if a in labels and b in labels
            ]
            if not shared:
                mat.loc[a, b] = np.nan
                continue
            agree = sum(1 for la, lb in shared if la == lb)
            mat.loc[a, b] = agree / len(shared)
    return mat


# -----------------------------------------------------------------------------
# Tier 3: model-based flagship gap metric (JSD between the two hier_bayes twins)
# -----------------------------------------------------------------------------

def load_hier_bayes_models(results_dir):
    with open(os.path.join(results_dir, "hier_bayes_model_agent.pkl"), "rb") as f:
        model_agent = pickle.load(f)
    with open(os.path.join(results_dir, "hier_bayes_model_human.pkl"), "rb") as f:
        model_human = pickle.load(f)
    return model_agent, model_human


def js_divergence(P, Q, eps=1e-10):
    """Row-wise Jensen-Shannon divergence, base-2 log -> bounded in [0,1].
    P, Q: (n,4) probability arrays."""
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * np.log2(P / M), axis=1)
    kl_qm = np.sum(Q * np.log2(Q / M), axis=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def kl_divergence(P, Q, eps=1e-10):
    """Row-wise KL(P||Q), base-2, clipped for stability."""
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.sum(P * np.log2(P / Q), axis=1)


def model_based_gap(model_agent, model_human, eval_rows):
    """-> DataFrame: row_idx, category, true_label, P_agent_{A..D},
    P_human_{A..D}, JSD, KL_HA, KL_AH."""
    from src import model_hier_bayes as hb

    P_A = hb.predict_proba(model_agent, eval_rows)  # (n,4)
    P_H = hb.predict_proba(model_human, eval_rows)  # (n,4)

    jsd = js_divergence(P_A, P_H)
    kl_ha = kl_divergence(P_H, P_A)  # KL(P_H || P_A)
    kl_ah = kl_divergence(P_A, P_H)  # KL(P_A || P_H)

    records = []
    for i, r in enumerate(eval_rows):
        rec = {
            "row_idx": i,
            "category": r["category"],
            "true_label": r["label"],
            "JSD": jsd[i],
            "KL_HA": kl_ha[i],
            "KL_AH": kl_ah[i],
        }
        for j, l in enumerate(OPTION_LABELS):
            rec[f"P_agent_{l}"] = P_A[i, j]
            rec[f"P_human_{l}"] = P_H[i, j]
        records.append(rec)
    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_hier_bayes.yaml",
                     help="config whose output.results_dir holds the pickled hier_bayes models")
    ap.add_argument("--out_dir", default="results_gap_analysis")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    unified = cfg["data"]["unified_dir"]
    agent_rows = load_source(f"{unified}/agent.jsonl")
    human_rows = load_source(f"{unified}/human.jsonl")

    # --- Tier 1: empirical entropy / HHI ---
    print("[gap_analysis] Tier 1: empirical entropy / Herfindahl-Hirschman...")
    entropy_hhi_df = per_group_entropy_hhi(agent_rows, human_rows, by_category=True)
    entropy_hhi_df.to_csv(os.path.join(args.out_dir, "entropy_hhi_by_group.csv"), index=False)

    share_records = []
    for group_label, rows in list({g: [r for r in agent_rows if r["group"] == g] for g in sorted(set(r["group"] for r in agent_rows))}.items()) + [("Human (pooled)", human_rows)]:
        for cat in ["ALL"] + sorted(set(r["category"] for r in rows)):
            sub = rows if cat == "ALL" else [r for r in rows if r["category"] == cat]
            shares, n = empirical_label_shares(sub)
            for label, share in shares.items():
                share_records.append({"group": group_label, "category": cat, "label": label, "share": share, "n": n})
    pd.DataFrame.from_records(share_records).to_csv(
        os.path.join(args.out_dir, "empirical_label_shares.csv"), index=False
    )

    # --- Tier 2: inter-agent agreement ---
    print("[gap_analysis] Tier 2: inter-agent Fleiss' kappa + pairwise agreement...")
    rating_matrix, full_prompts, agent_names = build_agent_rating_matrix(agent_rows)
    kappa, n_items, n_raters = fleiss_kappa(rating_matrix)
    with open(os.path.join(args.out_dir, "fleiss_kappa.json"), "w") as f:
        json.dump(
            {
                "fleiss_kappa": kappa,
                "n_items": n_items,
                "n_raters": n_raters,
                "agent_names": agent_names,
                "interpretation": _kappa_interpretation(kappa),
            },
            f,
            indent=2,
        )
    pairwise_df = pairwise_agreement(agent_rows)
    pairwise_df.to_csv(os.path.join(args.out_dir, "pairwise_agent_agreement.csv"))
    print(f"  Fleiss' kappa = {kappa:.4f} ({_kappa_interpretation(kappa)}), n_items={n_items}, n_raters={n_raters}")

    # --- Tier 3: model-based flagship gap metric ---
    hier_bayes_dir = cfg["output"]["results_dir"]
    agent_pkl = os.path.join(hier_bayes_dir, "hier_bayes_model_agent.pkl")
    human_pkl = os.path.join(hier_bayes_dir, "hier_bayes_model_human.pkl")
    gap_summary = {}
    if os.path.exists(agent_pkl) and os.path.exists(human_pkl):
        print("[gap_analysis] Tier 3: model-based JSD gap (using pickled hier_bayes twins)...")
        model_agent, model_human = load_hier_bayes_models(hier_bayes_dir)
        splits = build_splits(cfg)

        for eval_name, eval_rows in [("agent_test", splits["agent"]["test"]), ("human_test", splits["human"]["test"])]:
            df = model_based_gap(model_agent, model_human, eval_rows)
            df.to_csv(os.path.join(args.out_dir, f"model_based_gap_on_{eval_name}.csv"), index=False)
            gap_summary[eval_name] = {
                "mean_JSD": float(df["JSD"].mean()),
                "median_JSD": float(df["JSD"].median()),
                "mean_JSD_by_category": df.groupby("category")["JSD"].mean().to_dict(),
                "n": len(df),
            }
            print(f"  {eval_name}: mean JSD(P_agent, P_human) = {df['JSD'].mean():.4f} (n={len(df)})")
    else:
        print(
            f"[gap_analysis] SKIPPING Tier 3: pickled hier_bayes models not found in "
            f"'{hier_bayes_dir}' (run `python -m src.run_all --config {args.config} "
            f"--models hier_bayes` first)"
        )

    with open(os.path.join(args.out_dir, "gap_summary.json"), "w") as f:
        json.dump(gap_summary, f, indent=2)

    # --- Figures + report ---
    try:
        _make_figures(entropy_hhi_df, pairwise_df, kappa, gap_summary, args.out_dir)
    except Exception as e:
        print(f"[gap_analysis] WARNING: figure generation skipped ({e})")

    _write_report(entropy_hhi_df, kappa, n_items, n_raters, gap_summary, args.out_dir)

    print(f"\n[gap_analysis] DONE. Wrote outputs to {args.out_dir}/")


def _make_figures(entropy_hhi_df, pairwise_df, kappa, gap_summary, out_dir):
    import matplotlib.pyplot as plt

    # fig 1: entropy + HHI by group (ALL-category rows only)
    df_all = entropy_hhi_df[entropy_hhi_df["category"] == "ALL"].copy()
    df_all = df_all.sort_values("group")
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(df_all) + 3, 4.5))
    x = np.arange(len(df_all))
    axes[0].bar(x, df_all["entropy_norm"], color="#4C72B0")
    axes[0].axhline(1.0, ls="--", c="gray", lw=1, label="uniform (max)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_all["group"], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Normalized Shannon entropy")
    axes[0].set_title("Choice entropy by group")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)

    axes[1].bar(x, df_all["hhi"], color="#DD8452")
    axes[1].axhline(0.25, ls="--", c="gray", lw=1, label="uniform (min)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_all["group"], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Herfindahl-Hirschman index")
    axes[1].set_title("Choice concentration by group")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_entropy_hhi_by_agent.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig 2: pairwise agent kappa/agreement heatmap
    fig, ax = plt.subplots(figsize=(5.5, 5))
    mat = pairwise_df.astype(float).values
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pairwise_df.columns)))
    ax.set_xticklabels(pairwise_df.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pairwise_df.index)))
    ax.set_yticklabels(pairwise_df.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Pairwise agent agreement (overall Fleiss' κ = {kappa:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_pairwise_agent_kappa_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig 3: JSD distribution (if Tier 3 ran)
    if gap_summary:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = {"agent_test": "#4C72B0", "human_test": "#DD8452"}
        for eval_name in gap_summary:
            path = os.path.join(out_dir, f"model_based_gap_on_{eval_name}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                ax.hist(df["JSD"], bins=30, alpha=0.6, label=f"{eval_name} (mean={df['JSD'].mean():.3f})",
                        color=colors.get(eval_name))
        ax.set_xlabel("Jensen-Shannon divergence: JSD(P_agent_twin, P_human_twin)")
        ax.set_ylabel("count")
        ax.set_title("Human-Agent twin divergence (D_HA) distribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_jsd_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _write_report(entropy_hhi_df, kappa, n_items, n_raters, gap_summary, out_dir):
    lines = []
    lines.append("HUMAN-AGENT GAP ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Tier 1 -- empirical choice entropy / concentration (ALL categories pooled):")
    df_all = entropy_hhi_df[entropy_hhi_df["category"] == "ALL"].sort_values("group")
    for _, row in df_all.iterrows():
        lines.append(
            f"  {row['group']:<20s} entropy_norm={row['entropy_norm']:.4f}  hhi={row['hhi']:.4f}  (n={row['n']})"
        )
    lines.append("")
    lines.append(f"Tier 2 -- inter-agent agreement (n_items={n_items}, n_raters={n_raters}):")
    lines.append(f"  Fleiss' kappa = {kappa:.4f} ({_kappa_interpretation(kappa)}, Landis & Koch scale)")
    lines.append("")
    if gap_summary:
        lines.append("Tier 3 -- model-based flagship gap metric D_HA = JSD(P_human_twin, P_agent_twin):")
        for eval_name, s in gap_summary.items():
            lines.append(f"  on {eval_name}: mean JSD = {s['mean_JSD']:.4f}  median = {s['median_JSD']:.4f}  (n={s['n']})")
    else:
        lines.append("Tier 3 -- SKIPPED (no fitted hier_bayes models found)")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 60)
    lines.append(
        "Higher entropy_norm / lower hhi means a group's choices are less\n"
        "predictable (closer to uniform over A/B/C/D). Fleiss' kappa quantifies\n"
        "how much the 6 LLM agents agree with each other on identical tasks,\n"
        "beyond chance. D_HA (mean JSD) is the flagship number: how much the\n"
        "agent-trained twin's implied choice distribution diverges from the\n"
        "human-trained twin's, on the same held-out decision tasks -- a value\n"
        "of 0 would mean the twins are indistinguishable; this study's premise\n"
        "is that it is meaningfully greater than 0."
    )
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
