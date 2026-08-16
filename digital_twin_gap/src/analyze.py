"""
analyze.py
==========
Turns results/all_results.json into paper-ready figures and a text summary.

Produces, in results/:
  - fig_transfer_matrix.png   : per-architecture 2x2 heatmap of accuracy
  - fig_gap_bars.png          : matched vs transferred accuracy, side by side
  - fig_per_agent.png         : per-agent accuracy (agent->agent cell)
  - report.txt                : plain-language summary of the gap

Run:
    python -m src.analyze --config configs/config.yaml
"""

import argparse
import json
import os

import yaml


def _load(cfg):
    path = os.path.join(cfg["output"]["results_dir"], "all_results.json")
    with open(path) as f:
        return json.load(f)


def plot_transfer_matrices(results, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    archs = list(results.keys())
    n = len(archs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]

    order = [("agent", "agent"), ("agent", "human"), ("human", "agent"), ("human", "human")]
    for ax, arch in zip(axes, archs):
        cells = results[arch]["cells"]
        mat = np.array(
            [
                [cells["agent->agent"]["accuracy"], cells["agent->human"]["accuracy"]],
                [cells["human->agent"]["accuracy"], cells["human->human"]["accuracy"]],
            ]
        )
        im = ax.imshow(mat, cmap="viridis", vmin=0.2, vmax=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["test:agent", "test:human"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["train:agent", "train:human"])
        ax.set_title(arch)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.55 else "black", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Digital-Twin transfer matrix (choice-prediction accuracy)", y=1.03)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_transfer_matrix.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_gap_bars(results, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    archs = list(results.keys())
    x = np.arange(len(archs))
    w = 0.2

    matched_agent = [results[a]["gaps"]["acc_agent_matched"] for a in archs]
    transfer_agent = [results[a]["gaps"]["acc_human_on_agent"] for a in archs]
    matched_human = [results[a]["gaps"]["acc_human_matched"] for a in archs]
    transfer_human = [results[a]["gaps"]["acc_agent_on_human"] for a in archs]

    fig, ax = plt.subplots(figsize=(2.2 * len(archs) + 3, 4.5))
    ax.bar(x - 1.5 * w, matched_agent, w, label="agent twin \u2192 agents (matched)")
    ax.bar(x - 0.5 * w, transfer_agent, w, label="human twin \u2192 agents (transfer)")
    ax.bar(x + 0.5 * w, matched_human, w, label="human twin \u2192 humans (matched)")
    ax.bar(x + 1.5 * w, transfer_human, w, label="agent twin \u2192 humans (transfer)")
    ax.axhline(0.25, ls="--", c="gray", lw=1, label="random (4-way)")
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("Accuracy")
    ax.set_title("Matched vs transferred twin accuracy")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_gap_bars.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_per_agent(results, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    # Use the agent->agent cell's per-agent breakdown from each architecture.
    archs = list(results.keys())
    agents = None
    data = {}
    for a in archs:
        pa = results[a]["cells"]["agent->agent"].get("per_agent")
        if not pa:
            continue
        if agents is None:
            agents = list(pa.keys())
        data[a] = [pa.get(g, {}).get("accuracy", 0.0) for g in agents]

    if not data:
        return None

    x = np.arange(len(agents))
    w = 0.8 / max(len(data), 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(agents) + 3, 4.5))
    for k, (arch, vals) in enumerate(data.items()):
        ax.bar(x + k * w, vals, w, label=arch)
    ax.set_xticks(x + w * (len(data) - 1) / 2)
    ax.set_xticklabels(agents, rotation=30, ha="right", fontsize=8)
    ax.axhline(0.25, ls="--", c="gray", lw=1)
    ax.set_ylabel("Accuracy (agent-matched twin)")
    ax.set_title("How well the twin models each agent")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_per_agent.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def write_report(results, out_dir):
    lines = []
    lines.append("DIGITAL-TWIN GAP REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Accuracy of predicting the chosen product option (A/B/C/D).")
    lines.append("Random baseline for a 4-way choice = 0.25.")
    lines.append("")
    for arch, res in results.items():
        g = res["gaps"]
        lines.append(f"[{arch}]")
        lines.append(f"  Agent twin on agents (matched)   : {g['acc_agent_matched']:.4f}")
        lines.append(f"  Human twin on humans (matched)   : {g['acc_human_matched']:.4f}")
        lines.append(f"  Agent twin on humans (transfer)  : {g['acc_agent_on_human']:.4f}")
        lines.append(f"  Human twin on agents (transfer)  : {g['acc_human_on_agent']:.4f}")
        lines.append(f"  GAP on humans (matched-transfer) : {g['gap_on_human']:+.4f}")
        lines.append(f"  GAP on agents (matched-transfer) : {g['gap_on_agent']:+.4f}")
        lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 60)
    lines.append(
        "A large positive gap means a twin built from one population is a poor\n"
        "stand-in for the other. If agent-matched accuracy is much higher than\n"
        "human-matched accuracy, it means LLM agents make far more predictable\n"
        "(consistent, rule-like) choices than real people do -- so an 'agentic\n"
        "digital twin' overstates how well it would model human consumers."
    )
    report = "\n".join(lines)
    p = os.path.join(out_dir, "report.txt")
    with open(p, "w") as f:
        f.write(report)
    print(report)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["output"]["results_dir"]
    results = _load(cfg)

    written = []
    try:
        written.append(plot_transfer_matrices(results, out_dir))
        written.append(plot_gap_bars(results, out_dir))
        pa = plot_per_agent(results, out_dir)
        if pa:
            written.append(pa)
    except Exception as e:
        print(f"[warn] plotting skipped ({e}). Text report still produced.")
    written.append(write_report(results, out_dir))

    print("\nWrote:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
