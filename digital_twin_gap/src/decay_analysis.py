"""
decay_analysis.py
==================
Reports the scientific output of the cognitive-decay model: the fitted decay
rate `delta` and the implied decision-sharpness curve s(t) over the session,
for each training source (agent vs human) and, for agents, per individual model.

Unlike run_all.py (which only records accuracy), this refits the cognitive-decay
model on each source / agent and dumps the interpretable decay parameters.

Outputs (results_cognitive_decay/):
  decay_by_source.csv          delta + sharpness start/end for agent & human
  decay_by_agent.csv           delta per individual agent (train on that agent)
  fig_sharpness_curves.png     s(t) curves, agent vs human
  decay_report.md              short reading

Run:
  python -m src.decay_analysis --config configs/config.yaml
"""

import argparse
import csv
import os

import numpy as np
import yaml

from src import model_cognitive_decay as cd
from src.splits import build_splits


def _fit_and_summarize(rows, cfg, tmax_norm):
    model = cd.fit(rows, cfg)
    ts = np.linspace(0, tmax_norm, 50)
    s_curve = model["sp"] * np.exp(-model["delta"] * ts) + model["floor"]
    return model, ts, s_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="results_cognitive_decay")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)

    splits = build_splits(cfg, verbose=False)
    t_scale = float(cfg["models"].get("cognitive_decay", {}).get("trial_scale", 12.0))

    # ---- by source (agent pooled vs human) ----
    by_source = {}
    curves = {}
    for src in ("agent", "human"):
        rows = splits[src]["train"]
        trials = [r.get("trial") for r in rows if r.get("trial") is not None]
        tmax_norm = (max(trials) - 1) / t_scale if trials else 1.0
        model, ts, s_curve = _fit_and_summarize(rows, cfg, tmax_norm)
        by_source[src] = model
        curves[src] = (ts * t_scale + 1, s_curve)  # x back in trial units

    src_csv = os.path.join(args.out, "decay_by_source.csv")
    with open(src_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "decay_rate_delta", "sharpness_start",
                    "sharpness_end", "sharpness_drop_frac", "n_train"])
        for src, m in by_source.items():
            w.writerow([src, f"{m['decay_rate']:.4f}", f"{m['sharpness_start']:.4f}",
                        f"{m['sharpness_end']:.4f}", f"{m['sharpness_drop_frac']:.4f}",
                        len(splits[src]["train"])])

    # ---- per individual agent ----
    agent_rows = splits["agent"]["train"]
    agents = sorted(set(r["group"] for r in agent_rows))
    agent_csv = os.path.join(args.out, "decay_by_agent.csv")
    with open(agent_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "decay_rate_delta", "sharpness_start",
                    "sharpness_end", "sharpness_drop_frac", "n_train"])
        for a in agents:
            rows_a = [r for r in agent_rows if r["group"] == a]
            if len(rows_a) < 30:
                continue
            m = cd.fit(rows_a, cfg)
            w.writerow([a, f"{m['decay_rate']:.4f}", f"{m['sharpness_start']:.4f}",
                        f"{m['sharpness_end']:.4f}", f"{m['sharpness_drop_frac']:.4f}",
                        len(rows_a)])

    # ---- figure: sharpness curves (log-y because agent sharpness >> human) ----
    fig_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for src, style in [("agent", "-"), ("human", "--")]:
            x, y = curves[src]
            ax.plot(x, y, style, color="black",
                    label=f"{src} (\u03b4={by_source[src]['decay_rate']:.3f})")
        ax.set_yscale("log")
        ax.set_xlabel("trial index within session/run")
        ax.set_ylabel("decision sharpness s(t)  (log scale)")
        ax.set_title("Cognitive-decay: decision sharpness over a session")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(args.out, "fig_sharpness_curves.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")

    # ---- report ----
    md = ["# Cognitive-decay model -- findings", ""]
    md.append("`delta` is the fitted decay rate of decision sharpness over a "
              "session; higher = faster degradation into noise. `sharpness_start`"
              " is the initial inverse-temperature (higher = more decisive).\n")
    for src, m in by_source.items():
        md.append(f"- **{src}**: \u03b4 = {m['decay_rate']:.4f}, "
                  f"sharpness {m['sharpness_start']:.2f} \u2192 {m['sharpness_end']:.2f} "
                  f"({100*m['sharpness_drop_frac']:.1f}% drop)")
    md.append("")
    md.append("Note: agent decision sharpness starts far higher than human "
              "(agents are near-deterministic given the option features, humans "
              "are much noisier from the first trial). The decay term captures "
              "*within-run* drift on top of that baseline difference. See "
              "`decay_by_agent.csv` for per-model rates and "
              "`fig_sharpness_curves.png` for the curves.")
    md_path = os.path.join(args.out, "decay_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("Wrote:")
    for pth in [src_csv, agent_csv, fig_path, md_path]:
        if pth:
            print(" ", pth)


if __name__ == "__main__":
    main()
