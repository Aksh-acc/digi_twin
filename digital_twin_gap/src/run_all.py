"""
run_all.py
==========
The single entry point that produces every result in the paper.

For each ENABLED architecture it runs the full 2x2 transfer matrix:

               test = agent        test = human
    train=agent   A->A                A->H
    train=human   H->A                H->H

The diagonal (A->A, H->H) is the "matched" twin — trained and tested on the
same population. The off-diagonal (A->H, H->A) is the "transferred" twin. The
difference between them is the GAP: how much fidelity you lose when your twin
was built from the wrong population.

Everything is written to results/:
    results/all_results.json     -- every cell, full metrics + per-agent
    results/summary_matrix.csv   -- flat table (one row per arch x cell)
    results/gap_report.csv       -- the headline gap numbers per architecture

Run (after data_prep):
    python -m src.run_all --config configs/config.yaml
    python -m src.run_all --config configs/config.yaml --models tfidf_logreg
"""

import argparse
import csv
import json
import os
import time

import yaml

from src import metrics
from src.splits import build_splits

# Registry mapping config key -> module implementing fit/predict.
MODEL_REGISTRY = {
    "tfidf_logreg": "src.model_tfidf",
    "embed_mlp": "src.model_embed_mlp",
    "distilbert": "src.model_distilbert",
    # Alias for a hyperparameter-tuned rerun of the same architecture (more
    # epochs / lower lr) -- lets a tuned config live under its own model key
    # ("distilbert_tuned") without touching the original "distilbert" results.
    "distilbert_tuned": "src.model_distilbert",
    # Hierarchical-twin track: Model 1 (statistical baseline, hand-rolled
    # discrete-choice MNL) and Model 2 (hierarchical Bayesian choice model).
    "mnl_baseline": "src.model_mnl_baseline",
    "hier_bayes": "src.model_hier_bayes",
    # Reference baselines. `consensus` is the per-task memorisation ceiling
    # (a prompt->modal-label lookup, no learning); `majority` is the floor.
    # Keeping both as permanent rows makes task-identity leakage visible in
    # every results table -- see the LEAKAGE NOTE in src/splits.py.
    "consensus": "src.model_consensus",
    "majority": "src.model_majority",
}


def _import(path):
    import importlib

    return importlib.import_module(path)


def run_architecture(name, module, splits, cfg):
    """Run all four transfer cells for one architecture."""
    cells = {}
    train_sources = ["agent", "human"]
    test_sources = ["agent", "human"]

    # Train once per train-source, then evaluate on both test-sources.
    for tr in train_sources:
        print(f"\n--- [{name}] training on {tr.upper()} ({len(splits[tr]['train'])} rows) ---")
        t0 = time.time()
        model = module.fit(splits[tr]["train"], cfg)
        train_secs = time.time() - t0

        for te in test_sources:
            test_rows = splits[te]["test"]
            y_true = [r["label"] for r in test_rows]
            y_pred = module.predict(model, test_rows)
            m = metrics.score(y_true, y_pred, test_rows)
            m["train_seconds"] = round(train_secs, 1)
            cells[f"{tr}->{te}"] = m
            print(
                f"    {tr}->{te}: acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  (n={m['n']})"
            )
    return cells


def compute_gaps(cells):
    """Derive the headline gap numbers for one architecture."""
    aa = cells["agent->agent"]["accuracy"]
    ah = cells["agent->human"]["accuracy"]
    ha = cells["human->agent"]["accuracy"]
    hh = cells["human->human"]["accuracy"]
    return {
        "acc_agent_matched": aa,          # train agent, test agent
        "acc_human_matched": hh,          # train human, test human
        "acc_agent_on_human": ah,         # agent twin used on humans
        "acc_human_on_agent": ha,         # human twin used on agents
        # Gap = matched minus transferred (positive => transfer hurts).
        "gap_on_human": hh - ah,          # cost of using an agent twin for humans
        "gap_on_agent": aa - ha,          # cost of using a human twin for agents
    }


def write_outputs(all_results, results_dir):
    """Writes all_results.json, summary_matrix.csv, gap_report.csv. Called
    after EVERY architecture (not just once at the end) so a later crash --
    e.g. a transient CUDA OOM in a subsequent architecture -- can never erase
    results already computed earlier in the same run."""
    json_path = os.path.join(results_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    matrix_path = os.path.join(results_dir, "summary_matrix.csv")
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "cell", "accuracy", "macro_f1", "n", "train_seconds"])
        for name, res in all_results.items():
            for cell, m in res["cells"].items():
                w.writerow(
                    [name, cell, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", m["n"], m.get("train_seconds", "")]
                )

    gap_path = os.path.join(results_dir, "gap_report.csv")
    with open(gap_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "architecture",
                "acc_agent_matched",
                "acc_human_matched",
                "acc_agent_on_human",
                "acc_human_on_agent",
                "gap_on_human",
                "gap_on_agent",
            ]
        )
        for name, res in all_results.items():
            g = res["gaps"]
            w.writerow(
                [
                    name,
                    f"{g['acc_agent_matched']:.4f}",
                    f"{g['acc_human_matched']:.4f}",
                    f"{g['acc_agent_on_human']:.4f}",
                    f"{g['acc_human_on_agent']:.4f}",
                    f"{g['gap_on_human']:+.4f}",
                    f"{g['gap_on_agent']:+.4f}",
                ]
            )
    return json_path, matrix_path, gap_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model keys to run (default: all enabled in config).",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("Building fixed train/test splits (grouped, no leakage)...")
    splits = build_splits(cfg)
    for src in ["agent", "human"]:
        print(
            f"  {src}: train={len(splits[src]['train'])}  test={len(splits[src]['test'])}"
        )

    # Decide which architectures to run.
    if args.models:
        selected = args.models
    else:
        selected = [k for k, v in cfg["models"].items() if v.get("enabled", False)]

    all_results = {}
    failed = []
    for name in selected:
        if name not in MODEL_REGISTRY:
            print(f"[skip] unknown model '{name}'")
            continue
        print("\n" + "=" * 70)
        print(f"ARCHITECTURE: {name}")
        print("=" * 70)
        try:
            module = _import(MODEL_REGISTRY[name])
            cells = run_architecture(name, module, splits, cfg)
            gaps = compute_gaps(cells)
            all_results[name] = {"cells": cells, "gaps": gaps}

            print(f"\n  >> {name} GAP SUMMARY:")
            print(f"     agent-matched acc : {gaps['acc_agent_matched']:.4f}")
            print(f"     human-matched acc : {gaps['acc_human_matched']:.4f}")
            print(f"     gap on humans     : {gaps['gap_on_human']:+.4f}  (human twin - agent twin, tested on humans)")
            print(f"     gap on agents     : {gaps['gap_on_agent']:+.4f}  (agent twin - human twin, tested on agents)")
        except Exception:
            # One architecture failing (e.g. a transient CUDA OOM) must never
            # take down the whole run or erase results already computed for
            # earlier architectures -- log it, skip it, keep going. Outputs
            # are re-written after every architecture (see below), so
            # whatever succeeded before the failure is already safely on disk.
            import traceback

            print(f"\n  !! [{name}] FAILED -- skipping. Traceback:")
            traceback.print_exc()
            failed.append(name)

        # Re-write outputs after every architecture (success or not) so
        # partial progress always survives a later failure.
        write_outputs(all_results, results_dir)

    json_path, matrix_path, gap_path = write_outputs(all_results, results_dir)

    print("\n" + "=" * 70)
    print("DONE. Wrote:")
    print(f"  {json_path}")
    print(f"  {matrix_path}")
    print(f"  {gap_path}")
    if failed:
        print(f"  !! {len(failed)} architecture(s) FAILED and were skipped: {failed}")
        print(f"     Re-run with --models {' '.join(failed)} to retry just those.")
    print("=" * 70)


if __name__ == "__main__":
    main()
