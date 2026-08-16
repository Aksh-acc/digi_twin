"""
merge_results.py
=================
Safely merges several `all_results.json` files (each holding one or more
architecture keys, e.g. one per `python -m src.run_all --models <subset>`
invocation into its own results_<name>/ directory) into a single NEW
directory, using the exact same JSON/CSV schema `run_all.py` itself writes.

Deliberately standalone -- NOT refactored into run_all.py -- so it carries
zero risk to that already-working file. Raises on architecture-key collisions
across inputs rather than silently overwriting, since a silent merge here is
exactly the kind of mistake that would clobber prior results.

Run:
    python -m src.merge_results \
        --base results_distilbert_tuned/all_results.json \
               results_mnl_baseline/all_results.json \
               results_hier_bayes/all_results.json \
        --out_dir results_full_comparison
"""

import argparse
import csv
import json
import os


def merge_all_results(paths):
    merged = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for key, value in data.items():
            if key in merged:
                raise ValueError(
                    f"architecture key '{key}' appears in more than one input file "
                    f"(latest: {path}) -- refusing to silently overwrite"
                )
            merged[key] = value
    return merged


def write_summary_csv(all_results, out_dir):
    path = os.path.join(out_dir, "summary_matrix.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "cell", "accuracy", "macro_f1", "n", "train_seconds"])
        for name, res in all_results.items():
            for cell, m in res["cells"].items():
                w.writerow(
                    [name, cell, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", m["n"], m.get("train_seconds", "")]
                )
    return path


def write_gap_csv(all_results, out_dir):
    path = os.path.join(out_dir, "gap_report.csv")
    with open(path, "w", newline="") as f:
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
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="one or more all_results.json paths to merge")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    merged = merge_all_results(args.base)
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(merged, f, indent=2)

    matrix_path = write_summary_csv(merged, args.out_dir)
    gap_path = write_gap_csv(merged, args.out_dir)

    print(f"Merged {len(args.base)} file(s) -> {len(merged)} architecture(s): {list(merged.keys())}")
    print("Wrote:")
    print(f"  {json_path}")
    print(f"  {matrix_path}")
    print(f"  {gap_path}")


if __name__ == "__main__":
    main()
