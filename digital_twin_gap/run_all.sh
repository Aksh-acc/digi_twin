#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot runner for the Digital-Twin Gap study.
# Runs: data prep -> all experiments -> analysis/plots.
#
# Usage:
#   bash run_all.sh                 # all three architectures
#   bash run_all.sh tfidf_logreg    # just one (quick, CPU only)
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

MODELS="${@:-tfidf_logreg embed_mlp distilbert}"

echo ">>> [1/3] Building unified dataset ..."
python -m src.data_prep --config configs/config.yaml

echo ">>> [2/3] Running experiments for: $MODELS"
python -m src.run_all --config configs/config.yaml --models $MODELS

echo ">>> [3/3] Analyzing + plotting ..."
python -m src.analyze --config configs/config.yaml

echo ">>> Done. See results/ for figures, CSVs, and report.txt"
