"""
option_features.py
===================
Shared per-option feature-engineering utility for the discrete-choice models
(`model_mnl_baseline.py`, `model_hier_bayes.py`).

The unified schema (`data/unified/{agent,human}.jsonl`) only carries a `prompt`
blob -- there is no separate per-option attribute column anywhere upstream. But
the prompt text is a consistently templated block per option:

    OPTION: A
    ...
    PRICE_USD: 295
    STAR_RATING: 4.7          <-- agent template: plain, + a separate REVIEW_COUNT line
    REVIEW_COUNT: 330
    ...

    OPTION: A
    ...
    PRICE_USD: 160
    STAR_RATING: 4.7 (290 reviews)   <-- human template: review count embedded in parens
    ...

This module regexes PRICE_USD / STAR_RATING / review-count out of each of the
4 option blocks and turns them into a small, interpretable, WITHIN-ROW-normalized
feature matrix suitable for a discrete-choice utility model. "Within-row" matters:
every feature is computed relative to the other 3 options in the same row, so
`predict()` needs no persisted scaler/statistics from training.

Verified against the full 8270-row unified corpus (both files): every row has
exactly 4 `OPTION:` markers in A,B,C,D order, and PRICE_USD/STAR_RATING are
present in all 4 options of all 8270 rows (0% missing). The missing-value
fallback below is defensive -- it is not expected to fire on this dataset, but
must not silently corrupt the pipeline if the data ever changes.
"""

import re
from collections import OrderedDict

import numpy as np

OPTION_LABELS = ["A", "B", "C", "D"]
FEATURE_NAMES = [
    "price_z", "price_rank_c",
    "rating_z", "rating_rank_c",
    "review_log_z", "review_rank_c",
]

_OPTION_RE = re.compile(r"OPTION:\s*([A-D])\b")
_PRICE_RE = re.compile(r"PRICE_USD:\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_STAR_RE = re.compile(
    r"STAR_RATING:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\(\s*([0-9][0-9,]*)\s*reviews?\s*\))?"
)
_REVIEW_RE = re.compile(r"REVIEW_COUNT:\s*([0-9][0-9,]*)")

# Fallback constants used only if an option is missing a field AND every other
# option in the same row is also missing it (never observed on real data).
_FALLBACK = {"price_usd": 100.0, "star_rating": 4.0, "review_count": 100.0}

# In-process memo: parsing the whole 8270-row corpus benchmarks at ~0.15s, so a
# disk cache isn't worth the staleness risk -- but fit() and predict() (and
# gap_analysis.py reusing the same rows) can end up parsing identical prompts
# more than once within a single process, so a cheap per-process memo avoids
# redundant regex work.
_memo = {}


def split_option_blocks(prompt):
    """Return {'A': block_text, 'B': ..., 'C': ..., 'D': ...}.

    Slices between successive `OPTION: X` markers (not a whole-prompt
    `findall`) so that a field missing from one option's block can never
    silently misalign the fields read from a later option.
    """
    matches = list(_OPTION_RE.finditer(prompt))
    letters = [m.group(1) for m in matches]
    if letters != OPTION_LABELS:
        raise ValueError(
            f"expected OPTION markers {OPTION_LABELS} in order, got {letters!r}"
        )
    blocks = OrderedDict()
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        blocks[letters[i]] = prompt[start:end]
    return blocks


def _to_float(numeric_str):
    return float(numeric_str.replace(",", ""))


def parse_option_fields(block_text):
    """-> {'price_usd': float|None, 'star_rating': float|None,
           'review_count': float|None, 'missing': set[str]}"""
    out = {"price_usd": None, "star_rating": None, "review_count": None}
    missing = set()

    m = _PRICE_RE.search(block_text)
    if m:
        out["price_usd"] = _to_float(m.group(1))
    else:
        missing.add("price_usd")

    m = _STAR_RE.search(block_text)
    review_from_star = None
    if m:
        out["star_rating"] = _to_float(m.group(1))
        if m.group(2):
            review_from_star = _to_float(m.group(2))
    else:
        missing.add("star_rating")

    m = _REVIEW_RE.search(block_text)
    if m:
        out["review_count"] = _to_float(m.group(1))
    elif review_from_star is not None:
        out["review_count"] = review_from_star
    else:
        missing.add("review_count")

    out["missing"] = missing
    return out


def parse_row_raw(prompt):
    """4 dicts (A..D order), missing-value fallback applied.
    Each dict additionally carries 'imputed': set[str]."""
    blocks = split_option_blocks(prompt)
    parsed = [parse_option_fields(blocks[letter]) for letter in OPTION_LABELS]

    for field in ("price_usd", "star_rating", "review_count"):
        vals = [p[field] for p in parsed if field not in p["missing"]]
        for p in parsed:
            if field in p["missing"]:
                p.setdefault("imputed", set()).add(field)
                if vals:
                    p[field] = float(np.median(vals))
                else:
                    p[field] = _FALLBACK[field]
    for p in parsed:
        p.setdefault("imputed", set())
    return parsed


def engineer_features(raw4):
    """raw4: 4 dicts (A..D order) from parse_row_raw(). -> (4,6) float array,
    column order = FEATURE_NAMES. Every feature is computed WITHIN this row
    only (relative to the other 3 options), so no persisted training
    statistics are needed at predict time."""

    def _z_and_rank(values):
        values = np.asarray(values, dtype=float)
        mean, std = values.mean(), values.std()
        z = (values - mean) / (std + 1e-6)
        # average rank (1=lowest..4=highest), centered so it's mean-zero
        order = values.argsort()
        ranks = np.empty(4)
        ranks[order] = np.arange(1, 5)
        # average ranks for ties
        for v in np.unique(values):
            idx = np.where(values == v)[0]
            if len(idx) > 1:
                ranks[idx] = ranks[idx].mean()
        rank_c = ranks - 2.5
        return z, rank_c

    price = [r["price_usd"] for r in raw4]
    rating = [r["star_rating"] for r in raw4]
    review_log = [np.log1p(r["review_count"]) for r in raw4]

    price_z, price_rank_c = _z_and_rank(price)
    rating_z, rating_rank_c = _z_and_rank(rating)
    review_log_z, review_rank_c = _z_and_rank(review_log)

    return np.stack(
        [price_z, price_rank_c, rating_z, rating_rank_c, review_log_z, review_rank_c],
        axis=1,
    )  # (4, 6)


def compute_option_features(prompt):
    """Memoized. -> (X: (4,6) float array, meta: dict)"""
    key = hash(prompt)
    if key in _memo:
        return _memo[key]
    raw4 = parse_row_raw(prompt)
    X = engineer_features(raw4)
    has_missing = any(r["imputed"] for r in raw4)
    meta = {"raw": raw4, "has_missing": has_missing}
    _memo[key] = (X, meta)
    return X, meta


def build_feature_matrix(rows):
    """rows: list of unified-schema dicts (must have 'prompt', 'label').
    -> (X: (n,4,6) float, y_idx: (n,) int in {0..3}, has_missing: list[bool])

    Rows whose label isn't one of A/B/C/D are dropped (logged), matching
    `model_distilbert.py`'s existing convention that NONE is negligible/absent
    in this dataset (verified: 0 NONE rows in either unified file).
    """
    label2idx = {l: i for i, l in enumerate(OPTION_LABELS)}
    X_list, y_list, has_missing = [], [], []
    n_dropped = 0
    for r in rows:
        if r["label"] not in label2idx:
            n_dropped += 1
            continue
        X, meta = compute_option_features(r["prompt"])
        X_list.append(X)
        y_list.append(label2idx[r["label"]])
        has_missing.append(meta["has_missing"])
    if n_dropped:
        print(f"  [option_features] dropped {n_dropped} row(s) with label not in A/B/C/D")
    return np.stack(X_list, axis=0), np.array(y_list, dtype=int), has_missing


def build_feature_matrix_for_predict(rows):
    """Like build_feature_matrix, but NEVER drops a row (predict() must return
    exactly one label per input row). Rows with a non-A-D label (shouldn't
    occur at predict time) get a dummy y_idx of 0; callers must not use the
    returned y_idx for anything at predict time."""
    label2idx = {l: i for i, l in enumerate(OPTION_LABELS)}
    X_list, y_list, has_missing = [], [], []
    for r in rows:
        X, meta = compute_option_features(r["prompt"])
        X_list.append(X)
        y_list.append(label2idx.get(r["label"], 0))
        has_missing.append(meta["has_missing"])
    return np.stack(X_list, axis=0), np.array(y_list, dtype=int), has_missing


def _cli():
    import argparse
    import random

    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--sample", type=int, default=8)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    unified = cfg["data"]["unified_dir"]

    import json

    for source in ["agent", "human"]:
        path = f"{unified}/{source}.jsonl"
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        print(f"\n=== {source}: {len(rows)} rows ===")

        random.seed(0)
        sample_rows = random.sample(rows, min(args.sample, len(rows)))
        for r in sample_rows:
            raw4 = parse_row_raw(r["prompt"])
            X, meta = compute_option_features(r["prompt"])
            print(f"  label={r['label']} category={r['category']} group={r['group']}")
            for letter, raw, feat in zip(OPTION_LABELS, raw4, X):
                imputed = f" IMPUTED={raw['imputed']}" if raw["imputed"] else ""
                print(
                    f"    {letter}: price=${raw['price_usd']:.0f} "
                    f"rating={raw['star_rating']:.1f} reviews={raw['review_count']:.0f}"
                    f"  |  z/rank=[{feat[0]:+.2f}/{feat[1]:+.1f}, "
                    f"{feat[2]:+.2f}/{feat[3]:+.1f}, {feat[4]:+.2f}/{feat[5]:+.1f}]{imputed}"
                )

        # Corpus-wide sanity aggregates
        n_missing = 0
        prices, ratings, reviews = [], [], []
        n_bad_label = 0
        for r in rows:
            if r["label"] not in OPTION_LABELS:
                n_bad_label += 1
            X, meta = compute_option_features(r["prompt"])
            if meta["has_missing"]:
                n_missing += 1
            for raw in meta["raw"]:
                prices.append(raw["price_usd"])
                ratings.append(raw["star_rating"])
                reviews.append(raw["review_count"])
        print(
            f"  corpus: has_missing={n_missing}/{len(rows)} "
            f"({100*n_missing/len(rows):.2f}%), non-A-D labels={n_bad_label}"
        )
        print(
            f"  price range=[{min(prices):.0f}, {max(prices):.0f}] mean={np.mean(prices):.0f}"
        )
        print(
            f"  rating range=[{min(ratings):.1f}, {max(ratings):.1f}] mean={np.mean(ratings):.2f}"
        )
        print(
            f"  review range=[{min(reviews):.0f}, {max(reviews):.0f}] mean={np.mean(reviews):.0f}"
        )


if __name__ == "__main__":
    _cli()
