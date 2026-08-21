# Side-Study: Real-vs-Synthetic Likert Response Gap

This is a **decoupled side-study**, separate from the main Digital-Twin Gap
Study (LLM-agent vs. human shopping choice — see [../README.md](../README.md)
and [hierarchical_twin_spec.md](hierarchical_twin_spec.md)). It uses a
different data source (`Datasets_total/`, provided by the user's mentor) and
answers a structurally analogous but distinct question:

> If you generate synthetic survey respondents as a stand-in for a real human
> population, does the synthetic data reproduce that population's actual
> response *distribution*?

This is the same underlying question the main study asks about LLM agents
standing in for humans — but here the "twin" is a hand-coded synthetic-data
generator rather than an LLM, the task is answering ordinal Likert-scale
survey items rather than choosing among 4 described product options, and
there is **no train/test split and no model fit**: the synthetic data is
already a fixed, pre-generated CSV someone else produced. The analysis is
purely a distributional comparison between two already-realized samples.

---

## 1. Motivation & framing

`Datasets_total/` (a mentor-provided data drop, surveyed but not used
elsewhere in this repo) contains consumer-behavior survey datasets, none of
which match the main study's schema (a prompt with 4 described options and a
chosen label — see the main README's data section). Several of them, however,
ship as **matched real/synthetic pairs**: a real survey response file
alongside a same-schema synthetic file with 400 generated respondents. That
pairing is directly usable for a real-vs-synthetic distributional gap
analysis, with zero new data collection.

**Stated limitation, up front**: a full-repository search found **no
generator script for either dataset pair analyzed here**. (Generator scripts
do exist in `Datasets_total/Experiments/` for two *other*, unrelated study
sets — ResearchBox 207 and 412 — but not for `green_purchase_behavior` or
`non_alcoholic_beverages`.) So this study cannot check "does the output match
what the generator's code claims to do," the way the main study's
`behavioral_profiles.py` checks the hierarchical model's fitted decision
temperature against independently-measured empirical entropy. It can only
**describe the empirical gap** between the two realized samples. That's still
a legitimate and informative comparison — it just isn't a validation of any
known, inspectable behavioral model.

---

## 2. Dataset selection & harmonization

Of 7 candidate real/synthetic pairs surveyed in `Datasets_total/`, two were
selected for clean schema alignment (exact column-name match, consistent
numeric encoding on both sides, zero missing/out-of-range values):

| | Real | Synthetic | Items | Scale |
|---|---|---|---|---|
| **Primary** | `Datasets_total/OS_Samples/green_purchase_behavior.xlsx` (n=253) | `..._synthetic_400.csv` (n=400) | 28 | 1–7 |
| **Secondary** | `Datasets_total/Raw data_Non-alcoholic beverages.xlsx` (n=436) | `Datasets_total/OS_Samples/non_alcoholic_beverages_synthetic_400.csv` (n=400) | 25 | 1–5 |

The secondary dataset exists specifically to check whether any finding
**replicates** across two independently-chosen survey domains — the same
logic as the main study's two independently-designed corrected splits
agreeing with each other.

5 other candidate pairs were surveyed and **not** used: `indonesia_paylater`
(real is numeric 1–5, synthetic uses text Likert labels for the identical
items — would need remapping before any comparison is even possible);
`vietnam_ecommerce_privacy` (one item was redesigned from single-choice in
the real data to a 7-column multi-select in the synthetic data — not a
renaming, a schema change); `starbucks_china` (smallest real sample, n=212,
and the synthetic data encodes a 4-arm experimental design absent from the
real single-shot survey); `chatbot` and `smi_attitude` (both usable, held in
reserve — not needed once two clean pairs already agreed).

**Harmonization** (`load_and_harmonize()` in `src/likert_gap_analysis.py`):
item columns = (columns common to both files) − an **explicit,
per-invocation `--exclude` list** (IDs, demographics, derived/composite
columns) — never auto-inferred, so the item set is always an auditable,
documented choice rather than a heuristic guess that could silently swallow
a composite column (`green_purchase_behavior` ships derived construct-mean
and z-score columns, e.g. `PB`, `ZPB`, `EOXEC`, identically named in both
files — these must be excluded explicitly, not detected). Every surviving
candidate column is additionally verified numeric-coercible and within
`[scale_min, scale_max]` in **both** files; any column that fails this raises
a named error rather than being silently dropped, catching an incomplete
`--exclude` list immediately. The scale bounds are passed explicitly, not
auto-detected from the data's own min/max — synthetic data can have a
compressed range relative to real (e.g. `green_purchase_behavior`'s `PBC1`:
real min=1, synthetic min=3), and auto-detecting from the union would mask
exactly the kind of gap this study exists to find.

Exact `--exclude` lists used (see §5 for full commands):
- `green_purchase_behavior`: `SN, CaseNo` (IDs); `PB, EC, EO, PI, PBC, SNs,
  ATT` (construct means); `ZPB, ZEC, ZEO, ZPI, ZPBC, ZSNs, ZATT` (z-scores);
  `EOXEC, ATTXSN, PBCXSN, ATTXPBC` (interaction terms); `Gender, Age,
  Marital_Status` (demographics with mismatched real/synthetic encodings).
- `non_alcoholic_beverages`: `Q1_GENDER, Q2_BIRTHYEAR, Q3_PROVINCE/CITY`.

---

## 3. Metrics & formulas

For each harmonized item, with `real_shares`/`synth_shares` = normalized
value counts over the **full fixed** `scale_min..scale_max` range (zero-
filled for unobserved categories, so real and synthetic share vectors align
positionally):

```
entropy_norm(shares, k) = -sum(p*log2(p) for p in shares if p>0) / log2(k)      in [0,1]
herfindahl(shares)      = sum(p^2 for p in shares)                              in [1/k, 1.0]
```
New, generic functions in `src/likert_gap_analysis.py` — **not** imported
from `src/gap_analysis.py`, whose `shannon_entropy_norm`/`herfindahl_index`
hardcode the main study's 4-way `A/B/C/D` alphabet (two of them literally
re-read that module's global constant instead of the input dict's own keys,
so passing a k≠4 shares dict would silently miscompute rather than error).

```
JSD(real_shares, synth_shares)   -- reused directly from src.gap_analysis.js_divergence,
                                     already fully generic over category count, in [0,1]
```

**Per-item significance**: `scipy.stats.mannwhitneyu` (two-sided) — a rank
test, chosen over chi-square because Likert items are ordinal, not nominal,
and a rank test uses that ordering. With ~25–28 simultaneous per-item tests,
raw p-values are Benjamini-Hochberg corrected via
`scipy.stats.false_discovery_control(pvalues, method="bh")` → `q_value`
(requires scipy ≥1.11; `requirements.txt`'s floor was bumped for this, no new
dependency added).

**Classifier two-sample test** (`run_c2st_likert`): the harmonized item
values as features, `y = 0` (real) / `1` (synthetic). Three deliberate
departures from the main study's `src/c2st.py`:
1. **Plain stratified `train_test_split`**, not prompt-blocked — there is no
   shared task/prompt structure across Likert survey rows to leak (each row
   is one respondent's full answer set; the main study's leakage concern was
   specific to its per-task structure, which has no analogue here).
2. **`StandardScaler`, fit on train only** — unlike `c2st.py`'s features
   (already within-row z-scored/ranked), raw Likert values need this step
   before coefficient magnitudes are comparable across items.
3. The fit/AUC/balanced-accuracy glue (~15 lines) is **reimplemented
   locally** rather than importing `c2st.py`'s `run_c2st`, which is tied to
   the main study's prompt schema — this avoids coupling a side-study to a
   file it shouldn't need to depend on. `src.c2st._permutation_test` **is**
   imported directly, since it's already fully generic (operates only on
   `predict_proba`/labels/observed AUC, no schema dependency).

Same permutation-test design as the main study: held-out AUC vs. a null
distribution built by 1000 label-shuffled refits, two-sided on distance from
0.5 (so a below-chance AUC is treated as a genuine effect in the other
direction, not silently reported as "no effect").

**Cross-dataset replication check** (`compare_with`, triggered by
`--compare_with <other_out_dir>`): reads the other run's
`aggregate_summary.json`, reports both datasets' mean JSD / C2ST AUC /
permutation p-value side by side, and a plain verdict — `REPLICATES` if both
show C2ST AUC significantly above chance (p<0.05); otherwise flagged as not
cleanly replicating. Modeled on `src/behavioral_profiles.py`'s
`validate_against_entropy` precedent (read one script's artifact from
another's output directory, gate on it existing, report a comparison) — but
here there's no independently-known ground truth to validate *against*, only
agreement in direction *between* two independently-chosen dataset pairs.

---

## 4. Limitations

- **No generator script exists for either dataset** (see §1) — findings are
  purely empirical/descriptive, not a validation of a known synthetic model's
  stated assumptions.
- Only 2 of 7 surveyed candidate pairs were analyzed; the other 5 were either
  schema-mismatched (`indonesia_paylater`, `vietnam_ecommerce_privacy`) or
  simply not needed once two clean pairs already agreed
  (`starbucks_china`, `chatbot`, `smi_attitude` — held in reserve).
- Demographic fields (gender, age, marital status, province) were
  deliberately excluded, not harmonized — their real/synthetic encodings
  differ (numeric codes vs. text labels vs. different units) and they're
  peripheral to the Likert-item response-distribution question this study
  asks.
- `green_purchase_behavior`'s pre-computed composite/z-score/interaction
  columns (`PB`, `ZPB`, `EOXEC`, etc.) were excluded from the item-level
  analysis, not compared at the construct level — a natural v2 extension
  using `scipy.stats.ks_2samp` on the continuous composite scores, not
  built here.
- No correction for household/respondent-level covariates (e.g. comparing
  matched subpopulations) — this is a marginal, per-item distributional
  comparison only.

---

## 5. Results

**`green_purchase_behavior`** (n_real=253, n_synth=400, 28 items):
mean JSD = 0.1152 (median 0.0669, max 0.3211, item `GPB5`); 19/28 items
(67.9%) significant after BH correction (q<0.05); C2ST held-out AUC = 0.8841
(train-AUC 0.9091, gap 0.025 — small, no overfitting red flag), permutation
p = 0.0010, balanced accuracy = 0.8207. Full outputs:
`results_likert_green_purchase/`.

**`non_alcoholic_beverages`** (n_real=436, n_synth=400, 25 items):
mean JSD = 0.0372 (median 0.0326, max 0.0909, item `MAR2`); 20/25 items
(80.0%) significant (q<0.05); C2ST held-out AUC = 0.8270, permutation
p = 0.0010, balanced accuracy = 0.7511. Full outputs:
`results_likert_non_alcoholic_beverages/`.

**Cross-dataset replication**: `REPLICATES` — both datasets show C2ST AUC
significantly above chance (p<0.05 in both), with the majority of items
significantly different after correction in both (68% and 80%). Neither AUC
is anywhere near 1.0 (0.88 and 0.83), which rules out a trivial
leak/artifact (e.g. a stray ID column) as the explanation — the two
synthetic generators produce data that is genuinely, robustly
distinguishable from the real survey population it stands in for, on both
tested domains. See `results_likert_non_alcoholic_beverages/cross_study_comparison.json`
for the full numeric comparison.

Sanity bounds verified on both runs: `JSD` ∈ [0,1] (observed max 0.32);
`entropy_norm` ∈ [0,1]; `herfindahl` ∈ [1/k, 1.0] (k=7: floor 0.143, observed
min 0.192; k=5: floor 0.2, observed min 0.288 — both comfortably above the
uniform floor, as expected for real survey data); `q_value` ∈ [0,1]; item
counts matched the pre-verified expectation exactly (28 and 25) on both runs.

---

## 6. Reproducibility

```bash
python -m src.likert_gap_analysis \
  --real_path "Datasets_total/OS_Samples/green_purchase_behavior.xlsx" \
  --synthetic_path "Datasets_total/OS_Samples/green_purchase_behavior_400_synthetic.csv" \
  --exclude SN CaseNo PB EC EO PI PBC SNs ATT ZPB ZEC ZEO ZPI ZPBC ZSNs ZATT EOXEC ATTXSN PBCXSN ATTXPBC Gender Age Marital_Status \
  --scale_min 1 --scale_max 7 --dataset_name green_purchase_behavior \
  --out_dir results_likert_green_purchase

python -m src.likert_gap_analysis \
  --real_path "Datasets_total/Raw data_Non-alcoholic beverages.xlsx" \
  --synthetic_path "Datasets_total/OS_Samples/non_alcoholic_beverages_synthetic_400.csv" \
  --exclude Q1_GENDER Q2_BIRTHYEAR "Q3_PROVINCE/CITY" \
  --scale_min 1 --scale_max 5 --dataset_name non_alcoholic_beverages \
  --out_dir results_likert_non_alcoholic_beverages \
  --compare_with results_likert_green_purchase
```

Seed: `--seed 42` (default) for both the C2ST train/test split and the
permutation test, in both runs.
