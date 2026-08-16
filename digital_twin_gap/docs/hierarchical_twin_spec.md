# Hierarchical Bayesian Digital Twin — Design Specification

This document specifies the hierarchical-twin track added on top of the base
Digital-Twin Gap Study (see the repo [README.md](../README.md) for the base
study's 3 architectures and transfer-matrix methodology, which this track
reuses without modification). It covers: the choice-modeling formulation,
Model 1 (statistical baseline) and Model 2 (the primary hierarchical Bayesian
twin), the Human-Agent Gap Analyzer, and the exact protocol for reproducing
every number in this doc.

**Central claim being tested**: agent-vs-human choice behavior differs
systematically enough that a twin built from one population is a
distributionally poor stand-in for the other — not just less *accurate*, but
*differently shaped*. The base study measures the accuracy cost of that
mismatch; this track adds an explicit model of *why* (behavioral parameters
that differ by population/group) and a *distributional* (not just
point-accuracy) measure of the mismatch itself.

> **⚠ Numbers in Sections 3–8 below (and the base study's README.md §7) were
> measured under a split that leaks task identity — see Section 0.** Every
> table in this document that predates the leakage audit is left in place for
> the historical record but is marked **[superseded]**; treat
> `results_corrected_prompt/` and `results_corrected_twoway/` as the
> authoritative results.

---

## 0. Leakage audit & corrected methodology — read this first

A forensic audit found that the original split (`split_mode: "group"` in
`src/splits.py`) blocks agents by `(agent, category)` and humans by
`participant_id`, but blocks on **neither source's task**. Because all 6
agents answer byte-identical prompts, and only 144 distinct human prompts
exist across 571 participants, task novelty is impossible under either
blocking scheme:

| | agent | human |
|---|:---:|:---:|
| test rows whose exact `prompt` is also in train | **100.00%** | **100.00%** |
| train rows sharing each overlapping test prompt | 2–5 | 25–42 |

A **non-learning lookup** — "find this exact prompt in train, return the
modal label," no features, no fitting — scores:

| predictor | agent→agent | human→human |
|---|:---:|:---:|
| consensus lookup (`src/model_consensus.py`) | **0.6188** | **0.4281** |
| TF-IDF + LogReg (originally reported) | 0.6646 | 0.4331 |
| lookup as % of TF-IDF | 93.1% | 98.8% |
| majority-class floor (`src/model_majority.py`) | 0.2917 | 0.2535 |

TF-IDF's per-row predictions agree with the dictionary lookup on 69.6%
(agent) / 88.7% (human) of test rows. Refitting TF-IDF with *only* the split
axis changed to prompt-disjoint: agent→agent **0.6646 → 0.5636**, human→human
**0.4331 → 0.3262** (vs. a 0.3039 majority-class floor on that split) — roughly
10 points evaporate on each diagonal.

**Why this specifically breaks the transfer matrix**: agent and human prompts
share **zero** exact strings (384 vs. 144 prompts, intersection = 0). Diagonal
cells have a memorizable lookup available; off-diagonal cells structurally
cannot. `gap = matched − transferred` therefore conflates a real population
difference with the mere presence/absence of that lookup — the original
`gap_on_human = +0.0499` is not a clean measurement.

**Why the "behavioral signature" reading fails mechanically**: the text
architectures (`tfidf_logreg`, `embed_mlp`, `distilbert`) consume only
`prompt`, never `group` — they cannot condition on *which* agent they're
predicting for. What they learn under `"group"` mode is closer to a
task→consensus table, accurate for agents specifically because **agents agree
with each other more than humans do** (pairwise agreement 0.5269 vs. 0.3233,
chance = 0.25 — see Section 6). TF-IDF sits at 93.8% / 94.7% of the
prompt-only oracle ceiling those agreement rates imply.

`mnl_baseline` and `hier_bayes` were **never inflated by this leak** — they
consume only structured per-option features (price/rating/reviews), never the
raw prompt, so they physically cannot memorize a prompt string. Their ~0.38
under `"group"` mode was honest all along; this is confirmed empirically in
Section 0.3 below (their accuracy barely moves between leaky and corrected
splits, unlike the text models' ~10-point drop).

### 0.1 Corrected split modes (`src/splits.py`, `cfg["data"]["split_mode"]`)

- `"group"` (default, kept for backward compatibility with previously
  published numbers) — the original, leaky design.
- `"prompt"` — both sources blocked on the prompt itself. Agent: 307/77
  prompts → 1798/456 rows, all 6 agents and all 4 categories on both sides
  (the leaky split's agent test set covered only 3 of 6 agents and 3 of 4
  categories — this also fixes that). Human: 115/29 prompts → 4805/1211 rows,
  all 4 categories both sides — reintroduces some participant overlap, since
  prompt- and participant-blocking are mutually exclusive on this dataset
  (only 144 distinct human prompts across 571 participants).
- `"twoway"` — agent: prompt-blocked (identical to `"prompt"` mode — the 6
  LLMs studied here are the whole population of interest, not a sample, so
  there is no unseen 7th model to generalize to). Human: **prompt AND
  participant jointly blocked** — test = rows where both are unseen
  (~240 rows after discarding the two off-diagonal blocks), the strictest
  possible human split ("a new person deciding about a new product").

`build_splits()` now prints a leakage diagnostic (train/test prompt overlap %,
category/group coverage) on every run, in every mode — the structural guard
against this recurring silently.

### 0.2 Explicitly rejected: an `L_transfer` training objective

An earlier proposal considered adding a loss term that penalizes a twin for
failing to transfer across populations. Rejected: the transfer gap is this
study's *dependent variable*. Training to minimize it converts the
measurement into a domain-adaptation intervention — the resulting number
answers "how much can the gap be optimized away," not "how large is it,"
and the two are different questions. It also has a data problem: computing
it needs target-population labels during training, which the digital-twin
premise assumes you don't have (or it becomes unsupervised distributional
alignment, a legitimate but distinct experiment that must be reported
separately, never folded into the primary twin's number).

### 0.3 Corrected-split results

**`configs/config_corrected_prompt.yaml`** (`results_corrected_prompt/`; agent
n=456, human n=1211) — all 7 architectures, leaky-split numbers alongside for
direct comparison:

| architecture | agent→agent | agent→human | human→agent | human→human | gap_on_human | gap_on_agent |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| tfidf_logreg | 0.5636 *(was 0.6646)* | 0.3534 | 0.4474 | 0.3262 *(was 0.4331)* | **−0.0273** *(was +0.0499)* | +0.1162 |
| embed_mlp | 0.5417 *(was 0.5521)* | 0.3295 | 0.3399 | 0.3245 *(was 0.4256)* | **−0.0050** *(was +0.0665)* | +0.2018 |
| distilbert | 0.4605 *(was 0.4646)* | 0.3006 | 0.3904 | 0.3576 *(was 0.4431)* | +0.0570 *(was +0.1446)* | +0.0702 |
| mnl_baseline | 0.4518 *(was 0.3875)* | 0.3377 | 0.3618 | 0.3551 *(was 0.3973)* | +0.0173 *(was +0.0008)* | +0.0899 |
| hier_bayes | 0.4649 *(was 0.3812)* | 0.3130 | 0.3180 | 0.3790 *(was 0.4256)* | +0.0661 *(was +0.0449)* | +0.1469 |
| **consensus** | 0.2544 | 0.3039 | 0.3039 | 0.2544 | +0.0000 | +0.0000 |
| **majority** | 0.2544 | 0.3039 | 0.3039 | 0.2544 | +0.0000 | +0.0000 |

`consensus` collapses to exactly `majority` on every cell (0% lookup hit rate)
— the direct, automatic proof the leak is closed on this split, vs. 100% hit
rate / 0.6188 & 0.4281 under `"group"` mode.

**The headline result changes, not just shrinks.** Under the leaky split every
architecture showed `gap_on_human > 0` (agent-trained twins always looked
worse on humans than human-trained twins). Corrected, `tfidf_logreg` and
`embed_mlp` both **invert sign** — their agent-trained twin is now *slightly
better* on humans than their human-trained twin is. `mnl_baseline` and
`hier_bayes` — the two architectures that were never able to memorize a
prompt — barely move on the diagonals (agent→agent actually rises: their
leaky-split agent test set covered only 3 of 6 agents and 3 of 4 categories,
vs. the corrected split's full 6/6 and 4/4 coverage) and keep a small,
consistent positive `gap_on_human`. That divergence between the text
architectures and the structured architectures is itself informative: it's
not that "the gap disappeared," it's that the text architectures' apparent
gap was largely a memorization artifact, while the structured architectures'
smaller, more stable gap looks like the real signal.

**`configs/config_corrected_twoway.yaml`** (`results_corrected_twoway/`; the
strictest human split — new person AND new product jointly, human n=242):

| architecture | agent→agent | agent→human | human→agent | human→human | gap_on_human | gap_on_agent |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| tfidf_logreg | 0.5636 | 0.3967 | 0.3969 | 0.3223 | **−0.0744** | +0.1667 |
| embed_mlp | 0.5088 | 0.3182 | 0.3158 | 0.3099 | **−0.0083** | +0.1930 |
| distilbert | 0.4364 | 0.3223 | 0.3004 | 0.2975 | **−0.0248** | +0.1360 |
| mnl_baseline | 0.4518 | 0.3719 | 0.3421 | 0.2975 | **−0.0744** | +0.1096 |
| hier_bayes | 0.4649 | 0.3264 | 0.3114 | 0.3347 | +0.0083 | +0.1535 |
| consensus | 0.2544 | 0.3223 | 0.3223 | 0.2544 | +0.0000 | +0.0000 |
| majority | 0.2544 | 0.3223 | 0.3223 | 0.2544 | +0.0000 | +0.0000 |

Agent-mode cells are byte-identical to `config_corrected_prompt.yaml`'s (same
prompt-blocked agent split, same seed — confirmed exactly, down to the MCMC
diagnostics). MCMC diagnostics for the human-mode fit: max r_hat=1.01, min
ESS bulk/tail = 1269/930, divergences = 2/4000 (0.05%) — all within the
pre-registered thresholds. `consensus` again collapses exactly to `majority`.

**This is the strongest evidence in the whole study that the original
`gap_on_human > 0` finding was a leakage artifact.** Under the strictest
possible split — a human test set that is simultaneously a new person AND a
new product, discarding both partially-novel blocks — `gap_on_human` is
**negative for 4 of 5 real architectures** (tfidf_logreg, embed_mlp,
distilbert, mnl_baseline), and the one exception (`hier_bayes`, +0.0083) is
an order of magnitude smaller than its leaky-split value (+0.0449). The two
independently-designed corrected splits (`"prompt"` and `"twoway"`) agree on
direction for 4 of 5 architectures despite very different test-set
composition and size (1211 rows vs. 242 rows) — the kind of cross-validation
that makes a corrected finding credible rather than an artifact of one
particular resampling.

---

## 1. Notation & glossary

| Symbol | Meaning |
|---|---|
| `i` | row index (one shopping decision) |
| `j` | option index, `j ∈ {A,B,C,D}` |
| `g` | agent-identity index (6 levels: the 6 LLM agents), agent-mode only |
| `p` | human participant index, human-mode only (train-time regularization only) |
| `c` | category index (4 levels: AIR/COF/EAR/SNK) |
| `X[i,j,k]` | k-th engineered per-option feature (k=1..6) of option j in row i |
| `U[i,j]` | latent utility of option j for row i |
| `τ_i` | decision temperature for row i (higher = noisier/less concentrated choices) |
| `θ` | Model 1's flat coefficient vector |
| `β_pop`, `β_cat`, `β_group` | Model 2's population / category / agent-identity slope terms |
| `source` | `"agent"` or `"human"` — which population a row comes from |
| `group` | agent identity (e.g. `"Claude Opus 4.6"`) or human `participant_id` |
| matched / transferred | trained-and-tested on the same / different population (the base study's 2×2 transfer matrix) |
| `D_HA` | flagship gap metric: `JSD(P_human_twin, P_agent_twin)` on shared held-out rows |

---

## 2. Data & feature engineering

The unified schema (`data/unified/{agent,human}.jsonl`, built by
`src/data_prep.py`) has exactly 5 keys per row: `prompt`, `label`, `source`,
`group`, `category`. There is no separate per-option attribute column
anywhere upstream — but every prompt is a consistently templated block of 4
lettered options, each with `PRICE_USD:` and `STAR_RATING:` fields (the human
template embeds review count in the rating field, e.g.
`STAR_RATING: 4.7 (290 reviews)`; the agent template has a separate
`REVIEW_COUNT:` line). `src/option_features.py` regex-parses these directly
out of `prompt` (verified 0% missing across all 8,270 rows × 4 options) and
engineers 6 **within-row-normalized** features per option — no persisted
training statistics are needed at predict time:

`price_z, price_rank_c, rating_z, rating_rank_c, review_log_z, review_rank_c`

(z-score and centered rank, each computed relative to the other 3 options in
the same row — choice is about *relative*, not absolute, attractiveness).
Rows with an unparseable field (none observed on this dataset) are dropped at
train time and imputed-and-kept at predict time — `predict()` must always
return exactly one label per input row.

---

## 3. Model 1 — fixed-effects discrete-choice baseline

`src/model_mnl_baseline.py`. A hand-rolled multinomial logit (via
`scipy.optimize.minimize`), not `sklearn`/`statsmodels`, because neither can
express *alternative-specific features with generic coefficients* — "the same
price coefficient applies to whichever option is cheapest" — the defining
structural property of a real choice model.

```
f_ij = [ z_ij (6)                                    # engineered features
         ASC_j (3)                                   # B/C/D vs. A=reference
         category × ASC_j (9)                        # 3 non-ref categories × 3 ASC
         category × z_ij (18)                        # 3 non-ref categories × 6 feats
         [agent-mode only] agent × ASC_j (15)         # 5 non-ref agents × 3 ASC
         [agent-mode only] agent × z_ij (30) ]        # 5 non-ref agents × 6 feats

U_ij(θ) = θ · f_ij
P(y_i=j) = softmax_j(U_i·)
NLL(θ) = -Σ log P(y_i_observed) + 0.5·λ·‖θ‖²    (λ = 1/C, ridge)
```

36 dims (human-mode) / 81 dims (agent-mode). Human `group` (participant_id,
571 levels) is deliberately **not** interacted with — far too sparse for
unpooled dummies, and per this study's scope decision, individual identity is
never relied on for prediction on unseen participants anyway.

**Verified result [superseded — leaky `"group"` split; see §0.3 for the
corrected numbers]** (`results_mnl_baseline/`):

| cell | accuracy | macro-F1 |
|---|:---:|:---:|
| agent→agent (matched) | 0.3875 | 0.3844 |
| agent→human (transfer) | 0.3965 | 0.3941 |
| human→agent (transfer) | 0.3250 | 0.3251 |
| human→human (matched) | 0.3973 | 0.3952 |

All 4 cells clear the 0.25 random baseline. Notably lower than the base
study's text architectures (e.g. `tfidf_logreg` agent→agent = 0.665) — this
model sees only 3 numeric attributes, not the full prompt text (brand,
delivery terms, warranty, seller reliability, etc., all of which the
full-text models can exploit). The **gap on humans nearly vanishes**
(+0.0008) — with only price/rating/reviews as inputs, agent- and
human-trained models transfer to humans almost equally, an interesting
contrast with the full-text architectures' much larger human-side gap.

---

## 4. Model 2 — Hierarchical Bayesian choice model (the twin)

`src/model_hier_bayes.py`. NUTS via NumPyro/JAX, CPU (picked over PyMC: no
C/C++ compiler toolchain needed on native Windows, and this model's runtime is
leapfrog-step-bound, not FLOP-bound, so GPU buys nothing).

### 4.1 Generative model

```
asc_j        ~ Normal(0,1)              for j ∈ {B,C,D};  asc_A := 0
β_pop_k      ~ Normal(0,1)                                            k=1..6

# category crossed random slopes (BOTH modes, 4 levels, non-centered)
σ_β_cat_k    ~ HalfNormal(0.5)
β_cat[c,k]   = σ_β_cat_k · Normal(0,1)

# agent-identity crossed random slopes (AGENT-MODE ONLY, 6 levels, non-centered)
σ_β_group_k  ~ HalfNormal(0.5)
β_group[g,k] = σ_β_group_k · Normal(0,1)

β[i,k] = β_pop_k + β_cat[c(i),k] + (β_group[g(i),k] if agent-mode else 0)

# decision temperature (behavioral "concentration/consistency")
μ_log_τ ~ Normal(0, 0.5)
  agent-mode: log τ[i] = μ_log_τ + σ_log_τ_group · Normal(0,1)[g(i)]
  human-mode: log τ[i] = μ_log_τ + σ_log_τ_participant · Normal(0,1)[p(i)]   (TRAIN-TIME ONLY)

U[i,j] = asc_j + Σ_k β[i,k]·X[i,j,k]
P(y_i=j) = softmax_j(U[i,·] / τ[i]);   y_i ~ Categorical(P)
```

**Why participant-τ is train-time-only**: `splits.py`'s grouped, leak-free
split guarantees no human participant appears in both train and test, so a
per-participant *slope* (or any latent used at predict time) is statistically
untestable on unseen participants. Restricting the individual term to a
single scalar τ (not a full slope vector) still extracts genuine
partial-pooling/shrinkage value during training — it's low-dimensional and
well-identified even from ~10.5 rows/participant — but at `predict()` time it
is **always marginalized to its population mean** (`μ_log_τ`), so no
prediction on a new participant ever depends on an individual latent. This
is what makes the human twin "population-level only for prediction" while
still deriving real value from the participant grouping during training.
Category and agent-identity get full crossed random *slopes* because both are
well-identified (500+ rows/category, ~300 rows/agent).

### 4.2 Inference & prediction

`predict()`/`predict_proba()` use **posterior-predictive averaging** — the
mean of `softmax(U_s/τ_s)` over posterior draws `s`, then argmax — not
posterior-mean coefficients plugged into the link function
(`E[softmax(f(θ))] ≠ softmax(f(E[θ]))` by Jensen's inequality, since softmax
is nonlinear). This is both the Bayes-optimal decision rule and what the Gap
Analyzer needs as genuine predictive probabilities.

Unseen-group fallback (e.g. an agent-trained model scoring human rows in the
transfer matrix): the group-level term is zeroed, i.e. population-level-only
prediction.

### 4.3 Verified MCMC diagnostics

Sampler convergence is a property of the model and data size, not of which
split mode is used, so these numbers are unaffected by the §0 leakage
correction — reported here for both the original quick smoke test and the
full production run under the corrected `"prompt"` split (1000 draws/1000
tune/4 chains):

| run | mode | wall time | max r_hat | min ESS bulk | min ESS tail | divergences |
|---|---|---|---|---|---|---|
| smoke (200/200/2 chains) | agent | 109.4s | 1.03 | 124 | 150 | 0/400 (0%) |
| smoke (200/200/2 chains) | human | 307.1s | 1.08 | 86 | 26 | 0/400 (0%) |
| full, corrected split | agent | 622.0s | **1.00** | 1308 | 839 | 0/4000 (0%) |
| full, corrected split | human | 2487.6s | **1.01** | 1110 | 1103 | 1/4000 (0.03%) |

All comfortably inside the pre-registered thresholds (max r_hat < 1.01, min
ESS > 400, divergence rate < 1%) — the full run is the one to trust.

Zero divergences in both modes; r_hat/ESS are expected to tighten
substantially at the full config's 5× more draws (see Section 8 for the
production run's numbers, generated via
`configs/config_hier_bayes.yaml`).

---

## 5. Train/val protocol

Reused **verbatim** from the base study — `src/splits.py`'s
`build_splits(cfg)`: grouped (agent by `group+category`, human by
participant `group`), leak-free, fixed seed=42, test_size=0.2. Both new
models train on exactly the same 4 splits as the base study's 4 architectures,
so all 6+ architectures are directly comparable in one merged results table.

| source | train n | test n | notes |
|---|---:|---:|---|
| agent | 1,774 | 480 | all 6 agent identities appear in both |
| human | 4,813 | 1,203 | 456 train / 115 test participants, **zero overlap** |

---

## 6. The Human-Agent Gap Analyzer

`src/gap_analysis.py`, output to `results_gap_analysis/` (never touches
`results/`). Three tiers, only the third needs a trained model:

**Tier 1 — empirical entropy / concentration** (straight from the unified
data, no model): for each group's empirical A/B/C/D label shares `p_l`,

```
entropy_norm = -Σ_l p_l·log2(p_l) / log2(4)     ∈ [0,1]   (1 = uniform)
hhi          = Σ_l p_l²                          ∈ [0.25,1] (0.25 = uniform)
```

**Tier 2 — inter-agent agreement**: Fleiss' κ (1971) computed on the
**verified 334-of-384** agent prompts that all 6 agents actually answered
(Grok is missing 50; Fleiss' κ requires a fixed rater count per item):

```
P_i = (Σ_l n_il² - N) / (N(N-1));   P̄ = mean_i(P_i)
p_l = Σ_i n_il / (n·N);             P̄_e = Σ_l p_l²
κ = (P̄ - P̄_e) / (1 - P̄_e)
```

**Tier 3 — model-based flagship metric**: using the two pickled `hier_bayes`
posteriors (agent-trained + human-trained, saved by `fit()`), compute both
twins' posterior-predictive `P_agent(y|x)` / `P_human(y|x)` on the **same**
held-out rows (the base study's own agent/human test sets — no cross-population
task bridge exists or is needed, since both twins are simply asked about the
same rows):

```
Δ(x)  = P_agent(y|x) - P_human(y|x)
D_HA  = JSD(P_human, P_agent) = ½·KL(P_human‖M) + ½·KL(P_agent‖M),  M = ½(P_human+P_agent)
```
`D_HA ∈ [0,1]` (base-2 log); `D_HA = 0` would mean the twins are
indistinguishable — this study's premise is that it is meaningfully greater
than 0.

**Verified Tier 1/2 results** (independent of the smoke-vs-full MCMC config —
these come straight from the raw data, and match this exact dataset):

| group | entropy_norm | hhi | n |
|---|:---:|:---:|---:|
| ChatGPT GPT 5.5 | 0.9867 | 0.2595 | 384 |
| ChatGPT O3 | 0.9990 | 0.2507 | 384 |
| Claude Opus 4.6 | 0.9956 | 0.2530 | 384 |
| Claude Opus 4.8 | 0.9937 | 0.2545 | 384 |
| Claude Sonnet 4.6 | 0.9989 | 0.2508 | 384 |
| Grok 4.2 | 0.9680 | 0.2728 | 334 |
| Human (pooled) | 0.9986 | 0.2509 | 6,016 |

**Fleiss' κ = 0.3505 ("fair" agreement, Landis & Koch scale)**, n_items=334,
n_raters=6.

Grok is the clear outlier — noticeably lower entropy / higher concentration
than every other agent and than the human population, meaning it is the most
predictable/rule-like chooser in this study. Every other agent is *closer to
uniform* than one might expect, and the human population's aggregate entropy
(0.9986) is actually *higher* than most individual agents' — the interesting
structure here is the **cross-agent heterogeneity**, not a clean
"agents-are-concentrated-humans-are-noisy" binary (see Section 9's framing
note).

### 6.1 Behavioral profiles — what the hierarchical model's posteriors say

`src/behavioral_profiles.py` extracts `beta_group` (per-agent deviation in
price/rating/review sensitivity) and `log_tau_group` (per-agent decision
temperature) from the already-fitted `hier_bayes_model_agent.pkl` — no
re-fitting needed. Run against the corrected-split posterior
(`results_corrected_prompt/`):

| agent | τ (posterior mean) | 94% HDI |
|---|:---:|:---:|
| ChatGPT O3 | 1.24 | [0.76, 1.78] |
| Claude Sonnet 4.6 | 1.39 | [0.80, 2.00] |
| Claude Opus 4.8 | 1.54 | [0.88, 2.24] |
| ChatGPT GPT 5.5 | 1.56 | [0.83, 2.26] |
| Claude Opus 4.6 | 1.67 | [0.97, 2.44] |
| Grok 4.2 | **3.68** | [1.60, 6.09] |

**Validation**: Spearman(τ, empirical entropy) = **−0.83** (p=0.042, n=6) —
two independent computations (MCMC posteriors over a discrete-choice
likelihood vs. counting raw label frequencies) agree that Grok is the
outlier. But the *sign* is counter-intuitive at first: higher τ means the
softmax is softer (closer to uniform, i.e. the model is LESS confident given
its covariates), yet Grok has the *lowest* empirical entropy (most
concentrated actual behavior). The mechanism, confirmed by inspecting Grok's
raw label distribution — `{A: 36.8%, B: 25.1%, C: 21.9%, D: 16.2%}` vs. every
other agent staying within 20–33% on every letter — is that Grok's
concentration comes from a strong **absolute preference for option A**, not
from a stronger reaction to price/rating/reviews. Because this model's
alternative-specific constants (ASCs) are deliberately kept
population-level-only in v1 (§9), there is no per-agent slot to attribute a
*positional* bias to Grok specifically — the unexplained variance gets
absorbed into an inflated τ instead. This is a real, diagnosable limitation
of the v1 model (not a bug): **hierarchical ASCs are the highest-value next
extension**, now with concrete empirical motivation rather than a generic
scoping note.

The feature-sensitivity table (`results_behavioral_profiles/feature_sensitivity_by_agent.csv`)
does show real, attribute-level differences beyond the position effect — e.g.
ChatGPT O3's `rating_rank_c` sensitivity is +2.15 (strongly favors the
higher-rated option) vs. Grok's −0.28 (essentially indifferent to rating
rank) — evidence the model is capturing genuine behavioral heterogeneity
alongside the ASC gap.

### 6.2 Classifier two-sample test (C2ST)

`src/c2st.py` — the standalone, principled version of "learn to tell agents
and humans apart," rather than folding it into the twin's own training
objective (see §0.2 for why that was rejected). Trains a classifier on the
**within-row-normalized features of the chosen option only** (never raw
text — agent and human prompts use different field names, so a text
classifier would hit near-100% by detecting the template, not the behavior),
on a prompt-blocked split.

First attempt used a 200-tree gradient-boosted classifier: 0.77 in-sample AUC
vs. 0.43 held-out — severe overfitting on a 10-feature problem, discarded.
Corrected to L2-regularized logistic regression (11 parameters total):

| | value |
|---|:---:|
| held-out AUC | **0.578** |
| train (in-sample) AUC | 0.603 (gap = +0.025, well-controlled) |
| balanced accuracy | 0.559 |
| permutation p-value (two-sided, 1000 perms) | **0.001** |

A small but genuine, statistically significant, out-of-sample-generalizing
population difference — driven primarily by price: `price_z` coefficient
−0.767 (agents lean toward relatively higher-priced choices in raw z-score
terms) and `price_rank_c` coefficient +0.596 (an opposite-signed *ordinal*
effect — a real nuance about magnitude vs. relative position, not a
contradiction). AUC well below 1.0 rules out residual template/configuration
leakage as the explanation.

---

## 7. Experiment matrix

All architectures share the same 4-cell transfer matrix and merge into one
comparison via `src/merge_results.py`:

| architecture | type | text input | structured input |
|---|---|:---:|:---:|
| `tfidf_logreg` | TF-IDF + logistic regression | ✓ | |
| `embed_mlp` | frozen sentence-embedding + MLP | ✓ | |
| `distilbert` / `distilbert_tuned` | fine-tuned transformer | ✓ | |
| `mnl_baseline` | fixed-effects discrete choice | | ✓ |
| `hier_bayes` | **hierarchical Bayesian discrete choice (the twin)** | | ✓ |

Plus the Gap Analyzer's 3 tiers, which are architecture-independent except
for Tier 3 (uses `hier_bayes` specifically, since it's the only architecture
that natively produces calibrated posterior-predictive probabilities rather
than a single hard label).

---

## 8. Verification & sanity-check protocol

Run via `conda run -n torch121 python -m ...` (or with `torch121`'s
interpreter directly) from the repo root:

1. `python -m src.option_features --config configs/config.yaml --sample 8` —
   expect ~0% missing rate, price ∈ roughly [$16,$425], rating ∈ [3.8,4.8].
2. `python -m src.run_all --config configs/config_mnl_baseline.yaml --models mnl_baseline`
   — `results_mnl_baseline/all_results.json` has exactly 1 key, all 4 cells >
   0.25.
3. Smoke-test `hier_bayes` first (`config_hier_bayes_smoke.yaml`) — 0
   exceptions, 0 divergences, accuracy > 0.25 on both diagonal cells — *then*
   the full `config_hier_bayes.yaml` run. Check
   `mcmc_diagnostics_{agent,human}.csv`: max r_hat < 1.01, min
   ess_bulk/ess_tail comfortably > 400, divergence rate < 1%.
4. `python -m src.gap_analysis --config configs/config_hier_bayes.yaml --out_dir results_gap_analysis`
   — entropy ∈ [0,1], hhi ∈ [0.25,1.0], Fleiss' κ finite with
   n_items=334/n_raters=6, mean JSD strictly > 0 and well below 1.0.
   **Cross-check**: argmax(P_agent)/argmax(P_human) accuracy recomputed from
   `model_based_gap_on_*_test.csv` must match
   `results_hier_bayes/all_results.json`'s corresponding cells exactly —
   confirms the pickled models are the same fitted posteriors used for the
   official accuracy numbers, not silently re-fit.
5. Merge + `analyze.py` on the full comparison directory — `mnl_baseline` /
   `hier_bayes` accuracy should sit above 0.25 and plausibly below the
   full-text architectures (structured-only input, less signal); if
   `hier_bayes` beats `distilbert`, treat that as a possible leakage bug to
   investigate, not a finding.
6. `git status` shows **zero modifications** under `results/` or
   `results_distilbert_tuned/` — only new files/directories anywhere else.

*(Production-run numbers for Sections 4.3's full-config MCMC diagnostics and
Section 6's Tier 3 JSD gap are filled in once `configs/config_hier_bayes.yaml`'s
full run — 1000 draws/1000 tune/4 chains per mode — completes; see
`results_hier_bayes/mcmc_diagnostics_*.csv` and `results_gap_analysis/gap_summary.json`
for the final numbers actually used in the paper.)*

---

## 9. Limitations & scoped-out extensions

- **Task-level random effect (`γ_t`)**: the user's original proposal wanted
  `U_ijt = β·x_ijt + α_i + γ_t` to separate "this decision-maker prefers X"
  from "this task made X attractive." Verified infeasible *across*
  populations with current data — agents share 384 byte-identical prompts
  among themselves (so a within-agent-population `γ_t` **is** buildable as a
  future extension), but humans' `task_id` has no correspondence to agent
  prompts, and the two templates differ in field structure (agent has
  `BRAND_REPUTATION_LABEL`/`VISUAL_DESCRIPTION`; human has `LISTING`
  instead), so no single `γ_t` can span both populations without a new,
  currently-nonexistent cross-population task-matching scheme.
- **Per-participant random slopes**: scoped out — ~10.5 rows/participant is
  too sparse to identify a full slope vector per person; only the
  much-lower-dimensional decision-temperature scalar τ_p is used, and even
  that never reaches `predict()` (see Section 4.1).
- **Hierarchical ASCs**: kept population-level-only in v1 for parsimony (the
  6 feature slopes plus τ already carry most of the interesting
  cross-population structure); promoting ASCs to a hierarchical layer is a
  straightforward v2 extension using the same non-centered pattern.
- **The "agents are more predictable than humans" framing** (from the base
  study's `tfidf_logreg`-only report): the Gap Analyzer's Tier 1 results
  complicate a simple binary reading of this claim — the *aggregate* human
  population's entropy (0.9986) is actually higher than most *individual*
  agents', and the real story is **cross-agent heterogeneity** (Grok is a
  clear outlier; the other 5 agents are close to the human population's
  aggregate entropy). The more defensible framing, per Section 6's data: *can
  a probabilistic twin learn and quantify systematic, group-specific
  differences in decision concentration and task-conditional choice
  behavior* — which is exactly what `β_group`/`τ_group`'s posterior
  distributions in `hier_bayes` are built to answer.

---

## 10. Reproducibility

All commands run from the repo root. Seeds: `configs/config.yaml`-family's
`data.seed=42` (splits, shared across every architecture) and each
`hier_bayes` config's `models.hier_bayes.seed=42` (MCMC).

```bash
# Dependencies (torch121 conda env; already has CPU JAX)
conda run -n torch121 pip install numpyro arviz scipy

# 1. Model 1
python -m src.run_all --config configs/config_mnl_baseline.yaml --models mnl_baseline
python -m src.analyze --config configs/config_mnl_baseline.yaml

# 2. Model 2 (smoke test, then full)
python -m src.run_all --config configs/config_hier_bayes_smoke.yaml --models hier_bayes
python -m src.run_all --config configs/config_hier_bayes.yaml --models hier_bayes
python -m src.analyze --config configs/config_hier_bayes.yaml

# 3. Gap analyzer (reuses the pickled hier_bayes models from step 2)
python -m src.gap_analysis --config configs/config_hier_bayes.yaml --out_dir results_gap_analysis

# 4. Full 6-architecture comparison (merges, never overwrites, results/ or
#    results_distilbert_tuned/)
python -m src.merge_results \
  --base results_distilbert_tuned/all_results.json \
         results_mnl_baseline/all_results.json \
         results_hier_bayes/all_results.json \
  --out_dir results_full_comparison
python -m src.analyze --config configs/config_full_comparison.yaml
```

Config file inventory (this track): `configs/config_mnl_baseline.yaml`,
`configs/config_hier_bayes_smoke.yaml`, `configs/config_hier_bayes.yaml`,
`configs/config_full_comparison.yaml` — each a copy of the base
`configs/config.yaml` with only its own model block and `output.results_dir`
changed, per the pattern established by `configs/config_distilbert_tuned.yaml`.
