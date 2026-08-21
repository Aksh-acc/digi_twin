# Digital-Twin Gap Study

Measuring **how well a "digital twin" of shopping decisions transfers between LLM
agents and real humans** — across several model architectures — for a research
paper.

You already built a two-model digital twin (a cVAE numeric twin + a Qwen2.5-3B
LoRA LLM twin) and ran the agent-vs-human comparison on it. This repo is the
**complementary, architecture-comparison track**: it takes the *same* decision
task (given a shopping prompt with four options, predict which option is chosen)
and runs it through **seven architectures**, each evaluated on the full
agent↔human **transfer matrix**, so the *gap* between agents and humans is
measured directly and comparably.

Everything runs on **Google Colab, Kaggle, or locally** with no code changes.

---

## Final verdict (read this first)

An early run of this pipeline reported a striking result — agent-matched
accuracy of 0.665 vs. a human-matched 0.433 — that looked like strong evidence
LLM agents make far more predictable choices than humans. **That result was
substantially a data-leakage artifact.** The original train/test split blocked
on `(agent, category)` for agents and `participant_id` for humans, but never
blocked on the *task itself* — and because all 6 agents answer byte-identical
prompts (and only 144 distinct human prompts exist across 571 participants),
**100% of test rows in both sources had their exact prompt sitting in their own
training split.** A non-learning lookup table ("find this exact prompt in
train, return the modal label") recovered 93–99% of the reported accuracy.

Once the split was corrected to block on the task (two independent corrected
designs, agreeing with each other — see below), the headline finding **mostly
inverted**: 4 of 5 real architectures now show the agent-trained twin
performing *as well or better* on humans than the human-trained twin does —
the opposite of the original "agents overstate their fit to humans" claim.

**What survives, and is the actual finding of this study:**
- The **text-classification architectures' large diagonal accuracy was mostly
  memorization**, not a learned behavioral signature — they only ever see raw
  prompt text, never *which* agent or person is choosing, so they structurally
  cannot learn a per-decision-maker signature; corrected, their agent-matched
  accuracy drops ~10 points and the human-side gap flips sign or vanishes.
- The **structured discrete-choice architectures** (`mnl_baseline`,
  `hier_bayes` — new in this track, see below) were **never inflated** by this
  leak, since they consume only parsed price/rating/review features, never raw
  text. Their honest accuracy (~0.38–0.46, well above the 0.25 random and
  ~0.30 majority-class floors) is the more trustworthy signal in this study,
  and it shows LLM-agent choice carries real, structured, generalizable
  signal that individual human choice mostly doesn't.
- A **classifier two-sample test** on structured chosen-option features finds
  a small but statistically significant, out-of-sample-generalizing
  population difference (AUC = 0.578, permutation p = 0.001), driven mainly by
  price — humans and agents *are* behaviorally distinguishable, just not
  nearly as dramatically as the leaky numbers implied.
- **Inter-agent agreement is real and substantial** (Fleiss' κ = 0.35 "fair"
  agreement among the 6 LLMs on identical tasks, vs. ~0.32 pairwise agreement
  among humans) — this is a leak-free, model-free fact about the raw data, and
  it's the honest version of "agents are more predictable than humans": they
  agree with each other, not that any one of them has a stronger personal
  signature. One agent (Grok 4.2) is a clear behavioral outlier, driven by a
  strong absolute preference for option "A" rather than by price/rating
  sensitivity — see the hierarchical model's behavioral profiles below.

**Full forensic audit, the two corrected split designs, the corrected
7-architecture comparison table, behavioral profiles, and the two-sample
test: [docs/hierarchical_twin_spec.md §0](docs/hierarchical_twin_spec.md#0-leakage-audit--corrected-methodology--read-this-first).**
Reproduce it end-to-end with the exact command sequence in that doc's §10.

---

## 1. The research question

> If you build a digital twin from **LLM-agent** shopping decisions, how well
> does it stand in for **real human** consumers — and vice-versa?

We answer it with a 2×2 **transfer matrix** per architecture:

|              | test = agent | test = human |
|--------------|:------------:|:------------:|
| train = agent | A→A (matched) | A→H (transfer) |
| train = human | H→A (transfer) | H→H (matched) |

- **Diagonal** = matched twin (trained and tested on the same population).
- **Off-diagonal** = transferred twin (trained on one, tested on the other).
- **Gap** = matched accuracy − transferred accuracy. A large gap means a twin
  built from the wrong population is a poor stand-in — **but see the Final
  verdict above**: this number is only trustworthy under a task-blocked split.

---

## 2. Data

Two sources, reduced to one shared schema so the same models consume both:

```
{ "prompt": "<full shopping prompt: mandate + 4 options>",
  "label" : "A" | "B" | "C" | "D" | "NONE",     # the chosen option
  "source": "agent" | "human",
  "group" : "<agent name>" | "<participant_id>", # for splitting
  "category": "AIR" | "COF" | "EAR" | "SNK" }
```

**Agentic data** — LLM "shopping agents" answering the *same* product-choice
prompts. Six agents are included:

| Agent | Rows |
|---|---|
| Claude Opus 4.6 | 384 |
| Claude Sonnet 4.6 | 384 |
| Claude Opus 4.8 | 384 |
| ChatGPT O3 | 384 |
| ChatGPT GPT-5.5 | 384 |
| Grok 4.2 | 334¹ |
| **Total** | **2,254** |

¹ Grok's first batch (NO1–50) ships without a `full_chat_prompt` column, so
those 50 rows are skipped (a model needs the prompt as input). Everything else
is complete. All 6 agents answer the same 384 distinct prompts — verified by
exact string match — of which 334 have a response from all 6 (used as the
task set for the Fleiss' κ computation below; the earlier claim of "72 shared
product configurations" undercounts the true figure and should be treated as
approximate, not exact).

**Human data** — real survey respondents' actual choices on the real product
options they were shown (the pre-built `llm_train_human.jsonl` /
`llm_val_human.jsonl` from the original project). **6,016** decisions from 571
participants, but only **144 distinct prompts** — i.e. each product
configuration was shown to ~42 different participants on average.

**Splitting** (`src/splits.py`, `cfg["data"]["split_mode"]`) — three modes:

- `"group"` (default, kept only for backward compatibility with previously
  published numbers) — agents grouped by `(agent, category)`, humans by
  `participant_id`. **Blocks on neither source's task** — see Final verdict.
- `"prompt"` — both sources blocked on the prompt itself, so a held-out task
  is unseen by every decision-maker in training. Recommended default for new
  work.
- `"twoway"` — agents prompt-blocked (the 6 LLMs are the whole population of
  interest, not a sample); humans blocked **jointly** on prompt AND
  participant, so the test set is "a new person deciding about a new
  product." The strictest design; smaller test set as a result.

`build_splits()` prints a leakage diagnostic (train/test prompt overlap %,
category/group coverage) on every run, in every mode, so this class of bug
can't recur silently.

---

## 3. The architectures

Seven architectures, sharing the same `fit()` / `predict()` interface so
`run_all.py` treats them interchangeably and every one flows through the same
2×2 transfer matrix.

**Text-classification track** (consume only `prompt`, never see `group`):

### (1) `tfidf_logreg` — TF-IDF + Logistic Regression
`src/model_tfidf.py`. Prompt → sparse bag-of-n-grams (uni+bigrams, 20k features,
sublinear TF) → multinomial logistic regression, `class_weight="balanced"`.
Pure CPU, trains in ~2 seconds.

### (2) `embed_mlp` — frozen sentence embeddings + MLP head
`src/model_embed_mlp.py`. Prompt → **frozen** `all-MiniLM-L6-v2` sentence encoder
(384-dim) → 2-layer MLP classifier (256→128→4) with dropout and class-weighted
cross-entropy. Runs on CPU, faster on GPU.

### (3) `distilbert` / `distilbert_tuned` — fine-tuned transformer classifier
`src/model_distilbert.py`. `distilbert-base-uncased` fine-tuned end-to-end as a
4-way sequence classifier via Hugging Face `Trainer`. 66M params: fits
free-tier Colab/Kaggle GPUs; falls back to (slow) CPU. `distilbert_tuned` is
the same architecture with more epochs / lower LR (`configs/config_distilbert_tuned.yaml`).

**Discrete-choice track** (consume parsed per-option price/rating/review
features via `src/option_features.py` — structurally cannot memorize prompt
text, since they never see it):

### (4) `mnl_baseline` — fixed-effects discrete-choice multinomial logit
`src/model_mnl_baseline.py`. Hand-rolled (via `scipy.optimize`) alternative-specific
multinomial logit — the textbook discrete-choice estimator, not
`sklearn.LogisticRegression` (which can't express "the same price coefficient
applies to whichever option is cheapest"). The honest statistical baseline.

### (5) `hier_bayes` — hierarchical Bayesian choice model (the primary twin)
`src/model_hier_bayes.py`. NUTS via NumPyro/JAX (CPU). Per-category and
per-agent-identity partial-pooling on price/rating/review sensitivity, plus a
per-group decision-temperature parameter. `src/behavioral_profiles.py`
extracts each agent's fitted price/rating sensitivity and decision
temperature from the posterior — a genuine behavioral profile per LLM, with
uncertainty. Full model spec: [docs/hierarchical_twin_spec.md §4](docs/hierarchical_twin_spec.md#4-model-2--hierarchical-bayesian-choice-model-the-twin).

**Reference baselines** (no learning — the memorization ceiling and the floor):

### (6) `consensus` — per-task modal-label lookup
`src/model_consensus.py`. No features, no fitting: memorizes `{prompt: modal
train label}`. Under a task-blocked split this collapses to exactly
`majority`'s accuracy — the automatic, built-in proof that a given split
isn't leaking task identity. Under the original leaky split it alone recovers
93–99% of `tfidf_logreg`'s reported accuracy, which is how the leak was found.

### (7) `majority` — always predict the training set's modal label
`src/model_majority.py`. The floor every other architecture must clear.

### (8) `tabular_llm` — attention-based deep tabular choice model
`src/model_tabular_llm.py`. The **tabular** counterpart to the text twins: it
consumes the same engineered per-option feature table the discrete-choice models
use (price / rating / review z-scores + within-row ranks, plus a category
one-hot), but processes it with a small **Transformer** — each of the 4 options
becomes a feature *token*, and cross-option self-attention (the LLM mechanism)
lets it learn non-linear comparisons a linear MNL cannot ("cheapest *and* well
rated", "much cheaper than the next option"). Trained from scratch, no text, no
pretrained weights; CPU-fast and deterministic given the seed.

> ⚠ Its first reported number (agent→agent ≈ 0.65, "strongest twin in the
> study") was measured under the default leaky `"group"` split and does not
> hold up: even though it never sees raw text, it's high-capacity enough to
> memorize near-unique price/rating/review-count *combinations* per task —
> the same failure mode as the text models, in a different disguise. Under
> `split_mode: "prompt"` its agent→agent accuracy drops to **0.5526** (vs.
> `tfidf_logreg`'s corrected 0.5636 — no longer the strongest) and
> `gap_on_human` flips from +0.0648 to **−0.0017**, the same inversion the
> text architectures show. See
> [docs/hierarchical_twin_spec.md §0.3](docs/hierarchical_twin_spec.md#03-corrected-split-results)
> for the full corrected numbers under both split designs.

### (9) `cognitive_decay` — choice model with within-session fatigue
`src/model_cognitive_decay.py`. An MNL whose decision *sharpness* (inverse
temperature) decays with the trial index:
`s(t) = softplus(a)·exp(−δ·t_norm) + floor`. The fitted **decay rate δ** is the
scientific output — it measures how much a decision-maker drifts toward noise
over a session. It uses the trial ordering now preserved end-to-end: humans get
their within-session trial index (parsed from `task_id`, e.g. `…_S2_T13` → trial
13); agents get a global item index (parsed from the `NO###` file ordering).
Run `python -m src.decay_analysis --config configs/config_corrected_prompt.yaml
--out results_cognitive_decay_corrected` for the per-source and per-agent
decay-rate tables and the decision-sharpness curves. **Use the corrected
config** — the original `results_cognitive_decay/` (default `config.yaml`)
predates the leakage fix and also hit a Windows-only crash writing the `δ`
character with the platform's default encoding (both fixed; kept only as a
historical/leaky-split artifact). Corrected finding, same direction as the
original: agent decision sharpness starts ~5× higher than human (27.7 vs
5.9) and decays similarly in relative terms (21% vs 7.5% drop); among agents
**ChatGPT O3 shows the strongest within-run decay** (δ=0.077, 91% drop)
**while GPT-5.5 and Grok show none** (δ=0.000, at the boundary — the pooled-
agent and GPT-5.5 fits didn't fully converge, consistent with a genuinely
near-zero effect there rather than an optimizer failure).

### Separate experiment — entropy introduced in training
`src/training_entropy.py` (not part of the transfer matrix). Injects a
controlled amount of label noise (entropy) into the **training** labels while
keeping test labels clean, then plots matched-cell accuracy vs the injected
noise for each twin and source. It isolates how much of the human–agent gap is
explained by the human target simply being higher-entropy: agent-trained twins
collapse steeply as noise rises (they rely on clean, low-entropy supervision),
whereas human-trained twins barely move (their labels are already near-noise).
Run `python -m src.training_entropy --config configs/config_corrected_prompt.yaml
--models tfidf_logreg tabular_llm mnl_baseline cognitive_decay --seeds 3`
(outputs in `results_training_entropy_corrected/`) — the full default sweep
(6 noise levels × 4 models × 2 sources × 3 seeds), under the corrected split.
An earlier partial run (1 model, 2 noise levels, 2 seeds, leaky split) has
been removed. Clean-baseline training-label entropy is nearly identical for
both sources (agent 1.994 bits, human 1.998 bits, both near the 2.0 max) —
so the *raw* entropy of the training target barely distinguishes the two
populations; what differs is how each **architecture** responds as more
noise is injected: `tfidf_logreg` degrades steeply (0.56→0.39 on agent — it
needs clean supervision), while `cognitive_decay` is essentially flat
(0.36→0.38) — it was never exploiting fine label structure, consistent with
its low-but-honest baseline accuracy.

Plus a standalone **Human-Agent Gap Analyzer** (`src/gap_analysis.py`) that
quantifies the population gap directly rather than only through accuracy —
empirical entropy / Herfindahl concentration per group, inter-agent Fleiss' κ,
and a model-based flagship metric `D_HA = JSD(P_human_twin, P_agent_twin)` —
and a **classifier two-sample test** (`src/c2st.py`) that asks directly "can a
classifier tell agents and humans apart from their chosen-option features
alone?" Full design, math, and every verified number:
**[docs/hierarchical_twin_spec.md](docs/hierarchical_twin_spec.md)**.

---

## 4. Quick start

### On Colab / Kaggle
1. Zip this whole folder → `digital_twin_gap.zip`.
2. Open `notebooks/run_on_colab_or_kaggle.ipynb` in Colab (or import to Kaggle).
3. Set the runtime to **GPU** (for `distilbert`): *Runtime → Change runtime type → GPU*.
4. Run cells top to bottom.

### Locally / on a server
```bash
pip install -r requirements.txt

# 1. Build the unified dataset from data/raw/
python -m src.data_prep --config configs/config.yaml

# 2. Run experiments — use a task-blocked config, not the default leaky one
python -m src.run_all --config configs/config_corrected_prompt.yaml
python -m src.run_all --config configs/config_corrected_prompt.yaml --models tfidf_logreg   # quick, one architecture

# 3. Make figures + report
python -m src.analyze --config configs/config_corrected_prompt.yaml

# 4. Gap analyzer, behavioral profiles, two-sample test (see docs/hierarchical_twin_spec.md §10
#    for the full sequence, including the strict two-way-blocked config and merging into one
#    7-architecture comparison table)
python -m src.gap_analysis --config configs/config_hier_bayes.yaml --out_dir results_gap_analysis
python -m src.behavioral_profiles --hier_bayes_dir results_corrected_prompt --out_dir results_behavioral_profiles
python -m src.c2st --config configs/config_corrected_prompt.yaml --out_dir results_c2st

# 5. New additions
#    (a) tabular LLM + cognitive-decay run in the transfer matrix like any model:
python -m src.run_all --config configs/config.yaml --models tabular_llm cognitive_decay
#    (b) cognitive-decay scientific output (decay rates + sharpness curves):
python -m src.decay_analysis --config configs/config.yaml
#    (c) entropy-introduced-in-training sweep (separate experiment):
python -m src.training_entropy --config configs/config.yaml \
       --models tfidf_logreg tabular_llm mnl_baseline cognitive_decay --seeds 3
```
Or just: `bash run_all.sh` (runs the original 3-architecture, leaky-split
config — kept only for backward compatibility; prefer the commands above).

---

## 5. Where your data goes

`configs/config.yaml` expects (relative to the project root):

```
data/raw/
├── Agentic response/          # the 6-agent .xlsx tree
│   ├── Claude/{Opus 4.6, Sonnet 4.6, Opus 4.8}/*.xlsx
│   ├── Chatgpt/{O3, GPT 5.5}/*.xlsx
│   └── Grok 4.2/Grok 4.2/*.xlsx
└── human/
    ├── llm_train_human.jsonl
    └── llm_val_human.jsonl
```

The data that shipped with this project is already staged there. If yours lives
elsewhere, edit only the two paths under `data:` in `configs/config.yaml`
(`agentic_root` and `human_files`).

---

## 6. Outputs

Each config writes to its own `output.results_dir` (never shared — every
config used in this study points at a different directory, so nothing gets
silently overwritten):

| File | What |
|---|---|
| `all_results.json` | Every cell, full metrics, per-agent breakdown |
| `summary_matrix.csv` | Flat table: one row per architecture × transfer cell |
| `gap_report.csv` | Headline gap numbers per architecture |
| `report.txt` | Plain-language summary |
| `fig_transfer_matrix.png` | 2×2 accuracy heatmap per architecture |
| `fig_gap_bars.png` | Matched vs transferred accuracy, side by side |
| `fig_per_agent.png` | Per-agent accuracy (which agents the twin models best) |

`hier_bayes` additionally writes `hier_bayes_model_{agent,human}.pkl`
(posterior samples, reused by `gap_analysis.py` / `behavioral_profiles.py`
without re-fitting) and `mcmc_diagnostics_{agent,human}.csv` (r_hat, ESS,
divergences — check these before trusting any `hier_bayes` number).

| Directory | What's in it |
|---|---|
| `results/` | Leaky-split (`"group"`) run, 5 architectures (`tfidf_logreg`, `embed_mlp`, `distilbert`, `tabular_llm`, `cognitive_decay`) — historical only, see Final verdict |
| `results_distilbert_tuned/` | `distilbert` with more epochs / lower LR, leaky split |
| `results_mnl_baseline/`, `results_hier_bayes/` | Leaky-split runs of the two discrete-choice architectures |
| `results_corrected_prompt/` | **All 9 architectures, prompt-blocked split — primary corrected result** |
| `results_corrected_twoway/` | All 9 architectures, strict two-way-blocked human split |
| `results_full_comparison/` | 6-architecture merge of the leaky-split runs (historical) |
| `results_gap_analysis/` | Entropy/HHI/Fleiss' κ (split-independent) + JSD (Tier 3, now re-run against `results_corrected_prompt/`'s `hier_bayes` posteriors — an earlier version of this directory used the leaky-split posteriors) |
| `results_behavioral_profiles/` | Per-agent price/rating sensitivity + decision temperature |
| `results_c2st/` | Classifier two-sample test result (corrected split) |
| `results_cognitive_decay/` | `decay_analysis.py`, leaky split — historical only, see §3 |
| `results_cognitive_decay_corrected/` | **`decay_analysis.py`, corrected split — trust this one** |
| `results_training_entropy_corrected/` | **`training_entropy.py`, full sweep, corrected split — trust this one** |

`run_all.py` **merges** with whatever is already in a `results_dir/all_results.json`
rather than overwriting it — running `--models X` only ever adds/updates `X`'s
row, it never deletes other architectures' rows from that directory. Pass
`--fresh` to explicitly discard prior results in that directory instead.

---

## 7. Results

**See the Final verdict at the top of this README, and
[docs/hierarchical_twin_spec.md §0.3](docs/hierarchical_twin_spec.md#03-corrected-split-results)
for the complete 9-architecture corrected tables under both split designs.**
Headline corrected numbers (`results_corrected_prompt/`, random baseline =
0.25):

| architecture | agent→agent | human→human | gap_on_human |
|---|:---:|:---:|:---:|
| tfidf_logreg | 0.564 | 0.326 | **−0.027** |
| embed_mlp | 0.542 | 0.325 | **−0.005** |
| distilbert | 0.461 | 0.358 | +0.057 |
| mnl_baseline | 0.452 | 0.355 | +0.017 |
| hier_bayes | 0.465 | 0.379 | +0.066 |
| tabular_llm | 0.553 | 0.366 | **−0.002** |
| cognitive_decay | 0.364 | 0.347 | +0.026 |
| consensus / majority | 0.254 | 0.304 | 0.000 |

---

## 8. Project layout

```
digital_twin_gap/
├── README.md
├── requirements.txt
├── run_all.sh                       # one-shot: prep -> experiments -> analysis (leaky-split config)
├── docs/
│   └── hierarchical_twin_spec.md    # full math spec + leakage audit + every verified number
├── configs/
│   ├── config.yaml                  # original config -- default split_mode is the LEAKY one
│   ├── config_distilbert_tuned.yaml
│   ├── config_mnl_baseline.yaml
│   ├── config_hier_bayes.yaml / config_hier_bayes_smoke.yaml
│   ├── config_corrected_prompt.yaml # <- recommended: task-blocked split, all 9 architectures
│   ├── config_corrected_twoway.yaml # <- strictest human split
│   └── config_full_comparison.yaml
├── notebooks/
│   └── run_on_colab_or_kaggle.ipynb
├── src/
│   ├── data_prep.py                 # agent .xlsx + human .jsonl -> unified schema
│   ├── splits.py                    # split_mode: group (leaky, default) | prompt | twoway
│   ├── option_features.py           # per-option price/rating/review feature parsing
│   ├── model_tfidf.py               # architecture 1
│   ├── model_embed_mlp.py           # architecture 2
│   ├── model_distilbert.py          # architecture 3
│   ├── model_mnl_baseline.py        # architecture 4 -- discrete-choice statistical baseline
│   ├── model_hier_bayes.py          # architecture 5 -- hierarchical Bayesian twin (NumPyro)
│   ├── model_consensus.py           # architecture 6 -- memorization-ceiling reference baseline
│   ├── model_majority.py            # architecture 7 -- floor reference baseline
│   ├── model_tabular_llm.py         # architecture 8 -- attention/Transformer over tabular features
│   ├── model_cognitive_decay.py     # architecture 9 -- MNL with within-session sharpness decay
│   ├── metrics.py                   # accuracy, macro-F1, per-agent, gap
│   ├── gap_analysis.py              # entropy/HHI/Fleiss' kappa/JSD human-agent gap analyzer
│   ├── behavioral_profiles.py       # per-agent price/rating sensitivity + decision temperature
│   ├── c2st.py                      # classifier two-sample test (agent vs human, structured features)
│   ├── decay_analysis.py            # per-source/per-agent decay-rate + sharpness report
│   ├── training_entropy.py          # standalone: accuracy vs. injected training-label noise
│   ├── merge_results.py             # safely merges multiple results dirs into one comparison
│   ├── run_all.py                   # runs the full transfer matrix (merges with existing results_dir; --fresh to discard)
│   └── analyze.py                   # figures + text report
├── data/
│   ├── raw/                         # input data (agent xlsx + human jsonl)
│   └── unified/                     # generated by data_prep.py (gitignored)
└── results*/                        # generated outputs, one dir per config -- see §6
```

---

## 9. Adding a new architecture

Drop a `src/model_yours.py` exposing `fit(train_rows, cfg)` and
`predict(model, test_rows)`, add its key to `MODEL_REGISTRY` in `run_all.py` and
a block under `models:` in your config, then
`python -m src.run_all --config <your_config>.yaml --models yours`. The transfer
matrix, metrics, gap computation, and plots all pick it up automatically.
**Use a task-blocked split** (`split_mode: "prompt"` or `"twoway"`) in your
config — see the Final verdict above for why the default `"group"` mode
cannot be trusted for accuracy claims.

---

## 10. Side-study: real-vs-synthetic Likert response gap

A **decoupled side-study**, unrelated to the agent-vs-human shopping-choice
task above — different data, different question, no train/test/fit. Uses
mentor-provided consumer-behavior survey data (`Datasets_total/`, not part of
the main pipeline) to ask: does a hand-coded synthetic-data generator's
output reproduce the *distribution* of the real human population it stands
in for? Two matched real/synthetic Likert-survey pairs
(`green_purchase_behavior`, `non_alcoholic_beverages`) show a real, robust
gap that replicates across both — real and synthetic respondents are
classifiably distinguishable (AUC 0.88 and 0.83, both p=0.001) on their raw
item-level response distributions, well short of the ~1.0 that would suggest
a trivial leak. Full method, exact metrics, limitations (no generator script
was found for either dataset, so this only describes the empirical gap, not
a validation of a known model), and reproduction commands:
**[docs/likert_side_study.md](docs/likert_side_study.md)**.
