"""
model_hier_bayes.py
====================
ARCHITECTURE -- Model 2: Hierarchical Bayesian choice model ("the twin").

The primary architecture of the hierarchical-twin track. A discrete-choice
(multinomial-logit) model whose slopes on the per-option features are given
partial-pooling hierarchical priors -- crossed random effects for CATEGORY
(both modes) and AGENT IDENTITY (agent-mode only, 6 well-identified groups),
plus a per-group "decision temperature" tau capturing how concentrated vs.
noisy each group's choices are.

Inference: NUTS via NumPyro/JAX, run on CPU (this model's runtime is
leapfrog-step-bound, not FLOP-bound -- a GPU buys nothing at this parameter
count, and official JAX GPU wheels are Linux/WSL2-only on native Windows
anyway; NumPyro was picked over PyMC specifically because it needs no
C/C++-compiler toolchain to JIT-compile the model).

Generative model (row i, option j in {A,B,C,D}; g=agent-identity index
[agent-mode only], p=participant index [human-mode only], c=category index;
X[i,j,k] = k-th engineered per-option feature, k=1..6, from option_features.py):

    asc_j        ~ Normal(0,1)                    for j in {B,C,D}; asc_A := 0
    beta_pop_k   ~ Normal(0,1)

    # category crossed random slopes (both modes, 4 levels, non-centered)
    sigma_beta_cat_k ~ HalfNormal(0.5)
    beta_cat[c,k] = sigma_beta_cat_k * Normal(0,1)

    # agent-identity crossed random slopes (AGENT-MODE ONLY, 6 levels, non-centered)
    sigma_beta_group_k ~ HalfNormal(0.5)
    beta_group[g,k] = sigma_beta_group_k * Normal(0,1)

    beta[i,k] = beta_pop_k + beta_cat[c(i),k] + (beta_group[g(i),k] if agent_mode else 0)

    # decision temperature (behavioral "concentration/consistency")
    mu_log_tau ~ Normal(0, 0.5)
      agent-mode: log_tau[i] = mu_log_tau + sigma_log_tau_group * Normal(0,1)[g(i)]
      human-mode: log_tau[i] = mu_log_tau + sigma_log_tau_participant * Normal(0,1)[p(i)]
                  (participant term is TRAIN-TIME ONLY -- never applied at
                   predict() time, so no unseen participant's prediction
                   depends on an individual latent; this is what makes the
                   human twin "population-level only for prediction" while
                   still getting legitimate shrinkage value during training)

    U[i,j] = asc_j + sum_k beta[i,k] * X[i,j,k]
    P(y_i=j) = softmax_j(U[i,.] / tau[i]);   y_i ~ Categorical(P)

predict()/predict_proba() use POSTERIOR-PREDICTIVE AVERAGING (mean of
softmax(U_s/tau_s) over posterior draws s, then argmax) rather than plugging
in posterior-mean coefficients -- the correct Bayes-optimal rule under a
nonlinear link (Jensen's inequality: E[softmax(f(theta))] != softmax(f(E[theta]))),
and it's what gap_analysis.py needs as genuine predictive probabilities.

Interface matches the other architectures (fit/predict), PLUS an extra
predict_proba() used by gap_analysis.py (not part of the required contract --
run_all.py/metrics.py never call it).

Task-level random effects (the study's gamma_t, "this task made option j
attractive independent of who's choosing"), a per-participant SLOPE hierarchy,
and hierarchical ASCs are explicitly out of scope for v1 -- see
docs/hierarchical_twin_spec.md Sec. 9 for why (no task ID bridges the agent
and human populations; participant-level slopes are too sparse at ~10.5
rows/participant; ASCs kept population-level for parsimony).
"""

import os
import pickle
import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # numerical stability for the funnel geometry
# induced by non-centered hierarchical scales.

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from src import option_features as optfeat

OPTION_LABELS = optfeat.OPTION_LABELS  # ["A","B","C","D"]
N_FEAT = len(optfeat.FEATURE_NAMES)  # 6
CATEGORY_NAMES_CANON = ["AIR", "COF", "EAR", "SNK"]


def _hier_model(X, cat_idx, group_idx, tau_group_idx, n_cats, n_groups, n_tau_groups, agent_mode, y=None):
    asc_bcd = numpyro.sample("asc_bcd", dist.Normal(0.0, 1.0).expand([3]).to_event(1))
    asc_full = jnp.concatenate([jnp.zeros(1), asc_bcd])  # [A=0, B, C, D]
    beta_pop = numpyro.sample("beta_pop", dist.Normal(0.0, 1.0).expand([N_FEAT]).to_event(1))

    sigma_beta_cat = numpyro.sample("sigma_beta_cat", dist.HalfNormal(0.5).expand([N_FEAT]).to_event(1))
    beta_cat_raw = numpyro.sample("beta_cat_raw", dist.Normal(0.0, 1.0).expand([n_cats, N_FEAT]).to_event(2))
    beta_cat = numpyro.deterministic("beta_cat", beta_cat_raw * sigma_beta_cat)

    if agent_mode:
        sigma_beta_group = numpyro.sample("sigma_beta_group", dist.HalfNormal(0.5).expand([N_FEAT]).to_event(1))
        beta_group_raw = numpyro.sample(
            "beta_group_raw", dist.Normal(0.0, 1.0).expand([n_groups, N_FEAT]).to_event(2)
        )
        beta_group = numpyro.deterministic("beta_group", beta_group_raw * sigma_beta_group)
    else:
        beta_group = jnp.zeros((n_groups, N_FEAT))  # n_groups==1 dummy; never sampled

    mu_log_tau = numpyro.sample("mu_log_tau", dist.Normal(0.0, 0.5))
    if agent_mode:
        sigma_log_tau_group = numpyro.sample("sigma_log_tau_group", dist.HalfNormal(0.3))
        log_tau_group_raw = numpyro.sample(
            "log_tau_group_raw", dist.Normal(0.0, 1.0).expand([n_groups]).to_event(1)
        )
        log_tau_group = numpyro.deterministic("log_tau_group", mu_log_tau + sigma_log_tau_group * log_tau_group_raw)
        tau_i = jnp.exp(log_tau_group[group_idx])
    else:
        sigma_log_tau_participant = numpyro.sample("sigma_log_tau_participant", dist.HalfNormal(0.3))
        log_tau_participant_raw = numpyro.sample(
            "log_tau_participant_raw", dist.Normal(0.0, 1.0).expand([n_tau_groups]).to_event(1)
        )
        # Deliberately NOT a numpyro.deterministic site: this per-participant
        # term is train-time-only regularization and is never read back at
        # predict() time (see module docstring / gap-analysis scope decision).
        log_tau_p = mu_log_tau + sigma_log_tau_participant * log_tau_participant_raw
        tau_i = jnp.exp(log_tau_p[tau_group_idx])

    beta_row = beta_pop[None, :] + beta_cat[cat_idx] + beta_group[group_idx]  # (n, N_FEAT)
    util = asc_full[None, :] + jnp.einsum("nk,njk->nj", beta_row, X)  # (n, 4)
    logits = util / tau_i[:, None]
    numpyro.sample("obs", dist.CategoricalLogits(logits), obs=y)


def _prepare_index_arrays(rows, mode, category_names, group_names, participant_names):
    cat_idx = np.array([category_names.index(r["category"]) for r in rows], dtype=np.int32)
    if mode == "agent":
        group_idx = np.array([group_names.index(r["group"]) for r in rows], dtype=np.int32)
        tau_group_idx = np.zeros(len(rows), dtype=np.int32)  # unused in agent-mode
    else:
        group_idx = np.zeros(len(rows), dtype=np.int32)  # unused (n_groups=1 dummy) in human-mode
        tau_group_idx = np.array([participant_names.index(r["group"]) for r in rows], dtype=np.int32)
    return cat_idx, group_idx, tau_group_idx


def fit(train_rows, cfg):
    p = cfg["models"]["hier_bayes"]
    mode = train_rows[0]["source"]  # "agent" | "human"
    assert all(r["source"] == mode for r in train_rows), "fit() expects a single-source batch"

    X, y, has_missing = optfeat.build_feature_matrix(train_rows)
    kept_rows = [r for r in train_rows if r["label"] in OPTION_LABELS]
    keep_mask = [not hm for hm in has_missing]
    X = X[keep_mask]
    y = y[keep_mask]
    rows = [r for r, k in zip(kept_rows, keep_mask) if k]
    n_dropped = len(kept_rows) - len(rows)
    if n_dropped:
        print(f"  [hier_bayes] dropped {n_dropped} row(s) with unparseable option fields")

    category_names = CATEGORY_NAMES_CANON
    n_cats = len(category_names)
    group_names = None
    participant_names = None
    if mode == "agent":
        group_names = sorted(set(r["group"] for r in rows))
        n_groups = len(group_names)
        n_tau_groups = 1
    else:
        n_groups = 1
        participant_names = sorted(set(r["group"] for r in rows))
        n_tau_groups = len(participant_names)

    cat_idx, group_idx, tau_group_idx = _prepare_index_arrays(
        rows, mode, category_names, group_names, participant_names
    )

    Xj = jnp.asarray(X)
    yj = jnp.asarray(y)
    cat_idx_j = jnp.asarray(cat_idx)
    group_idx_j = jnp.asarray(group_idx)
    tau_group_idx_j = jnp.asarray(tau_group_idx)

    kernel = NUTS(
        _hier_model,
        target_accept_prob=p.get("target_accept", 0.9),
        max_tree_depth=p.get("max_tree_depth", 10),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=p.get("tune", 1000),
        num_samples=p.get("draws", 1000),
        num_chains=p.get("chains", 4),
        chain_method=p.get("chain_method", "vectorized"),
        progress_bar=True,
    )
    rng_key = jax.random.PRNGKey(p.get("seed", 42))

    print(f"  [hier_bayes] fitting mode={mode} n={len(rows)} n_cats={n_cats} n_groups={n_groups} "
          f"n_tau_groups={n_tau_groups} (draws={p.get('draws',1000)} tune={p.get('tune',1000)} "
          f"chains={p.get('chains',4)})...")
    t0 = time.time()
    mcmc.run(
        rng_key,
        X=Xj, cat_idx=cat_idx_j, group_idx=group_idx_j, tau_group_idx=tau_group_idx_j,
        n_cats=n_cats, n_groups=n_groups, n_tau_groups=n_tau_groups, agent_mode=(mode == "agent"),
        y=yj,
    )
    elapsed = time.time() - t0

    samples = mcmc.get_samples()
    posterior = {k: np.asarray(v) for k, v in samples.items()}

    n_chains = p.get("chains", 4)
    n_draws = p.get("draws", 1000)
    diagnostics_df = None
    try:
        import arviz as az

        idata = az.from_numpyro(mcmc)
        diagnostics_df = az.summary(idata)
        n_divergent = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        n_total = n_chains * n_draws
        print(
            f"  [hier_bayes] done in {elapsed:.1f}s | max_r_hat={diagnostics_df['r_hat'].max():.4f} "
            f"min_ess_bulk={diagnostics_df['ess_bulk'].min():.0f} "
            f"min_ess_tail={diagnostics_df['ess_tail'].min():.0f} "
            f"divergences={n_divergent}/{n_total} ({100*n_divergent/n_total:.2f}%)"
        )
    except Exception as e:
        print(f"  [hier_bayes] WARNING: diagnostics computation failed: {e}")

    model = {
        "mode": mode,
        "agent_mode": (mode == "agent"),
        "posterior": posterior,
        "category_names": category_names,
        "group_names": group_names,  # None for human-mode
        "n_groups": n_groups,
    }

    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    if p.get("save_model_pickle", True):
        with open(os.path.join(out_dir, f"hier_bayes_model_{mode}.pkl"), "wb") as f:
            pickle.dump(model, f)
    if p.get("save_diagnostics", True) and diagnostics_df is not None:
        diagnostics_df.to_csv(os.path.join(out_dir, f"mcmc_diagnostics_{mode}.csv"))

    return model


def predict_proba(model, rows):
    """-> np.ndarray (n,4): posterior-predictive P(A/B/C/D | x) per row.
    Extra function beyond the fit/predict contract -- used by gap_analysis.py."""
    X, _, _ = optfeat.build_feature_matrix_for_predict(rows)  # (n,4,6), never drops a row
    X = X.astype(np.float32)
    n = X.shape[0]
    post = model["posterior"]
    category_names = model["category_names"]

    cat_idx_raw = np.array(
        [category_names.index(r["category"]) if r["category"] in category_names else -1 for r in rows]
    )
    cat_known = cat_idx_raw >= 0
    cat_idx_safe = np.where(cat_known, cat_idx_raw, 0)

    beta_pop_s = post["beta_pop"].astype(np.float32)  # (S,6)
    S = beta_pop_s.shape[0]
    beta_cat_s = post["beta_cat"].astype(np.float32)  # (S, n_cats, 6)
    beta_cat_row_s = beta_cat_s[:, cat_idx_safe, :] * cat_known[None, :, None]  # (S,n,6)

    if model["agent_mode"]:
        group_names = model["group_names"]
        group_idx_raw = np.array(
            [group_names.index(r["group"]) if r["group"] in group_names else -1 for r in rows]
        )
        group_known = group_idx_raw >= 0
        group_idx_safe = np.where(group_known, group_idx_raw, 0)
        beta_group_s = post["beta_group"].astype(np.float32)  # (S, n_groups, 6)
        beta_group_row_s = beta_group_s[:, group_idx_safe, :] * group_known[None, :, None]  # (S,n,6)
        log_tau_group_s = post["log_tau_group"].astype(np.float32)  # (S, n_groups)
        mu_log_tau_s = post["mu_log_tau"].astype(np.float32)  # (S,)
        log_tau_row_s = np.where(
            group_known[None, :],
            log_tau_group_s[:, group_idx_safe],
            mu_log_tau_s[:, None],
        )  # (S,n)
    else:
        # Human-mode: no group-level slope hierarchy exists at all, and the
        # participant-tau term is train-time-only -- prediction always uses
        # only the population-mean decision temperature (see module docstring).
        beta_group_row_s = np.zeros((S, n, N_FEAT), dtype=np.float32)
        mu_log_tau_s = post["mu_log_tau"].astype(np.float32)  # (S,)
        log_tau_row_s = np.broadcast_to(mu_log_tau_s[:, None], (S, n))

    beta_row_s = beta_pop_s[:, None, :] + beta_cat_row_s + beta_group_row_s  # (S,n,6)
    asc_bcd_s = post["asc_bcd"].astype(np.float32)  # (S,3)
    asc_full_s = np.concatenate([np.zeros((S, 1), dtype=np.float32), asc_bcd_s], axis=1)  # (S,4)

    util_s = asc_full_s[:, None, :] + np.einsum("snk,njk->snj", beta_row_s, X)  # (S,n,4)
    tau_s = np.exp(log_tau_row_s)  # (S,n)
    logits_s = util_s / tau_s[:, :, None]
    logits_s = logits_s - logits_s.max(axis=2, keepdims=True)
    exp_s = np.exp(logits_s)
    P_s = exp_s / exp_s.sum(axis=2, keepdims=True)  # (S,n,4)
    return P_s.mean(axis=0)  # (n,4) posterior-predictive probabilities


def predict(model, rows):
    P = predict_proba(model, rows)
    idx = P.argmax(axis=1)
    return [OPTION_LABELS[i] for i in idx]
