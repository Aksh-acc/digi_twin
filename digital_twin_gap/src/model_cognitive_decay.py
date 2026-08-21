"""
model_cognitive_decay.py
=========================
ARCHITECTURE -- Cognitive-decay discrete-choice model.

Motivation
----------
A real decision-maker does not answer the 30th question the way they answer the
1st. Attention drifts, fatigue accumulates, and choices drift toward noise (or
toward a fixed heuristic) as a session wears on. A standard choice model assumes
a *constant* decision sharpness across all trials and therefore cannot express
this. This model adds an explicit **cognitive-decay** term: the sharpness of the
softmax over option-utilities is modulated by the trial index.

Every human row carries a within-session trial index (parsed from task_id, e.g.
"EAR_TN_S2_T13" -> trial 13); every agent row carries a global item index
(parsed from the NO### file ordering). Both are exposed as `row["trial"]`.

Model
-----
Per option j in row i (features from the shared option_features module):

    u_ij            = beta . f_ij                         (base utility, generic coefs)
    s_i(t)          = softplus(a) * exp(-delta * that_i) + floor
    P(y_i = j)      = softmax_j( s_i(t) * u_ij )

where
    that_i          = (trial_i - 1) / T_scale             (normalized trial, >=0)
    s_i(t)          = the *decision sharpness* at trial t (inverse temperature)
    delta >= 0      = the COGNITIVE-DECAY RATE (the parameter of interest)
    softplus(a)     = initial sharpness at trial 0 (kept positive)
    floor           = small constant so very late trials are noisy, not degenerate.

delta = 0 recovers a constant-sharpness MNL (no decay). delta > 0 means later
trials have a flatter choice distribution -- the decision-maker is being modelled
as progressively noisier / more fatigued. We fit beta, a, delta jointly by
maximum likelihood (L-BFGS-B, analytic gradient), with an L2 penalty on beta.

The fitted `delta` (and the implied sharpness curve) is the scientific output:
it is reported per training source so the paper can state, e.g., "human choices
decay in quality over a session at rate delta_H, whereas agent choices show
delta_A ~ 0 (no within-run fatigue)".

Interface matches the other architectures: fit(train_rows, cfg) / predict(model, rows).
"""

import numpy as np
from scipy.optimize import minimize

from src import option_features as optfeat

OPTION_LABELS = optfeat.OPTION_LABELS  # ["A","B","C","D"]
N_FEAT = len(optfeat.FEATURE_NAMES)    # 6


def _softplus(x):
    # numerically stable softplus
    return np.logaddexp(0.0, x)


def _prep(rows, t_scale):
    """Build (X, y, that) from unified rows, dropping non-A-D and unparseable.
    that = normalized, non-negative trial index; rows with no trial default to 0
    (== trial 1), which simply places them at the no-decay end of the curve."""
    X, y, has_missing = optfeat.build_feature_matrix(rows)
    kept = [r for r in rows if r["label"] in OPTION_LABELS]
    keep = [not hm for hm in has_missing]
    X = X[keep]
    y = y[keep]
    kept = [r for r, k in zip(kept, keep) if k]

    trials = np.array(
        [(r.get("trial") if r.get("trial") is not None else 1) for r in kept],
        dtype=float,
    )
    that = np.maximum(trials - 1.0, 0.0) / float(t_scale)
    return X, y, that, len(rows) - len(kept)


def _unpack(theta):
    beta = theta[:N_FEAT]
    a = theta[N_FEAT]
    delta = theta[N_FEAT + 1]
    return beta, a, delta


def _nll_and_grad(theta, X, y, that, lam, floor):
    """X:(n,4,6) y:(n,) that:(n,). Returns (nll, grad) with analytic gradient."""
    n = X.shape[0]
    beta, a, delta = _unpack(theta)

    u = np.einsum("njf,f->nj", X, beta)          # (n,4) base utilities
    sp = _softplus(a)                             # scalar initial sharpness
    decay = np.exp(-delta * that)                 # (n,)
    s = sp * decay + floor                        # (n,) per-row sharpness

    z = s[:, None] * u                            # (n,4) scaled utilities
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    P = ez / ez.sum(axis=1, keepdims=True)        # (n,4)

    logP_true = np.log(P[np.arange(n), y] + 1e-12)
    nll = -logP_true.sum() + 0.5 * lam * np.sum(beta ** 2)

    onehot = np.zeros_like(P)
    onehot[np.arange(n), y] = 1.0
    diff = onehot - P                             # (n,4); dNLL/dz = -(diff)

    # grad wrt beta: dz/dbeta = s * X  ->  sum_n sum_j -diff_nj * s_n * X_njf
    grad_beta = -np.einsum("nj,n,njf->f", diff, s, X) + lam * beta

    # For a and delta we need dz/ds * ds/dparam.
    # dz_nj/ds_n = u_nj ; sum_j -diff_nj * u_nj  == g_n
    g = -np.einsum("nj,nj->n", diff, u)           # (n,) = dNLL/ds_n
    # ds/da     = sigmoid(a) * decay           (since d softplus/da = sigmoid)
    sig = 1.0 / (1.0 + np.exp(-a))
    ds_da = sig * decay                           # (n,)
    grad_a = np.sum(g * ds_da)
    # ds/ddelta = sp * decay * (-that)
    ds_ddelta = sp * decay * (-that)              # (n,)
    grad_delta = np.sum(g * ds_ddelta)

    grad = np.concatenate([grad_beta, [grad_a, grad_delta]])
    return nll, grad


def fit(train_rows, cfg):
    p = cfg["models"].get("cognitive_decay", {})
    lam = 1.0 / max(p.get("C", 1.0), 1e-6)
    t_scale = float(p.get("trial_scale", 12.0))   # normalizer for trial index
    floor = float(p.get("sharpness_floor", 0.05))
    max_iter = int(p.get("max_iter", 300))

    mode = train_rows[0]["source"]
    X, y, that, n_dropped = _prep(train_rows, t_scale)
    if n_dropped:
        print(f"  [cognitive_decay] dropped {n_dropped} row(s) (bad label/fields)")

    # init: small beta, sharpness ~1 (softplus(0.5)~0.97), delta 0 (no decay)
    theta0 = np.concatenate([np.zeros(N_FEAT), [0.5, 0.0]])
    # bound delta >= 0 so "decay" can't flip into "sharpening"; a unbounded.
    bounds = [(None, None)] * N_FEAT + [(None, None), (0.0, 5.0)]

    res = minimize(
        fun=lambda th: _nll_and_grad(th, X, y, that, lam, floor),
        x0=theta0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter},
    )
    beta, a, delta = _unpack(res.x)
    sp = float(_softplus(a))

    # Report the decay: sharpness at first vs last trial actually seen.
    tmax = float(that.max()) if len(that) else 0.0
    s0 = sp + floor
    s_last = sp * np.exp(-delta * tmax) + floor
    print(
        f"  [cognitive_decay] mode={mode} n={len(y)} converged={res.success} "
        f"nll={res.fun:.2f} | delta(decay)={delta:.4f} "
        f"sharpness: start={s0:.3f} -> end={s_last:.3f} "
        f"({100*(1-s_last/s0):.1f}% drop over session)"
    )

    return {
        "beta": beta, "a": a, "delta": float(delta),
        "sp": sp, "floor": floor, "t_scale": t_scale, "mode": mode,
        # scientific summary surfaced for downstream reporting
        "decay_rate": float(delta),
        "sharpness_start": float(s0),
        "sharpness_end": float(s_last),
        "sharpness_drop_frac": float(1 - s_last / s0) if s0 > 0 else 0.0,
    }


def predict(model, test_rows):
    X, _, _ = optfeat.build_feature_matrix_for_predict(test_rows)
    trials = np.array(
        [(r.get("trial") if r.get("trial") is not None else 1) for r in test_rows],
        dtype=float,
    )
    that = np.maximum(trials - 1.0, 0.0) / model["t_scale"]

    u = np.einsum("njf,f->nj", X, model["beta"])
    s = model["sp"] * np.exp(-model["delta"] * that) + model["floor"]
    z = s[:, None] * u
    idx = z.argmax(axis=1)   # argmax is temperature-invariant, but predict()
    # only needs the label; the decay term matters for the fitted likelihood
    # and the reported sharpness curve, not for which option is the argmax.
    return [OPTION_LABELS[i] for i in idx]


def predict_proba(model, test_rows):
    """Exposed for the entropy/analysis tooling: returns (n,4) choice
    probabilities WITH the trial-dependent sharpness applied."""
    X, _, _ = optfeat.build_feature_matrix_for_predict(test_rows)
    trials = np.array(
        [(r.get("trial") if r.get("trial") is not None else 1) for r in test_rows],
        dtype=float,
    )
    that = np.maximum(trials - 1.0, 0.0) / model["t_scale"]
    u = np.einsum("njf,f->nj", X, model["beta"])
    s = model["sp"] * np.exp(-model["delta"] * that) + model["floor"]
    z = s[:, None] * u
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)
