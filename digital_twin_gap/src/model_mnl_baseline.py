"""
model_mnl_baseline.py
======================
ARCHITECTURE -- Model 1: hand-rolled discrete-choice multinomial logit (MNL).

This is the statistical baseline for the hierarchical-twin track. It is
deliberately NOT `sklearn.LogisticRegression` or `statsmodels.MNLogit`: both
fit one weight vector per CLASS given a single feature vector per ROW, which
cannot express the defining structural property of a real choice model --
"the same price coefficient applies to whichever option happens to be
cheapest," i.e. alternative-specific features with GENERIC (shared-across-
alternatives) coefficients. A hand-rolled softmax-NLL minimization over
per-option features is the textbook discrete-choice estimator and stays
lightweight (scipy only).

Feature vector per option j, row i (all option-varying so nothing cancels out
of the softmax):

    f_ij = [ z_ij (6 dims: price/rating/review, z-score + within-row rank)
             ASC_j (3 dims: 1{j=B}, 1{j=C}, 1{j=D}; A is reference)
             category x ASC_j        (9 dims: 3 non-ref categories x 3 ASC)
             category x z_ij         (18 dims: 3 non-ref categories x 6 feats)
             [agent-mode only] agent x ASC_j   (15 dims: 5 non-ref agents x 3 ASC)
             [agent-mode only] agent x z_ij    (30 dims: 5 non-ref agents x 6 feats) ]

Total: 36 dims (human-mode) / 81 dims (agent-mode). Human `group`=participant_id
has 571 levels -- far too sparse for unpooled dummies, and per this study's
scope decision individual identity must never be relied on for prediction on
unseen participants anyway -- so agent-identity interactions are included only
when training on agent data.

    U_ij(theta) = theta . f_ij
    P(y_i=j)    = softmax_j(U_i.)
    NLL(theta)  = -sum_i log P(y_i=y_i_observed) + 0.5*lambda*||theta||^2   (lambda = 1/C)

Interface matches the other architectures: fit(train_rows, cfg) / predict(model, rows).
"""

import numpy as np
from scipy.optimize import minimize

from src import option_features as optfeat

OPTION_LABELS = optfeat.OPTION_LABELS  # ["A","B","C","D"]
N_FEAT = len(optfeat.FEATURE_NAMES)  # 6
CATEGORY_NAMES_CANON = ["AIR", "COF", "EAR", "SNK"]  # AIR = reference


def _dummy(value, names_non_ref):
    """One-hot over `names_non_ref` (the reference level is implicit all-zero).
    Values not present in names_non_ref (unseen at predict time) also fall
    back to the reference (all-zero) -- a safe, documented fallback that is
    never expected to trigger given the verified split composition (every
    category and every agent identity appears in both train and test)."""
    v = np.zeros(len(names_non_ref))
    if value in names_non_ref:
        v[names_non_ref.index(value)] = 1.0
    return v


def _build_design(X, rows, mode, category_names_non_ref, group_names_non_ref):
    """X: (n,4,6) engineered option features (from option_features).
    rows: same length list of unified-schema dicts (for category/group).
    -> f: (n,4,D) design matrix."""
    n = X.shape[0]
    asc_dim = 3
    cat_dim = len(category_names_non_ref)
    group_dim = len(group_names_non_ref) if mode == "agent" else 0
    D = N_FEAT + asc_dim + cat_dim * asc_dim + cat_dim * N_FEAT
    if mode == "agent":
        D += group_dim * asc_dim + group_dim * N_FEAT

    f = np.zeros((n, 4, D), dtype=float)
    asc_eye = np.eye(4)[:, 1:]  # (4,3): A-row is all zero, B/C/D rows are one-hot

    for i, row in enumerate(rows):
        cat_d = _dummy(row["category"], category_names_non_ref)  # (cat_dim,)
        grp_d = _dummy(row["group"], group_names_non_ref) if mode == "agent" else None

        for j in range(4):
            z = X[i, j, :]  # (6,)
            asc = asc_eye[j]  # (3,)
            parts = [z, asc, np.outer(cat_d, asc).ravel(), np.outer(cat_d, z).ravel()]
            if mode == "agent":
                parts.append(np.outer(grp_d, asc).ravel())
                parts.append(np.outer(grp_d, z).ravel())
            f[i, j, :] = np.concatenate(parts)
    return f


def _nll_and_grad(theta, f, y, lam):
    U = np.einsum("njd,d->nj", f, theta)  # (n,4)
    U = U - U.max(axis=1, keepdims=True)
    expU = np.exp(U)
    P = expU / expU.sum(axis=1, keepdims=True)
    n = f.shape[0]
    logP_true = np.log(P[np.arange(n), y] + 1e-12)
    nll = -logP_true.sum() + 0.5 * lam * np.sum(theta ** 2)
    onehot = np.zeros_like(P)
    onehot[np.arange(n), y] = 1.0
    diff = onehot - P  # (n,4)
    grad = -np.einsum("nj,njd->d", diff, f) + lam * theta
    return nll, grad


def fit(train_rows, cfg):
    p = cfg["models"]["mnl_baseline"]
    mode = train_rows[0]["source"]  # "agent" | "human"
    assert all(r["source"] == mode for r in train_rows), "fit() expects a single-source batch"

    X, y, has_missing = optfeat.build_feature_matrix(train_rows)
    # build_feature_matrix() drops non-A-D-label rows internally; has_missing is
    # aligned to that same filtered set, so re-derive it here to stay in lockstep.
    kept_rows = [r for r in train_rows if r["label"] in OPTION_LABELS]
    assert len(kept_rows) == len(has_missing)
    keep_mask = [not hm for hm in has_missing]
    X = X[keep_mask]
    y = y[keep_mask]
    rows = [r for r, k in zip(kept_rows, keep_mask) if k]
    n_dropped_missing = len(kept_rows) - len(rows)
    if n_dropped_missing:
        print(f"  [mnl_baseline] dropped {n_dropped_missing} row(s) with unparseable option fields")

    category_names_non_ref = [c for c in CATEGORY_NAMES_CANON if c != "AIR"]
    group_names_non_ref = None
    if mode == "agent":
        all_groups = sorted(set(r["group"] for r in rows))
        ref_group = all_groups[0]
        group_names_non_ref = [g for g in all_groups if g != ref_group]
    else:
        ref_group = None

    f = _build_design(X, rows, mode, category_names_non_ref, group_names_non_ref or [])
    D = f.shape[2]
    theta0 = np.zeros(D)
    lam = 1.0 / max(p.get("C", 1.0), 1e-6)

    res = minimize(
        fun=lambda th: _nll_and_grad(th, f, y, lam),
        x0=theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": p.get("max_iter", 200)},
    )
    print(
        f"  [mnl_baseline] mode={mode} n={len(rows)} D={D} "
        f"converged={res.success} nll={res.fun:.2f} iters={res.nit}"
    )

    return {
        "theta": res.x,
        "mode": mode,
        "category_names_non_ref": category_names_non_ref,
        "group_names_non_ref": group_names_non_ref,  # None for human-mode
        "ref_group": ref_group,
    }


def predict(model, test_rows):
    X, _, _ = optfeat.build_feature_matrix_for_predict(test_rows)
    f = _build_design(
        X,
        test_rows,
        model["mode"],
        model["category_names_non_ref"],
        model["group_names_non_ref"] or [],
    )
    U = np.einsum("njd,d->nj", f, model["theta"])
    idx = U.argmax(axis=1)
    return [OPTION_LABELS[i] for i in idx]
