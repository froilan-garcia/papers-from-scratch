"""Choosing the lasso parameter — Sec. 4.

Two of the paper's three methods: fivefold cross-validation and generalized
cross-validation (Eq. 10).  Stein's unbiased risk estimate (Eq. 11) is not
implemented here; it is only derived for the orthogonal design.

Both index the fit by s = t / sum_j |beta^o_j|, "over a grid of values of s from
0 to 1 inclusive" (Sec. 4) — not by lambda on a log scale.
"""

import numpy as np

from lasso import l1_norm, lasso, ols, rss, t_max

DEFAULT_GRID = np.linspace(0.0, 1.0, 21)


# --------------------------------------------------------------------------
# Step 6 — fivefold cross-validation (Sec. 4)
# --------------------------------------------------------------------------
#
# Why the training RSS cannot be used instead: Eq. (8) says
#     PE = E{Y - eta_hat(X)}^2 = ME + sigma^2,
# so prediction error and model error differ by the constant sigma^2 and are
# minimized at the same s.  But the *training* RSS is not an estimate of PE:
# beta was chosen to make it small on those very points, so it falls
# monotonically as s grows and would always pick s = 1 (OLS).  Held-out data is
# what removes that optimism.

def cv_curve(X, y, s_grid=DEFAULT_GRID, n_folds=5, seed=0):
    """Fivefold CV estimate of prediction error over a grid of s.

    Declared choice: within each fold, t is set from *that fold's* training OLS,
    t = s * sum_j |beta^o_j(train)|.  This keeps s meaning "fraction of the
    unconstrained L1 norm" for the model actually being fitted.  The paper does
    not spell this out.
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(N), n_folds)

    errors = np.zeros((n_folds, len(s_grid)))
    for k, test in enumerate(folds):
        train = np.setdiff1d(np.arange(N), test)
        Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
        t0 = t_max(Xtr, ytr)
        for j, s in enumerate(s_grid):
            beta = lasso(Xtr, ytr, s * t0)
            errors[k, j] = np.mean((yte - Xte @ beta) ** 2)

    pe = errors.mean(axis=0)
    se = errors.std(axis=0, ddof=1) / np.sqrt(n_folds)
    return pe, se


def cv_select(X, y, s_grid=DEFAULT_GRID, n_folds=5, seed=0):
    """s_hat minimizing the CV estimate of prediction error."""
    pe, _ = cv_curve(X, y, s_grid, n_folds, seed)
    return float(s_grid[int(np.argmin(pe))]), pe


# --------------------------------------------------------------------------
# Step 7 — generalized cross-validation, Eq. (10)
# --------------------------------------------------------------------------
#
# The lasso is not a linear smoother, so GCV does not apply directly.  Sec. 2.5
# supplies the bridge: write the penalty sum_j |beta_j| as sum_j beta_j^2/|beta_j|,
# which turns the constrained fit into the *ridge* estimator of Eq. (9),
#     beta_tilde = (X'X + lambda W^-)^{-1} X'y,   W = diag(|beta_tilde_j|),
# a linear smoother whose hat matrix has trace
#     p(t) = tr{X (X'X + lambda W^-)^{-1} X'},
# the effective number of parameters.  Then Eq. (10):
#     GCV(t) = (1/N) rss(t) / {1 - p(t)/N}^2.

def lambda_from_kkt(X, y, beta, tol=1e-8):
    """The lambda of Eq. (9) implied by a lasso fit.

    Stationarity of ||y - X beta||^2 + lambda sum_j beta_j^2/|beta_j| reads
    (X'X + lambda W^-) beta = X'y, i.e. x_j'(y - X beta) = lambda sign(beta_j)
    for every active j.  So lambda = |x_j'(y - X beta)|, and it must come out
    the *same* for all active j — which is checked and returned as a diagnostic.
    """
    active = np.abs(beta) > tol
    if not active.any():
        return 0.0, 0.0
    c = np.abs(X[:, active].T @ (y - X @ beta))
    return float(c.mean()), float(c.max() - c.min())


def effective_parameters(X, beta, lam, tol=1e-8):
    """p(t) = tr{X (X'X + lambda W^-)^{-1} X'}, Sec. 4.

    Declared interpretation: W = diag(|beta_j|) is singular whenever a
    coefficient is 0, and the paper only says "W^- denotes a generalized
    inverse".  Read as the Moore-Penrose inverse it would put *zero* penalty on
    the zeroed coefficients, which is backwards.  The reading consistent with
    the paper's own text is 1/|beta_j| -> infinity, i.e. the zeroed predictors
    drop out of the fit entirely — that is exactly what makes formula (7) "give
    an estimated variance of 0 for predictors with beta_j = 0" (Sec. 2.5), as
    reported in Table 2.  So the trace is taken over the active set only.
    """
    active = np.abs(beta) > tol
    if not active.any():
        return 0.0
    XA = X[:, active]
    A = XA.T @ XA
    M = A + lam * np.diag(1.0 / np.abs(beta[active]))
    return float(np.trace(np.linalg.solve(M, A)))


def gcv_curve(X, y, s_grid=DEFAULT_GRID):
    """GCV(t) of Eq. (10) over a grid of s, plus p(t) at each grid point."""
    N = X.shape[0]
    t0 = t_max(X, y)
    gcv, dof = np.zeros(len(s_grid)), np.zeros(len(s_grid))
    for j, s in enumerate(s_grid):
        beta = lasso(X, y, s * t0)
        lam, _ = lambda_from_kkt(X, y, beta)
        dof[j] = effective_parameters(X, beta, lam)
        gcv[j] = (rss(X, y, beta) / N) / (1.0 - dof[j] / N) ** 2
    return gcv, dof


def gcv_select(X, y, s_grid=DEFAULT_GRID):
    """s_hat minimizing Eq. (10)."""
    gcv, dof = gcv_curve(X, y, s_grid)
    return float(s_grid[int(np.argmin(gcv))]), gcv, dof
