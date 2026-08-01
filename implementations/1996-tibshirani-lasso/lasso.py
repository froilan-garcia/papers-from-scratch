"""Lasso core solver — Tibshirani (1996).

Objective of Eq. (1) and the sequential sign-constraint algorithm of Sec. 6.
Deliberately NOT coordinate descent: that is Friedman et al. (2007), eleven
years later.  See README.md, "Reglas de esta implementacion".

Convention: the objective is the paper's, unscaled.

    g(beta) = sum_i (y_i - sum_j beta_j x_ij)^2      subject to  sum_j |beta_j| <= t

Predictors are assumed standardized (sum_i x_ij / N = 0, sum_i x_ij^2 / N = 1)
and y centred, so the intercept drops out of the optimization: alpha = mean(y).
"""

import numpy as np


# --------------------------------------------------------------------------
# Step 1 — evaluate before solving
# --------------------------------------------------------------------------

def rss(X, y, beta):
    """g(beta), Sec. 6.  Residual sum of squares, no 1/2 and no 1/N."""
    r = y - X @ beta
    return float(r @ r)


def l1_norm(beta):
    """sum_j |beta_j|, the left-hand side of the constraint in Eq. (1)."""
    return float(np.abs(beta).sum())


def is_feasible(beta, t, tol=1e-9):
    """Does beta satisfy the constraint of Eq. (1)?"""
    return l1_norm(beta) <= t + tol


def ols(X, y):
    """beta^o, the full least squares estimate (Sec. 2.1)."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def standardize(X, y):
    """Sec. 2.1: sum_i x_ij / N = 0, sum_i x_ij^2 / N = 1, and y centred.

    Note the 1/N (population) scaling, not 1/(N-1): the paper's normalisation
    makes each column have unit *second moment*, so X.T @ X has diagonal N.
    Returns the standardized (X, y) plus the shift/scale needed to map
    coefficients back to the original units.
    """
    mean, scale = X.mean(axis=0), X.std(axis=0)  # np.std uses ddof=0
    return (X - mean) / scale, y - y.mean(), mean, scale, y.mean()


# --------------------------------------------------------------------------
# Step 4 — the algorithm of Sec. 6
# --------------------------------------------------------------------------
#
# The constraint sum_j |beta_j| <= t is equivalent to the 2^p linear
# constraints  delta_i^T beta <= t,  where delta_i ranges over all p-tuples of
# the form (+-1, ..., +-1).  Proof: max over delta of delta^T beta is attained
# at delta = sign(beta) and equals sum_j |beta_j|.  So the L1 ball is a polytope
# with 2^p faces.  Sec. 6 never builds all 2^p rows: it adds violated ones one
# at a time until the Kuhn-Tucker conditions hold.

def _sign_vector(beta):
    """delta = sign(beta), forced to +-1 (a zero coefficient may take either
    sign: delta_j * beta_j = 0 both ways, so the constraint is unchanged)."""
    return np.where(beta >= 0, 1.0, -1.0)


def constrained_ls(XtX, Xty, G, h, tol=1e-9):
    """min g(beta) subject to G beta <= h — steps (b) and (d) of Sec. 6.

    Written with a general right-hand side h (the lasso passes h = t*1) so the
    same solver handles the garotte of Eq. (2), whose constraints are c_j >= 0
    and sum_j c_j <= t.

    Sec. 6 delegates this inner solve to Lawson & Hansen (1974); this is the
    textbook primal active-set method it stands for.  Keep a working set W of
    constraints held at equality and repeat:

      - solve for the step d that minimizes g on the current face, G_W d = 0,
        from the KKT system  [[H, G_W'], [G_W, 0]] [d; mu] = [-grad; 0]
        with H = 2 X'X and grad = 2(X'X beta - X'y);
      - if d = 0, the multipliers decide: all mu >= 0 is exactly the
        Kuhn-Tucker condition of Sec. 6, so stop; otherwise drop the most
        negative constraint, which is the one worth moving off;
      - otherwise walk along d until some constraint blocks, and add it to W.

    Starting point: beta = 0, feasible because G 0 = 0 <= h for every h >= 0.
    """
    m, p = G.shape
    h = np.asarray(h, dtype=float)
    beta = np.zeros(p)
    W = []

    for _ in range(4 * (m + p) + 20):
        grad = 2.0 * (XtX @ beta - Xty)
        GW = G[W] if W else np.zeros((0, p))
        K = np.block([[2.0 * XtX, GW.T], [GW, np.zeros((len(W), len(W)))]])
        sol = np.linalg.lstsq(K, np.concatenate([-grad, np.zeros(len(W))]),
                              rcond=None)[0]
        d, mu = sol[:p], sol[p:]

        if np.abs(d).max() <= tol:
            if not W or mu.min() >= -tol:
                return beta                      # Kuhn-Tucker satisfied
            W.pop(int(np.argmin(mu)))
            continue

        # Longest step along d that keeps G beta <= h.
        slack = np.maximum(h - G @ beta, 0.0)
        move = G @ d
        alpha, blocker = 1.0, None
        for i in range(m):
            if i in W or move[i] <= tol:
                continue
            ratio = slack[i] / move[i]
            if ratio < alpha:
                alpha, blocker = ratio, i
        beta = beta + alpha * d
        if blocker is not None:
            W.append(blocker)

    raise RuntimeError("inner active-set QP did not converge")


def lasso(X, y, t, tol=1e-9, return_info=False):
    """Solve Eq. (1) by the algorithm of Sec. 6, steps (a)-(d).

    (a) start with E = {i0} where delta_i0 = sign(beta^o), beta^o the OLS fit;
    (b) minimize g(beta) subject to G_E beta <= t*1;
    (c) while sum_j |beta_j| > t:
    (d) add to E the sign vector of the current beta, and re-solve.
    """
    p = X.shape[1]
    if t <= 0:
        beta = np.zeros(p)
        return (beta, {"n_iter": 0, "E": []}) if return_info else beta

    XtX, Xty = X.T @ X, X.T @ y

    E = [_sign_vector(ols(X, y))]                        # (a)
    beta = constrained_ls(XtX, Xty, np.array(E), np.full(len(E), float(t)), tol)  # (b)

    n_iter = 0
    while l1_norm(beta) > t + tol:                        # (c)
        delta = _sign_vector(beta)
        if any(np.array_equal(delta, e) for e in E):
            # Cannot happen in exact arithmetic: delta^T beta = sum|beta_j| > t
            # would violate a constraint the QP just enforced.
            raise RuntimeError("Sec. 6 loop repeated a sign vector")
        E.append(delta)                                   # (d)
        beta = constrained_ls(XtX, Xty, np.array(E), np.full(len(E), float(t)), tol)
        n_iter += 1

    if return_info:
        return beta, {"n_iter": n_iter, "E": E, "n_constraints": len(E)}
    return beta


# --------------------------------------------------------------------------
# Step 5 — the coefficient path
# --------------------------------------------------------------------------

def t_max(X, y):
    """t_0 = sum_j |beta^o_j| (Sec. 2.1): the smallest t at which the
    constraint stops binding, so s = t / t_0 lives in [0, 1]."""
    return l1_norm(ols(X, y))


def lasso_path(X, y, s_grid, tol=1e-9):
    """Coefficients along a grid of s = t / sum_j |beta^o_j| (Sec. 3, Fig. 5).

    Returns an array of shape (len(s_grid), p).
    """
    t0 = t_max(X, y)
    return np.array([lasso(X, y, s * t0, tol) for s in s_grid])


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # --- Sec. 6 claim: the algorithm converges in 0.5p to 0.75p iterations ---
    print("Iterations of the Sec. 6 loop (paper claims 0.5p-0.75p):\n")
    print(f"{'p':>4}  {'0.5p':>6}  {'0.75p':>6}  {'mean':>7}  {'max':>5}  {'2^p':>8}")
    for p in [4, 6, 8, 10]:
        counts = []
        for _ in range(40):
            N = 5 * p
            Xr = rng.standard_normal((N, p))
            Xr = (Xr - Xr.mean(0)) / Xr.std(0)
            yr = Xr @ rng.standard_normal(p) + rng.standard_normal(N)
            for s in [0.2, 0.4, 0.6, 0.8]:
                counts.append(lasso(Xr, yr, s * t_max(Xr, yr), return_info=True)[1]["n_iter"])
        counts = np.array(counts)
        print(f"{p:4d}  {0.5*p:6.1f}  {0.75*p:6.1f}  {counts.mean():7.2f}"
              f"  {counts.max():5d}  {2**p:8d}")

    # --- Step 5: what the path must satisfy ---
    N, p = 60, 8
    X = rng.standard_normal((N, p))
    X = (X - X.mean(0)) / X.std(0)
    y = X @ np.array([2.0, 0.0, -1.0, 0.0, 0.0, 1.5, 0.0, 0.0]) + rng.standard_normal(N)
    y = y - y.mean()

    s_grid = np.linspace(0.0, 1.0, 41)
    path = lasso_path(X, y, s_grid)
    beta_o = ols(X, y)

    print("\nPath checks:")
    print(f"  s=1 recovers OLS         : max|diff| = {np.abs(path[-1] - beta_o).max():.2e}")
    print(f"  s=0 gives all zeros      : max|beta| = {np.abs(path[0]).max():.2e}")
    norms = np.abs(path).sum(axis=1)
    print(f"  sum|beta| non-decreasing : {bool(np.all(np.diff(norms) >= -1e-9))}")
    print(f"  budget never exceeded    : "
          f"{bool(np.all(norms <= s_grid * l1_norm(beta_o) + 1e-9))}")

    # Why the zeros are structural rather than numerical: if two active sign
    # vectors differ only in coordinate j, subtracting their equalities
    # delta'beta = t and delta''beta = t gives 2 delta_j beta_j = 0, so
    # beta_j = 0 exactly.  What is left below is only KKT round-off.
    mid = path[len(s_grid) // 3]
    tiny = np.abs(mid)[np.abs(mid) < 1e-6]
    print(f"  zeros at machine precision: {tiny.size} zeros, "
          f"largest magnitude {tiny.max() if tiny.size else 0.0:.2e}")
