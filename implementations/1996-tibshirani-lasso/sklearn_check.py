"""Cross-check of the Sec. 6 solver against scikit-learn.

The point of this file is that the two objectives are NOT the same, so nothing
can be compared until the conventions are lined up.  Ours is the paper's, Eq. (1):

    g(beta) = ||y - X beta||^2          subject to  sum_j |beta_j| <= t

whose Lagrangian form, with the lambda of DEDUCCIONES sec. 14, is

    ||y - X beta||^2 + 2 lambda ||beta||_1,      x_j'(y - X beta) = lambda s_j.

scikit-learn minimizes

    (1 / 2N) ||y - X beta||^2 + alpha ||beta||_1,

which multiplied by 2N is  ||y - X beta||^2 + 2 N alpha ||beta||_1.  Matching the
two gives the only conversion used below:

    alpha = lambda / N.

Three checks, in increasing strength:

  1. one point   — the paper's s = 0.44 on the prostate data;
  2. whole path  — every s on a grid, via the same conversion;
  3. no conversion at all — against LARS, matching on ||beta||_1, so a bug in
     the conversion cannot hide a bug in the solver.
"""

import numpy as np
from sklearn.linear_model import Lasso, lars_path

from lasso import l1_norm, lasso, lasso_path, ols, rss, t_max
from selection import lambda_from_kkt
import prostate

PREDICTORS = prostate.PREDICTORS


def alpha_for(X, y, beta):
    """The sklearn alpha equivalent to a fit of ours.  See the module docstring."""
    lam, spread = lambda_from_kkt(X, y, beta)
    return lam / X.shape[0], lam, spread


def sklearn_at(X, y, alpha):
    """sklearn's lasso with the intercept already removed by standardize()."""
    m = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=200_000)
    return m.fit(X, y).coef_


# --------------------------------------------------------------------------
# 1. One point: the model of Table 1
# --------------------------------------------------------------------------

def check_point(X, y, s=0.44):
    beta = lasso(X, y, s * t_max(X, y))
    alpha, lam, spread = alpha_for(X, y, beta)
    skl = sklearn_at(X, y, alpha)

    print(f"--- 1. One point: s = {s} (the model of Table 1) ---\n")
    print(f"lambda from KKT = {lam:.6f}  (spread across active coords {spread:.1e})")
    print(f"alpha = lambda / N = {alpha:.6f}\n")
    print(f"{'predictor':>9} {'Sec. 6':>10} {'sklearn':>10} {'diff':>10}")
    for name, a, b in zip(PREDICTORS, beta, skl):
        print(f"{name:>9} {a:10.6f} {b:10.6f} {abs(a-b):10.2e}")
    print(f"\nmax|diff| = {np.abs(beta - skl).max():.2e}")
    print(f"same support: {np.array_equal(np.abs(beta) > 1e-8, np.abs(skl) > 1e-8)}")
    print(f"RSS  Sec. 6 = {rss(X, y, beta):.10f}   sklearn = {rss(X, y, skl):.10f}")
    return np.abs(beta - skl).max()


# --------------------------------------------------------------------------
# 2. The whole path
# --------------------------------------------------------------------------

def check_path(X, y, n=41):
    grid = np.linspace(0.05, 0.95, n)
    ours = lasso_path(X, y, grid)
    worst, worst_s = 0.0, None
    for s, beta in zip(grid, ours):
        alpha, _, _ = alpha_for(X, y, beta)
        d = np.abs(beta - sklearn_at(X, y, alpha)).max()
        if d > worst:
            worst, worst_s = d, s
    print(f"\n--- 2. The whole path: {n} values of s ---\n")
    print(f"max|diff| over the grid = {worst:.2e}  (at s = {worst_s:.3f})")
    return worst


# --------------------------------------------------------------------------
# 3. Against LARS, with no conversion of conventions
# --------------------------------------------------------------------------

def check_lars(X, y):
    """LARS returns the exact path at its own breakpoints.  Comparing at equal
    ||beta||_1 needs no alpha and no lambda, so it is independent of check 1-2.
    """
    _, _, coefs = lars_path(X, y, method="lasso")     # (p, n_breakpoints)
    coefs = coefs.T
    norms = np.abs(coefs).sum(axis=1)

    print("\n--- 3. Against LARS, with no conversion of conventions ---\n")
    print(f"LARS gives {len(norms)} breakpoints; compared at each, at equal ||beta||_1\n")
    print(f"{'||beta||_1':>11} {'#active':>9} {'max|diff|':>10}")
    worst = 0.0
    for nrm in norms:
        if nrm < 1e-12:
            continue
        i = int(np.argmin(np.abs(norms - nrm)))
        ours = lasso(X, y, nrm)
        d = np.abs(ours - coefs[i]).max()
        worst = max(worst, d)
        print(f"{nrm:11.6f} {int((np.abs(ours) > 1e-8).sum()):9d} {d:10.2e}")
    print(f"\nmax|diff| over all breakpoints = {worst:.2e}")
    return worst


# --------------------------------------------------------------------------
# Where the two differ: what sklearn will not do
# --------------------------------------------------------------------------

def check_zeros(X, y, s=0.44):
    """Both agree on the fit and on the support; they print their zeros
    differently, and not in the direction one would guess.

    Ours are *algebraically* exact (DEDUCCIONES sec. 13: two active sign vectors
    differing in one coordinate force that coordinate to 0), but they reach the
    output through a linear solve, so round-off leaves them near 1e-14.
    Coordinate descent assigns 0 literally -- the soft threshold of sec. 6 is its
    update -- so its zeros print as 0.0 while the *support* it selects is only
    correct up to its convergence tolerance.  Exactness in a different place.
    """
    beta = lasso(X, y, s * t_max(X, y))
    alpha, _, _ = alpha_for(X, y, beta)
    skl = sklearn_at(X, y, alpha)
    ours_z = np.abs(beta)[np.abs(beta) < 1e-6]
    skl_z = np.abs(skl)[np.abs(skl) < 1e-6]
    print("\n--- The zeros: exact in different places ---\n")
    print(f"Sec. 6 : {ours_z.size} zeros, largest {ours_z.max():.2e}"
          "   (exact by algebra, rounded by the linear solve)")
    print(f"sklearn: {skl_z.size} zeros, largest {skl_z.max():.2e}"
          "   (assigned by the soft threshold, support subject to tolerance)")


def main():
    X, y, _ = prostate.load(paper_data=True)
    print(f"Prostate data: N = {X.shape[0]}, p = {X.shape[1]}\n")
    d1 = check_point(X, y)
    d2 = check_path(X, y)
    d3 = check_lars(X, y)
    check_zeros(X, y)

    print("\n" + "=" * 62)
    ok = max(d1, d2, d3) < 1e-6
    print(f"All three checks {'pass' if ok else 'DO NOT pass'} "
          f"(worst discrepancy {max(d1, d2, d3):.2e})")


if __name__ == "__main__":
    main()
