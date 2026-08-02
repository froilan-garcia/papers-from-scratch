"""Orthonormal design case — Eq. (3), Sec. 2.2.

When X'X = I the objective separates into p one-dimensional problems that share
a single budget t.  The derivation, in three lines:

    ||y - X beta||^2 = ||y||^2 - 2 beta' X'y + beta' X'X beta
                     = ||y||^2 - 2 beta' beta^o + beta' beta      (X'X = I, beta^o = X'y)
                     = ||beta - beta^o||^2 + const

so Eq. (1) becomes the projection of beta^o onto the L1 ball of radius t.  Its
stationarity condition, with a single multiplier gamma >= 0 shared by every
coordinate, is beta_j = beta^o_j - gamma * sign(beta_j), i.e. Eq. (3):

    beta_j = sign(beta^o_j) (|beta^o_j| - gamma)^+ ,   gamma set by sum_j |beta_j| = t.

Note the paper switches normalisation here: Sec. 2.1 standardizes columns to
sum_i x_ij^2 / N = 1 (so X'X has diagonal N), while Sec. 2.2 assumes X'X = I.
"""

import numpy as np


def soft_threshold(b, gamma):
    """The shrinkage function of Eq. (3), for a *fixed* gamma."""
    return np.sign(b) * np.maximum(np.abs(b) - gamma, 0.0)


def gamma_for_budget(beta_ols, t):
    """The gamma of Eq. (3): the value making sum_j |beta_j| = t.

    sum_j (|beta^o_j| - gamma)^+ is piecewise linear and decreasing in gamma, so
    the root is found exactly by sorting (no bisection).  Let a be |beta^o|
    sorted decreasingly; on the interval where exactly the top k terms survive,
    the equation reads sum_{i<=k} a_i - k*gamma = t.  The valid k is the largest
    one whose implied gamma keeps a_k above the threshold.
    """
    a = np.sort(np.abs(beta_ols))[::-1]
    if a.sum() <= t:
        return 0.0                                  # constraint inactive
    csum = np.cumsum(a)
    k = np.arange(1, len(a) + 1)
    candidate = (csum - t) / k
    valid = candidate < a                           # a_k still above threshold
    k_star = k[valid][-1]
    return float((csum[k_star - 1] - t) / k_star)


def lasso_orthonormal(beta_ols, t):
    """Closed-form solution of Eq. (1) when X'X = I — Eq. (3)."""
    return soft_threshold(beta_ols, gamma_for_budget(beta_ols, t))


# --- the other three shrinkage functions of Fig. 1, same design ------------

def hard_threshold(b, lam):
    """Best subset selection, Sec. 2.2: keep beta^o_j if |beta^o_j| > lam."""
    return np.where(np.abs(b) > lam, b, 0.0)


def ridge_shrinkage(b, gamma):
    """Ridge, Sec. 2.2: beta^o_j / (1 + gamma) — proportional shrinkage."""
    return b / (1.0 + gamma)


def garotte_shrinkage(b, gamma):
    """Non-negative garotte (Breiman, 1993), Sec. 2.2: (1 - gamma/beta^o_j^2)^+ beta^o_j."""
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(b == 0.0, 0.0, 1.0 - gamma / np.where(b == 0.0, 1.0, b ** 2))
    return np.maximum(factor, 0.0) * b


if __name__ == "__main__":
    # Validation: Eq. (3) against the Sec. 6 solver on an orthonormal design.
    from lasso import lasso, l1_norm, ols

    rng = np.random.default_rng(0)
    N, p = 40, 6
    X = np.linalg.qr(rng.standard_normal((N, p)))[0]     # X'X = I exactly
    y = X @ np.array([3.0, -1.5, 0.0, 0.8, 0.0, -2.0]) + 0.3 * rng.standard_normal(N)

    print(f"||X'X - I||_max = {np.abs(X.T @ X - np.eye(p)).max():.2e}\n")
    beta_o = ols(X, y)
    t0 = l1_norm(beta_o)

    print(f"{'s':>6}  {'max|Eq.3 - Sec.6|':>18}  {'sum|beta|':>10}  {'t':>8}")
    worst = 0.0
    for s in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0, 1.2]:
        t = s * t0
        closed = lasso_orthonormal(beta_o, t)
        solved = lasso(X, y, t)
        gap = np.abs(closed - solved).max()
        worst = max(worst, gap)
        print(f"{s:6.2f}  {gap:18.2e}  {l1_norm(solved):10.4f}  {t:8.4f}")

    print(f"\nworst disagreement over the grid: {worst:.2e}")

    # ---------------------------------------------------------------------
    # Stein's unbiased risk estimate for soft thresholding (Sec. 4, Eq. 11).
    # DERIVATIONS.md section 13 derives  p - 2#{|z_i| < g} + sum_i min(|z_i|, g)^2
    # and argues the paper prints max(|.|, g^2) where min(|.|, g)^2 belongs.
    # An unbiased estimate must average to the true risk; only one of them does.
    # ---------------------------------------------------------------------
    print("\nStein's risk estimate for soft thresholding (Sec. 4):")
    print("  an unbiased estimate must match the true risk on average.\n")
    p, n_rep = 8, 40000
    mu = np.array([3.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    Z = mu + rng.standard_normal((n_rep, p))          # z ~ N(mu, I)

    print(f"  {'gamma':>6}  {'true risk':>10}  {'with min (ours)':>16}  "
          f"{'with max (as printed)':>22}")
    for g in [0.0, 0.5, 1.0, 2.0, 4.0, 12.0]:
        true = np.mean(((soft_threshold(Z, g) - mu) ** 2).sum(axis=1))
        below = (np.abs(Z) < g).sum(axis=1)
        ours = np.mean(p - 2 * below + (np.minimum(np.abs(Z), g) ** 2).sum(axis=1))
        printed = np.mean(p - 2 * below + np.maximum(np.abs(Z), g ** 2).sum(axis=1))
        print(f"  {g:6.1f}  {true:10.3f}  {ours:16.3f}  {printed:22.3f}")
    print(f"\n  (as gamma -> infinity the true risk tends to ||mu||^2 = "
          f"{mu @ mu:.2f})")
