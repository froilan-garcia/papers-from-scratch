"""Closed-form efficient frontier — Markowitz (1952), Piece 1.

The paper poses the problem (Eqs. 1-4, p. 81) and solves it GEOMETRICALLY for
three and four assets, saying on p. 79 that it does not treat the n-asset case
analytically.  This module does, for the version without the sign constraint:

    min_w  w' Sigma w      subject to      mu' w = E,      1' w = 1        (*)

Dropping  w >= 0  is exactly what makes a closed form possible.  Restoring it
is Piece 3, in constrained.py, and the whole distance between the two is the
distance between a linear system and a quadratic program.

Everything below is four scalars (DERIVATIONS.md, sec. 4):

    A = 1' Sigma^-1 1,   B = 1' Sigma^-1 mu,   C = mu' Sigma^-1 mu,   D = AC - B^2

Sigma is never inverted explicitly.  np.linalg.solve is better conditioned, and
it keeps visible the fact that only two vectors of the market are ever needed,
Sigma^-1 1 and Sigma^-1 mu.
"""

import numpy as np


def scalars(market):
    """The four scalars A, B, C, D of sec. 4.  D > 0 unless mu is proportional to 1."""
    mu, Sigma = market.mu, market.Sigma
    ones = np.ones_like(mu)
    z1, zmu = np.linalg.solve(Sigma, ones), np.linalg.solve(Sigma, mu)
    A, B, C = ones @ z1, ones @ zmu, mu @ zmu
    D = A * C - B * B
    if D <= 0:
        raise ValueError("D <= 0: the expected returns are (numerically) all equal, "
                         "and the isomean lines of Fig. 2 are not defined (footnote 9)")
    return float(A), float(B), float(C), float(D)


def two_funds(market):
    """The pair (g, h) with  w(E) = g + E h,  sec. 6.

    Every efficient portfolio is an affine function of its own target return, so
    the whole frontier is spanned by two fixed portfolios.  g is the one with
    E = 0 and h is a zero-cost, unit-return direction:  1'g = 1, mu'g = 0,
    1'h = 0, mu'h = 1.  This is Tobin's (1958) two-fund separation in embryo,
    and in three assets it is the paper's *critical line* (sec. 10).
    """
    mu, Sigma = market.mu, market.Sigma
    ones = np.ones_like(mu)
    A, B, C, D = scalars(market)
    z1, zmu = np.linalg.solve(Sigma, ones), np.linalg.solve(Sigma, mu)
    g = (C * z1 - B * zmu) / D
    h = (A * zmu - B * z1) / D
    return g, h


def weights(market, E):
    """w(E), the minimum-variance portfolio with expected return E.  Sec. 4-6.

    E may be a scalar or an array; with an array the result is (len(E), n).
    """
    g, h = two_funds(market)
    E = np.asarray(E, float)
    return g + E[..., None] * h if E.ndim else g + E * h


def multipliers(market, E):
    """(lambda, gamma) of the Lagrangian.  Sec. 4, and sec. 7 for the reading of lambda.

    lambda = (AE - B)/D is half the slope of the frontier, dV/dE = 2 lambda: the
    shadow price of the return target, exactly as the lasso's multiplier prices
    its budget.
    """
    A, B, C, D = scalars(market)
    return (A * E - B) / D, (C - B * E) / D


def variance(market, E):
    """V(E) = (A E^2 - 2 B E + C)/D, sec. 5.  A parabola, opening upwards."""
    A, B, C, D = scalars(market)
    E = np.asarray(E, float)
    return (A * E**2 - 2 * B * E + C) / D


def volatility(market, E):
    """sqrt(V(E)).  The same frontier drawn as the hyperbola of the modern picture."""
    return np.sqrt(variance(market, E))


def min_variance(market):
    """The vertex of the parabola: E = B/A, V = 1/A, w = Sigma^-1 1 / A.  Sec. 5.

    Note what is missing: mu.  The minimum-variance portfolio does not depend on
    the expected returns at all, which is why it survives estimation error far
    better than any other point of the frontier.
    """
    A, B, C, D = scalars(market)
    z1 = np.linalg.solve(market.Sigma, np.ones_like(market.mu))
    return z1 / A, B / A, 1.0 / A


def is_efficient(market, E):
    """Efficient means minimum V for that E *and* maximum E for that V (p. 82).

    The lower half of the parabola, E < B/A, is minimum-variance but not
    efficient: for the same V there is a portfolio above with more return.
    """
    A, B, _, _ = scalars(market)
    return np.asarray(E) >= B / A


def evaluate(market, w):
    """(E, V) of an arbitrary portfolio -- the paper's Eqs. (1) and (2), p. 81."""
    w = np.asarray(w, float)
    return float(market.mu @ w), float(w @ market.Sigma @ w)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _check(market, label):
    """Everything the closed form claims, verified numerically."""
    from scipy.optimize import minimize

    mu, Sigma, n = market.mu, market.Sigma, len(market.mu)
    A, B, C, D = scalars(market)
    print(f"\n{label}: {n} assets")
    print(f"  A = {A:10.3f}   B = {B:8.3f}   C = {C:8.4f}   D = {D:8.4f}")

    w_mv, E_mv, V_mv = min_variance(market)
    print(f"  minimum variance: E = {E_mv:.4f}, sd = {np.sqrt(V_mv):.4f}, "
          f"w = {np.round(w_mv, 3)}")

    grid = np.linspace(E_mv - 0.02, E_mv + 0.06, 9)
    W = weights(market, grid)

    # 1. The constraints hold identically along the frontier.
    err_budget = np.abs(W.sum(axis=1) - 1).max()
    err_return = np.abs(W @ mu - grid).max()

    # 2. V(E) as a formula agrees with V(E) computed from the weights it predicts.
    err_parabola = np.abs(np.einsum("ij,jk,ik->i", W, Sigma, W) - variance(market, grid)).max()

    # 3. dV/dE = 2 lambda, to first order in a finite difference.
    lam, _ = multipliers(market, grid)
    slope = np.gradient(variance(market, grid), grid, edge_order=2)
    err_lambda = np.abs(slope - 2 * lam).max()

    # 4. The same problem handed to a general-purpose optimizer.
    err_qp = 0.0
    for E in grid:
        r = minimize(lambda w: w @ Sigma @ w, np.ones(n) / n, method="SLSQP",
                     jac=lambda w: 2 * Sigma @ w,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                  {"type": "eq", "fun": lambda w, E=E: mu @ w - E}],
                     options={"maxiter": 500, "ftol": 1e-14})
        err_qp = max(err_qp, np.abs(r.x - weights(market, E)).max())

    print(f"  1' w = 1                          max error {err_budget:.2e}")
    print(f"  mu' w = E                         max error {err_return:.2e}")
    print(f"  w' Sigma w = (AE^2-2BE+C)/D       max error {err_parabola:.2e}")
    print(f"  dV/dE = 2 lambda                  max error {err_lambda:.2e}")
    print(f"  closed form vs SLSQP              max error {err_qp:.2e}")

    # 5. No random portfolio lies to the left of the frontier.
    rng = np.random.default_rng(1952)
    R = rng.normal(size=(20000, n))
    R /= R.sum(axis=1, keepdims=True)
    E_r = R @ mu
    V_r = np.einsum("ij,jk,ik->i", R, Sigma, R)
    margin = (V_r - variance(market, E_r)).min()
    print(f"  20000 random portfolios, all above the parabola: "
          f"worst margin {margin:+.3e}")


if __name__ == "__main__":
    import markets

    _check(markets.sectors(), "sectors()")
    _check(markets.triple_inside(), "triple_inside()")
