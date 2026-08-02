"""The two-predictor case — Sec. 2.4, Eq. (5), Eq. (6) and Fig. 4.

The paper states Eq. (5) and Eq. (6) and asserts that they hold "even if the
predictors are correlated", without saying why.  DERIVATIONS.md, section 6,
derives it: with standardized predictors X'X has equal diagonal entries, so
(1,1)' is an eigenvector of X'X, hence X'X^{-1}(1,1)' is again proportional to
(1,1)'.  The lasso therefore steps away from the OLS estimate along the direction
(-1,-1) whatever the correlation is.  This file checks that numerically.

Exact design: x1 = u1, x2 = rho*u1 + sqrt(1-rho^2)*u2 with u1, u2 orthogonal,
centred and of squared norm N.  Then X'X = N[[1, rho], [rho, 1]] exactly and, with
noiseless y = 6 x1 + 3 x2, the OLS estimate is exactly (6, 3).
"""

import numpy as np
import matplotlib.pyplot as plt

from lasso import lasso, ols

BETA_OLS = np.array([6.0, 3.0])          # the paper's example, Sec. 2.4
RHOS = [0.0, 0.23, 0.45, 0.68, 0.90]     # the five curves of Fig. 4
N = 100


def design(rho, n=N, seed=0):
    """X with X'X = n[[1, rho], [rho, 1]] exactly, and y = 6 x1 + 3 x2."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, 2))
    M -= M.mean(axis=0)                                  # centred
    Q = np.linalg.qr(M)[0] * np.sqrt(n)                  # Q'Q = n I
    X = np.column_stack([Q[:, 0], rho * Q[:, 0] + np.sqrt(1 - rho ** 2) * Q[:, 1]])
    return X, X @ BETA_OLS


def eq6(t):
    """Eq. (6): beta_1 = (t/2 + (b1-b2)/2)^+,  beta_2 = (t/2 - (b1-b2)/2)^+."""
    half = 0.5 * (BETA_OLS[0] - BETA_OLS[1])
    return np.array([max(0.5 * t + half, 0.0), max(0.5 * t - half, 0.0)])


def ridge_locus(X, y, lams):
    S, Xty = X.T @ X, X.T @ y
    return np.array([np.linalg.solve(S + lam * np.eye(2), Xty) for lam in lams])


def check():
    gap = BETA_OLS[0] - BETA_OLS[1]      # = 3; below this t, beta_2 leaves
    total = BETA_OLS.sum()               # = 9; above this t, the constraint sleeps

    print("1. Does the OLS estimate come out exactly (6, 3) for every rho?\n")
    for rho in RHOS:
        X, y = design(rho)
        err = np.abs(X.T @ X / N - np.array([[1, rho], [rho, 1]])).max()
        print(f"   rho = {rho:4.2f}   beta_o = {np.round(ols(X, y), 10)}"
              f"   ||X'X/N - target|| = {err:.1e}")

    print("\n2. Eq. (5): is the step away from beta_o a COMMON gamma on both")
    print("   coordinates, and is it the same for every rho?\n")
    print(f"   {'t':>5}  " + "  ".join(f"gamma(rho={r:.2f})" for r in RHOS))
    for t in [8.0, 7.0, 6.0, 5.0, 4.0, 3.0]:
        gammas = []
        for rho in RHOS:
            X, y = design(rho)
            b = lasso(X, y, t)
            g = BETA_OLS - b                       # should be (gamma, gamma)
            gammas.append(g if np.abs(g[0] - g[1]) > 1e-9 else g[0])
        print(f"   {t:5.1f}  " + "  ".join(f"{g:13.9f}" for g in gammas))

    print("\n3. Eq. (6) against the Sec. 6 solver.  The paper says the formula")
    print("   holds for t <= b1+b2 = 9; it also needs t >= b1-b2 = 3, which the")
    print("   paper does not say.  Below 3 the solution is (t, 0), not Eq. (6).\n")
    print(f"   {'t':>5}  {'solver':>18}  {'Eq. (6)':>18}  {'agree':>6}")
    for t in [9.0, 7.0, 5.0, 3.0, 2.5, 2.0, 1.0]:
        X, y = design(0.45)
        b, e = lasso(X, y, t), eq6(t)
        ok = "yes" if np.abs(b - e).max() < 1e-8 else "NO"
        print(f"   {t:5.1f}  {str(np.round(b, 4)):>18}  {str(np.round(e, 4)):>18}"
              f"  {ok:>6}"
              + ("   <- solver gives (t, 0)" if ok == "NO" else ""))

    print(f"\n   (t = {gap} is where beta_2 reaches 0; t = {total} is where the"
          f" constraint stops binding)")

    print("\n4. Ridge: DERIVATIONS.md section 7 predicts beta_2 rises as the bound")
    print("   is tightened exactly when rho > 1/2.  d(beta_2)/d(lambda) at 0:\n")
    for rho in RHOS:
        X, y = design(rho)
        a, b = N, N * rho
        # beta = 4.5*(a+b)/(a+b+lam)*(1,1) + 1.5*(a-b)/(a-b+lam)*(1,-1)
        slope = -4.5 / (a + b) + 1.5 / (a - b)
        num = np.gradient(ridge_locus(X, y, np.linspace(0, 1e-6, 3))[:, 1],
                          np.linspace(0, 1e-6, 3))[1]
        print(f"   rho = {rho:4.2f}   theory {slope:+.3e}   numeric {num:+.3e}"
              f"   beta_2 {'rises' if slope > 0 else 'falls'}")


def figure_4(path="fig4_two_predictors.png"):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    lams = np.concatenate([[0.0], np.logspace(-1, 4.5, 400)])
    for rho in RHOS:
        X, y = design(rho)
        r = ridge_locus(X, y, lams)
        ax.plot(r[:, 0], r[:, 1], ls="--", lw=1.1,
                label=f"ridge, $\\rho$ = {rho:.2f}")

    # One lasso curve is enough: it is the same for all rho (checked above).
    X, y = design(0.45)
    ts = np.linspace(0.0, BETA_OLS.sum(), 300)
    l1 = np.array([lasso(X, y, t) for t in ts])
    ax.plot(l1[:, 0], l1[:, 1], c="k", lw=2.0, label="lasso (all $\\rho$)")

    ax.plot(*BETA_OLS, "k.", ms=9)
    ax.annotate(r"$\hat\beta^{\,o}=(6,3)$", BETA_OLS, xytext=(-78, 4),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel(r"$\beta_1$")
    ax.set_ylabel(r"$\beta_2$")
    ax.set_title("Fig. 4 — lasso and ridge as the bound varies")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlim(1.6, 6.4)
    ax.set_ylim(-0.15, 3.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    check()
    figure_4()
