"""Figures 1 and 2 — Sec. 2.2 and Sec. 2.3.

These illustrate; nothing else depends on them, which is why they come late.

Fig. 1: the four shrinkage functions in the orthonormal design.
Fig. 2: why the L1 corner produces zeros and the L2 ball does not.
"""

import numpy as np
import matplotlib.pyplot as plt

from lasso import l1_norm, lasso, ols, standardize
from orthonormal import (garotte_shrinkage, hard_threshold, ridge_shrinkage,
                         soft_threshold)


def figure_1(path="fig1_shrinkage_functions.png"):
    """Fig. 1: (a) subset, (b) ridge, (c) lasso, (d) garotte.

    Parameters chosen so the three thresholding methods have their knee at the
    same place, beta = 2, as in the paper's panels: hard threshold at 2, soft
    threshold gamma = 2, garotte gamma = 4 (its knee is at sqrt(gamma)).  Ridge
    has no knee; gamma = 4 reproduces the paper's slope of about 1/5.
    """
    b = np.linspace(0.0, 5.0, 501)
    panels = [
        ("(a) subset regression", hard_threshold(b, 2.0)),
        ("(b) ridge regression", ridge_shrinkage(b, 4.0)),
        ("(c) the lasso", soft_threshold(b, 2.0)),
        ("(d) the garotte", garotte_shrinkage(b, 4.0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6.4), sharex=True, sharey=True)
    for ax, (title, curve) in zip(axes.ravel(), panels):
        ax.plot(b, b, ls=":", c="0.6", lw=1)          # 45-degree reference
        ax.plot(b, curve, c="k", lw=1.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\hat\beta^{\,o}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\hat\beta$")
    fig.suptitle("Fig. 1 — coefficient shrinkage, orthonormal design", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def figure_2(path="fig2_geometry.png", seed=3):
    """Fig. 2: contours of (beta - beta^o)' X'X (beta - beta^o) against the
    L1 diamond (a) and the L2 disc (b).

    The contour drawn as a solid line is the one that passes exactly through the
    solution returned by the Sec. 6 solver, so the picture is a consequence of
    the code rather than a drawing of what the answer ought to look like.
    """
    rng = np.random.default_rng(seed)
    N = 60
    z = rng.standard_normal((N, 2))
    Xraw = np.column_stack([z[:, 0], 0.7 * z[:, 0] + 0.7 * z[:, 1]])  # correlated
    yraw = Xraw @ np.array([0.35, 1.6]) + 0.35 * rng.standard_normal(N)
    X, y, *_ = standardize(Xraw, yraw)

    beta_o = ols(X, y)
    A = X.T @ X                              # the quadratic form of Sec. 2.3
    t = 0.55 * l1_norm(beta_o)
    beta_l1 = lasso(X, y, t)

    # Ridge (Eq. 4) with lambda tuned to the same L2 radius as the L1 solution's,
    # so the two panels shrink by a comparable amount.
    radius = np.linalg.norm(beta_l1)
    lam = _ridge_lambda_for_radius(A, X.T @ y, radius)
    beta_l2 = np.linalg.solve(A + lam * np.eye(2), X.T @ y)

    def quad(g1, g2, centre):
        d1, d2 = g1 - centre[0], g2 - centre[1]
        return (A[0, 0] * d1 ** 2 + 2 * A[0, 1] * d1 * d2 + A[1, 1] * d2 ** 2)

    lim = 1.35 * max(np.abs(beta_o).max(), t)
    g = np.linspace(-lim, lim, 400)
    G1, G2 = np.meshgrid(g, g)
    Z = quad(G1, G2, beta_o)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7))
    theta = np.linspace(0, 2 * np.pi, 400)
    regions = [
        ("(a) the lasso", np.array([[t, 0], [0, t], [-t, 0], [0, -t], [t, 0]]),
         beta_l1),
        ("(b) ridge regression",
         np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]),
         beta_l2),
    ]

    for ax, (title, poly, sol) in zip(axes, regions):
        ax.fill(poly[:, 0], poly[:, 1], color="0.82", ec="0.35", lw=1.2)
        ax.contour(G1, G2, Z, levels=quad(*sol, beta_o) * np.array([0.25, 0.55]),
                   colors="0.65", linewidths=0.9)
        ax.contour(G1, G2, Z, levels=[quad(*sol, beta_o)],
                   colors="crimson", linewidths=1.6)
        ax.plot(*beta_o, "k.", ms=9)
        ax.annotate(r"$\hat\beta^{\,o}$", beta_o, xytext=(7, 4),
                    textcoords="offset points", fontsize=10)
        ax.plot(*sol, "o", c="crimson", ms=6)
        ax.annotate(rf"$\hat\beta=({sol[0]:.2f},\,{sol[1]:.2f})$", sol,
                    xytext=(8, -12), textcoords="offset points", fontsize=8,
                    c="crimson")
        ax.axhline(0, c="0.4", lw=0.8)
        ax.axvline(0, c="0.4", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\beta_1$")
        ax.set_ylabel(r"$\beta_2$")
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle("Fig. 2 — estimation picture", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    print(f"  lasso solution {np.round(beta_l1, 4)}"
          f"   (a corner: beta_1 exactly {beta_l1[0]:.1e})")
    print(f"  ridge solution {np.round(beta_l2, 4)}   (no zero)")


def _ridge_lambda_for_radius(A, Xty, radius, lo=0.0, hi=1e6):
    """Bisection for the lambda whose ridge solution has the given L2 norm."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        n = np.linalg.norm(np.linalg.solve(A + mid * np.eye(A.shape[0]), Xty))
        lo, hi = (mid, hi) if n > radius else (lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    figure_1()
    figure_2()
