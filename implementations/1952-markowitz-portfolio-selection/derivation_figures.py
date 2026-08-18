"""Figures for DERIVATIONS.md.

These are not the paper's figures (those are in geometry.py and figures.py).
Each one checks a step of the development, and each is computed with the
solvers of this implementation rather than drawn.

    ded_frontier.png    sec. 5   the parabola in (E,V) and the hyperbola in (sigma,E)
    ded_paths.png       sec. 6   w(E) affine without the constraint, polygonal with it
    ded_curvature.png   sec. 12  lambda continuous, curvature jumping at each corner
"""

import numpy as np
import matplotlib.pyplot as plt

import markets
from constrained import corners, frontier as qp_frontier, min_variance_at, segment_market
from frontier import min_variance, scalars, variance, volatility, weights

GREY, RED, BLUE, GOLD = "0.55", "crimson", "#1f77b4", "#b8860b"


def frontier_shapes(market, path="ded_frontier.png"):
    """Sec. 5: one curve, two pictures, and the vertex that does not depend on mu."""
    A, B, C, D = scalars(market)
    w_mv, E_mv, V_mv = min_variance(market)
    span = np.linspace(E_mv - 0.045, E_mv + 0.085, 400)
    upper, lower = span >= E_mv, span <= E_mv

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax.plot(span[lower], variance(market, span[lower]), c=BLUE, lw=1.6, ls="--",
            label="minimum $V$, not efficient")
    ax.plot(span[upper], variance(market, span[upper]), c=RED, lw=2.4,
            label="efficient half")
    ax.plot(E_mv, V_mv, "o", c=GOLD, ms=7, mec="k", mew=0.6, zorder=5)
    ax.annotate(rf"$(B/A,\ 1/A) = ({E_mv:.4f},\ {V_mv:.4f})$",
                (E_mv, V_mv), textcoords="offset points", xytext=(14, 8), fontsize=9)
    ax.axvline(E_mv, c=GREY, lw=0.7, ls=":")
    ax.set_xlabel("$E$"), ax.set_ylabel("$V$")
    ax.set_title(r"the paper's axes: $V(E)=(AE^2-2BE+C)/D$, a parabola", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")

    ax2.plot(volatility(market, span[lower]), span[lower], c=BLUE, lw=1.6, ls="--")
    ax2.plot(volatility(market, span[upper]), span[upper], c=RED, lw=2.4)
    ax2.plot(np.sqrt(V_mv), E_mv, "o", c=GOLD, ms=7, mec="k", mew=0.6, zorder=5)
    for i, (nm, m_i) in enumerate(zip(market.names, market.mu)):
        ax2.plot(np.sqrt(market.Sigma[i, i]), m_i, "x", c="k", ms=6)
        ax2.annotate(nm, (np.sqrt(market.Sigma[i, i]), m_i), fontsize=8,
                     textcoords="offset points", xytext=(6, -3))
    ax2.set_xlabel(r"$\sigma=\sqrt{V}$"), ax2.set_ylabel("$E$")
    ax2.set_title(r"the modern axes: the same curve is a hyperbola", fontsize=10)
    ax2.margins(x=0.18)

    fig.suptitle("The frontier without the sign constraint", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def weight_paths(market, path="ded_paths.png"):
    """Sec. 6 and 12: two funds give straight lines, the constraint breaks them."""
    mu = market.mu
    span = np.linspace(mu.min() - 0.01, mu.max() + 0.01, 300)
    W = weights(market, span)

    grid = np.linspace(mu.min(), mu.max(), 400)
    Wc, Vc, _, _ = qp_frontier(market, grid)
    ks = [E for E, _, _ in corners(market)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for i, nm in enumerate(market.names):
        ax.plot(span, W[:, i], lw=1.7, label=nm)
        ax2.plot(grid, Wc[:, i], lw=1.7)
    ax.axhline(0, c="k", lw=0.8)
    ax2.axhline(0, c="k", lw=0.8)

    E_mv = min_variance(market)[1]
    ax.axvline(E_mv, c=GREY, lw=0.7, ls=":")
    ax.set_xlabel("$E$"), ax.set_ylabel("weight")
    ax.set_title(r"$w(E)=g+Eh$: every weight is affine in $E$", fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    for E in ks:
        ax2.axvline(E, c=GREY, lw=0.7, ls=":")
    ax2.set_xlabel("$E$")
    ax2.set_title(r"with $w\geq 0$: affine between corners, and only there",
                  fontsize=10)

    fig.suptitle("Where the polygonal chain of the paper comes from", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def curvature(market, path="ded_curvature.png"):
    """Sec. 12: the value function is C^1 and not C^2, with the corners as evidence."""
    mu = market.mu
    ks = [E for E, _, _ in corners(market)]
    grid = np.linspace(mu.min() + 1e-5, mu.max() - 1e-5, 800)
    _, V, _, lam = qp_frontier(market, grid)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax.plot(grid, 2 * lam, c=RED, lw=2.0, label=r"with $w\geq 0$")
    A, B, _, D = scalars(market)
    ax.plot(grid, 2 * (A * grid - B) / D, c=GOLD, lw=1.3, ls="--",
            label="shorts allowed")
    for E in ks:
        ax.axvline(E, c=GREY, lw=0.7, ls=":")
    ax.set_xlabel("$E$"), ax.set_ylabel(r"$dV/dE = 2\lambda$")
    ax.set_title(r"the slope is continuous: no jump at any corner", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")

    edges = [mu.min()] + ks + [mu.max()]
    for a, b in zip(edges[:-1], edges[1:]):
        sub, _ = segment_market(market, min_variance_at(market, 0.5 * (a + b))[1])
        A_s, _, _, D_s = scalars(sub)
        ax2.plot([a, b], [2 * A_s / D_s] * 2, c=RED, lw=2.6, solid_capstyle="butt")
        ax2.plot([a, b], [2 * A_s / D_s] * 2, "o", c=RED, ms=3.5)
    for E in ks:
        ax2.axvline(E, c=GREY, lw=0.7, ls=":")
    ax2.set_yscale("log")
    ax2.set_xlabel("$E$"), ax2.set_ylabel(r"$d^2V/dE^2 = 2A_F/D_F$")
    ax2.set_title("the curvature is not: each face has its own parabola", fontsize=10)

    fig.suptitle(r"The value function of the constrained problem", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    market = markets.sectors()
    frontier_shapes(market)
    weight_paths(market)
    curvature(market)
