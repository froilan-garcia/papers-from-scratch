"""The paper's own figures in the (E, V) plane — Figs. 1 and 6.

Faithful to the paper's axes, which are not today's: Markowitz puts E on the
horizontal axis and V on the vertical one (Figs. 1, 5 and 6), where the modern
picture plots sigma horizontally and E vertically.  The derivation figures show
the other convention; here we stay with his.

    fig1_attainable.png   the attainable set and its efficient boundary (p. 82)
    fig6_frontier.png     the efficient frontier as CONNECTED PARABOLIC SEGMENTS

Fig. 6 is the one that has to be earned.  The paper draws it as a chain of
parabolic arcs and explains why (p. 87): each time a sign constraint starts or
stops binding, the surviving assets change, and the section of the paraboloid
changes with them.  Here each arc is drawn together with the full parabola it
belongs to, so that "different arcs, different parabolas" is visible rather
than asserted.
"""

import numpy as np
import matplotlib.pyplot as plt

import markets
from constrained import corners, frontier, min_variance_at, segment_market
from frontier import scalars, variance

GREY, RED, BLUE, GOLD = "0.55", "crimson", "#1f77b4", "#b8860b"


def attainable_cloud(market, n=40000, seed=1952):
    """Random long-only portfolios: the attainable set of Fig. 1, sampled.

    Dirichlet, not normalised gaussians: the constraint w >= 0 with 1'w = 1 is
    exactly the simplex, and Dirichlet(1) is uniform on it.
    """
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(market.mu)), size=n)
    return W @ market.mu, np.einsum("ij,jk,ik->i", W, market.Sigma, W)


def fig1(market, path="fig1_attainable.png"):
    """The attainable set and the efficient boundary (p. 82, Fig. 1)."""
    E_c, V_c = attainable_cloud(market)
    grid = np.linspace(market.mu.min(), market.mu.max(), 400)
    _, V, _, _ = frontier(market, grid)
    start = int(np.argmin(V))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(E_c, V_c, s=2, c=GREY, alpha=0.25, lw=0, label="attainable portfolios")
    ax.plot(grid[:start + 1], V[:start + 1], c=BLUE, lw=1.6, ls="--",
            label="minimum $V$ for each $E$, but not efficient")
    ax.plot(grid[start:], V[start:], c=RED, lw=2.6, label="efficient set")
    ax.plot(grid[start], V[start], "s", c=RED, ms=6, mec="k", mew=0.6)

    ax.set_xlabel("$E$ — expected return")
    ax.set_ylabel("$V$ — variance")
    ax.set_title("Fig. 1 — the attainable set and its efficient boundary", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def fig6(market, path="fig6_frontier.png"):
    """The efficient frontier as connected parabolic segments (p. 87, Fig. 6)."""
    mu = market.mu
    ks = corners(market)
    grid = np.linspace(mu.min(), mu.max(), 600)
    _, V, _, _ = frontier(market, grid)
    start = int(np.argmin(V))
    E0 = grid[start]

    edges = [mu.min()] + [E for E, _, _ in ks] + [mu.max()]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    # Left: each arc together with the parabola of its own sub-market.
    span = np.linspace(mu.min() - 0.004, mu.max() + 0.004, 300)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= E0:
            continue                              # inefficient branch
        a = max(a, E0)
        sub, _ = segment_market(market, min_variance_at(market, 0.5 * (a + b))[1])
        ax.plot(span, variance(sub, span), c=BLUE, lw=0.8, ls=":", alpha=0.9)
        arc = np.linspace(a, b, 120)
        ax.plot(arc, variance(sub, arc), c=RED, lw=2.8, solid_capstyle="round")

    for E, _, _ in ks:
        if E > E0:
            ax.plot(E, variance(segment_market(market, min_variance_at(market, E)[1])[0], E),
                    "o", c="k", ms=5, zorder=6)
    ax.plot(E0, V[start], "s", c=RED, ms=6, mec="k", mew=0.6, zorder=6)

    ax.set_ylim(0, V[start:].max() * 1.05)
    ax.set_xlim(E0 - 0.004, mu.max() + 0.004)
    ax.set_xlabel("$E$ — expected return")
    ax.set_ylabel("$V$ — variance")
    ax.set_title("each arc belongs to a different parabola", fontsize=10)

    # Right: the same chain against the single parabola of the unconstrained problem.
    ax2.plot(grid[start:], V[start:], c=RED, lw=2.6, label=r"with $w\geq 0$")
    ax2.plot(span, variance(market, span), c=GOLD, lw=1.4, ls="--",
             label="shorts allowed (Piece 1)")
    for E, _, _ in ks:
        if E > E0:
            ax2.plot(E, variance(segment_market(market, min_variance_at(market, E)[1])[0], E),
                     "o", c="k", ms=5, zorder=6)
    ax2.set_ylim(0, V[start:].max() * 1.05)
    ax2.set_xlim(E0 - 0.004, mu.max() + 0.004)
    ax2.set_xlabel("$E$ — expected return")
    ax2.set_title("the price of the constraint", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle("Fig. 6 — the efficient frontier is a chain of parabolic arcs",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    market = markets.sectors()
    fig1(market)
    fig6(market)
