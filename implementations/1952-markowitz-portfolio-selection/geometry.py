"""The geometry of the paper — Markowitz (1952), Piece 2: Figs. 2 and 3.

With three assets the budget constraint  X1 + X2 + X3 = 1  eliminates one
variable (the paper's Eq. 3', p. 83) and the whole problem becomes plane
geometry in (X1, X2):

    attainable set   the triangle X1 >= 0, X2 >= 0, X1 + X2 <= 1
    isomeans         parallel straight lines, because E is affine in X
    isovariances     concentric ellipses, because V is a quadratic form
    critical line    the locus of tangency between the two families

Everything drawn here is computed: the ellipses from the reduced quadratic form,
the critical line from Piece 1 and the efficient set from the active set solver
of Piece 3.  The figure is therefore a check of the paper's claims and not an
illustration of them -- in particular of the one the paper asserts without any
algebra (p. 85), that the tangency points lie on a straight line.

Two cases, which is why the paper needs both figures: the centre of the ellipses
inside the triangle (Fig. 2) or outside it (Fig. 3).
"""

import numpy as np
import matplotlib.pyplot as plt

import markets
from constrained import corners, min_variance_at

GREY, RED, BLUE, GOLD = "0.55", "crimson", "#1f77b4", "#b8860b"


def reduce_to_plane(market):
    """Substitute X3 = 1 - X1 - X2 and return the reduced problem.

    Writing w = e3 + Dx with D = [e1 - e3, e2 - e3] and x = (X1, X2),

        E(x) = mu3 + e' x,                e = (mu1 - mu3, mu2 - mu3)
        V(x) = sigma33 + 2 c' x + x' Q x, Q = D' Sigma D,  c = D' Sigma e3

    Q is the object footnote 12 (p. 89) is about: the isovariances are genuine
    ellipses exactly when Q is positive definite, and Q is positive definite
    exactly when no two distinct portfolios have perfectly correlated returns.
    """
    mu, Sigma = market.mu, market.Sigma
    D = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    Q = D.T @ Sigma @ D
    c = D.T @ Sigma @ np.array([0.0, 0.0, 1.0])
    e = np.array([mu[0] - mu[2], mu[1] - mu[2]])
    return Q, c, e, float(Sigma[2, 2]), float(mu[2])


def variance_on_plane(market, x):
    """V at the point x = (X1, X2) of the plane."""
    Q, c, _, s33, _ = reduce_to_plane(market)
    x = np.atleast_2d(np.asarray(x, float))
    return s33 + 2 * x @ c + np.einsum("ij,jk,ik->i", x, Q, x)


def expected_on_plane(market, x):
    """E at the point x = (X1, X2).  Affine, which is why the isomeans are parallel."""
    _, _, e, _, mu3 = reduce_to_plane(market)
    return mu3 + np.atleast_2d(np.asarray(x, float)) @ e


def centre(market):
    """X-hat, the centre of the ellipses: the unconstrained minimiser of V.

    It is the same portfolio as frontier.min_variance, seen in the plane -- the
    figure's first check.
    """
    Q, c, _, _, _ = reduce_to_plane(market)
    return -np.linalg.solve(Q, c)


def critical_line(market):
    """(point, direction) of the paper's critical line l, from the plane algebra.

    Minimising V(x) subject to e'x = E - mu3 gives  2Qx + 2c = theta e,  hence

        x(theta) = X-hat + (theta/2) Q^-1 e,

    a straight line through X-hat with direction Q^-1 e, independent of E.  That
    is the assertion of p. 85, and the derivation the paper omits (sec. 10).
    """
    Q, _, e, _, _ = reduce_to_plane(market)
    d = np.linalg.solve(Q, e)
    return centre(market), d / np.linalg.norm(d)


def efficient_set(market, n=600):
    """The efficient set in the plane, from the active set solver of Piece 3.

    Runs from the constrained minimum-variance portfolio to the asset of maximum
    expected return, which is the paper's description on p. 85.
    """
    mu = market.mu
    grid = np.linspace(mu.min(), mu.max(), n)
    W = np.array([min_variance_at(market, E)[0] for E in grid])
    V = np.einsum("ij,jk,ik->i", W, market.Sigma, W)
    start = int(np.argmin(V))
    return W[start:, :2], grid[start:]


def simplex_figure(market, path, title, subtitle):
    """The paper's Fig. 2 (or Fig. 3) for a three-asset market."""
    names = market.names
    x_hat = centre(market)
    p0, d = critical_line(market)
    eff, _ = efficient_set(market)

    lo = min(-0.25, x_hat[0] - 0.15, x_hat[1] - 0.15)
    hi = max(1.15, x_hat[0] + 0.15, x_hat[1] + 0.15)
    gx = np.linspace(lo, hi, 400)
    X1, X2 = np.meshgrid(gx, gx)
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    Vg = variance_on_plane(market, pts).reshape(X1.shape)
    Eg = expected_on_plane(market, pts).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(6.8, 6.9))

    # Isovariance ellipses, at levels taken along the critical line so that each
    # one is tangent to an isomean somewhere in view.
    ts = np.linspace(0.05, 1.6, 7)
    levels = np.sort(variance_on_plane(market, p0 + np.outer(ts, d)))
    cs = ax.contour(X1, X2, Vg, levels=levels, colors=BLUE, linewidths=0.9, alpha=0.8)
    ax.clabel(cs, fmt=lambda v: f"$\\sigma$={np.sqrt(v):.2f}", fontsize=7, inline=True)

    # Isomean lines: contours of an affine function, hence parallel.
    e_lv = np.linspace(market.mu.min() - 0.005, market.mu.max() + 0.005, 7)
    cm = ax.contour(X1, X2, Eg, levels=e_lv, colors=GREY, linewidths=0.8,
                    linestyles="--")
    ax.clabel(cm, fmt=lambda v: f"E={v:.3f}", fontsize=7, inline=True)

    # The critical line, drawn long enough to leave the picture.
    ts = np.linspace(-4, 4, 2)
    line = p0 + np.outer(ts, d)
    ax.plot(line[:, 0], line[:, 1], c=GOLD, lw=1.4, ls="-",
            label="critical line $l$")

    # The attainable set: the triangle of Eq. (4).
    tri = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], c="k", lw=1.6)
    ax.fill(tri[:, 0], tri[:, 1], color="k", alpha=0.04)

    ax.plot(eff[:, 0], eff[:, 1], c=RED, lw=3.2, solid_capstyle="round",
            label="efficient set", zorder=5)
    ax.plot(*eff[0], "s", c=RED, ms=6, mec="k", mew=0.6, zorder=7)
    ax.annotate("min $V$", eff[0] + np.array([0.05, -0.075]), fontsize=8, color=RED)

    # The corners of the chain, located by bisection rather than read off a grid.
    for E, _, _ in corners(market):
        if E < expected_on_plane(market, eff[0])[0] - 1e-9:
            continue                                  # below the efficient branch
        k = min_variance_at(market, E)[0][:2]
        ax.plot(*k, "o", c="k", ms=4.5, zorder=7)
    ax.plot(*x_hat, "o", c=GOLD, ms=7, mec="k", mew=0.6, zorder=6)
    ax.annotate(r"$\hat X$", x_hat + np.array([0.03, 0.03]), fontsize=11)

    for xy, nm in [((1, 0), names[0]), ((0, 1), names[1]), ((0, 0), names[2])]:
        ax.plot(*xy, "o", c="k", ms=4, zorder=6)
        ax.annotate(f"100% {nm}", np.array(xy) + np.array([0.02, -0.05]), fontsize=8)

    ax.set_xlabel(f"$X_1$  ({names[0]})")
    ax.set_ylabel(f"$X_2$  ({names[1]})")
    ax.set_title(subtitle, fontsize=9.5, color="0.35", pad=8)
    fig.suptitle(title, fontsize=12, y=0.965)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _check(market, label):
    """The paper's three geometric claims, verified rather than drawn."""
    from frontier import min_variance, weights

    print(f"\n{label}: {market.names}")
    Q, c, e, _, _ = reduce_to_plane(market)
    eig = np.linalg.eigvalsh(Q)
    shape = "ellipses" if eig.min() > 0 else "degenerate (footnote 12)"
    print(f"  Q eigenvalues {eig.round(5)}  ->  the level sets are {shape}")

    # 1. The centre of the ellipses is the minimum-variance portfolio.
    w_mv = min_variance(market)[0]
    print(f"  X-hat = {centre(market).round(6)} vs min-variance weights "
          f"{w_mv[:2].round(6)}   error {np.abs(centre(market) - w_mv[:2]).max():.2e}")

    # 2. The tangency points are collinear, which is what p. 85 asserts.
    grid = np.linspace(market.mu.min() - 0.02, market.mu.max() + 0.02, 40)
    T = weights(market, grid)[:, :2]
    centred = T - T.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    print(f"  tangency points, singular values {sv.round(6)}: "
          f"the second is {sv[1] / sv[0]:.1e} of the first, so they lie on a line")

    # 3. That line is the one the plane algebra predicts.
    p0, d = critical_line(market)
    dir_closed = T[-1] - T[0]
    dir_closed /= np.linalg.norm(dir_closed)
    cross = abs(float(d[0] * dir_closed[1] - d[1] * dir_closed[0]))
    print(f"  direction from Q^-1 e vs from the n-asset closed form: "
          f"cross product {cross:.2e}")

    # 4. Where the efficient set turns.
    print(f"  X-hat is {'inside' if (centre(market) >= 0).all() and centre(market).sum() <= 1 else 'OUTSIDE'} "
          f"the triangle")
    for E, left, right in corners(market):
        gone, back = set(right) - set(left), set(left) - set(right)
        move = (f"{market.names[next(iter(gone))]} leaves" if gone
                else f"{market.names[next(iter(back))]} enters")
        print(f"     corner at E = {E:.5f}: {move}")


def _footnote12():
    """Footnote 12 (p. 89), and the half of it that does not hold.

    The footnote says that to draw the isovariance curves as ellipses it is
    "both necessary and sufficient to assume that no two distinct portfolios
    have perfectly correlated returns".  Sufficient, yes.  Necessary, no: what
    the algebra asks for is that no two distinct portfolios have returns
    differing by a CONSTANT -- perfect correlation *and* equal variance -- and
    perfect correlation between assets of different variance leaves the level
    sets as genuine ellipses.  Sec. 9 of DERIVATIONS.md.
    """
    from markets import Market

    def case(label, sd, rho):
        sd = np.asarray(sd, float)
        corr = np.eye(3)
        corr[0, 1] = corr[1, 0] = rho
        Sigma = corr * np.outer(sd, sd)
        m = Market(["1", "2", "3"], np.array([0.10, 0.08, 0.05]), Sigma)
        Q = reduce_to_plane(m)[0]
        shape = "ellipses" if np.linalg.eigvalsh(Q).min() > 1e-12 else "DEGENERATE"
        print(f"  {label:34s} min eig Sigma {np.linalg.eigvalsh(Sigma).min():7.4f}   "
              f"det Q {np.linalg.det(Q):7.4f}   {shape}")

    print()
    print("Footnote 12: two distinct portfolios with perfectly correlated returns")
    case("rho = 1, variances 1, 4", [1, 2, 1], 1.0)
    case("rho = 1, variances 1, 1", [1, 1, 1], 1.0)
    Sigma = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    w = np.array([2.0, -1.0, 0.0])
    print(f"  in the first market, the portfolio 2*(1) - (2) sums to {w.sum():.0f} "
          f"and has variance {w @ Sigma @ w:.1e}: riskless, so Sigma is singular "
          f"and Piece 1 has no closed form there, while the plane picture survives")


if __name__ == "__main__":
    inside, outside = markets.triple_inside(), markets.triple_outside()
    _check(inside, "Fig. 2 case")
    _check(outside, "Fig. 3 case")
    _footnote12()
    simplex_figure(inside, "fig2_simplex.png",
                   "Fig. 2 — the attainable set, the isomeans, the isovariances",
                   r"$\hat X$ inside the triangle: the efficient set starts there")
    simplex_figure(outside, "fig3_simplex.png",
                   "Fig. 3 — the same geometry with the centre outside",
                   r"$\hat X$ unattainable: the efficient set starts on an edge")
