"""The frontier with no short selling — Markowitz (1952), Piece 3.

The paper's problem is the one WITH the sign constraint (Eq. 4, p. 81):

    min_w  w' Sigma w   subject to   mu' w = E,   1' w = 1,   w >= 0        (*)

which is what Piece 1 dropped in order to get a closed form.  Restoring it
turns a linear system into a quadratic program, and the whole of the paper's
geometry -- critical line, polygonal efficient set, connected parabolic
segments of Fig. 6 -- is a description of what that constraint does.

The solver here is a primal ACTIVE SET method, written out rather than called
from a library, for two reasons.  It is the algorithm the paper is missing (the
systematic version is Markowitz 1956, four years later, and its multipliers are
the *critical lines* of p. 85).  And it is the same method already written for
the lasso in this repository -- guess which constraints bind, solve the
resulting linear system, correct the guess -- so the two can be read against
each other.  scipy is used only to check the answer.
"""

import numpy as np

from frontier import scalars


def _equality_step(Sigma, w, free, A_rows):
    """One step of the equality-constrained subproblem, restricted to `free`.

    Minimise (1/2)(w+p)' Sigma (w+p) over the directions p that keep the two
    equalities and hold the working set at zero.  With g = Sigma w that is

        min_p  (1/2) p' Sigma p + g' p    s.t.  A p = 0,  p_j = 0 for j not free,

    whose KKT conditions are the saddle-point system

        [ Sigma_FF   A_F' ] [  p_F ]   [ -g_F ]
        [ A_F        0    ] [ -nu  ] = [   0  ]
    """
    F = np.flatnonzero(free)
    k, m = len(F), A_rows.shape[0]
    g = Sigma[np.ix_(F, np.arange(len(w)))] @ w
    A_F = A_rows[:, F]

    K = np.zeros((k + m, k + m))
    K[:k, :k] = Sigma[np.ix_(F, F)]
    K[:k, k:] = A_F.T
    K[k:, :k] = A_F
    rhs = np.concatenate([-g, np.zeros(m)])

    sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    p = np.zeros_like(w)
    p[F] = sol[:k]
    return p


def _bound_multipliers(Sigma, mu, w, free):
    """The multipliers s_j >= 0 of the constraints w_j >= 0, plus (lambda, gamma).

    Stationarity of (*) reads  Sigma w = lambda mu + gamma 1 + s,  with s_j = 0
    on the free set.  Those |F| equations fix (lambda, gamma), and s is then read
    off the rest.  A negative s_j says the constraint w_j >= 0 is pushing the
    wrong way: releasing it lowers the variance.
    """
    F = np.flatnonzero(free)
    M = np.column_stack([mu[F], np.ones(len(F))])
    lam, gam = np.linalg.lstsq(M, (Sigma @ w)[F], rcond=None)[0]
    s = Sigma @ w - lam * mu - gam * np.ones_like(w)
    s[F] = 0.0
    return s, float(lam), float(gam)


def min_variance_at(market, E, tol=1e-12, max_iter=200):
    """Solve (*) for one target return.  Returns (w, active set, lambda, gamma).

    The active set is the tuple of indices held at zero; on it, sec. 12 shows
    the solution is the *unconstrained* frontier of the sub-market of the
    remaining assets, which is what makes the answer piecewise affine in E.
    """
    mu, Sigma, n = market.mu, market.Sigma, len(market.mu)
    lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
    if not mu[lo] - 1e-12 <= E <= mu[hi] + 1e-12:
        raise ValueError(f"E = {E:.4f} is outside [{mu[lo]:.4f}, {mu[hi]:.4f}]: "
                         "with w >= 0 no portfolio attains it")

    # A feasible start: the two-asset mixture that brackets E.  Any feasible
    # point would do; this one needs no solve.
    theta = (E - mu[lo]) / (mu[hi] - mu[lo])
    w = np.zeros(n)
    w[lo], w[hi] = 1 - theta, theta
    free = w > tol
    A_rows = np.vstack([mu, np.ones(n)])

    for _ in range(max_iter):
        p = _equality_step(Sigma, w, free, A_rows)
        if np.abs(p).max() <= tol:
            s, lam, gam = _bound_multipliers(Sigma, mu, w, free)
            j = int(np.argmin(s))
            if s[j] >= -tol:                        # KKT complete: done
                return np.where(free, w, 0.0), tuple(np.flatnonzero(~free)), lam, gam
            free[j] = True                          # release the worst constraint
            continue

        # Step length: how far before a free weight hits zero.
        down = free & (p < -tol)
        ratios = np.where(down, -w / np.where(down, p, 1.0), np.inf)
        alpha = min(1.0, float(ratios.min()))
        w = w + alpha * p
        if alpha < 1.0:
            free[int(np.argmin(ratios))] = False     # a new constraint binds
            w = np.maximum(w, 0.0)

    raise RuntimeError(f"active set did not converge at E = {E}")


def solve_scipy(market, E):
    """The same problem handed to a general-purpose solver, for checking only."""
    from scipy.optimize import minimize

    mu, Sigma, n = market.mu, market.Sigma, len(market.mu)
    r = minimize(lambda w: w @ Sigma @ w, np.full(n, 1 / n), method="SLSQP",
                 jac=lambda w: 2 * Sigma @ w, bounds=[(0, None)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                              {"type": "eq", "fun": lambda w: mu @ w - E}],
                 options={"maxiter": 1000, "ftol": 1e-16})
    return r.x


def frontier(market, E_grid):
    """Sweep the target return.  Returns (W, V, active_sets, lambdas)."""
    out = [min_variance_at(market, E) for E in E_grid]
    W = np.array([w for w, _, _, _ in out])
    V = np.einsum("ij,jk,ik->i", W, market.Sigma, W)
    return W, V, [a for _, a, _, _ in out], np.array([l for _, _, l, _ in out])


def corners(market, n_scan=400, tol=1e-13):
    """Where the efficient set turns: the E at which the active set changes.

    A scan finds the intervals in which it changes and bisection locates the
    corner inside each one.  The kinks are exactly the points at which some
    asset enters or leaves the portfolio, and they are what makes the paper's
    efficient set a polygonal chain rather than a straight line (sec. 12).
    """
    mu = market.mu
    grid = np.linspace(mu.min(), mu.max(), n_scan)
    sets = [min_variance_at(market, E)[1] for E in grid]

    out = []
    for i in range(n_scan - 1):
        if sets[i] == sets[i + 1]:
            continue
        a, b, left = grid[i], grid[i + 1], sets[i]
        while b - a > tol:
            m = 0.5 * (a + b)
            if min_variance_at(market, m)[1] == left:
                a = m
            else:
                b = m
        out.append((0.5 * (a + b), sets[i], sets[i + 1]))
    return out


def segment_market(market, active):
    """The sub-market that survives on a given face.

    On a face the sign constraints in `active` are equalities, so the problem is
    Piece 1 on the remaining assets and V(E) there is that sub-market's
    parabola.  Different faces, different sub-markets, different parabolas:
    that is Fig. 6.
    """
    from markets import Market

    free = [i for i in range(len(market.mu)) if i not in active]
    return Market([market.names[i] for i in free],
                  market.mu[free], market.Sigma[np.ix_(free, free)]), free


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _check(market, label):
    from frontier import weights as closed_form

    mu = market.mu
    print(f"\n{label}: {len(mu)} assets, E in [{mu.min():.3f}, {mu.max():.3f}]")

    grid = np.linspace(mu.min() + 1e-4, mu.max() - 1e-4, 120)
    W, V, sets, lam = frontier(market, grid)

    err_scipy = max(np.abs(w - solve_scipy(market, E)).max() for w, E in zip(W, grid))
    print(f"  1' w = 1                        max error {np.abs(W.sum(1) - 1).max():.2e}")
    print(f"  mu' w = E                       max error {np.abs(W @ mu - grid).max():.2e}")
    print(f"  w >= 0                          most negative {W.min():+.2e}")
    print(f"  active set vs SLSQP             max error {err_scipy:.2e}")

    ks = corners(market)
    print(f"  corners of the polygonal chain: {len(ks)}")
    for E, left, right in ks:
        gone, back = set(right) - set(left), set(left) - set(right)
        move = (f"{market.names[next(iter(gone))]} leaves" if gone
                else f"{market.names[next(iter(back))]} enters")
        held = [market.names[i] for i in range(len(mu)) if i not in right]
        print(f"     E = {E:.5f}   {move:<22} held after: {held}")

    # On each segment the QP solution is Piece 1 applied to the surviving assets.
    edges = [mu.min()] + [E for E, _, _ in ks] + [mu.max()]
    print("  each segment is the closed-form frontier of its sub-market:")
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        w, active, _, _ = min_variance_at(market, mid)
        sub, free = segment_market(market, active)
        A, B, C, D = scalars(sub)
        err = np.abs(closed_form(sub, mid) - w[free]).max()
        print(f"     E in [{a:.4f}, {b:.4f}]   V'' = 2A/D = {2 * A / D:9.3f}   "
              f"closed form matches to {err:.1e}")

    # C^1 but not C^2: lambda continuous across a corner, curvature jumping.
    # The two are told apart by shrinking h: a continuous quantity closes the
    # gap in proportion, a discontinuous one does not move at all.
    if ks:
        E0 = ks[0][0]
        print("  across the first corner, the slope is continuous:")
        for h in (1e-4, 1e-6, 1e-8):
            lo, hi = min_variance_at(market, E0 - h)[2], min_variance_at(market, E0 + h)[2]
            print(f"     h = {h:.0e}:  lambda {lo:.8f} -> {hi:.8f}, "
                  f"difference {abs(hi - lo):.1e}")
        left = segment_market(market, min_variance_at(market, E0 - 1e-8)[1])[0]
        right = segment_market(market, min_variance_at(market, E0 + 1e-8)[1])[0]
        A_l, _, _, D_l = scalars(left)
        A_r, _, _, D_r = scalars(right)
        print(f"  and the curvature is not: {2 * A_l / D_l:.3f} -> {2 * A_r / D_r:.3f}, "
              f"a gap that no h makes smaller")


if __name__ == "__main__":
    import markets

    _check(markets.sectors(), "sectors()")
    _check(markets.triple_inside(), "triple_inside()")
    _check(markets.triple_outside(), "triple_outside()")
