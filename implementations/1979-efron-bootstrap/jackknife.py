"""The jackknife as a linearisation of the bootstrap — Efron (1979), Sec. 5.

This is the paper's central thesis turned into code.  The bootstrap
distribution of R is a function of the resampling proportions
P* = N*/n (Eq. 5.1); expanding that function in a Taylor series about the
observed sample P* = e/n and keeping the first two terms (Eq. 5.4) turns the
bootstrap into a formula.  That formula is Jaeckel's infinitesimal jackknife
(Eq. 5.11), and replacing its derivatives by finite differences (Eq. 5.12)
gives the ordinary jackknife.  So the three methods are one method, read at
three degrees of approximation:

    bootstrap        R itself, over the whole simplex
    infinitesimal    R linearised at the centre of the simplex
    ordinary         the same linearisation, with the derivatives estimated
                     by moving to the edges of the simplex

Everything here therefore depends on the statistic being differentiable in
the resampling weights, and the module is organised so that the place where
that assumption fails is visible: for the median every derivative below comes
out exactly zero, and the whole construction collapses.  The bootstrap of
median.py needs no such assumption and does not collapse.

Note what has to be supplied to any of this.  The bootstrap only ever
evaluates the statistic on actual resamples, that is on weight vectors with
entries in {0, 1/n, 2/n, ...}.  The expansion needs R at *fractional* weights,
which is extra information not contained in the statistic; the paper concedes
as much (p. 13: the interpolation "will be obvious in most specific cases, but
a general recipe is difficult to provide").  Hence the functionals below take
a weight vector rather than a sample, and there is deliberately no generic
wrapper turning a sample statistic into one.

The last part of the module is Remark J, which proposes the repair — delete
observations in groups of size g rather than one at a time — and gives no
numbers for it.  Those are computed here exactly, without enumerating or
sampling the groups.
"""

from itertools import combinations

import numpy as np
from scipy.special import gammaln

from bootstrap import resample_indices
from median import median_pmf


# --------------------------------------------------------------------------
# Step 1 — statistics as functions of the weight vector
# --------------------------------------------------------------------------
#
# Each factory below returns R(p) for a fixed data set: the value of the
# statistic when observation i carries weight p_i.  The vectors are NOT
# required to be normalised, because Eq. (5.6) extends R to all non-negative
# p by R(p) = R(p / sum p).  That homogeneity is what makes the unconstrained
# derivatives of Eq. (5.5) legitimate, and it is worth having in the code
# rather than in a comment: it is used by every routine that follows, and it
# is why `jackknife_replicates` can simply write down a vector of ones with a
# zero in it.

def mean_functional(x):
    """R(p) = sum p_i x_i / sum p_i, centred so that R(e/n) = 0.

    The linear case, and the one where all three methods must agree.
    """
    x = np.asarray(x, dtype=float)
    xbar = x.mean()
    return lambda p: float(p @ x / p.sum()) - xbar


def median_functional(x):
    """R(p) = weighted median of x, centred so that R(e/n) = 0.

    The weighted median is the smallest x_(k) whose cumulative weight reaches
    one half, averaging the two middle values when the weight lands exactly
    there — so that with p = e it is np.median, for n even and odd alike.

    Observations of zero weight are dropped before that reading and not
    merely given no mass, which matters exactly once but decisively: at a
    tie the value averaged in must be the next one *present*.  Deleting the
    median itself is that case, and it is the one leave-one-out replicate
    that distinguishes the median from its neighbours.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    centre = float(np.median(x))

    def R(p):
        w = np.asarray(p, dtype=float)[order]
        present = w > 0
        xw, ww = xs[present], w[present]
        c = np.cumsum(ww) / ww.sum()
        k = int(np.searchsorted(c, 0.5))
        if k + 1 < xw.size and abs(c[k] - 0.5) < 1e-12:
            return 0.5 * (xw[k] + xw[k + 1]) - centre
        return float(xw[k]) - centre

    return R


def ratio_functional(y, z):
    """R(p) = (weighted mean of y / weighted mean of z) / (y-bar / z-bar).

    The worked example of Eqs. (5.14)-(5.15): theta(F) = E_F Y / E_F Z
    estimated by t(X) = y-bar / z-bar, with R = t(X) / theta(F) a RATIO rather
    than a difference, so that R(e/n) = 1.  It is the only case in the paper
    where U and V are written down in closed form, which makes it the natural
    test of the differentiation below.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    scale = y.mean() / z.mean()
    return lambda p: float((p @ y) / (p @ z)) / scale


def ratio_derivatives_exact(y, z):
    """U and V for the ratio estimator, Eq. (5.14)."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    u, v = y / y.mean(), z / z.mean()
    U = u - v
    V = 2 * np.outer(v, v) - (np.outer(u, v) + np.outer(v, u))
    return U, V


def ratio_moments_exact(y, z):
    """(E_* R*, Var_* R*) by the delta method, Eq. (5.15)."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = y.shape[0]
    u, v = y / y.mean() - 1.0, z / z.mean() - 1.0
    mean = 1.0 - (np.sum(u * v) - np.sum(v ** 2)) / n ** 2
    var = np.sum((y / y.mean() - z / z.mean()) ** 2) / n ** 2
    return float(mean), float(var)


# --------------------------------------------------------------------------
# Step 2 — Method 3, the Taylor expansion (Eqs. 5.4-5.11)
# --------------------------------------------------------------------------
#
# U_i and V_ij are the first and second derivatives of R at p = e/n
# (Eq. 5.5), computed here by central differences on the homogeneous
# extension.  The step is taken relative to the weights themselves, h/n
# rather than h, since the point of expansion has all coordinates equal to
# 1/n and an absolute step would mean different things at different n.
#
# Cost is O(n^2) evaluations of R, which is the price of the second
# derivatives; U alone costs 2n.

def simplex_derivatives(R, n, step=1e-3, second=True):
    """(U, V) of Eq. (5.5) by finite differences at P* = e/n.

    With `second=False` only U is computed and V is returned as None, which
    is all Eq. (5.10) needs.
    """
    p0 = np.full(n, 1.0 / n)
    eps = step / n

    def bump(*moves):
        p = p0.copy()
        for i, s in moves:
            p[i] += s * eps
        return R(p)

    R0 = R(p0)
    Rp = np.array([bump((i, +1)) for i in range(n)])
    Rm = np.array([bump((i, -1)) for i in range(n)])
    U = (Rp - Rm) / (2 * eps)
    if not second:
        return U, None

    V = np.empty((n, n))
    np.fill_diagonal(V, (Rp - 2 * R0 + Rm) / eps ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            V[i, j] = V[j, i] = (
                bump((i, +1), (j, +1)) - bump((i, +1), (j, -1))
                - bump((i, -1), (j, +1)) + bump((i, -1), (j, -1))
            ) / (4 * eps ** 2)
    return U, V


def infinitesimal_var(U, n):
    """Var_* R(P*) = sum U_i^2 / n^2 — Eqs. (5.10), (5.11).

    The multinomial covariance of Eq. (5.2) is Cov_* P* = I/n^2 - e'e/n^3, so
    the variance of the linear term is U'[I/n^2 - e'e/n^3]U, and the second
    piece drops out because eU = 0 by homogeneity (Eq. 5.7).  That identity is
    checked numerically in __main__ rather than assumed.
    """
    return float(np.sum(np.asarray(U) ** 2) / n ** 2)


def infinitesimal_bias(V, n):
    """E_* R(P*) - R(e/n) = V-bar / 2n — Eqs. (5.8), (5.9), (5.11)."""
    return float(np.trace(V) / n) / (2 * n)


# --------------------------------------------------------------------------
# Step 3 — the ordinary jackknife (Eq. 5.12)
# --------------------------------------------------------------------------
#
# The ordinary jackknife is the same expansion with the derivatives replaced
# by finite differences taken all the way out to the edge of the simplex:
# instead of nudging the weight of x_i, delete it outright.  Eq. (5.6) is what
# lets us write that deletion as the weight vector e_(i), a vector of ones
# with a zero in position i -- no normalisation needed, and no separate notion
# of "the sample with x_i removed".

def jackknife_replicates(R, n):
    """R*_(i) = R(e_(i)/(n-1)) for i = 1, ..., n — Eq. (5.12)."""
    e = np.ones(n)
    return np.array([R(e - np.eye(1, n, i)[0]) for i in range(n)])


def jackknife_U(R, n):
    """U-tilde_i = (n-1)(R*_. - R*_(i)) — Eq. (5.12).

    The paper's own gloss for R = theta(F-hat) - theta(F) prints
    U-tilde_i = (n-1)(theta-hat - theta-hat_(i)), with theta-hat where
    Eq. (5.12) has the average theta-hat_(.).  It has to be the average: the
    U-tilde_i must sum to zero, mirroring eU = 0 of Eq. (5.7), and it is that
    centring which turns sum U-tilde_i^2 / n(n-1) into Tukey's variance
    formula.  Both are checked in __main__.
    """
    rep = jackknife_replicates(R, n)
    return (n - 1) * (rep.mean() - rep)


def jackknife_var(R, n):
    """Ordinary jackknife variance, sum U-tilde_i^2 / [n(n-1)].

    Identical to Tukey's (n-1)/n * sum (theta-hat_(i) - theta-hat_(.))^2, and
    equal to Eq. (5.10) up to a factor 1 + O(1/n) whenever the expansion is
    legitimate (p. 14).
    """
    return float(np.sum(jackknife_U(R, n) ** 2) / (n * (n - 1)))


def jackknife_bias(R, n):
    """Ordinary jackknife bias estimate, (n-1)(R*_. - R(e/n))."""
    rep = jackknife_replicates(R, n)
    return float((n - 1) * (rep.mean() - R(np.full(n, 1.0 / n))))


# --------------------------------------------------------------------------
# Step 4 — the median, where the linearisation has nothing to linearise
# --------------------------------------------------------------------------
#
# The parity of n matters here, and it is the whole of Sec. 7.6 of
# DERIVATIONS.md.  Both cases are implemented, and the reason is worth stating
# plainly: an implementation that accepts only odd n cannot discover that the
# answer depends on the parity, and this one did not until the even case was
# put in beside it.
#
# ODD, n = 2m - 1.  Deleting one observation leaves an even sample, whose
# median is the average of its two middle values.  Which two depends only on
# whether the deleted point sat below the median, above it, or was the median
# itself, so with a = x_(m) - x_(m-1) and b = x_(m+1) - x_(m) the n
# leave-one-out medians take THREE values relative to x_(m):
#
#     b/2        (m - 1 times)   deleting any point below the median
#     -a/2       (m - 1 times)   deleting any point above it
#     (b - a)/2  (once)          deleting the median
#
# EVEN, n = 2m.  Deleting leaves an odd sample and a single order statistic,
# and the middle case disappears: the n leave-one-out medians take TWO values,
# x_(m+1) for each of the m deletions at or below x_(m) and x_(m) for each of
# the m above.  One spacing enters instead of two, and that single difference
# changes the limiting law.
#
# Either way the estimate is a function of two or three order statistics out
# of n, however large n is, which is the disease itself.

def median_jackknife_var(x):
    """Ordinary jackknife variance of the sample median, in closed form.

    For n = 2m - 1 odd, summing the three values above and their squares,

        v = (m-1)^2 / [2(2m-1)] * [ a^2 + b^2 - (m-1)(a-b)^2 / (2m-1) ];

    for n = 2m even, with d = x_(m+1) - x_(m), the two-valued case collapses to

        v = (n-1) d^2 / 4.

    Both are checked against the brute-force jackknife in __main__.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.shape[0]
    m = n // 2
    if n % 2 == 0:
        return float((n - 1) * (x[m] - x[m - 1]) ** 2 / 4)
    m = (n + 1) // 2
    a = x[m - 1] - x[m - 2]
    b = x[m] - x[m - 1]
    bracket = a ** 2 + b ** 2 - (m - 1) * (a - b) ** 2 / (2 * m - 1)
    return float((m - 1) ** 2 / (2 * (2 * m - 1)) * bracket)


def median_jackknife_var_batch(samples):
    """median_jackknife_var applied row-wise to a (trials, n) array.

    Only two or three middle order statistics are needed, so the rows are
    partitioned rather than sorted.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[1]
    if n % 2 == 0:
        m = n // 2
        part = np.partition(samples, [m - 1, m], axis=1)
        return (n - 1) * (part[:, m] - part[:, m - 1]) ** 2 / 4
    m = (n + 1) // 2
    part = np.partition(samples, [m - 2, m - 1, m], axis=1)
    a = part[:, m - 1] - part[:, m - 2]
    b = part[:, m] - part[:, m - 1]
    bracket = a ** 2 + b ** 2 - (m - 1) * (a - b) ** 2 / (2 * m - 1)
    return (m - 1) ** 2 / (2 * (2 * m - 1)) * bracket


def delete_d_median_pmf(n, d):
    """Prob{the retained median is x_(j)} when a random d-subset is deleted.

    Remark J diagnoses the failure above as an overdependence on P* within
    1/n of e/n, and proposes the cure: delete observations in groups of size
    g, with g large enough — "the calculations above suggest g = O(n^{1/2})" —
    that the deleted samples sit as far from the observed one as bootstrap
    resamples do.

    The same trick that made the bootstrap median tractable works again.
    Keeping r = n - d observations at random, the retained median is the
    k = (r+1)/2-th kept one, and it sits at original position j exactly when
    j is kept, k-1 of the j-1 positions below it are kept and r-k of the n-j
    above it are:

        Prob{J = j} = C(j-1, k-1) C(n-j, r-k) / C(n, r),

    a negative hypergeometric.  So the whole delete-d distribution is one
    vector of length n depending on (n, d) alone — the data again enter only
    through the support — and no groups need to be enumerated or sampled.

    r is required odd, so that the retained median is a single order
    statistic; with n odd this means d even.
    """
    r = n - d
    if r % 2 == 0:
        raise ValueError("n - d must be odd, so that the retained median is x_(k)")
    k = (r + 1) // 2
    j = np.arange(1, n + 1)
    below, above = j - 1, n - j
    ok = (below >= k - 1) & (above >= r - k)

    def log_comb(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(np.maximum(a - b, 0) + 1)

    logp = np.full(n, -np.inf)
    logp[ok] = (log_comb(below[ok], k - 1) + log_comb(above[ok], r - k)
                - log_comb(n, r))
    return np.exp(logp)


def delete_d_median_var(x, d, pmf=None):
    """Delete-d jackknife variance of the median — Shao and Wu's estimator,

        v = (n-d) / [d * C(n,d)] * sum_S (theta-hat_S - theta-hat_.)^2,

    which at d = 1 is Tukey's formula.  The sum over the C(n,d) subsets is the
    exact variance of theta-hat_S, so `delete_d_median_pmf` evaluates it in
    full: this is enumeration, not sampling.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.shape[0]
    p = delete_d_median_pmf(n, d) if pmf is None else pmf
    var = (x ** 2) @ p - (x @ p) ** 2
    return float((n - d) / d * var)


def delete_d_median_var_batch(samples, d):
    """delete_d_median_var applied row-wise to a (trials, n) array."""
    samples = np.sort(np.asarray(samples, dtype=float), axis=1)
    n = samples.shape[1]
    p = delete_d_median_pmf(n, d)
    return (n - d) / d * ((samples ** 2) @ p - (samples @ p) ** 2)


def median_bootstrap_var_batch(samples):
    """Var_* R* for the sample median, row-wise, from Eq. (3.5).

    The exact bootstrap variance, with no Monte Carlo: median.py's pmf
    depends on n alone, so one dot product per sample does it.
    """
    samples = np.sort(np.asarray(samples, dtype=float), axis=1)
    n = samples.shape[1]
    centred = samples - samples[:, [(n - 1) // 2]]
    p = median_pmf(n)
    return (centred ** 2) @ p - ((centred @ p) ** 2)


if __name__ == "__main__":
    # The demonstrations that produce a table quoted in the README seed their
    # own generator, so that inserting a block above them does not silently
    # change numbers written down elsewhere.  It has happened twice.
    rng = np.random.default_rng(0)

    # --- The differentiator, against Eq. (5.14) ----------------------------
    #
    # Before trusting any number produced by Method 3, the finite differences
    # have to be checked against something exact.  The ratio estimator is the
    # only place the paper writes U and V down, and it also lets us check the
    # homogeneity identities of Eq. (5.7) -- eU = 0, eV = -nU', eVe' = 0 --
    # which hold for reasons having nothing to do with the data, so any
    # violation is the differentiation's fault and not the statistic's.

    n = 8
    y, z = rng.uniform(1, 3, n), rng.uniform(1, 3, n)
    R_ratio = ratio_functional(y, z)
    U, V = simplex_derivatives(R_ratio, n)
    U_exact, V_exact = ratio_derivatives_exact(y, z)

    print("Method 3 for the ratio estimator, n = 8 (Eqs. 5.5, 5.7, 5.14):\n")
    print(f"  max |U  - Eq. (5.14)|   = {np.abs(U - U_exact).max():.2e}")
    print(f"  max |V  - Eq. (5.14)|   = {np.abs(V - V_exact).max():.2e}")
    print(f"  |eU|        (should be 0)      = {abs(U.sum()):.2e}")
    print(f"  max |eV + nU'|  (Eq. 5.7)      = {np.abs(V.sum(0) + n * U).max():.2e}")
    print(f"  |eVe'|      (should be 0)      = {abs(V.sum()):.2e}")

    # --- Eq. (5.15) against the bootstrap it approximates -------------------
    #
    # The delta method replaces the bootstrap by a formula.  The formula is
    # not the bootstrap, and the gap is what the whole section is about: it
    # closes like 1/n for a smooth statistic, and never for the median.

    print("\nEq. (5.15) against Monte Carlo (N = 200000 resamples):\n")
    print(f"{'n':>5}  {'E_* R* (5.8)':>13}  {'E_* R* (5.15)':>14}  {'E_* R* (MC)':>12}"
          f"  {'Var_* (5.15)':>13}  {'Var_* (MC)':>11}")
    for n in [10, 40, 160]:
        y, z = rng.uniform(1, 3, n), rng.uniform(1, 3, n)
        mean_delta, var_delta = ratio_moments_exact(y, z)
        U, V = simplex_derivatives(ratio_functional(y, z), n)
        # R(e/n) = 1 here, so Eq. (5.8) reads E_* R* = 1 + V-bar/2n, and it
        # must land on Eq. (5.15): the latter is the former with Eq. (5.14)
        # substituted in.  Likewise Eq. (5.10) is the variance of Eq. (5.15).
        idx = resample_indices(n, 200_000, rng)
        stars = (y[idx].mean(axis=1) / z[idx].mean(axis=1)) / (y.mean() / z.mean())
        print(f"{n:5d}  {1 + infinitesimal_bias(V, n):13.8f}  {mean_delta:14.8f}"
              f"  {stars.mean():12.8f}  {var_delta:13.8f}  {stars.var():11.8f}")
    print("  the first two columns are Eq. (5.8) computed numerically and")
    print("  algebraically; the third is what they approximate")

    # --- The mean: the three methods are one method ------------------------
    #
    # For a linear statistic the expansion is exact -- V = 0, no remainder --
    # so Method 3 must return the bootstrap variance sigma-hat^2/n to machine
    # precision.  The ordinary jackknife lands a factor n/(n-1) away, which is
    # the "1 + O(1/n)" of p. 14 in its simplest possible instance: the finite
    # difference is taken over a step of size 1/(n-1), not an infinitesimal
    # one, and for a quadratic form that discrepancy never vanishes at fixed n.

    print("\nThe mean, where the expansion is exact:\n")
    print(f"{'n':>5}  {'bootstrap':>12}  {'infinitesimal':>14}  {'|diff|':>9}"
          f"  {'ordinary':>11}  {'ratio to boot':>14}  {'n/(n-1)':>9}")
    for n in [5, 10, 20, 50]:
        x = rng.normal(size=n)
        R = mean_functional(x)
        v_boot = float(np.mean((x - x.mean()) ** 2) / n)
        U, _ = simplex_derivatives(R, n, second=False)
        v_ij = infinitesimal_var(U, n)
        v_j = jackknife_var(R, n)
        print(f"{n:5d}  {v_boot:12.9f}  {v_ij:14.9f}  {abs(v_ij - v_boot):9.1e}"
              f"  {v_j:11.8f}  {v_j / v_boot:14.6f}  {n / (n - 1):9.6f}")

    # --- The variance: the expansion is exact there too, and says so -------
    #
    # R* is quadratic in P* for the sample variance, so Eq. (5.4) terminates
    # with no remainder and the SECOND-order term must reproduce the bias
    # computed by hand in bootstrap.py: V-bar/2n = -sigma-hat^2/n, that is
    # V-bar = -2 sigma-hat^2.  The first-order term must likewise reproduce
    # the leading part of Var_* R*, which is (mu-hat_4 - sigma-hat^4)/n.
    # Nothing here is fitted: both are consequences of Eqs. (5.8) and (5.10)
    # meeting a case already solved.

    print("\nThe variance, where the expansion is exact for a different reason:\n")
    print(f"{'n':>5}  {'V-bar':>12}  {'-2 sigma^2':>12}  {'sum U^2/n^2':>13}"
          f"  {'(mu_4 - s^4)/n':>15}")
    for n in [5, 8, 12]:
        x = rng.normal(size=n)
        d = x - x.mean()
        s2, m4 = float(np.mean(d ** 2)), float(np.mean(d ** 4))
        R = lambda p: float((p @ (x ** 2)) / p.sum() - ((p @ x) / p.sum()) ** 2) - s2
        U, V = simplex_derivatives(R, n)
        print(f"{n:5d}  {np.trace(V) / n:12.8f}  {-2 * s2:12.8f}"
              f"  {infinitesimal_var(U, n):13.9f}  {(m4 - s2 ** 2) / n:15.9f}")
    print("  the bias of Sec. 5 and the second-order term of Eq. (5.8) are the")
    print("  same number reached from two directions")

    # --- Eq. (5.13): the two jackknives differ by O(1/n) --------------------
    #
    # U-tilde_i = (n-2)/(n-1) U_i - (V_ii - V-bar)/[2(n-1)] + smaller, so the
    # ordinary jackknife is the infinitesimal one seen through a finite step.
    # Multiplying the discrepancy by n shows it is that order and no worse;
    # so does the variance ratio, which sits at 1 + O(1/n).

    print("\nEq. (5.13) for the ratio estimator:\n")
    print(f"{'n':>5}  {'n * max|U-tilde - (5.13)|':>26}  {'|sum U-tilde|':>14}"
          f"  {'n(v_ord/v_inf - 1)':>19}")
    y_all, z_all = rng.uniform(1, 3, 160), rng.uniform(1, 3, 160)
    for n in [10, 20, 40, 80, 160]:
        # Nested samples, so that what moves down the column is n and not the data.
        y, z = y_all[:n], z_all[:n]
        R = ratio_functional(y, z)
        U, V = simplex_derivatives(R, n)
        Ut = jackknife_U(R, n)
        pred = (n - 2) / (n - 1) * U - (np.diag(V) - np.trace(V) / n) / (2 * (n - 1))
        v_ij = infinitesimal_var(U, n)
        v_j = float(np.sum(Ut ** 2) / (n * (n - 1)))
        print(f"{n:5d}  {n * np.abs(Ut - pred).max():26.2e}  {abs(Ut.sum()):14.1e}"
              f"  {n * (v_j / v_ij - 1):19.4f}")

    # --- The median, step one: every derivative is zero --------------------
    #
    # F-hat is discrete, so the weighted median is a step function of the
    # weights: it sits at x_(m) until some weight has moved by about 1/2n, and
    # then jumps.  Its derivative at e/n is therefore not merely hard to
    # estimate, it is exactly 0, and so is every second derivative.  Method 3
    # reports that the median has no sampling variability at all.
    #
    # This is not a numerical accident of the step size -- the function is
    # locally constant, so every step small enough gives the same 0.

    print("\nThe median, Method 3 (n = 13):\n")
    x13 = rng.normal(size=13)
    R_med = median_functional(x13)
    for step in [1e-2, 1e-3, 1e-6]:
        U, V = simplex_derivatives(R_med, 13, step=step)
        print(f"  step = {step:6.0e}:  max|U| = {np.abs(U).max():.1e}"
              f"   max|V| = {np.abs(V).max():.1e}"
              f"   infinitesimal variance = {infinitesimal_var(U, 13):.1e}")
    print("  the estimate is exactly zero, for every sample and every n")

    # --- The median, step two: the ordinary jackknife sees three points ----

    print("\nThe median, ordinary jackknife (Eq. 5.12) vs the closed form:\n")
    print(f"{'n':>5}  {'leave-one-out':>15}  {'via Eq. (5.12)':>15}"
          f"  {'closed form':>15}  {'distinct U-tilde':>17}")
    for n in [5, 8, 9, 12, 13, 25]:
        x = rng.normal(size=n)
        R = median_functional(x)
        # Tukey's formula on the n leave-one-out medians, written out by hand:
        # what Eq. (5.12) says in weights, said in deletions.
        loo = np.array([np.median(np.delete(x, i)) for i in range(n)])
        v_loo = (n - 1) / n * np.sum((loo - loo.mean()) ** 2)
        distinct = np.unique(np.round(jackknife_U(R, n), 12)).size
        print(f"{n:5d}  {v_loo:15.10f}  {jackknife_var(R, n):15.10f}"
              f"  {median_jackknife_var(x):15.10f}  {distinct:17d}")
    print("  three distinct pseudo-values at odd n and two at even n, whatever")
    print("  n is: the jackknife variance of the median is a function of the")
    print("  two or three central order statistics and of nothing else, and")
    print("  that count is what decides the limit law below")

    # --- The median, step three: it does not converge ----------------------
    #
    # n * Var-hat should settle at 1/4f^2(theta) = pi/2 for F = N(0,1).  The
    # bootstrap column does.  The jackknife column cannot: it is built from
    # the spacings around the median, and n times a spacing converges in
    # DISTRIBUTION to an exponential instead of concentrating.  How many
    # spacings enter depends on the parity of n, and so does the limit --
    # see the parity table further down.

    limit = np.pi / 2                       # 1/(4 f^2(0)) for the standard normal
    rng = np.random.default_rng(101)
    print(f"\nThe median, consistency (F = N(0,1), n * Var -> {limit:.4f}):\n")
    print(f"{'n':>6}  {'truth':>8}  {'jackknife: mean':>16}  {'s.d.':>7}"
          f"  {'bootstrap: mean':>16}  {'s.d.':>7}")
    n_trials = 20_000
    for n in [13, 51, 201, 1001]:
        v_j, v_b, truth = [], [], []
        for _ in range(n_trials // 2000):
            s = rng.normal(size=(2000, n))
            v_j.append(median_jackknife_var_batch(s))
            v_b.append(median_bootstrap_var_batch(s))
            truth.append(np.median(s, axis=1))
        v_j = n * np.concatenate(v_j)
        v_b = n * np.concatenate(v_b)
        truth = n * float(np.var(np.concatenate(truth)))
        print(f"{n:6d}  {truth:8.4f}  {v_j.mean():16.4f}  {v_j.std():7.4f}"
              f"  {v_b.mean():16.4f}  {v_b.std():7.4f}")
    print("  the bootstrap's spread shrinks with n and the jackknife's does not")

    # --- the limit law, and the parity it depends on ------------------------
    #
    # Sec. 3 states the limit as (1/4f^2)[chi^2_2/2]^2, of mean 2 and variance
    # 20, in a section that has assumed n = 2m - 1 odd.  That law is the EVEN
    # one.  With n even the leave-one-out medians take two values and a single
    # spacing enters, giving [chi^2_2/2]^2 exactly as printed; with n odd they
    # take three and two spacings enter, giving [chi^2_4/4]^2, of mean 1.5 and
    # variance 5.25.  Both are simulated below against both candidates, which
    # is the only honest way to present it: neither law is wrong, and which
    # one applies is decided by a parity the statement does not mention.

    rng = np.random.default_rng(102)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    laws = [("[chi^2_2/2]^2  (as printed)", (rng.chisquare(2, 400_000) / 2) ** 2),
            ("[chi^2_4/4]^2", (rng.chisquare(4, 400_000) / 4) ** 2)]

    print("\nThe limiting law of n * v_jack / (1/4f^2), by the parity of n:\n")
    print(f"{'':>28}  {'mean':>7}  {'var':>7}  "
          + " ".join(f"{f'q{q:.2f}':>6}" for q in qs))
    rows = []
    for n in [4000, 4001]:
        ratios = []
        for _ in range(40_000 // 2000):
            s = rng.normal(size=(2000, n))
            ratios.append(n * median_jackknife_var_batch(s) / limit)
        rows.append((f"simulated, n = {n} ({'even' if n % 2 == 0 else 'odd'})",
                     np.concatenate(ratios)))
    for label, v in rows + laws:
        quant = " ".join(f"{q:6.3f}" for q in np.quantile(v, qs))
        print(f"{label:>28}  {v.mean():7.3f}  {v.var():7.3f}  {quant}")
    print("\n  each simulated row follows one of the two laws quantile by")
    print("  quantile and neither is a rounding of the other.  The printed law")
    print("  is the even one; Sec. 3 works with n odd, where a second spacing")
    print("  enters and the estimate is less wild than advertised -- biased by")
    print("  50% rather than 100%.  The inconsistency, which is the point, is")
    print("  the same in both cases: the limit is random, not a number")

    # --- Remark J: deleting in groups repairs it ---------------------------
    #
    # The diagnosis is that the jackknife only ever looks at P* within 1/n of
    # e/n (Eq. 8.14), where the median is locally constant, while the
    # bootstrap looks at distance n^(-1/2).  Deleting d at a time moves the
    # deleted samples out to distance d/n, so d of order sqrt(n) puts the
    # jackknife back on the bootstrap's own scale.  Efron asserts that this
    # "gives the correct asymptotic variance for the median" and shows no
    # numbers; these are the numbers.
    #
    # First the estimator itself, against two independent routes: the
    # ordinary jackknife (which is the case d = 1) and a literal enumeration
    # of the C(n, d) deletions.

    x8 = rng.normal(size=8)
    R8 = median_functional(x8)
    groups = list(combinations(range(9), 2))
    x9 = rng.normal(size=9)
    thetas = np.array([np.median(np.delete(x9, list(g))) for g in groups])
    v_enum = (9 - 2) / 2 * thetas.var()

    print("\nThe delete-d estimator, checked two ways:\n")
    print(f"  n = 8, d = 1:  ordinary jackknife  {jackknife_var(R8, 8):.12f}")
    print(f"                 delete-d formula    {delete_d_median_var(x8, 1):.12f}")
    print(f"  n = 9, d = 2:  all C(9,2) = {len(groups)} deletions   {v_enum:.12f}")
    print(f"                 delete-d formula            "
          f"{delete_d_median_var(x9, 2):.12f}")

    # d must be even for n odd, so that n - d observations are retained and
    # their median is a single order statistic.
    def even(v):
        return max(2, 2 * int(round(v / 2)))

    rng = np.random.default_rng(103)
    print(f"\nRemark J: does delete-d converge? (limit {limit:.4f}, 2000 trials)\n")
    print(f"{'n':>6}  " + "  ".join(f"{lbl:<20}" for lbl in
          ["d = 2", "d ~ sqrt(n)", "d ~ n^(3/5)", "bootstrap"]).rstrip())
    for n in [101, 401, 1601, 6401]:
        ds = [2, even(np.sqrt(n)), even(n ** 0.6)]
        chunks, boot = [[] for _ in ds], []
        for _ in range(4):
            s = rng.normal(size=(500, n))
            for c, d in zip(chunks, ds):
                c.append(delete_d_median_var_batch(s, d))
            boot.append(median_bootstrap_var_batch(s))
        cells = []
        for c, d in zip(chunks, ds):
            v = n * np.concatenate(c)
            cells.append(f"d={d:<4}{v.mean():5.3f} ({v.std():5.3f})")
            if d == ds[1]:
                last_sqrt = (v.mean(), v.std())
        v = n * np.concatenate(boot)
        last_boot = (v.mean(), v.std())
        cells.append(f"     {v.mean():5.3f} ({v.std():5.3f})")
        print(f"{n:6d}  " + "  ".join(cells))
    print("\n  mean (s.d. across trials).  Grouping is not by itself the cure:")
    print("  at d = 2 the estimator is as lost as at d = 1, because two")
    print("  deletions still move P* by O(1/n).  What repairs it is letting d")
    print("  GROW -- both growing rules walk towards the limit with a spread")
    print("  that shrinks, the faster rule faster -- which is Remark J's claim,")
    print(f"  though the constants are poor: at n = {n} the sqrt(n) rule is")
    print(f"  still {last_sqrt[0] / limit - 1:.0%} high and"
          f" {last_sqrt[1] / last_boot[1]:.1f} times as variable as the bootstrap,")
    print("  which needed no repair and no choice of d")
