"""Bootstrap core — Efron (1979).

The three steps of Sec. 2: put mass 1/n on each datum to get F-hat, resample
from it (Eq. 2.4), and read off the distribution of R* = R(X*, F-hat)
(Eq. 2.5).

Two of the paper's three ways of computing that distribution live here:

    Method 1 (exact)        bootstrap_exact  — sum over every distinct
                            resample, weighted by its multinomial probability.
                            Feasible only for small n, and the reason it is
                            here at all is that it separates two errors that
                            a single Monte Carlo run mixes together.
    Method 2 (Monte Carlo)  bootstrap_mc     — draw N resamples at random.
                            This is what everyone means by "the bootstrap".

Method 3 (Taylor expansion about F-hat) is the infinitesimal jackknife of
Sec. 5 and is not here yet.

Nothing in this module knows what statistic it is applied to: the paper's
point is that R is arbitrary, so `statistic` is a callable throughout.
"""

from itertools import combinations_with_replacement
from math import comb, factorial

import numpy as np


# --------------------------------------------------------------------------
# Step 1 — F-hat and how to draw from it
# --------------------------------------------------------------------------
#
# F-hat puts mass 1/n on each observed x_i (Sec. 2).  Drawing from it is
# drawing an index uniformly from {0, ..., n-1}, so a resample is fully
# described by the indices it picks.  Everything downstream — pairs for a
# correlation, rows of a design matrix for a regression — resamples the same
# index vector, which is why the index version is the primitive one.

def resample_indices(n, n_boot, rng):
    """Indices of n_boot bootstrap samples of size n, drawn with replacement."""
    return rng.integers(0, n, size=(n_boot, n))


def resample(x, n_boot, rng):
    """X* of Eq. (2.4): n_boot resamples of size n, drawn iid from F-hat.

    Returns an array of shape (n_boot, n).
    """
    x = np.asarray(x, dtype=float)
    return x[resample_indices(x.shape[0], n_boot, rng)]


# --------------------------------------------------------------------------
# Step 2 — Method 2, the Monte Carlo bootstrap
# --------------------------------------------------------------------------

def bootstrap_mc(x, statistic, n_boot, rng, vectorized=False):
    """Bootstrap distribution of `statistic` by Monte Carlo (Sec. 2, Method 2).

    Returns the N values statistic(x*^1), ..., statistic(x*^N), unsorted and
    unsummarised: what to do with them (a variance, a mean, a histogram, a
    quantile) is the caller's business, exactly as in the paper.

    `vectorized=True` promises that statistic maps an array of shape
    (n_boot, n) to one of shape (n_boot,) — true of np.mean/np.median with
    axis=-1, and worth using when N is large.
    """
    samples = resample(x, n_boot, rng)
    if vectorized:
        return np.asarray(statistic(samples), dtype=float)
    return np.array([statistic(s) for s in samples], dtype=float)


# --------------------------------------------------------------------------
# Step 3 — Method 1, the exact bootstrap
# --------------------------------------------------------------------------
#
# Given the data, the bootstrap distribution is not random: it is a finite
# weighted sum, and Monte Carlo is only a way of approximating it.  Two facts
# make the sum computable for small n.
#
# First, a resample matters only through how many times each x_i appears, so
# the distinct resamples are the multisets of size n drawn from n items:
#
#     C(2n - 1, n)      of them,
#
# which is far fewer than the n^n ordered draws (n = 10: 92378 against 10^10).
#
# Second — and this is the trap — those multisets are NOT equiprobable.  The
# count vector (N_1, ..., N_n) is multinomial(n; 1/n, ..., 1/n), so the
# multiset appears with probability
#
#     n! / (N_1! ... N_n!) * (1/n)^n,
#
# the weight computed below.  Treating the multisets as uniform is a genuinely
# different (and wrong) resampling scheme.

def n_distinct_resamples(n):
    """C(2n-1, n): how many distinct resamples of size n exist."""
    return comb(2 * n - 1, n)


def bootstrap_exact(x, statistic):
    """Exact bootstrap distribution (Sec. 2, Method 1) by enumeration.

    Returns (values, weights): the value of the statistic on every distinct
    resample, and its multinomial probability.  The weights sum to 1, so any
    bootstrap quantity is a weighted average over these — with no Monte Carlo
    error whatsoever.

    Cost is C(2n-1, n) evaluations, so this is for n of about a dozen at most.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    values = np.empty(n_distinct_resamples(n))
    weights = np.empty_like(values)

    n_fact = factorial(n)
    for k, idx in enumerate(combinations_with_replacement(range(n), n)):
        counts = np.bincount(idx, minlength=n)
        multiplicity = n_fact
        for c in counts:
            multiplicity //= factorial(int(c))
        values[k] = statistic(x[list(idx)])
        weights[k] = multiplicity

    return values, weights / float(n) ** n


def weighted_mean(values, weights=None):
    """E_* of a bootstrap quantity, weighted (exact) or not (Monte Carlo)."""
    if weights is None:
        return float(np.mean(values))
    return float(np.sum(weights * values))


def weighted_quantile(values, q, weights=None):
    """The q-quantile of the bootstrap distribution.

    The smallest value whose cumulative probability reaches q, which is the
    ordinary sample quantile when the weights are uniform.  Quantiles matter
    more here than the name suggests: they are the only summary that survives
    a monotone transformation of the statistic (Eq. 8.1), and the only one
    that survives at all when the bootstrap distribution has no moments.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    v = values[order]
    if weights is None:
        w = np.full(v.shape, 1.0 / v.size)
    else:
        w = np.asarray(weights, dtype=float)[order]
        w = w / w.sum()
    return float(v[np.searchsorted(np.cumsum(w), q)])


def weighted_var(values, weights=None):
    """Var_* of a bootstrap quantity.

    Note the 1/N (population) convention: this is a variance *under F-hat*,
    which is a fully specified distribution, not a sample from something else.
    There is no -1 to make here.
    """
    mu = weighted_mean(values, weights)
    return weighted_mean((values - mu) ** 2, weights)


# --------------------------------------------------------------------------
# Step 4 — what the paper says the answer must be, for the mean
# --------------------------------------------------------------------------

def var_mean_formula(x):
    """Var_*(X-bar* - x-bar) worked out by hand (Sec. 2).

    Resampling makes X_1*, ..., X_n* iid under F-hat, so the variance of their
    average is Var_*(X_1*)/n, and Var_*(X_1*) is the variance OF F-HAT, that
    is sum_i (x_i - x-bar)^2 / n.  Hence sigma-hat^2 / n with a 1/n, not the
    1/(n-1) of the unbiased sample variance: F-hat is the population here, and
    nothing about it is being estimated.

    For 0/1 data this collapses to x-bar (1 - x-bar) / n, the paper's Eq. (2.8).
    """
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - x.mean()) ** 2) / x.shape[0])


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- The exact bootstrap is exactly the hand formula --------------------
    #
    # If the enumeration and the weights are right, Method 1 applied to the
    # mean must return sigma-hat^2/n to machine precision.  Nothing is being
    # estimated in this line: both sides are functions of the same fixed data.

    print("Method 1 (enumeration) against the hand formula, for the mean:\n")
    print(f"{'n':>3}  {'C(2n-1,n)':>10}  {'sum of weights':>15}"
          f"  {'exact Var_*':>13}  {'sigma^2/n':>12}  {'|diff|':>9}")
    for n in [3, 5, 8, 10]:
        x = rng.normal(size=n)
        values, weights = bootstrap_exact(x, np.mean)
        v_exact = weighted_var(values, weights)
        v_formula = var_mean_formula(x)
        print(f"{n:3d}  {n_distinct_resamples(n):10d}  {weights.sum():15.12f}"
              f"  {v_exact:13.9f}  {v_formula:12.9f}  {abs(v_exact - v_formula):9.2e}")

    # --- Eq. (2.8), the 0/1 case -------------------------------------------

    x01 = np.array([1.0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    values, weights = bootstrap_exact(x01, np.mean)
    xbar = x01.mean()
    print(f"\nEq. (2.8) with 0/1 data, x-bar = {xbar:.1f}:")
    print(f"  exact bootstrap Var_*  = {weighted_var(values, weights):.12f}")
    print(f"  x-bar(1 - x-bar)/n     = {xbar * (1 - xbar) / x01.size:.12f}")

    # --- The two errors, separated -----------------------------------------
    #
    # Monte Carlo approximates the exact sum, so its error must die as
    # N^{-1/2} and the ratio below must hover around 1.  What it converges TO
    # is the exact bootstrap value, not the truth about F: that gap is the
    # bootstrap's own error and no N ever touches it.

    x = rng.normal(size=8)
    v_exact = weighted_var(*bootstrap_exact(x, np.mean))

    print("\nMonte Carlo error against the exact value (n = 8, 200 repetitions):\n")
    print(f"{'N':>7}  {'RMS error':>11}  {'x sqrt(N)':>10}")
    for n_boot in [100, 400, 1600, 6400]:
        errs = [weighted_var(bootstrap_mc(x, np.mean, n_boot, rng, vectorized=False))
                - v_exact for _ in range(200)]
        rms = float(np.sqrt(np.mean(np.square(errs))))
        print(f"{n_boot:7d}  {rms:11.3e}  {rms * np.sqrt(n_boot):10.3e}")
    print("  the last column is flat: the Monte Carlo error is O(N^-1/2), and")
    print("  the target it shrinks towards is the exact bootstrap, not the truth")
