"""Confidence statements, and the pivot that is not one — Efron (1979), Remark D.

The paper does not develop confidence intervals, and Remark D is the reason
why: it exhibits, in six lines, a bootstrap interval statement that agrees
strikingly with an exact one and is nevertheless wrong.  Everything the
bootstrap literature spent the next decade on — the percentile method (1981),
the bias-corrected and accelerated interval (1987) — starts here.

The example is the median of n = 13.  Two probability statements:

    (8.3)  Prob_F{x_(4) < theta < x_(10)} = Prob{4 <= Bi(13, 1/2) <= 9} = .908

           exact, and DISTRIBUTION-FREE: it holds for every continuous F,
           because the only random thing in it is how many observations fall
           below theta, and that is Binomial(n, 1/2) whatever F is.

    (8.4)  Prob_*{x_(4) < theta-hat* < x_(10)} = .914

           the bootstrap's own statement, read off the distribution of the
           resampled median from Eq. (3.6).

Six thousandths apart, which "looks striking".  The trap is in what happens
next.  Since theta-hat = x_(7), statement (8.4) can be rewritten as a
statement about theta-hat* - theta-hat; treating that difference as a PIVOTAL
quantity — one whose distribution does not depend on the unknown F — turns it
into a statement about theta-hat - theta, and inverting THAT gives

    (8.6)  Prob_F{2x_(7) - x_(10) < theta < 2x_(7) - x_(4)} = .914,

the reflection of (8.3) about the sample median.  Efron ends the derivation
with an exclamation mark and observes that the fault is not the bootstrap's:
it is the inferential step, which assumes a pivot where there is none.

What the paper does not do is say how wrong (8.6) is, and the answer below is
not "slightly".  It is also not confined to skewed F, which is the natural
guess: the reflection is about x_(7), a random point, so it misfires under
symmetric distributions too.
"""

import numpy as np
from scipy.stats import binom

from median import median_pmf


def order_statistic_coverage(n, r, s):
    """Prob_F{x_(r) < theta < x_(s)} for the median — Eq. (8.3).

    x_(r) < theta fails exactly when fewer than r observations fall below
    theta, and theta < x_(s) fails when at least s do.  The count is
    Binomial(n, 1/2) for every continuous F, so the coverage is
    Prob{r <= Bi(n, 1/2) <= s - 1} and depends on nothing else.  This is the
    strongest kind of statement in the whole paper: exact, and free of F.
    """
    return float(binom.cdf(s - 1, n, 0.5) - binom.cdf(r - 1, n, 0.5))


def bootstrap_median_coverage(n, r, s, continuity=True):
    """Prob_*{x_(r) < theta-hat* < x_(s)} from Eq. (3.5) — Eq. (8.4).

    The bootstrap distribution of the resampled median is discrete and sits
    ON the order statistics, so the endpoints are atoms and "inside" is
    ambiguous.  Efron's continuity correction counts half of each endpoint
    atom, which is what makes (8.4) come out at .914; counting neither gives
    .859, so the convention is carrying real weight and not tidying a
    rounding.
    """
    p = median_pmf(n)
    inner = float(p[r:s - 1].sum())
    if continuity:
        inner += 0.5 * float(p[r - 1] + p[s - 1])
    return inner


def percentile_endpoints(n, level):
    """The order statistics bounding the central `level` of Prob_*{theta-hat*}.

    Not in the paper — this is the percentile method of Efron (1981) — and it
    is here for one line of arithmetic: for the median it returns exactly the
    interval of Eq. (8.3), which is why the comparison below is worth making.
    """
    p = median_pmf(n)
    tail = (1.0 - level) / 2
    c = np.cumsum(p)
    r = int(np.searchsorted(c, tail)) + 1
    s = int(np.searchsorted(c, 1 - tail)) + 1
    return r, s


if __name__ == "__main__":
    n, r, s = 13, 4, 10
    rng = np.random.default_rng(0)

    exact = order_statistic_coverage(n, r, s)
    boot = bootstrap_median_coverage(n, r, s)

    plain = bootstrap_median_coverage(n, r, s, continuity=False)
    print(f"Remark D, the median of n = {n}:\n")
    for label, value, printed in [
            (f"(8.3)  Prob_F{{x_({r}) < theta < x_({s})}}", exact, ".908"),
            (f"(8.4)  Prob_*{{x_({r}) < theta-hat* < x_({s})}}", boot, ".914"),
            ("       without the continuity correction", plain, "")]:
        note = f"     paper: {printed}" if printed else ""
        print(f"  {label:<43} = {value:.5f}{note}")
    print(f"  the two agree to {abs(exact - boot):.3f}, which is Efron's 'striking'")

    # --- What (8.6) actually delivers --------------------------------------
    #
    # (8.3) is exact for every continuous F, so it is its own check: any
    # simulation must return .908 whatever we draw from.  (8.6) has the same
    # width by construction — it is (8.3) reflected about x_(7), and a
    # reflection preserves length — so whatever it loses, it loses purely by
    # sitting in the wrong place.

    print(f"\nCoverage of the two intervals, 400000 samples of size {n}:\n")
    print(f"{'F':>12}  {'(8.3) exact':>12}  {'(8.6) pivotal':>14}"
          f"  {'claimed':>8}  {'same width?':>12}")
    n_sim = 400_000
    for name, draw, theta in [
            ("N(0,1)", lambda size: rng.normal(size=size), 0.0),
            ("U(0,1)", lambda size: rng.uniform(size=size), 0.5),
            ("Exp(1)", lambda size: rng.exponential(size=size), np.log(2.0)),
            ("lognormal", lambda size: rng.lognormal(size=size), 1.0)]:
        x = np.sort(draw((n_sim, n)), axis=1)
        lo, hi = x[:, r - 1], x[:, s - 1]
        med = x[:, (n - 1) // 2]
        c_exact = float(np.mean((lo < theta) & (theta < hi)))
        c_pivot = float(np.mean((2 * med - hi < theta) & (theta < 2 * med - lo)))
        same = np.allclose((hi - lo), (2 * med - lo) - (2 * med - hi))
        print(f"{name:>12}  {c_exact:12.4f}  {c_pivot:14.4f}  {boot:8.3f}"
              f"  {str(same):>12}")

    print("\n  The first column is .908 for every F, as it must be.  The second")
    print("  claims .914 and delivers about .73 -- and it fails under the two")
    print("  SYMMETRIC distributions as badly as under the skewed ones, which")
    print("  the usual intuition about reflection does not predict.  The")
    print("  reason is that the reflection is about x_(7), which is random:")
    print("  the interval is the right length and in the wrong place.")

    # --- Which of the two the bootstrap actually recommends -----------------
    #
    # Post-1979, and one line: the percentile method reads the interval off
    # the bootstrap distribution of theta-hat* directly instead of using it
    # as a pivot.  For the median that returns Eq. (8.3) itself.

    pr, ps = percentile_endpoints(n, boot)
    print(f"\n  Read as percentiles instead of as a pivot (Efron 1981), the")
    print(f"  same bootstrap distribution gives (x_({pr}), x_({ps})) -- which is")
    print(f"  Eq. (8.3), the exactly correct interval.  The whole disaster is")
    print(f"  in the inferential step, exactly where Remark D puts it, and")
    print(f"  this single example is why intervals needed their own papers.")
