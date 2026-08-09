"""The median — Efron (1979), Sec. 3.

The flagship case of the paper: the statistic where the jackknife breaks and
the bootstrap does not.  It is also the case where the exact bootstrap
distribution can be written in closed form, which makes it the best possible
test of the Monte Carlo machinery in bootstrap.py.

Why a closed form exists.  A resample only moves the median around the n
observed values, so R* = X*_(m) - x_(m) lives on the n numbers
x_(1) - x_(m), ..., x_(n) - x_(m) whatever the data are.  All that is needed
is how much probability sits on each, and that is a binomial count: the
resampled median exceeds x_(l) exactly when fewer than m of the n draws land
in the l smallest observations, each of which is drawn with probability l/n.
That is Eq. (3.4), and differencing it in l gives Eq. (3.5).

The sample size is taken odd, n = 2m - 1, so that the median is a single
order statistic and no averaging of two middle values is needed (the paper's
convention in Eq. 3.1).
"""

import numpy as np
from scipy.stats import binom

from bootstrap import bootstrap_exact, bootstrap_mc, weighted_mean


# --------------------------------------------------------------------------
# Step 1 — the exact bootstrap distribution, in closed form
# --------------------------------------------------------------------------

def median_pmf(n):
    """Prob_*{R* = x_(l) - x_(m)} for l = 1, ..., n — Eq. (3.5).

    Returns an array indexed by l - 1.  It depends on n alone: the DATA never
    enter.  Only the support x_(l) - x_(m) does, which is what makes the
    median so clean here and so awkward for the jackknife.
    """
    if n % 2 == 0:
        raise ValueError("Sec. 3 takes n = 2m - 1 odd")
    m = (n + 1) // 2
    l = np.arange(1, n + 1)
    # Eq. (3.4): Prob_*{X*_(m) > x_(l)} = Prob{Binomial(n, l/n) <= m - 1}
    upper = binom.cdf(m - 1, n, (l - 1) / n)
    lower = binom.cdf(m - 1, n, l / n)
    return upper - lower


def median_exact_moments(x):
    """(E_* R*, E_* R*^2) for the sample median, from Eq. (3.5).

    E_* R*^2 is Eq. (3.7), the paper's estimate of E_F R^2 = E_F[t(X) - theta(F)]^2.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.shape[0]
    support = x - x[(n - 1) // 2]           # x_(l) - x_(m)
    p = median_pmf(n)
    return float(support @ p), float((support ** 2) @ p)


def median_abs_error_exact(x, scaled=True):
    """E_* R* for R of Eq. (3.12), |t(X*) - theta(F-hat)| / sigma-hat.

    The quantity tabulated in Table 1.  Efron bootstraps the absolute error
    rather than the squared one because it is more stable, and divides by
    sigma-hat to make R* scale invariant, which removes the variation due to
    sigma-hat differing from sigma(F).

    `scaled` multiplies by sqrt(n), and it has to: Eq. (3.12) as printed has no
    such factor, but without it the numbers are not the ones tabulated.  The
    absolute error of a median shrinks like n^(-1/2), so an unscaled Eq. (3.12)
    would drift towards 0 with n instead of settling anywhere.  See the
    reproduction of column (3.6) in __main__ for the arithmetic: unscaled it
    averages 0.276 against the paper's 1.01, and scaled it averages 1.05.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.shape[0]
    support = np.abs(x - x[(n - 1) // 2])
    sigma_hat = np.sqrt(np.mean((x - x.mean()) ** 2))   # mu-hat_2^(1/2)
    value = float((support @ median_pmf(n)) / sigma_hat)
    return value * np.sqrt(n) if scaled else value


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- Eq. (3.6): the paper prints these six numbers for n = 13 ----------

    paper_36 = {2: .0015, 3: .0142, 4: .0550, 5: .1242, 6: .1936, 7: .2230}
    pmf13 = median_pmf(13)

    print("Eq. (3.6), the bootstrap distribution of R* for n = 13:\n")
    print(f"{'l':>8}  {'paper':>7}  {'ours':>9}  {'|diff|':>8}")
    for l, p_paper in paper_36.items():
        ours = pmf13[l - 1]
        tag = f"{l} or {14 - l}" if l != 7 else "7"
        print(f"{tag:>8}  {p_paper:7.4f}  {ours:9.6f}  {abs(ours - p_paper):8.1e}")
    print(f"\n  total mass: {pmf13.sum():.15f}")
    # Symmetric because F-hat is not involved: the pmf depends on n alone, and
    # l <-> n+1-l is the same binomial read from the other end.  The residual
    # is binom.cdf round-off, not a broken symmetry.
    print(f"  max |p_l - p_(n+1-l)|: {np.abs(pmf13 - pmf13[::-1]).max():.1e}")

    # --- The closed form against brute force ------------------------------
    #
    # Eq. (3.5) is a derivation, and bootstrap_exact is an enumeration of all
    # C(2n-1, n) resamples.  They must agree exactly; if they do, both the
    # multinomial weights of Method 1 and the binomial argument of Sec. 3 are
    # right, since it would take a conspiracy for two wrong routes to meet.

    print("\nEq. (3.5) against enumerating every resample:\n")
    print(f"{'n':>3}  {'E_* R* (closed)':>16}  {'E_* R* (enum)':>14}"
          f"  {'E_* R*^2 (closed)':>18}  {'E_* R*^2 (enum)':>16}")
    for n in [5, 7, 9]:
        x = np.sort(rng.normal(size=n))
        m1, m2 = median_exact_moments(x)
        centre = x[(n - 1) // 2]
        vals, w = bootstrap_exact(x, np.median)
        e1 = weighted_mean(vals - centre, w)
        e2 = weighted_mean((vals - centre) ** 2, w)
        print(f"{n:3d}  {m1:16.12f}  {e1:14.12f}  {m2:18.12f}  {e2:16.12f}")

    # --- And against Monte Carlo ------------------------------------------

    x13 = np.sort(rng.normal(size=13))
    _, m2_exact = median_exact_moments(x13)
    print("\nMonte Carlo against the closed form (n = 13):\n")
    print(f"{'N':>8}  {'E_* R*^2':>12}  {'rel. error':>11}")
    for n_boot in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]:
        vals = bootstrap_mc(x13, lambda s: np.median(s, axis=-1), n_boot, rng,
                            vectorized=True)
        mc = float(np.mean((vals - x13[6]) ** 2))
        print(f"{n_boot:8d}  {mc:12.8f}  {abs(mc / m2_exact - 1):11.2%}")

    # --- The asymptotic claim of Sec. 3 -----------------------------------
    #
    # n E_*(R*)^2 -> 1/(4 f^2(theta)).  For the standard normal, f(0) is
    # 1/sqrt(2 pi), so the limit is pi/2 = 1.5708.  Two things converge there,
    # and it is worth separating them: the TRUTH, n E_F R^2, which is what we
    # would like to know and can only compute here because we chose F; and the
    # BOOTSTRAP ESTIMATE of it, averaged over samples, which is all a
    # practitioner ever has.

    limit = np.pi / 2
    print(f"\nSec. 3 asymptotics, F = N(0,1), limit 1/(4f^2) = {limit:.6f}:\n")
    print(f"{'n':>5}  {'n E_F R^2 (truth)':>18}  {'mean n E_* R*^2 (boot)':>23}"
          f"  {'ratio':>7}")
    n_trials = 4000
    for n in [13, 25, 51, 101, 201]:
        samples = rng.normal(size=(n_trials, n))
        samples.sort(axis=1)
        truth = n * float(np.mean(np.median(samples, axis=1) ** 2))
        centre = samples[:, (n - 1) // 2][:, None]
        p = median_pmf(n)
        boot = n * float(np.mean(((samples - centre) ** 2) @ p))
        print(f"{n:5d}  {truth:18.4f}  {boot:23.4f}  {boot / truth:7.3f}")
    print("  both columns walk towards pi/2, and towards each other: the")
    print("  bootstrap is estimating the right quantity, not merely a stable one")

    # --- Table 1, column (3.6), and the missing sqrt(n) --------------------
    #
    # The paper runs 200 trials with 100 bootstrap replications each and
    # reports AVE 1.01, S.D. .31 for the plain bootstrap.  Our column (3.6) is
    # computed exactly from Eq. (3.5) rather than by a secondary Monte Carlo,
    # so it is that experiment with its N -> infinity limit taken; the paper
    # notes each of its entries carries a standard error of about .15.
    #
    # Running it both ways is what pins the scaling down: Eq. (3.12) as
    # printed is off by sqrt(n).

    print("\nTable 1, column (3.6) — n = 13, F = N(0,1):\n")
    print(f"{'Eq. (3.12)':>22}  {'trials':>7}  {'AVE':>6}  {'S.D.':>6}")
    for label, scaled, n_trials in [("as printed", False, 200),
                                    ("times sqrt(n)", True, 200),
                                    ("times sqrt(n)", True, 20_000)]:
        vals = np.array([median_abs_error_exact(rng.normal(size=13), scaled=scaled)
                         for _ in range(n_trials)])
        print(f"{label:>22}  {n_trials:7d}  {vals.mean():6.3f}  {vals.std(ddof=1):6.3f}")
    print(f"{'paper':>22}  {200:7d}  {1.01:6.2f}  {0.31:6.2f}")
    # At the paper's own 200 trials both AVE and S.D. wander by about .02-.04
    # from seed to seed, so the 20000-trial row is the one to compare against.
    print("  only the scaled row can be what Table 1 tabulates.  It is not a")
    print("  typo of consequence -- R* is scale invariant either way -- but the")
    print("  unscaled quantity drifts to 0 with n and would tabulate nothing")

    # --- Eq. (3.13): the true value Table 1 is trying to hit ---------------
    #
    # With the scaling settled, the truth can be simulated directly, and it
    # does not come out at the paper's .95.  The gap is small but it is 30
    # times our simulation error, so it is not noise on our side.  Efron does
    # not say how he obtained .95; the asymptotic median distribution
    # N(0, pi/2n) would give exactly 1, so .95 is not that either.

    n, n_sim = 13, 400_000
    samples = rng.normal(size=(n_sim, n))
    med = np.abs(np.median(samples, axis=1))
    e_f_r = float(med.mean()) * np.sqrt(n)
    se = float(med.std(ddof=1)) * np.sqrt(n) / np.sqrt(n_sim)
    print(f"\nEq. (3.13), E_F R for n = 13 (sigma(F) = 1, so no division):")
    print(f"  paper: 0.95      ours: {e_f_r:.4f} +/- {se:.4f}"
          f"   ({abs(e_f_r - 0.95) / se:.0f} standard errors away)")
    print(f"  asymptotic value sqrt(n) E|median| -> 1: "
          f"{1.0:.4f}, approached from below")
