"""The correlation coefficient and Figure 1 — Efron (1979), Remark B.

The only figure in the paper, and the only place it works on real numbers
rather than simulated ones: nine data pairs with sample correlation
rho-hat = .945, bootstrapped N = 1000 times, and the same replications shown
again after Fisher's transformation z = arctanh(rho).

What the figure is for is Eq. (8.1).  If g is monotone, then a bootstrap
replication R* = t(X*) - theta(F-hat) becomes

    S* = g(R* + theta-hat) - g(theta-hat)

under the transformation, and since g preserves order, the q-quantile of one
distribution is carried to the q-quantile of the other.  The bootstrap is
therefore EQUIVARIANT under monotone transformations -- exactly, not
approximately, and at every N -- so the two halves of the figure contain the
same information and one is entitled to work in whichever scale is more
convenient.  Everything else in the paper is an approximation of some kind;
this is an identity, and the code below treats it as one to be checked.

Two things make these nine points worth more than the paper asks of them.
First, n = 9 puts the exact bootstrap of Sec. 2 within reach -- C(17, 9) =
24310 resamples -- so the histograms can be compared with the fixed
distribution they are approximating, and not merely with each other.  Second,
the correlation is the first statistic here whose bootstrap distribution is
badly behaved at the edges: it reaches +-1 with positive probability and is
undefined on a few resamples, which is invisible at N = 1000 and decides what
may legitimately be computed from the transformed scale.

The data are those printed in the caption of Fig. 1, credited there to
Miller [10]; the text of Remark B says Miller [14], page 12, which is the
Miller entry of the reference list.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from bootstrap import (bootstrap_exact, bootstrap_mc, n_distinct_resamples,
                       weighted_quantile)  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# The nine pairs of Fig. 1, in the order printed there.
PAIRS = np.array([(1.15, 1.38), (1.70, 1.72), (1.42, 1.59), (1.38, 1.47),
                  (2.80, 1.66), (4.70, 3.45), (4.80, 3.87), (1.41, 1.31),
                  (3.90, 3.75)])


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------
#
# This is the first statistic in this implementation that is not a function
# of one number per observation: an observation is a PAIR, and a resample
# must take both coordinates together.  Nothing in bootstrap.py needs
# changing for that — resampling is resampling of indices, and `bootstrap_mc`
# and `bootstrap_exact` index the data whatever shape it has — which is why
# the index vector was made the primitive there.

def pearson(pairs):
    """Sample correlation of an array of pairs, shape (..., n, 2).

    Returns nan where a resample has no variability in one coordinate, which
    happens when it draws a single pair n times.  That is not defensive
    programming: it is a real feature of the bootstrap distribution of a
    correlation, quantified in __main__.
    """
    pairs = np.asarray(pairs, dtype=float)
    x, y = pairs[..., 0], pairs[..., 1]
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(-1) * (yc ** 2).sum(-1))
    num = (xc * yc).sum(-1)
    r = np.divide(num, den, out=np.full(np.shape(num), np.nan), where=den > 0)
    # A resample drawing only two distinct pairs is exactly collinear, so
    # |r| = 1 up to rounding.  Clipping matters here rather than being
    # cosmetic: arctanh is nan just outside [-1, 1] and infinite at the ends,
    # and it is the infinity that is the true answer.
    return np.clip(r, -1.0, 1.0)


def fisher_z(rho):
    """arctanh, the transformation g of Eq. (8.1) used in Fig. 1.

    Monotone on (-1, 1) and nowhere else, which is the whole subtlety below.
    The infinities at +-1 are wanted rather than tolerated — they are what
    the last section of __main__ is about — so numpy is told not to complain.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctanh(rho)


def transformed(rho_star, rho_hat):
    """S* of Eq. (8.1) for g = arctanh: z(rho*) - z(rho-hat)."""
    return fisher_z(rho_star) - fisher_z(rho_hat)


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------

def fig1(path="fig1_correlation.png", n_boot=1000, seed=0):
    """Fig. 1: the same N = 1000 replications, in both scales.

    Drawn as in the paper — two histograms, with the 1/6, 1/2 and 5/6
    quantiles marked — with the exact bootstrap distribution of all
    C(17, 9) resamples added behind them, which the paper could not compute
    and which is what the histograms are approximating.
    """
    rng = np.random.default_rng(seed)
    rho_hat = float(pearson(PAIRS))

    mc = bootstrap_mc(PAIRS, pearson, n_boot, rng, vectorized=True)
    values, weights = bootstrap_exact(PAIRS, pearson)
    good = np.isfinite(values)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.6))
    panels = [
        (axes[0], mc - rho_hat, values[good] - rho_hat,
         "$\\hat{\\rho}^* - \\hat{\\rho}$", np.linspace(-0.26, 0.08, 35)),
        (axes[1], transformed(mc, rho_hat), transformed(values[good], rho_hat),
         "$\\tanh^{-1}\\hat{\\rho}^* - \\tanh^{-1}\\hat{\\rho}$",
         np.linspace(-1.5, 1.8, 34)),
    ]

    for ax, sample, exact, label, edges in panels:
        ax.hist(exact, bins=edges, weights=weights[good], color="0.87",
                label="exact, all $\\binom{17}{9}$ resamples")
        ax.hist(sample, bins=edges, weights=np.full(n_boot, 1 / n_boot),
                histtype="step", lw=1.4, color="#2b6cb0",
                label=f"Monte Carlo, $N={n_boot}$")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
        top = ax.get_ylim()[1]
        # Staggered, because 1/6 and the median land within a label's width
        # of each other once the scale is stretched.
        for k, (q, name) in enumerate([(1 / 6, "1/6"), (0.5, "median"),
                                       (5 / 6, "5/6")]):
            xq = weighted_quantile(sample, q)
            ax.annotate(name, (xq, top * 0.60),
                        xytext=(xq, top * (0.93 if k % 2 == 0 else 0.78)),
                        ha="center", fontsize=7.5, color="#b91c1c",
                        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1))
        ax.set_xlabel(label)
        ax.set_ylabel("probability")
        ax.set_xlim(edges[0], edges[-1])

    # The ceiling.  rho* cannot pass 1, which is only 1 - rho-hat above the
    # observed value, so the upper tail of the first panel is cut off by
    # construction and the distribution has nowhere to go but left.  arctanh
    # sends that ceiling to infinity, and the asymmetry changes sides.
    axes[0].axvline(1 - rho_hat, color="#b45309", ls="--", lw=1.1)
    axes[0].annotate("$\\hat{\\rho}^* = 1$",
                     (1 - rho_hat, axes[0].get_ylim()[1] * 0.20),
                     xytext=(5, 0), textcoords="offset points", ha="left",
                     va="bottom", rotation=90, fontsize=7.5, color="#b45309")
    axes[1].annotate("the same ceiling is now at $+\\infty$",
                     (1.75, axes[1].get_ylim()[1] * 0.5), ha="right",
                     fontsize=7.5, color="#b45309")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title("The nine pairs of Fig. 1, $\\hat{\\rho} = 0.945$:\n"
                      "the same replications before and after $\\tanh^{-1}$",
                      fontsize=9.5, pad=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rho_hat = float(pearson(PAIRS))
    quantiles = [1 / 6, 0.5, 5 / 6]

    print(f"Remark B, the nine pairs of Fig. 1 (n = {len(PAIRS)}):\n")
    print(f"  rho-hat = {rho_hat:.6f}     paper: .945")
    print(f"  z-hat   = {fisher_z(rho_hat):.6f}")

    # --- The paper's own experiment ----------------------------------------
    #
    # N = 1000, and the three quantiles the figure marks.  Computing them
    # separately in each scale and then checking Eq. (8.1) makes the point of
    # the figure: the transformation moves the histogram but not the ranking,
    # so the quantiles are carried across exactly.

    mc = bootstrap_mc(PAIRS, pearson, 1000, rng, vectorized=True)
    print("\nN = 1000 replications, as in Fig. 1:\n")
    print(f"{'q':>8}  {'rho* - rho-hat':>15}  {'z* - z-hat':>11}"
          f"  {'Eq. (8.1) from the left':>24}")
    for q in quantiles:
        a = weighted_quantile(mc - rho_hat, q)
        b = weighted_quantile(transformed(mc, rho_hat), q)
        print(f"{q:8.3f}  {a:15.4f}  {b:11.4f}"
              f"  {fisher_z(a + rho_hat) - fisher_z(rho_hat):24.4f}")
    print("  the last two columns agree to every digit: Eq. (8.1) is an")
    print("  identity, holding at any N and for any monotone g")

    # --- Against the distribution it is approximating -----------------------
    #
    # n = 9 is small enough to enumerate, so for once the exact bootstrap
    # distribution of a correlation is available.  Monte Carlo error at
    # N = 1000 is what separates the columns; nothing else does.

    values, weights = bootstrap_exact(PAIRS, pearson)
    good = np.isfinite(values)
    v_ok, w_ok = values[good], weights[good] / weights[good].sum()

    print(f"\nExact bootstrap, all C(17, 9) = {n_distinct_resamples(9)} "
          "resamples:\n")
    print(f"{'q':>8}  {'exact':>9}  {'N = 1000':>9}  {'N = 100000':>11}")
    big = bootstrap_mc(PAIRS, pearson, 100_000, rng, vectorized=True)
    for q in quantiles:
        print(f"{q:8.3f}  {weighted_quantile(v_ok, q, w_ok) - rho_hat:9.4f}"
              f"  {weighted_quantile(mc - rho_hat, q):9.4f}"
              f"  {weighted_quantile(big - rho_hat, q):11.4f}")

    # --- What the two histograms actually say -------------------------------
    #
    # Remark B reads them off: one straggles left, the other right, and the
    # median sits just above zero.  Both are quantifiable, and are read here
    # from the exact distribution rather than from a run of 1000 — the second
    # of them cannot be read from moments at all, since the transformed
    # distribution has none, as the next section shows.
    #
    # The mechanism is in the last row.  rho* cannot exceed 1, which is only
    # 1 - rho-hat = 0.055 above the observed value, while it can fall to -1:
    # the upper tail is cut off by construction and the mass has nowhere to
    # go but left.  arctanh stretches that ceiling to infinity, and the
    # straggle changes sides.  The transformation does not merely reshape the
    # distribution, it removes the boundary that was shaping it.

    print("\nShape, from the exact distribution (Remark B):\n")
    print(f"{'q':>8}  {'rho* - rho-hat':>15}  {'z* - z-hat':>12}")
    for q in [0.01, 1 / 6, 0.5, 5 / 6, 0.99]:
        r = weighted_quantile(v_ok, q, w_ok)
        print(f"{q:8.3f}  {r - rho_hat:15.4f}  {transformed(r, rho_hat):12.4f}")
    print(f"{'min':>8}  {v_ok.min() - rho_hat:15.4f}"
          f"  {transformed(v_ok.min(), rho_hat):12.4f}")
    print(f"{'max':>8}  {v_ok.max() - rho_hat:15.4f}"
          f"  {transformed(v_ok.max(), rho_hat):12.4f}")
    print()
    for label, v in [("rho* - rho-hat", v_ok - rho_hat),
                     ("z* - z-hat", transformed(v_ok, rho_hat))]:
        med = weighted_quantile(v, 0.5, w_ok)
        spread = (weighted_quantile(v, 5 / 6, w_ok)
                  - weighted_quantile(v, 1 / 6, w_ok))
        print(f"  {label:>15}:  median is {abs(med) / spread:5.1%}"
              f" of the 1/6-5/6 spread")
    print("  the first straggles left, the second right, and in both the")
    print("  median is a few per cent of the spread: no bias correction is")
    print("  worth making here, which is what Remark B concludes")

    # --- The edges, which N = 1000 cannot see -------------------------------
    #
    # A resample drawing only two distinct pairs has all its points on a
    # line, so rho* = +-1 exactly; drawing a single pair nine times leaves
    # rho* undefined.  Neither is a numerical artefact and both have exact
    # probabilities: 36 * [(2/9)^9 - 2/9^9] and 9 / 9^9.  They are invisible
    # in a run of 1000 and they decide what the transformed scale means,
    # because arctanh carries them to +-infinity.

    at_one = np.isclose(np.abs(values), 1.0)
    undefined = ~good
    print("\nThe edges of the bootstrap distribution of a correlation:\n")
    print(f"  |rho*| = 1 exactly:  {at_one.sum():4d} resamples,"
          f"  probability {weights[at_one].sum():.3e}"
          f"   (36[(2/9)^9 - 2/9^9] = {36 * ((2 / 9) ** 9 - 2 / 9 ** 9):.3e})")
    print(f"  rho* undefined:      {undefined.sum():4d} resamples,"
          f"  probability {weights[undefined].sum():.3e}"
          f"   (9/9^9 = {9 / 9 ** 9:.3e})")
    print(f"  expected number of either in N = 1000:"
          f" {1000 * weights[at_one | undefined].sum():.2f}")
    print("\n  so the bootstrap distribution of tanh^-1(rho*) has atoms at")
    print("  +-infinity, and no mean or variance whatsoever.  Its quantiles")
    print("  are unaffected -- the mass out there is 5 in 100000 -- which is")
    print("  why Fig. 1 marks quantiles and not moments, and why Eq. (8.1),")
    print("  a statement about quantiles alone, survives a transformation")
    print("  that destroys every moment.")

    print("\nwrote", fig1())
