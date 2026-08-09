"""Figures for DERIVATIONS.md.

Every figure is computed with the code in bootstrap.py rather than drawn, so
that it checks the section it accompanies instead of illustrating it.

    python derivation_figures.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from bootstrap import (bootstrap_exact, bootstrap_mc, n_distinct_resamples,
                       var_mean_formula)
from jackknife import median_jackknife_var_batch
from median import median_pmf

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------
# Sec. 3 — the simplex of resamples, and how uneven their weights are
# --------------------------------------------------------------------------

def fig_simplex(path="ded_simplex.png"):
    """The 10 distinct resamples of n = 3, placed on the simplex.

    Weights come from bootstrap_exact, so what is plotted is exactly what the
    enumeration uses.  Marker area is proportional to probability.
    """
    n = 3
    x = np.array([0.0, 1.0, 2.0])                # labels only; weights need no data
    _, weights = bootstrap_exact(x, np.mean)

    from itertools import combinations_with_replacement
    counts = np.array([np.bincount(idx, minlength=n)
                       for idx in combinations_with_replacement(range(n), n)])
    P = counts / n

    # Barycentric -> cartesian.
    cx = P[:, 1] + 0.5 * P[:, 2]
    cy = P[:, 2] * np.sqrt(3) / 2

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], color="0.75", lw=1)

    ax.scatter(cx, cy, s=weights * 4200, color="#2b6cb0", alpha=0.85, zorder=3)
    for k in range(len(P)):
        ax.annotate(f"{weights[k]*27:.0f}/27",
                    (cx[k], cy[k]), textcoords="offset points",
                    xytext=(0, 13 if weights[k] < 0.2 else 17),
                    ha="center", fontsize=7.5, color="0.35")

    centre = np.argmax(weights)
    ax.annotate("the observed sample\n$\\mathbf{P}^* = \\mathbf{e}/n$",
                (cx[centre], cy[centre]), textcoords="offset points",
                xytext=(0, -34), ha="center", fontsize=8)

    # The apex label has to go sideways: the title is above it and its own
    # weight annotation sits where a centred label would land.
    for corner, label, off in zip(
            corners[:3],
            ["$(x_1,x_1,x_1)$", "$(x_2,x_2,x_2)$", "$(x_3,x_3,x_3)$"],
            [(-14, -10), (14, -10), (52, 2)]):
        ax.annotate(label, corner, textcoords="offset points", xytext=off,
                    ha="center", fontsize=8, color="0.4")

    ax.set_title(f"$n=3$: all $\\binom{{2n-1}}{{n}} = {n_distinct_resamples(n)}$ "
                 "distinct resamples,\narea proportional to multinomial probability",
                 pad=16)
    ax.set_aspect("equal")
    ax.set_ylim(-0.12, np.sqrt(3) / 2 + 0.10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Sec. 4 — Monte Carlo approximates a fixed, finite object
# --------------------------------------------------------------------------

def fig_exact_vs_mc(path="ded_exact_vs_mc.png"):
    """The exact bootstrap distribution of the mean against Monte Carlo.

    Left: the weighted atoms of the exact distribution with Monte Carlo
    histograms on top.  Right: the two errors of Sec. 7, separated.
    """
    rng = np.random.default_rng(4)
    n = 8
    x = rng.normal(size=n)

    values, weights = bootstrap_exact(x, np.mean)
    centre = x.mean()
    v_exact = var_mean_formula(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    edges = np.linspace((values - centre).min(), (values - centre).max(), 26)
    ax1.hist(values - centre, bins=edges, weights=weights, color="0.85",
             label="exact (Method 1)")
    for n_boot, colour in [(200, "#d97706"), (20000, "#2b6cb0")]:
        mc = bootstrap_mc(x, lambda s: np.mean(s, axis=-1), n_boot, rng,
                          vectorized=True) - centre
        # Same vertical scale as the exact atoms: probability per bin.
        ax1.hist(mc, bins=edges, weights=np.full(n_boot, 1 / n_boot),
                 histtype="step", lw=1.4, color=colour,
                 label=f"Monte Carlo, $N={n_boot}$")
    ax1.set_xlabel("$R^* = \\bar{X}^* - \\bar{x}$")
    ax1.set_ylabel("probability")
    ax1.set_title(f"$n={n}$: Monte Carlo approximates a fixed object")
    ax1.legend(frameon=False, fontsize=8)

    grid = np.array([50, 200, 800, 3200, 12800])
    rms = []
    for n_boot in grid:
        errs = [np.var(bootstrap_mc(x, lambda s: np.mean(s, axis=-1), n_boot,
                                    rng, vectorized=True)) - v_exact
                for _ in range(300)]
        rms.append(np.sqrt(np.mean(np.square(errs))))
    rms = np.array(rms)

    ax2.loglog(grid, rms, "o-", color="#2b6cb0", label="Monte Carlo error")
    ax2.loglog(grid, rms[0] * np.sqrt(grid[0] / grid), "--", color="0.6",
               label="$N^{-1/2}$")
    ax2.axhline(abs(v_exact - 1.0 / n), color="#b91c1c", lw=1.3,
                label="bootstrap's own error")
    ax2.set_ylim(top=abs(v_exact - 1.0 / n) * 2.2)
    ax2.set_xlabel("$N$")
    ax2.set_ylabel("error in $\\mathrm{Var}_* R^*$")
    ax2.set_title("only one of the two errors responds to $N$")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Sec. 6 — the median: a distribution that does not depend on the data
# --------------------------------------------------------------------------

def fig_median(path="ded_median.png"):
    """The bootstrap distribution of the median, and what it converges to.

    Left: the pmf of Eq. (3.5) for three sample sizes, on the scale that makes
    them comparable.  Right: the quantity the section is about, n E_* R*^2,
    against the truth it is estimating and the limit both approach.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    for n, colour in [(13, "#d97706"), (51, "#2b6cb0"), (201, "#166534")]:
        p = median_pmf(n)
        m = (n + 1) // 2
        # The support is x_(l) - x_(m), which for a normal sample spreads like
        # 1/sqrt(n); plotting against (l - m)/sqrt(n) and scaling the mass by
        # sqrt(n) is what puts the three on one scale.
        u = (np.arange(1, n + 1) - m) / np.sqrt(n)
        ax1.plot(u, p * np.sqrt(n), ".-", ms=3, lw=1, color=colour,
                 label=f"$n = {n}$")
    ax1.set_xlim(-3, 3)
    ax1.set_xlabel("$(l - m)/\\sqrt{n}$")
    ax1.set_ylabel("$\\sqrt{n}\\;\\mathrm{Prob}_*\\{R^* = x_{(l)} - x_{(m)}\\}$")
    ax1.set_title("Eq. (3.5) depends on $n$ alone,\nand settles into one shape")
    ax1.legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(1)
    grid = [13, 25, 51, 101, 201, 401]
    truth, boot = [], []
    for n in grid:
        # 30000 trials, in chunks: at 4000 the truth wobbles by 2% and reads
        # as structure it does not have.
        t_acc, b_acc = [], []
        for _ in range(10):
            s = np.sort(rng.normal(size=(3000, n)), axis=1)
            t_acc.append(np.median(s, axis=1) ** 2)
            b_acc.append(((s - s[:, [(n - 1) // 2]]) ** 2) @ median_pmf(n))
        truth.append(n * float(np.mean(np.concatenate(t_acc))))
        boot.append(n * float(np.mean(np.concatenate(b_acc))))

    ax2.axhline(np.pi / 2, color="0.6", ls="--", lw=1.2,
                label="$1/4f^2(\\theta) = \\pi/2$")
    ax2.plot(grid, boot, "o-", color="#2b6cb0", label="bootstrap, $n E_* R^{*2}$")
    ax2.plot(grid, truth, "s-", color="#b91c1c", label="truth, $n E_F R^2$")
    ax2.set_xscale("log")
    ax2.set_xticks(grid)
    ax2.set_xticklabels([str(g) for g in grid])
    ax2.minorticks_off()
    ax2.set_xlabel("$n$")
    ax2.set_title("both converge, and the gap between\nthem closes slowly")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Sec. 7 — where each method looks, and what the jackknife converges to
# --------------------------------------------------------------------------

def fig_jackknife(path="ded_jackknife.png"):
    """The diagnosis of Remark J, and the limit law it leads to.

    Left: the distance from e/n at which each method evaluates R.  Right: the
    jackknife variance of the median at n = 4001, against the two candidate
    limit laws, whose densities are elementary.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    n = np.array([10, 30, 100, 300, 1000, 3000, 10000], dtype=float)
    # ||P* - e/n||: RMS sqrt(n-1)/n for a bootstrap resample, and exactly
    # sqrt(d / [n(n-d)]) for a deletion of d points -- both derived in Sec. 7.
    ax1.loglog(n, np.sqrt(n - 1) / n, "o-", color="#2b6cb0",
               label="bootstrap resample (r.m.s.)")
    ax1.loglog(n, np.sqrt(1 / (n * (n - 1))), "s-", color="#b91c1c",
               label="delete one, $\\;d = 1$")
    d = np.maximum(2, np.round(np.sqrt(n)))
    ax1.loglog(n, np.sqrt(d / (n * (n - d))), "^-", color="#166534",
               label="delete $d \\sim \\sqrt{n}$")
    ax1.set_xlabel("$n$")
    ax1.set_ylabel("$\\|\\mathbf{P}^* - \\mathbf{e}/n\\|$")
    ax1.set_title("where each method evaluates $R$")
    ax1.legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(2)
    n_big, trials = 4001, 40_000
    ratios = []
    for _ in range(trials // 2000):
        s = rng.normal(size=(2000, n_big))
        ratios.append(n_big * median_jackknife_var_batch(s) / (np.pi / 2))
    ratios = np.concatenate(ratios)

    edges = np.linspace(0, 8, 41)
    ax2.hist(ratios, bins=edges, density=True, color="0.85",
             label=f"simulated, $n = {n_big}$")
    w = np.linspace(1e-4, 8, 400)
    # Densities of the two candidates, by change of variable from the
    # exponential: 2 exp(-2 sqrt(w)) and exp(-sqrt(w)) / (2 sqrt(w)).
    ax2.plot(w, 2 * np.exp(-2 * np.sqrt(w)), color="#166534", lw=1.6,
             label="$[\\chi^2_4/4]^2$  (ours)")
    ax2.plot(w, np.exp(-np.sqrt(w)) / (2 * np.sqrt(w)), color="#b91c1c",
             lw=1.6, ls="--", label="$[\\chi^2_2/2]^2$  (as printed)")
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("$n\\,\\hat v_{\\mathrm{jack}} \\,/\\, (1/4f^2)$")
    ax2.set_ylabel("density")
    ax2.set_title("the limit law of the jackknife variance")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for f in (fig_simplex, fig_exact_vs_mc, fig_median, fig_jackknife):
        print("wrote", f())
