"""Prostate cancer example — Sec. 3, Table 1 and Fig. 5.

Stamey et al. (1989): N = 97 men, 8 clinical predictors, response lpsa.  The
paper uses all 97 observations; the train/test split carried in the data file is
from *Elements of Statistical Learning*, not from this paper, and is ignored.

Two things are declared up front.

1. THE DATA FILE HAS SINCE BEEN CORRECTED.  Row 32 carries lweight = 3.8044
   (44.9 g).  The file Tibshirani used in 1996 had 6.1076 (449 g), a decimal-point
   typo later fixed on the *Elements of Statistical Learning* website.  With the
   1996 value our least squares column matches Table 1 to within 0.01 on all
   eight predictors; with the corrected value the gap reaches 0.04.  Default here
   is the 1996 value, because the point is to reproduce the paper.

2. s_hat IS DERIVED, NOT HARD-CODED.  GCV (Eq. 10) is run over a grid and its
   minimizer reported, whatever it turns out to be.  It does not come out at the
   paper's 0.44 — see the printed comparison and the README.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lasso import l1_norm, lasso, lasso_path, ols, standardize
from selection import cv_select, gcv_select

PREDICTORS = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]

LWEIGHT_1996 = 6.1076        # row 32 as the paper had it
LWEIGHT_FIXED = 3.8044       # row 32 as distributed today

# Table 1 (paper, p. 274), standardized predictors.
PAPER_LS = np.array([0.69, 0.23, -0.15, 0.16, 0.32, -0.15, 0.03, 0.13])
PAPER_LASSO = np.array([0.56, 0.10, 0.00, 0.00, 0.16, 0.00, 0.00, 0.00])
PAPER_S_HAT = 0.44


def load(paper_data=True):
    df = pd.read_csv("data/prostate.data", sep="\t", index_col=0)
    if paper_data:
        df.loc[32, "lweight"] = LWEIGHT_1996
    X, y, *_ = standardize(df[PREDICTORS].to_numpy(float),
                           df["lpsa"].to_numpy(float))
    return X, y, df["lpsa"].mean()


def main():
    X, y, y_mean = load(paper_data=True)
    N, p = X.shape
    print(f"N = {N}, p = {p}, intercept = mean(lpsa) = {y_mean:.2f}"
          f"   (Table 1: 2.48)\n")

    # --- which version of row 32 reproduces the paper's least squares column? --
    print("Least squares column of Table 1, both versions of lweight[32]:\n")
    print(f"{'predictor':>9} {'paper':>7} {'1996 data':>10} {'fixed data':>11}")
    b_1996 = ols(*load(paper_data=True)[:2])
    b_fixed = ols(*load(paper_data=False)[:2])
    for n, ref, a, b in zip(PREDICTORS, PAPER_LS, b_1996, b_fixed):
        print(f"{n:>9} {ref:7.2f} {a:10.2f} {b:11.2f}")
    print(f"{'max|diff|':>9} {'':7} {np.abs(np.round(b_1996, 2) - PAPER_LS).max():10.2f}"
          f" {np.abs(np.round(b_fixed, 2) - PAPER_LS).max():11.2f}\n")

    # --- select s (steps 6 and 7) ------------------------------------------
    grid = np.linspace(0.0, 1.0, 101)
    s_gcv, gcv, dof = gcv_select(X, y, grid)
    s_cv, pe = cv_select(X, y, grid, seed=0)
    i44 = int(np.argmin(np.abs(grid - PAPER_S_HAT)))

    print("Choosing s:\n")
    print(f"  GCV (Eq. 10) minimizes at s = {s_gcv:.2f},  GCV = {gcv.min():.5f},"
          f"  p(t) = {dof[int(np.argmin(gcv))]:.2f}")
    print(f"  at the paper's s = {PAPER_S_HAT}     GCV = {gcv[i44]:.5f},"
          f"  p(t) = {dof[i44]:.2f}")
    print(f"  fivefold CV  minimizes at s = {s_cv:.2f}")
    print(f"  paper reports s_hat = {PAPER_S_HAT} by GCV  <-- NOT reproduced;"
          f" see README\n")

    # --- Table 1, at both values of s --------------------------------------
    t0 = l1_norm(ols(X, y))
    b_at_paper = lasso(X, y, PAPER_S_HAT * t0)
    b_at_ours = lasso(X, y, s_gcv * t0)

    print("Lasso column of Table 1:\n")
    print(f"{'predictor':>9} {'paper':>7} {'ours @0.44':>11} {'ours @%.2f':>11}"
          % s_gcv)
    for n, ref, a, b in zip(PREDICTORS, PAPER_LASSO, b_at_paper, b_at_ours):
        print(f"{n:>9} {ref:7.2f} {a:11.2f} {b:11.2f}")

    def kept(b):
        return [n for n, v in zip(PREDICTORS, b) if abs(v) > 1e-8]

    print(f"\n  retained at s = 0.44 : {kept(b_at_paper)}")
    print(f"  paper retains        : ['lcavol', 'lweight', 'svi']")
    print(f"  max|diff| at s = 0.44: "
          f"{np.abs(np.round(b_at_paper, 2) - PAPER_LASSO).max():.2f}")
    print(f"  retained at s = {s_gcv:.2f} : {kept(b_at_ours)}")

    _plot(X, y, s_gcv)


def _plot(X, y, s_gcv):
    grid = np.linspace(0.0, 1.0, 201)
    path = lasso_path(X, y, grid)

    fig, ax = plt.subplots(figsize=(7.2, 5))
    for j in range(len(PREDICTORS)):
        ax.plot(grid, path[:, j], lw=1.4)

    # Label each curve at its right end, nudged apart so that predictors ending
    # at the same coefficient (age and lcp both land near -0.15) stay readable.
    order = np.argsort(path[-1])
    gap = 0.035 * (path[-1].max() - path[-1].min())
    ypos = path[-1][order].astype(float).copy()
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + gap)
    for j, ylab in zip(order, ypos):
        ax.annotate(PREDICTORS[j], (1.0, ylab), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.axvline(PAPER_S_HAT, ls="--", c="0.35", lw=1.1)
    ax.annotate(r"paper $\hat s = 0.44$", (PAPER_S_HAT, ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", fontsize=8, c="0.35")
    ax.axvline(s_gcv, ls=":", c="crimson", lw=1.1)
    ax.annotate(f"our GCV = {s_gcv:.2f}", (s_gcv, ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", fontsize=8, c="crimson")
    ax.axhline(0.0, c="0.75", lw=0.8)
    ax.set_xlabel(r"$s = t\,/\,\sum_j |\hat\beta_j^{\,o}|$")
    ax.set_ylabel("coef")
    ax.set_title("Fig. 5 — lasso shrinkage, prostate cancer data")
    ax.set_xlim(0.0, 1.19)
    fig.tight_layout()
    fig.savefig("fig5_prostate_paths.png", dpi=150)
    print("\nwrote fig5_prostate_paths.png")


if __name__ == "__main__":
    main()
