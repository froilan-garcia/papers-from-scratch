"""
Coefficient paths on the prostate-cancer data — Tibshirani (1996), Fig. 5 + Table 1.

Piece 3: the lasso applied to the paper's real dataset, where variable selection
can actually be watched happening.

Data: Stamey et al. (1989), N = 97 men, 8 predictors, response `lpsa` (log
prostate-specific antigen). Same data the paper uses in Sec. 4.

The paper parametrises the path not by the penalty `lam` but by the normalised
budget (Sec. 2.1, Sec. 4):

    s = t / sum_j |beta_j^OLS|        in [0, 1]

so s = 1 is OLS (constraint inactive) and s = 0 kills every coefficient. We
solve the Lagrangian form for a grid of `lam`, then convert each solution to its
own s = sum_j |beta_j(lam)| / sum_j |beta_j^OLS|. That traces the same path.

The paper selects s_hat = 0.44 by generalised cross-validation and reports the
resulting model in Table 1: only `lcavol`, `lweight` and `svi` survive.

Run:  python prostate_paths.py   -> saves fig5_prostate_paths.png
Needs: numpy, pandas, matplotlib  (data/prostate.data must be present)
"""

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lasso import lasso_coordinate_descent, standardize

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# %%
# ---------------------------------------------------------------------------
# Data. The file is whitespace-separated with a leading index column and a
# trailing `train` flag; the paper uses all 97 observations, so we drop `train`.
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "data", "prostate.data"), sep="\t", index_col=0)

PREDICTORS = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
X_raw = df[PREDICTORS].to_numpy(dtype=float)
y_raw = df["lpsa"].to_numpy(dtype=float)

# Paper convention (Eq. 1): predictors standardized, response centered.
X, y = standardize(X_raw, y_raw)
N, p = X.shape
print(f"N = {N}, p = {p}")

# OLS reference: needed for the denominator of s.
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
l1_ols = np.abs(beta_ols).sum()
print(f"sum |beta_OLS| = {l1_ols:.4f}")


# %%
# ---------------------------------------------------------------------------
# Trace the path. lam_max is the smallest penalty that zeroes everything: above
# max_j |x_j^T y| / N every coordinate is soft-thresholded to 0 on the first
# sweep, so the path is fully covered by lam in (0, lam_max].
# ---------------------------------------------------------------------------
lam_max = np.abs(X.T @ y).max() / N
lambdas = np.linspace(lam_max, 0.0, 400)

betas = np.array([lasso_coordinate_descent(X, y, lam) for lam in lambdas])
s_values = np.abs(betas).sum(axis=1) / l1_ols   # normalised budget of each fit


# %%
# ---------------------------------------------------------------------------
# Table 1: the model the paper reports at s_hat = 0.44 (chosen by GCV).
# We pick the fit on the grid whose s is closest to that value.
# ---------------------------------------------------------------------------
S_HAT = 0.44
idx = int(np.argmin(np.abs(s_values - S_HAT)))
beta_hat = betas[idx]

print(f"\nModel at s = {s_values[idx]:.3f} (target {S_HAT}, lam = {lambdas[idx]:.4f})")
print(f"{'predictor':<10}{'OLS':>10}{'lasso':>10}")
for name, b_ols, b_las in zip(PREDICTORS, beta_ols, beta_hat):
    mark = "" if b_las != 0 else "   <- dropped"
    print(f"{name:<10}{b_ols:>10.3f}{b_las:>10.3f}{mark}")

kept = [n for n, b in zip(PREDICTORS, beta_hat) if b != 0]
print(f"\nRetained ({len(kept)}): {', '.join(kept)}")
print("Paper's Table 1 keeps: lcavol, lweight, svi")


# %%
# ---------------------------------------------------------------------------
# Fig. 5: each coefficient against the normalised budget s.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))

colors = plt.cm.tab10(np.linspace(0, 1, p))
for j, (name, color) in enumerate(zip(PREDICTORS, colors)):
    ax.plot(s_values, betas[:, j], color=color, lw=1.8, label=name)
    # Label each curve at the OLS end of the path.
    ax.annotate(name, (s_values[-1], betas[-1, j]), color=color, fontsize=8,
                textcoords="offset points", xytext=(4, 0), va="center")

ax.axhline(0, color="0.7", lw=0.8, zorder=0)
ax.axvline(S_HAT, color="crimson", ls="--", lw=1.2)
ax.annotate(rf"$\hat s = {S_HAT}$ (GCV)", (S_HAT, ax.get_ylim()[1]),
            color="crimson", fontsize=9, ha="right", va="top",
            textcoords="offset points", xytext=(-5, -5))

ax.set_xlabel(r"normalised budget  $s = t\,/\,\sum_j |\hat\beta_j^{OLS}|$")
ax.set_ylabel(r"coefficient  $\hat\beta_j$")
ax.set_title("Lasso coefficient paths, prostate-cancer data\n"
             "(Tibshirani 1996, Fig. 5)")
ax.set_xlim(0, 1.08)
fig.tight_layout()

out = os.path.join(HERE, "fig5_prostate_paths.png")
fig.savefig(out, dpi=150)
print(f"\nSaved {os.path.basename(out)}")
plt.show(block=False)
