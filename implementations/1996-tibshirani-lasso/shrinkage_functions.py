"""
The four shrinkage functions — Tibshirani (1996), Figure 1.

Piece 2 of the implementation: the picture that explains *why* the lasso zeroes
coefficients while ridge does not.

In an orthonormal design (X^T X / N = I) every estimator acts coordinate-wise on
the OLS coefficient beta_hat (Sec. 2.2). Each method is then just a scalar
function d(beta_hat) — its "form of the estimate", which is what Fig. 1 plots:

    Subset selection (hard):  d(b) = b * 1[ |b| > gamma ]     (keep or kill)
    Ridge:                    d(b) = b / (1 + lam)            (proportional)
    Lasso (soft):             d(b) = sign(b) (|b| - gamma)+   (Eq. 3)
    Non-negative garotte:     d(b) = b (1 - gamma^2 / b^2)+   (Breiman 1993)

The dashed 45-degree line is OLS itself (no shrinkage, d(b) = b). Reading the
plot: how far each curve sits *below* that line is how much it shrinks; whether
it *touches zero over an interval* is whether it does variable selection.

Run as a script (saves the figure) or cell-by-cell (# %% blocks) in VS Code.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# Reuse the exact operator from Piece 1 — the lasso curve *is* soft thresholding.
from lasso import soft_threshold


# %%
def hard_threshold(b, gamma):
    """Subset selection in orthonormal design: keep beta_hat if |b| > gamma, else 0.

    Discontinuous at |b| = gamma — the instability the paper criticises (Sec. 1).
    """
    return np.where(np.abs(b) > gamma, b, 0.0)


def ridge_shrink(b, lam):
    """Ridge in orthonormal design: proportional shrinkage b / (1 + lam) (Sec. 2.2).

    Never reaches zero for finite lam — no variable selection.
    """
    return b / (1.0 + lam)


def garotte_shrink(b, gamma):
    """Non-negative garotte (Breiman 1993): b * (1 - gamma^2 / b^2)+ .

    The direct precedent of the lasso; scales each OLS coefficient by a
    non-negative factor. Smoother than soft thresholding for large |b|.
    """
    b = np.asarray(b, dtype=float)
    factor = np.maximum(1.0 - gamma**2 / np.where(b == 0.0, np.inf, b**2), 0.0)
    return b * factor


# %%
# One threshold shared by the selection-type methods so the curves are
# comparable; ridge's lam is picked to shrink a mid-size coefficient by a similar
# amount (purely for a fair-looking overlay, as in the paper's Fig. 1).
gamma = 1.0
lam_ridge = 0.5

b = np.linspace(-3.0, 3.0, 601)

curves = {
    "Subset selection (hard)": (hard_threshold(b, gamma), "#d1495b", "-"),
    "Ridge (proportional)":    (ridge_shrink(b, lam_ridge), "#2e86ab", "-"),
    "Lasso (soft)":            (soft_threshold(b, gamma), "#1b9e77", "-"),
    "Garotte":                 (garotte_shrink(b, gamma), "#e6a817", "--"),
}


# %%
fig, ax = plt.subplots(figsize=(6.2, 6.0))

# OLS reference: the 45-degree line d(b) = b (no shrinkage).
ax.plot(b, b, color="0.6", lw=1.0, ls=":", label="OLS (no shrinkage)")

for name, (y, color, style) in curves.items():
    ax.plot(b, y, color=color, ls=style, lw=2.0, label=name)

# Guides at 0 and at the thresholds ±gamma.
ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.axvline(0, color="0.85", lw=0.8, zorder=0)
for x0 in (-gamma, gamma):
    ax.axvline(x0, color="0.9", lw=0.8, ls="--", zorder=0)

ax.set_xlabel(r"OLS coefficient  $\hat\beta_j$")
ax.set_ylabel(r"shrunk coefficient  $d(\hat\beta_j)$")
ax.set_title("The four shrinkage functions (Tibshirani 1996, Fig. 1)\n"
             r"orthonormal design, threshold $\gamma=1$")
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.tight_layout()

out = "fig1_shrinkage_functions.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")

# The Interactive Window renders the figure inline at the end of the cell.
# block=False keeps a plain `python shrinkage_functions.py` run from hanging on
# an open GUI window (the PNG is already saved above).
plt.show(block=False)
