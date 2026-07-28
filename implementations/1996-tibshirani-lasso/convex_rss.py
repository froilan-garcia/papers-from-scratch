"""Convexity of the least-squares objective — a pedagogical figure.

Companion to the "Fundamentos" section of the README. Shows, for a 2-predictor
problem, that RSS(b) = ||y - X b||^2 is a convex paraboloid:

    (a) 1D slice  -> an upward parabola (unique minimum, no local traps)
    (b) 3D surface -> a bowl
    (c) top-down   -> elliptical level sets, centered at b_hat (OLS),
                      with the L1 rhombus / L2 circle overlaid to connect
                      with the constraint geometry of Fig. 2.

RSS(b) = y^T y - 2 y^T X b + b^T (X^T X) b   is quadratic in b, and its Hessian
2 X^T X is positive semidefinite (v^T X^T X v = ||X v||^2 >= 0), hence convex.

Run:  python convex_rss.py   ->  saves convex_rss.png   (numpy + matplotlib)
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

rng = np.random.default_rng(0)

# --- a small correlated 2-predictor design so the ellipses come out tilted ---
N = 60
rho = 0.5                                      # corr(x1, x2) -> tilts the ellipses
x1 = rng.normal(size=N)
x2 = rho * x1 + np.sqrt(1 - rho**2) * rng.normal(size=N)
X = np.column_stack([x1, x2])
X = (X - X.mean(0)) / X.std(0)                 # standardize (paper convention)
beta_true = np.array([0.5, 2.0])               # small b1, large b2 -> lasso zeroes b1
y = X @ beta_true + rng.normal(scale=1.0, size=N)
y = y - y.mean()                               # center -> no intercept

# OLS minimum (bottom of the bowl): X^T X b = X^T y
b_hat = np.linalg.solve(X.T @ X, X.T @ y)

def rss(b):
    """b: (..., 2) -> RSS. Works on a grid via broadcasting."""
    r = y - b @ X.T                            # residuals per grid point
    return np.einsum("...i,...i->...", r, r)

# --- grid over (b1, b2) around the minimum ---
span = 2.6
g = np.linspace(-span, span, 300) + 0  # symmetric-ish window
B1, B2 = np.meshgrid(b_hat[0] + np.linspace(-2.5, 2.5, 300),
                     b_hat[1] + np.linspace(-2.5, 2.5, 300))
grid = np.stack([B1, B2], axis=-1)
Z = rss(grid)

# =============================================================================
fig = plt.figure(figsize=(15, 4.8))

# --- (a) 1D slice: fix b2 = b_hat[1], vary b1 -> a parabola --------------------
ax1 = fig.add_subplot(1, 3, 1)
b1_line = np.linspace(b_hat[0] - 2.5, b_hat[0] + 2.5, 400)
slice_pts = np.column_stack([b1_line, np.full_like(b1_line, b_hat[1])])
ax1.plot(b1_line, rss(slice_pts), color="#1f77b4", lw=2.2)
ax1.axvline(b_hat[0], ls="--", color="gray", lw=1)
ax1.plot(b_hat[0], rss(b_hat[None])[0], "o", color="#d62728", ms=8, zorder=5)
ax1.annotate("mínimo único\n(vértice)", (b_hat[0], rss(b_hat[None])[0]),
             textcoords="offset points", xytext=(12, 28),
             arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728")
ax1.set_title(r"(a) Corte 1D:  parábola convexa ($a=\sum x_i^2>0$)")
ax1.set_xlabel(r"$\beta_1$   (con $\beta_2$ fijo)")
ax1.set_ylabel(r"RSS")

# --- (b) 3D surface: the bowl -------------------------------------------------
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
step = 4  # thin the grid for a lighter surface
ax2.plot_surface(B1[::step, ::step], B2[::step, ::step], Z[::step, ::step],
                 cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
ax2.scatter(b_hat[0], b_hat[1], rss(b_hat[None])[0],
            color="#d62728", s=45, depthshade=False)
ax2.set_title("(b) Paraboloide:  cuenco (bowl)")
ax2.set_xlabel(r"$\beta_1$"); ax2.set_ylabel(r"$\beta_2$")
ax2.set_zlabel("RSS")
ax2.view_init(elev=32, azim=-58)

# --- (c) top-down: elliptical contours + constraint balls ---------------------
ax3 = fig.add_subplot(1, 3, 3)
levels = rss(b_hat[None])[0] + np.array([2, 8, 20, 40, 70, 110, 160]) * 1.0
cs = ax3.contour(B1, B2, Z, levels=levels, cmap="viridis", linewidths=1.2)
ax3.plot(*b_hat, "o", color="#d62728", ms=8, zorder=6, label=r"$\hat\beta^{OLS}$")
ax3.axhline(0, color="k", lw=0.6); ax3.axvline(0, color="k", lw=0.6)

# constraint regions centered at origin (schematic budget t)
t = 1.4
circle = Circle((0, 0), t, fill=False, ec="#2ca02c", lw=2, label=r"$L_2$: $\|\beta\|_2\leq t$")
rhombus = Polygon([[t, 0], [0, t], [-t, 0], [0, -t]], closed=True,
                  fill=False, ec="#9467bd", lw=2, label=r"$L_1$: $\|\beta\|_1\leq t$")
ax3.add_patch(circle); ax3.add_patch(rhombus)

# constrained optima on the grid (smallest RSS inside each ball)
flat = grid.reshape(-1, 2)
Zf = Z.reshape(-1)
in_l2 = np.linalg.norm(flat, axis=1) <= t
in_l1 = np.abs(flat).sum(1) <= t
b_ridge = flat[in_l2][np.argmin(Zf[in_l2])]
b_lasso = flat[in_l1][np.argmin(Zf[in_l1])]
ax3.plot(*b_ridge, "s", color="#2ca02c", ms=9, zorder=6)
ax3.plot(*b_lasso, "D", color="#9467bd", ms=9, zorder=6)
ax3.annotate("esquina →\n$\\beta_1=0$", b_lasso, textcoords="offset points",
             xytext=(-70, -34), color="#9467bd",
             arrowprops=dict(arrowstyle="->", color="#9467bd"))

ax3.set_title("(c) Vista cenital:  elipses + bolas de restricción (Fig. 2)")
ax3.set_xlabel(r"$\beta_1$"); ax3.set_ylabel(r"$\beta_2$")
ax3.set_aspect("equal")
ax3.legend(loc="upper right", fontsize=8)
ax3.set_xlim(B1.min(), B1.max()); ax3.set_ylim(B2.min(), B2.max())

fig.tight_layout()
fig.savefig(os.path.join(HERE, "convex_rss.png"), dpi=130, bbox_inches="tight")
print(f"b_hat (OLS) = {b_hat}")
print(f"b_ridge     = {b_ridge}   (|.|_2 = {np.linalg.norm(b_ridge):.3f})")
print(f"b_lasso     = {b_lasso}   (|.|_1 = {np.abs(b_lasso).sum():.3f})  -> b1 ~ 0")
print("saved convex_rss.png")
plt.show()
