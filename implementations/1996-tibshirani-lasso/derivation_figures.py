"""Figures for DEDUCCIONES.md.

These are not the paper's figures (those live in figures.py, prostate.py and
two_predictors.py).  Each one illustrates a step of a derivation that the paper
states without proving, and each is computed from the solver rather than drawn.

    ded_polytope.png       sec. 3   the L1 ball as a polytope with 2^p faces
    ded_clip_vs_soft.png   sec. 6   clipping (p=1) vs soft thresholding (p>=2)
    ded_gamma_root.png     sec. 7   why the gamma of Eq. (3) is unique
    ded_active_set.png     sec. 14  the shift is not coordinatewise shrinkage
    ded_value_function.png sec. 15  V(t) convex, lambda(t) monotone
    ded_gcv_cv.png         sec. 18  where GCV and CV put their minima
    ded_stein.png          sec. 19  the erratum in the Stein risk formula
    ded_priors.png         sec. 20  Laplace vs normal prior (the paper's Fig. 7)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lasso import l1_norm, lasso, ols, rss, t_max
from orthonormal import gamma_for_budget, soft_threshold
from selection import cv_curve, gcv_curve, lambda_from_kkt

GREY, RED, BLUE = "0.55", "crimson", "#1f77b4"


def clip_vs_soft(path="ded_clip_vs_soft.png"):
    """Sec. 6: with p = 1 the constrained solution CLIPS; soft thresholding needs
    a budget shared by p >= 2 coordinates."""
    b = np.linspace(-5, 5, 601)
    t = 2.0

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
    axes[0].plot(b, b, ls=":", c=GREY, lw=1)
    axes[0].plot(b, np.sign(b) * np.minimum(np.abs(b), t), c="k", lw=1.8)
    axes[0].set_title(r"$p=1$: recorte, $\mathrm{sign}(\hat b)\min(|\hat b|,t)$",
                      fontsize=10)
    axes[1].plot(b, b, ls=":", c=GREY, lw=1)
    axes[1].plot(b, soft_threshold(b, t), c=RED, lw=1.8)
    axes[1].set_title(r"$p\geq2$: soft thresholding, "
                      r"$\mathrm{sign}(\hat b)(|\hat b|-\gamma)^+$", fontsize=10)
    for ax in axes:
        ax.axhline(0, c="0.8", lw=0.8)
        ax.axvline(0, c="0.8", lw=0.8)
        ax.set_xlabel(r"$\hat\beta^{\,o}_j$")
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    axes[0].set_ylabel(r"$\hat\beta_j$")
    fig.suptitle("La distinción del caso $p=1$: recortar no es trasladar",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def gamma_root(path="ded_gamma_root.png"):
    """Sec. 7: phi(gamma) = sum_j (a_j - gamma)^+ is piecewise linear and strictly
    decreasing where it matters, so the budget equation has a unique root."""
    a = np.array([2.6, 1.8, 1.1, 0.7, 0.25])
    t = 2.4
    g = np.linspace(0, a.max() * 1.12, 800)
    phi = np.array([np.maximum(a - x, 0).sum() for x in g])
    root = gamma_for_budget(a, t)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(g, phi, c="k", lw=1.8, label=r"$\varphi(\gamma)=\sum_j(a_j-\gamma)^+$")
    ax.axhline(t, c=RED, ls="--", lw=1.2, label=f"$t = {t}$")
    ax.plot([root], [t], "o", c=RED, ms=7)
    ax.annotate(rf"$\gamma^\star={root:.3f}$", (root, t), xytext=(8, 10),
                textcoords="offset points", c=RED, fontsize=9)
    for k, val in enumerate(np.sort(a)[::-1]):
        ax.axvline(val, c=GREY, lw=0.7, ls=":")
        ax.annotate(f"$a_{{({k+1})}}$", (val, phi.max() * 0.97), fontsize=7.5,
                    c=GREY, ha="center")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\sum_j|\hat\beta_j|$")
    ax.set_title(r"El $\gamma$ de la Eq. 3: lineal a trozos, raíz única",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, g.max())
    ax.set_ylim(0, phi.max() * 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def priors(path="ded_priors.png"):
    """Sec. 20 (the paper's Fig. 7): the Laplace prior has a non-differentiable
    peak at 0 — which is exactly why the posterior mode lands on 0."""
    b = np.linspace(-5, 5, 801)
    tau = 1.0                                   # Laplace scale
    lap = np.exp(-np.abs(b) / tau) / (2 * tau)
    sd = np.sqrt(2) * tau                       # matched variance 2*tau^2
    nor = np.exp(-b ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(b, lap, c="k", lw=1.8, label="doble exponencial (lasso)")
    ax.plot(b, nor, c=GREY, lw=1.6, ls="--", label="normal (ridge)")
    ax.set_xlabel(r"$\beta_j$")
    ax.set_ylabel("densidad")
    ax.set_title("Fig. 7 — el pico no derivable en 0 es la selección de variables",
                 fontsize=10.5)
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def polytope(path="ded_polytope.png"):
    """Sec. 3: the L1 ball is the intersection of 2^p half-spaces — a cross
    polytope.  For p = 3 that is 8 triangular faces and 6 vertices, all on the
    axes, and a vertex is a point with p-1 coordinates exactly 0."""
    t = 1.0
    verts = np.array([[t, 0, 0], [-t, 0, 0], [0, t, 0],
                      [0, -t, 0], [0, 0, t], [0, 0, -t]])
    faces = [[verts[i], verts[j], verts[k]]
             for i in (0, 1) for j in (2, 3) for k in (4, 5)]

    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(faces, facecolor="0.75", edgecolor="0.25",
                                         linewidths=1.1, alpha=0.55))
    ax.scatter(*verts.T, c=RED, s=45, depthshade=False)
    for v in verts:
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], c=GREY, lw=0.7, ls=":")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel(r"$\beta_1$")
    ax.set_ylabel(r"$\beta_2$")
    ax.set_zlabel(r"$\beta_3$")
    ax.set_title(r"$\sum_j|\beta_j|\leq t$ con $p=3$:"
                 "\n" r"$2^p=8$ caras, $2p=6$ vértices sobre los ejes",
                 fontsize=10)
    ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def stein(path="ded_stein.png", n_rep=30000, seed=7):
    """Sec. 19: an unbiased risk estimate has to average to the true risk.  The
    formula with min does; the one printed in the paper, with max, does not."""
    rng = np.random.default_rng(seed)
    mu = np.array([3.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    p = len(mu)
    Z = mu + rng.standard_normal((n_rep, p))
    gs = np.linspace(0.0, 6.0, 61)

    true, ours, printed = [], [], []
    for g in gs:
        true.append(np.mean(((soft_threshold(Z, g) - mu) ** 2).sum(axis=1)))
        below = (np.abs(Z) < g).sum(axis=1)
        ours.append(np.mean(p - 2 * below
                            + (np.minimum(np.abs(Z), g) ** 2).sum(axis=1)))
        printed.append(np.mean(p - 2 * below
                               + np.maximum(np.abs(Z), g ** 2).sum(axis=1)))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(gs, true, c="k", lw=2.4, label="riesgo verdadero (Monte Carlo)")
    ax.plot(gs, ours, c=RED, lw=1.5, ls="--",
            label=r"con $\min(|z_i|,\gamma)^2$  (deducido)")
    ax.plot(gs, printed, c=BLUE, lw=1.5, ls=":",
            label=r"con $\max(|z_i|,\gamma^2)$  (como se imprime)")
    ax.axhline(mu @ mu, c=GREY, lw=0.8)
    ax.annotate(r"$\|\mu\|^2$", (5.8, mu @ mu), xytext=(0, 5),
                textcoords="offset points", fontsize=8, c=GREY)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("riesgo")
    ax.set_title("La errata de la fórmula de Stein (Sec. 4)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, 60)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def active_set(path="ded_active_set.png", n_rep=140, seed=3):
    """Sec. 14: on the active set the solution is OLS refit on A, shifted by
    lambda (X_A'X_A)^{-1} s_A.  That shift is NOT a coordinatewise shrinkage.

    (a) A design where [S^{-1} 1]_3 < 0, so beta_3 moves AWAY from zero as the
        budget tightens.  Only sum_j |beta_j| is monotone.
    (b) Over random correlated designs, beta_j against beta_j^o is a cloud, not
        a curve: the soft thresholding of Eq. (3) does not describe the lasso
        outside the orthonormal case.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))

    # --- (a) the coefficient that grows -----------------------------------
    r = 0.65                                  # r12 = 0, r13 = r23 = r
    R = np.array([[1.0, 0.0, r], [0.0, 1.0, r], [r, r, 1.0]])
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((400, 3)) @ np.linalg.cholesky(R).T
    Xa = (Z - Z.mean(0)) / Z.std(0)
    ya = Xa @ np.ones(3) + 0.5 * rng.standard_normal(400)
    ya = ya - ya.mean()

    bo = ols(Xa, ya)
    t0a = t_max(Xa, ya)
    ss = np.linspace(0.02, 1.0, 120)
    pa = np.array([lasso(Xa, ya, s * t0a) for s in ss])

    for j, colour in enumerate(["0.55", "0.75", RED]):
        axes[0].plot(ss, pa[:, j], c=colour, lw=2.0 if j == 2 else 1.4,
                     label=rf"$\beta_{j+1}$")
        axes[0].plot([1.0], [bo[j]], "o", c=colour, ms=4)
    axes[0].axhline(bo[2], c=RED, ls=":", lw=1.0)
    axes[0].annotate(rf"OLS de $\beta_3$ = {bo[2]:.2f}", (0.03, bo[2]),
                     xytext=(0, 5), textcoords="offset points", fontsize=8, c=RED)
    axes[0].plot(ss, np.abs(pa).sum(axis=1), c="k", lw=1.0, ls="--",
                 label=r"$\sum_j|\beta_j|$")
    axes[0].set_xlabel(r"$s = t/t_0$")
    axes[0].set_ylabel("coeficiente")
    axes[0].set_title(rf"$\beta_3$ crece hasta {pa[:, 2].max():.2f} "
                      "mientras el presupuesto baja", fontsize=9.5)
    axes[0].legend(fontsize=8, loc="upper left")

    # --- (b) same beta^o, many lassos --------------------------------------
    beta_true = np.array([3.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])   # Sec. 7.2
    p, N, s_fixed = len(beta_true), 60, 0.5
    obs, lst, gammas = [], [], []
    for _ in range(n_rep):
        A = rng.standard_normal((p, p + 2))
        S = A @ A.T
        d = np.sqrt(np.diag(S))
        Sigma = S / np.outer(d, d)                    # a random correlation matrix
        Zi = rng.standard_normal((N, p)) @ np.linalg.cholesky(Sigma).T
        Xi = (Zi - Zi.mean(0)) / Zi.std(0)
        yi = Xi @ beta_true + 2.0 * rng.standard_normal(N)
        yi = yi - yi.mean()
        bi = ols(Xi, yi)
        ti = s_fixed * l1_norm(bi)
        obs.append(bi)
        lst.append(lasso(Xi, yi, ti))
        gammas.append(gamma_for_budget(np.abs(bi), ti))
    obs, lst = np.concatenate(obs), np.concatenate(lst)
    g_med = float(np.median(gammas))

    grid = np.linspace(-2.4, 5.4, 400)
    axes[1].axhline(0, c="0.85", lw=0.8)
    axes[1].plot(grid, grid, ls=":", c=GREY, lw=1)
    axes[1].scatter(obs, lst, s=6, c="k", alpha=0.4, lw=0,
                    label="el lasso de verdad")
    axes[1].plot(grid, soft_threshold(grid, g_med), c=RED, lw=2.0,
                 label=rf"Eq. 3 con $\gamma={g_med:.2f}$ (la mediana)")

    # the vertical spread at one value of beta^o: a function would have none
    x0, half = 2.0, 0.12
    band = lst[np.abs(obs - x0) < half]
    axes[1].plot([x0, x0], [band.min(), band.max()], c=BLUE, lw=2.6,
                 solid_capstyle="butt")
    axes[1].annotate(rf"en $\hat\beta^{{\,o}}_j\approx{x0}$:"
                     "\n" rf"{band.min():.2f} a {band.max():.2f}",
                     (x0, band.max()), xytext=(6, 2), textcoords="offset points",
                     fontsize=8, c=BLUE)
    axes[1].set_xlabel(r"$\hat\beta^{\,o}_j$")
    axes[1].set_ylabel(r"$\hat\beta_j$")
    axes[1].set_title(rf"$s={s_fixed}$, {n_rep} diseños correlados al azar",
                      fontsize=9.5)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_xlim(-2.4, 5.4)
    axes[1].set_ylim(-2.4, 5.4)

    fig.suptitle("El desplazamiento no es un encogimiento por coordenadas",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    print(f"  (a) OLS = {np.round(bo, 4)},  max de beta_3 en la trayectoria "
          f"= {pa[:, 2].max():.4f}")
    print(f"  (a) (X'X)^-1 sign(beta^o) * N = "
          f"{np.round(np.linalg.solve(Xa.T @ Xa, np.sign(bo)) * len(Xa), 4)}")
    print(f"  (b) gamma mediana {g_med:.4f}; en beta^o ~ {x0} el lasso va de "
          f"{band.min():.4f} a {band.max():.4f} ({band.size} puntos)")


def _prostate():
    import pandas as pd
    from lasso import standardize
    from prostate import LWEIGHT_1996, PREDICTORS
    df = pd.read_csv("data/prostate.data", sep="\t", index_col=0)
    df.loc[32, "lweight"] = LWEIGHT_1996
    X, y, *_ = standardize(df[PREDICTORS].to_numpy(float),
                           df["lpsa"].to_numpy(float))
    return X, y


def value_function(path="ded_value_function.png"):
    """Sec. 15: V(t) is convex and non-increasing, so lambda(t) = -V'(t)/2 is
    non-increasing — which is what makes indexing by t or by lambda equivalent.
    The KKT lambda of section 14 is checked against the numerical derivative."""
    X, y = _prostate()
    t0 = t_max(X, y)
    ts = np.linspace(1e-3, t0, 160)

    V = np.array([rss(X, y, lasso(X, y, t)) for t in ts])
    lam_kkt = np.array([lambda_from_kkt(X, y, lasso(X, y, t))[0] for t in ts])
    lam_num = -0.5 * np.gradient(V, ts)          # V'(t) = -2 lambda

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    axes[0].plot(ts, V, c="k", lw=1.8)
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel(r"$V(t)=\min\,\|y-X\beta\|^2$")
    axes[0].set_title("convexa y no creciente", fontsize=10)

    axes[1].plot(ts, lam_kkt, c=RED, lw=2.2, label=r"$\lambda$ por KKT (Sec. 14)")
    axes[1].plot(ts, lam_num, c="k", lw=1.0, ls="--",
                 label=r"$-V'(t)/2$ numérico")
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel(r"$\lambda$")
    axes[1].set_title("monótona decreciente, lineal a trozos", fontsize=10)
    axes[1].legend(fontsize=8.5)
    fig.suptitle(r"La correspondencia $t \leftrightarrow \lambda$ (datos de próstata)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def gcv_cv(path="ded_gcv_cv.png"):
    """Sec. 18: both selectors on the prostate data, against the paper's 0.44."""
    X, y = _prostate()
    grid = np.linspace(0.0, 1.0, 101)
    gcv, dof = gcv_curve(X, y, grid)
    pe, se = cv_curve(X, y, grid, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, curve, name, colour in [(axes[0], gcv, "GCV (Eq. 10)", RED),
                                    (axes[1], pe, "CV quíntuple", BLUE)]:
        ax.plot(grid, curve, c=colour, lw=1.8)
        best = grid[int(np.argmin(curve))]
        ax.axvline(best, c=colour, ls=":", lw=1.2)
        ax.axvline(0.44, c="0.35", ls="--", lw=1.2)
        ax.annotate(f"mín. {best:.2f}", (best, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, c=colour)
        ax.annotate(r"paper 0.44", (0.44, ax.get_ylim()[1]), xytext=(-52, -12),
                    textcoords="offset points", fontsize=8, c="0.35")
        ax.set_xlabel("$s$")
        ax.set_title(name, fontsize=10)
    axes[1].fill_between(grid, pe - se, pe + se, color=BLUE, alpha=0.15)
    axes[0].set_ylabel("criterio")
    fig.suptitle("Ninguno de los dos selectores cae en 0.44", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    polytope()
    clip_vs_soft()
    gamma_root()
    active_set()
    value_function()
    gcv_cv()
    stein()
    priors()
