# Mean-variance portfolio selection — Markowitz (1952)

Implementation of the E-V rule from *Portfolio Selection* (Markowitz, 1952).
See the [review](../../reviews/1952-markowitz-portfolio-selection.md) for the
paper's context and results, and **[DERIVATIONS.md](DERIVATIONS.md)** for the
mathematics developed end to end — the problem, the closed form, the geometry of
the three-asset case, the sign constraint and the algorithm.

Built incrementally, same as the [lasso](../1996-tibshirani-lasso/). Current
status:

| # | Piece | Status |
|---|-------|--------|
| 0 | Fundamentals — deriving the frontier by Lagrange | ✅ now Part II of [DERIVATIONS.md](DERIVATIONS.md) |
| 1 | Closed-form efficient frontier (no sign constraint) | ✅ [frontier.py](frontier.py) |
| 2 | Figs. 2 and 3 — simplex, isomeans, isovariances, critical line | ✅ [geometry.py](geometry.py) |
| 3 | Fig. 6 — QP frontier with `w >= 0`, and the connected parabolic segments | ✅ [constrained.py](constrained.py) |
| 4 | The diversification floor — refuting the law of large numbers (p. 79) | ⬜ the identity is derived (sec. 2), the figure is not |
| 5 | Real data + bootstrap-resampled frontier (links to Efron 1979) | ⬜ pending |

## What matches and what does not

**The paper carries no numbers.** Figs. 1–7 are schematic and there is no table,
no data set and no worked example anywhere in the fifteen pages — Markowitz says
on closing (p. 91) that where the beliefs come from is another story. So there
is nothing to reproduce numerically, and validation here is **structural**: each
qualitative claim is turned into a statement that can fail, and then tested. The
market used is ours, and it is documented in [markets.py](markets.py).

| Claim of the paper | Result |
|---|---|
| Isomeans are parallel straight lines (p. 84) | ✅ $E$ is affine in $(X_1,X_2)$, so the slope has no $E$ in it |
| Isovariances are concentric ellipses centred at $\hat X$ (p. 84) | ✅ and $\hat X$ **is** the minimum-variance portfolio, to $2\times10^{-16}$ |
| The tangency points trace a straight line, the *critical line* (p. 85) | ✅ collinear to $1.5\times10^{-16}$; the paper asserts this without algebra, [DERIVATIONS.md](DERIVATIONS.md#s10) derives it |
| That line is the same object as the $n$-asset solution | ✅ directions parallel to $6\times10^{-17}$ in cross product |
| Fig. 2 — $\hat X$ inside: efficient set starts there, then an edge | ✅ reproduced, one corner at $E=0.0979$ |
| Fig. 3 — $\hat X$ outside: starts on an edge, meets the line, then another edge | ✅ reproduced, two corners; three phases exactly as p. 85 describes |
| The efficient set is a series of connected segments (p. 87) | ✅ 3 corners in the five-asset market, weights affine between them to $10^{-15}$ |
| Fig. 6 — $V$ against $E$ is a series of connected parabolic segments | ✅ each arc equals the closed-form parabola of **its own sub-market** to $10^{-15}$ |
| …and therefore a different parabola on each segment | ✅ curvature $17.49 \to 17.53 \to 59.03 \to 386.30$ |
| Footnote 9 — the degenerate case $\boldsymbol\mu \propto \mathbf 1$ | ✅ detected: $D=0$ and the solver refuses rather than dividing |
| Footnote 12 — ellipses **iff** no two distinct portfolios are perfectly correlated | ❌ **sufficient, not necessary** — see below |
| Against `scipy.optimize` (SLSQP) over the whole sweep | ✅ to $3\times10^{-8}$, which is that solver's tolerance and not ours |
| p. 79 — diversification has a floor at the average covariance | ⏳ derived as an identity (sec. 2); the figure is Piece 4 |

### The one discrepancy: footnote 12

The footnote (p. 89) says that to draw the isovariance curves as ellipses it is
necessary and sufficient that no two distinct portfolios have perfectly
correlated returns. Sufficient, yes. Necessary, no: what the algebra needs is
that no two distinct portfolios have returns differing by an **additive
constant** — perfect correlation *and* equal variance. Three assets, the first
two perfectly correlated and the third independent:

| variances of the correlated pair | $\min\operatorname{eig}\Sigma$ | $\det Q$ | level sets |
|---|---|---|---|
| $1$ and $4$ | $0$ | $1.00$ | ellipses |
| $1$ and $1$ | $0$ | $0.00$ | degenerate |

Both markets contain two distinct portfolios with perfectly correlated returns —
the two single-asset portfolios — and only the second loses its ellipses. The
first is more interesting than a counterexample: it contains a **riskless**
portfolio, $2\cdot(1)-(2)$, with variance exactly zero, so $\Sigma$ is singular
and the closed form of Piece 1 does not exist there, while the plane geometry of
Piece 2 is perfectly well defined. The check is `_footnote12` in
[geometry.py](geometry.py); the derivation is
[sec. 9](DERIVATIONS.md#s9).

## Rules of this implementation

- **The solver is ours.** The active set method of [constrained.py](constrained.py)
  is written out — feasible start, saddle-point system on the free assets,
  minimum-ratio test, multiplier test — rather than delegated to a library.
  It is the algorithm the paper is missing (footnote 10 describes the traversal
  but not the rules; the systematic version is Markowitz 1956), and it is the
  same method this repository already wrote for the lasso. `scipy` appears only
  to check the answer.
- **No implicit inverse.** $\Sigma^{-1}$ is never formed; `np.linalg.solve`
  does the work, and the four scalars $A,B,C,D$ keep visible that the market
  enters through only two vectors.
- **The paper's axes in the paper's figures** ($V$ vertical, $E$ horizontal),
  the modern ones in the derivation figures, and a note wherever it matters.
  The two conventions describe the same efficient set (p. 89).
- **Our numbers, declared as ours.** The five-asset market is named after the
  paper's sectors (p. 89) and built from annual returns, volatilities and
  correlations, which is the form beliefs actually take.
- Every figure is **computed with these solvers**. Nothing is drawn by hand,
  which is what lets a figure disagree with the text.

## Why this order and not the paper's

The paper opens with the constrained three-asset problem and draws it. This
implementation opens with the unconstrained $n$-asset problem and solves it,
because the constrained case is built out of it: on each face of the polygonal
efficient set, the answer is the closed form applied to the assets that survive.
Doing it the other way round means writing the QP first and having nothing to
check it against.

| What moves | Where the paper puts it | Here | Why |
|---|---|---|---|
| The $n$-asset closed form | absent (p. 79 declines it) | Piece 1 | Everything else is a corollary of it |
| The sign constraint | first, from Eq. (4) | Piece 3 | It is what destroys the closed form |
| The three-asset geometry | first, Figs. 2–3 | Piece 2 | It reads better as a picture *of* the general solution |

## The pieces

**Piece 1 — [frontier.py](frontier.py).** The four scalars, the frontier
$V(E)=(AE^2-2BE+C)/D$, the minimum-variance vertex $(B/A, 1/A)$, the two-fund
decomposition $\mathbf w(E)=\mathbf g+E\mathbf h$ and the multiplier reading
$dV/dE = 2\lambda$. Checks: the constraints along the whole frontier, the
parabola against the variance of the weights it predicts, the finite-difference
slope, agreement with SLSQP, and twenty thousand random portfolios none of which
falls below the curve.

**Piece 2 — [geometry.py](geometry.py).** The reduction to the plane, the
ellipses and their centre, the critical line derived from $Q^{-1}\mathbf e$, and
Figs. 2 and 3 with the efficient set drawn by the Piece 3 solver. Checks: the
centre against the minimum-variance portfolio, the collinearity of the tangency
points, the agreement of the two derivations of the critical line, and footnote
12 above.

**Piece 3 — [constrained.py](constrained.py).** The active set method, the sweep
over $E$, and the corners located by bisection to $10^{-13}$. Checks: agreement
with SLSQP; the identification of each segment with the closed-form frontier of
its sub-market; and the continuity of $\lambda$ against the jump in curvature at
every corner, which is the $C^1$-but-not-$C^2$ statement.

The five-asset market resolves into four efficient segments, with rails leaving
at $E = 0.0781$, bonds at $0.0856$ and utilities at $0.0985$. The first corner is
worth a look: the curvature changes by two parts in a thousand, so the kink is
real and invisible. Nothing guarantees that a corner can be seen.

## How to run

```bash
python frontier.py            # Piece 1 and its checks
python constrained.py         # Piece 3, the active set solver and its checks
python geometry.py            # Piece 2, its checks, and fig2/fig3
python figures.py             # the paper's Figs. 1 and 6
python derivation_figures.py  # the figures of DERIVATIONS.md
```

| File | Piece | What it does |
|---|---|---|
| [DERIVATIONS.md](DERIVATIONS.md) | — | The full mathematical development, with its figures |
| [markets.py](markets.py) | — | The example markets, and why each one is shaped as it is |
| [frontier.py](frontier.py) | 1 | Closed form, two funds, multipliers, minimum variance |
| [constrained.py](constrained.py) | 3 | Active set QP, the sweep, the corners |
| [geometry.py](geometry.py) | 2 | The plane reduction, the critical line, Figs. 2 and 3 |
| [figures.py](figures.py) | 1, 3 | The paper's Figs. 1 and 6 |
| [derivation_figures.py](derivation_figures.py) | — | The figures for the derivation (prefix `ded_`) |

## Stack

`numpy`, `scipy` (only to check Piece 3), `matplotlib`. `pandas` for the real
data of Piece 5, when it exists. Nothing outside the repository's base stack.
