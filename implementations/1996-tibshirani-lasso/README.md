# Lasso — Tibshirani (1996)

Implementation of *Regression Shrinkage and Selection via the Lasso*. Context and
results of the paper are in the [review](../../reviews/1996-tibshirani-lasso.md).

> 📐 **The mathematics is in [DERIVATIONS.md](DERIVATIONS.md)**, developed end to
> end: the problem, its geometry, the cases that can be solved by hand, the general
> algorithm, the closed form that survives on the active set, and the choice of the
> budget. This README only records what is run and what comes out.

> **Status: closed.** The solver, the orthonormal case, the $p=2$ case, the prostate
> data, the paper's figures and the cross-check against `scikit-learn`. An earlier
> version was discarded for using conventions and an algorithm that are not the
> paper's.
>
> **What is not here:** the reproduction of Table 3 (the Sec. 7 simulations). It runs
> and the ordering of methods comes out, but the error levels do not match under any
> $\sigma$, and the diagnosis depends on details of the simulation setup that I do not
> yet command. It stays local until I can say whose the mismatch is; claiming it now
> would be claiming more than is known.

## What matches and what does not

| Claim of the paper | Result |
|---|---|
| Eq. 3 (orthonormal case) | ✅ agrees with the solver to $1.8\times10^{-15}$ |
| Sec. 6, between $0.5p$ and $0.75p$ iterations | ✅ mean $\approx 0.5p$, never above $0.75p$ |
| Path: $s=1\Rightarrow$ least squares, $s=0\Rightarrow 0$ | ✅ to $3\times10^{-15}$ |
| Table 1, least squares column | ✅ to 0.01 — **but only with the uncorrected data file** |
| Table 1, lasso column (at $s=0.44$) | ✅ to 0.01, selecting exactly `lcavol`, `lweight`, `svi` |
| Fig. 5 (prostate coefficient paths) | ✅ |
| Fig. 1 and Fig. 2 | ✅ the lasso lands on the corner ($\beta_1 = 2.8\times10^{-17}$), ridge does not |
| Eqs. 5 and 6, "valid even if the predictors are correlated" | ✅ the same $\gamma$ in both coordinates and across all five $\rho$, to 9 decimals |
| Fig. 4 (two predictors) | ✅ including the ridge upturn for $\rho>1/2$, which comes out derived |
| Against `sklearn.linear_model.Lasso` and LARS | ✅ to $8\times10^{-13}$ over the whole path |
| "the stability of ridge regression" (abstract) | ✅ and **proved**, not merely simulated: it is non-expansiveness of the projection onto a convex set |
| $\hat s = 0.44$ by GCV (Eq. 10) | ❌ **our GCV minimises at 0.69** |

The GCV discrepancy is diagnosed below, and it is not a solver failure: at $s=0.44$ we
reproduce Table 1 exactly, which isolates the problem in the selector rather than in
the lasso.

## Rules of this implementation

Decisions taken up front, so that they do not turn up hidden in a comment halfway
through the code.

**1. The objective is the paper's, with no modern rescalings.** Eq. 1:

$$
\min_\beta \ \sum_{i=1}^N\Big(y_i - \alpha - \sum_j \beta_j x_{ij}\Big)^2
\qquad \text{subject to} \quad \sum_j |\beta_j| \le t
$$

Without the $\frac{1}{2N}$ of `glmnet`/`sklearn`, which is not in the paper: the
$\frac12$ is there to make the derivative come out clean and the $\frac1N$ to keep
$\lambda$ independent of sample size. The v1 used it without saying so, so that its
$\lambda$ was $\lambda_{\text{paper}}/2N$.

**2. The parameter is indexed as the paper indexes it**,
$s = t/\sum_j|\hat\beta_j^{OLS}| \in [0,1]$, because Section 4 says literally that CV
is done *"over a grid of values of $s$ from 0 to 1 inclusive"*. The v1 swept $\lambda$
on a log scale.

**3. The algorithm is the one in Section 6**, quadratic programming with the sign
constraints introduced sequentially. **Not** coordinate descent: that is Friedman,
Hastie, Höfling & Tibshirani (2007).

### Deviations from the paper

- **The inner QP.** Sec. 6 delegates each subproblem to Lawson & Hansen (1974). What is
  written here is the standard primal active-set method that this stands for
  (`constrained_ls` in [lasso.py](lasso.py)). Same QP, same solution, exact algebra;
  what changes is whose code it is.
- **The reading of $W^-$ in Eq. 9.** The paper only says "generalized inverse". Read as
  the Moore–Penrose pseudoinverse, the null coefficients would be left *unpenalised*,
  which is the opposite of what is needed. We use the reading
  $1/|\beta_j|\to\infty$ (the null ones drop out of the fit), which is the only one
  consistent with Eq. 7 giving them variance 0, as Table 2 reports. Both are
  implemented and compared.
- **CV within folds.** Inside each fold,
  $t = s\sum_j|\hat\beta_j^{OLS}(\text{train})|$, using *that fold's* least squares fit.
  The paper does not specify this.
- **Not implemented:** Stein's unbiased risk estimate (Eq. 11), which is only derived
  for the orthogonal design. The formula is checked — and an erratum found in it — in
  [orthonormal.py](orthonormal.py), but it is not used as a selector.
- **Not published:** the Sec. 7 simulations. See the status box.

## Why this order and not the paper's

The paper goes: definition → orthonormal case → geometry → prostate data → choice of
$t$ → Bayes → algorithm → simulations. That order is **expository**: it serves to
convince a reader.

An **implementation** order answers, at each step, *what needs to exist for the next
one to work?* It comes out different:

| What moves | Where the paper puts it | Here | Why |
|---|---|---|---|
| The algorithm (Sec. 6) | near the end | step 4 | Without a solver there is nothing to validate. |
| The prostate data (Sec. 3) | at the start | step 8 | It needs a selector for $s$, which is step 7. |
| Figures 1 and 2 (Sec. 2) | at the start | step 9 | They illustrate; they do not enable. |
| The $p=1$ case | does not appear | step 2 | It is the atom everything else comes from. |

## The steps

**1. Evaluate before solving** ✅ — `rss`, `l1_norm`, `is_feasible`.
$\text{RSS} = \beta^\top(X^\top X)\beta - 2y^\top X\beta + y^\top y$ has Hessian
$2X^\top X \succeq 0$, so it is convex, and the minimum over a convex set is unique if
$X^\top X\succ0$ — which is why any correct solver gives the same answer.

**2. The single-predictor case** ✅ — The unconstrained optimum is
$\hat b = x^\top y/x^\top x$, and since the parabola is symmetric about it, the
constrained solution is the closest point of $[-t,t]$:
$b^\star = \mathrm{sign}(\hat b)\min(|\hat b|,t)$. Note the nuance: with $p=1$ that is
**clipping**, not *soft thresholding*. Soft thresholding needs $p\ge2$ sharing one
budget — it is the common multiplier that translates every coefficient by the same
constant.

**3. The orthonormal case, and from it Eq. 3** ✅ — If $X^\top X = I$ then
$\hat\beta^{o}=X^\top y$ and

$$
\|y-X\beta\|^2 = \|y\|^2 - 2\beta^\top\hat\beta^o + \|\beta\|^2 = \|\beta-\hat\beta^o\|^2 + \text{const}
$$

which **separates** the problem into $p$ one-dimensional problems tied by a single
$\gamma\ge0$. Stationarity gives
$\beta_j = \hat\beta^o_j - \gamma\,\mathrm{sign}(\beta_j)$, that is Eq. 3. *Validated:*
agrees with the solver to $1.8\times10^{-15}$ ([orthonormal.py](orthonormal.py)).

**4. The paper's algorithm (Sec. 6)** ✅ — $\sum_j|\beta_j|\le t$ is equivalent to the
$2^p$ constraints $\delta_i^\top\beta\le t$ because $\max_\delta \delta^\top\beta$ is
attained at $\delta=\mathrm{sign}(\beta)$ and equals $\sum_j|\beta_j|$: the $L_1$ ball
is a polytope with $2^p$ faces. The algorithm never builds them all; it adds the
violated ones one at a time until Kuhn–Tucker holds. *Validated:* it reproduces step 3
in an orthonormal design, and the mean number of iterations comes out at $\approx0.5p$,
at the low end of the $[0.5p, 0.75p]$ range the paper claims.

*Where the exact zeros come from:* if two active sign vectors differ only in coordinate
$j$, subtracting their equalities $\delta^\top\beta=t$ and $\delta'^\top\beta=t$ gives
$2\delta_j\beta_j=0$, hence $\beta_j=0$. It is structural, not a numerical threshold.

**5. The full path** ✅ — *Validated:* at $s=1$ it gives the least squares fit to
$3\times10^{-15}$, at $s=0$ exact zeros, $\sum|\beta_j|$ is monotone and the budget is
never exceeded.

**6. Choosing $s$ (I): cross-validation** ✅ — Eq. 8, $PE = ME + \sigma^2$, says that
prediction error and model error differ by a constant and are minimised at the same
place. What does **not** work is the training RSS: $\beta$ was chosen to make it small
at those very points, so it decreases with $s$ and would always pick $s=1$.

**7. Choosing $s$ (II): GCV** ⚠️ — The lasso is not a *linear smoother*; the bridge in
Sec. 2.5 is to write $\sum_j|\beta_j| = \sum_j\beta_j^2/|\beta_j|$, which turns the fit
into the ridge estimator of Eq. 9 and allows the trace
$p(t) = \mathrm{tr}\{X(X^\top X+\lambda W^-)^{-1}X^\top\}$. The $\lambda$ comes from
Kuhn–Tucker: $|x_j^\top(y-X\hat\beta)| = \lambda$ for every active coordinate —
checked, the spread across coordinates is $10^{-13}$. **It does not reproduce the
paper's 0.44**; see below.

**8. Real data: prostate** ✅ — Table 1 and Fig. 5 reproduced to 0.01, with $\hat s$
derived rather than hard-coded.

![Fig. 5 — prostate coefficient paths](fig5_prostate_paths.png)

The order in which predictors enter is the paper's: `lcavol` from the start, `svi`
around 0.23, `lweight` around 0.32, and `lcp` and `age` entering negative at the end.
The grey line marks the paper's $\hat s = 0.44$ and the red one our GCV's 0.69.

**9. Figures 1 and 2** ✅ — The four shrinkage functions and the geometry of the
diamond against the circle. The contour in Fig. 2 is drawn through the solution the
solver returns, so the figure is a consequence of the code rather than a drawing of
what ought to come out.

**10. Against `scikit-learn`** ✅ — The first thing is not to compare but to **line up
the conventions**, because the two objectives are not the same: the paper's is
$\|y-X\beta\|^2$ with a constraint, the library's is
$\frac{1}{2N}\|y-X\beta\|^2 + \alpha\|\beta\|_1$. Equating the two Lagrangian forms
gives $\alpha = \lambda/N$, with the KKT $\lambda$ of
[section 14](DERIVATIONS.md) — which is the only conversion used.

Three checks of increasing strength ([sklearn_check.py](sklearn_check.py)):

| | Result |
|---|---|
| One point: $s=0.44$, the model of Table 1 | max. diff. $1.4\times10^{-13}$, same support, same RSS to 10 decimals |
| The whole path, 41 values of $s$ | max. diff. $8.1\times10^{-13}$ |
| Against LARS, **with no conversion at all**, matched at equal $\|\beta\|_1$ | max. diff. $4.9\times10^{-14}$ across the 8 breakpoints |

The third is the one that counts, because it uses neither $\alpha$ nor $\lambda$: an
error in the conversion cannot hide an error in the solver.

And one detail that comes out the opposite way round from what one would expect:
**`sklearn`'s zeros print as exact `0.0` and ours as $10^{-14}$**. Ours are exact by
algebra (step 4) but reach the output through a linear solve, which rounds; coordinate
descent assigns the 0 literally — the soft threshold *is* its update — although the
support it selects depends on its tolerance. Each is exact in a different place.

## The discrepancies with the paper

One is numerical and turned up on running; the other two turned up while deriving and
sit in their place inside [DERIVATIONS.md](DERIVATIONS.md):

- **Eq. 6 needs a lower limit the paper does not give.** It holds for
  $\hat\beta_1^o-\hat\beta_2^o\le t\le\hat\beta_1^o+\hat\beta_2^o$; the paper states
  only the upper one. Below it, one coordinate has already vanished and the solution is
  $(t,0)$. Verified in [two_predictors.py](two_predictors.py).
- **The Stein risk formula (Sec. 4) has an erratum:** it prints
  $\max(|\hat\beta_j^o/\hat\tau|,\gamma^2)$ where
  $\min(|\hat\beta_j^o/\hat\tau|,\gamma)^2$ belongs. The correct version tracks the
  true risk at every $\gamma$ (15.25 against 15.33 at large $\gamma$); the printed one
  blows up to 1144. Checked in [orthonormal.py](orthonormal.py).

### 1. The prostate data file changed after 1996

Row 32 today carries `lweight` = 3.8044 (44.9 g). The file Tibshirani used had 6.1076
(449 g), a decimal-point error later corrected on the *Elements of Statistical
Learning* website.

| | max. deviation vs Table 1 (LS column) |
|---|---|
| with `lweight[32] = 6.1076` (1996) | **0.01** |
| with `lweight[32] = 3.8044` (current) | 0.04 |

[prostate.py](prostate.py) defaults to the 1996 value and prints both columns, because
the point is to reproduce the paper.

### 2. GCV does not choose $\hat s = 0.44$

On the paper's data, Eq. 10 minimises at $s = 0.69$; fivefold CV, at 0.63. At $s=0.44$
GCV is 0.578 against 0.516 at its minimum, so it is not a matter of grid resolution or
of a flat curve.

It is not the solver: **at $s=0.44$ the lasso column of Table 1 comes out exactly**
(0.56 / 0.10 / 0.16 and precisely `lcavol`, `lweight`, `svi`, the rest zero). The
problem is in the selector. Also checked:

- the active-set reading of $W^-$ (ours) → 0.69;
- the Moore–Penrose reading → 0.75, further still;
- $p(t)$ would have to inflate about **6-fold** to move the minimum to 0.48;
- $p(t) := $ number of non-zero coefficients → 0.70.

No reasonable reading of Eq. 10 gives 0.44. This is consistent with the later
literature abandoning this GCV: degrees of freedom from a ridge approximation are
unreliable for the lasso.

## How to run

From this directory, with the `papers` environment:

```bash
python orthonormal.py && python lasso.py && python two_predictors.py && python prostate.py && python figures.py && python derivation_figures.py && python sklearn_check.py
```

| File | Steps | What it does |
|---|---|---|
| [DERIVATIONS.md](DERIVATIONS.md) | — | The full mathematical development, with its figures |
| [derivation_figures.py](derivation_figures.py) | — | The figures for that document (prefix `ded_`) |
| [lasso.py](lasso.py) | 1, 4, 5 | Objective, active-set QP, the Sec. 6 algorithm, the path, and the non-expansiveness check |
| [orthonormal.py](orthonormal.py) | 2, 3 | Eq. 3 in closed form, the four shrinkage functions and the check of the Stein formula |
| [two_predictors.py](two_predictors.py) | 3 | Eq. 5, Eq. 6 and Fig. 4 — validates the $p=2$ derivation |
| [selection.py](selection.py) | 6, 7 | Fivefold CV and GCV (Eq. 10) |
| [prostate.py](prostate.py) | 8 | Table 1 and Fig. 5 |
| [figures.py](figures.py) | 9 | Fig. 1 and Fig. 2 |
| [sklearn_check.py](sklearn_check.py) | 10 | Convention conversion and the cross-check against `sklearn` and LARS |

`data/prostate.data` is not versioned; it is downloaded from the *Elements of
Statistical Learning* website.

### Later, if the mood takes me, and labelled as post-1996

- Coordinate descent (Friedman et al. 2007), written out and compared against the
  Sec. 6 algorithm — not merely called through `sklearn`.
- LARS (Efron et al. 2004) implemented, taking advantage of the piecewise linearity
  already proved in section 14 of [DERIVATIONS.md](DERIVATIONS.md).
- Closing Table 3 and publishing it, once the simulation setup is understood in full.

## Stack

`numpy` and `matplotlib`; `pandas` to read the prostate data; `scikit-learn` **only**
in [sklearn_check.py](sklearn_check.py), as an external reference to check against —
nothing in the solver uses it. The quadratic programming is written by hand, so
`scipy.optimize` is not needed.
