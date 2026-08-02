# Regression Shrinkage and Selection via the Lasso

**Authors:** Robert Tibshirani · **Year:** 1996 · **Venue:** Journal of the Royal Statistical Society, Series B, 58(1), 267–288 · **Link/DOI:** [10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) · [JSTOR 2346178](https://www.jstor.org/stable/2346178)
**Field:** statistics / ML · **Read:** 2026-07-11

## TL;DR

Tibshirani proposes the *lasso* (least absolute shrinkage and selection operator): least squares subject to $\sum_j |\beta_j| \le t$. The geometry of the $L_1$ ball (a diamond with corners on the axes) makes some coefficients **exactly zero**, combining the best of subset selection (interpretability) and of ridge (stability, by being a continuous process). It is the founding paper of $L_1$ regularisation, now ubiquitous in high-dimensional statistics.

## Context and motivation

Least squares estimates have low bias but high variance, and with many predictors they are not interpretable. The two classical remedies fail on opposite sides: subset selection is interpretable but unstable (a discrete process: small changes in the data change the chosen model), and ridge is stable but never annihilates a coefficient. The direct precedent is Breiman's (1993) *non-negative garotte*, which rescales the least squares estimates by non-negative factors of bounded sum; its weakness is depending explicitly on those estimates, which behave badly under collinearity. The lasso avoids that explicit use.

## Methodology

**Definition (Eq. 1).** With standardized predictors ($\sum_i x_{ij}/N = 0$, $\sum_i x_{ij}^2/N = 1$):

$$\hat{\beta}^{lasso} = \arg\min_\beta \sum_{i=1}^N \Big(y_i - \alpha - \sum_j \beta_j x_{ij}\Big)^2 \quad \text{subject to} \quad \sum_j |\beta_j| \le t$$

Equivalent in Lagrangian form to penalising with $\lambda \sum_j |\beta_j|$. The parameter is usually normalised as $s = t / \sum_j |\hat{\beta}_j^{OLS}| \in [0, 1]$.

**Orthonormal case (Eq. 3).** If $X^T X = I$, the solution is *soft thresholding*:

$$\hat{\beta}_j = \mathrm{sign}(\hat{\beta}_j^{OLS})\,\big(|\hat{\beta}_j^{OLS}| - \gamma\big)^+$$

against ridge (which shrinks proportionally, $\hat{\beta}_j^{OLS}/(1+\gamma)$) and subset selection (*hard thresholding*). There is a direct connection with the wavelet soft shrinkage of Donoho & Johnstone (1994): the lasso asymptotically attains the risk of the ideal subset selector up to a factor $2\log p + 1$ (Section 10).

**Geometry (Section 2.3).** The elliptical RSS contours touch the feasible region first; with $L_1$ that region is a diamond and the contact usually happens at a corner ($\beta_j = 0$). With the $L_2$ ball of ridge there are no corners. This is the paper's most famous figure (Fig. 2). Note: for $p > 2$ with correlation, the lasso can even change sign relative to least squares.

**Bayesian reading (Section 5).** The lasso is the posterior mode under independent double-exponential (Laplace) priors: $f(\beta_j) \propto \exp(-|\beta_j|/\tau)$ — more mass at 0 and in the tails than the normal prior implicit in ridge.

**Choosing $t$ (Section 4).** Three methods: fivefold cross-validation over a grid of $s$, GCV using the effective number of parameters $p(t) = \mathrm{tr}\{X(X^TX + \lambda W^-)^{-1}X^T\}$ with $W = \mathrm{diag}(|\hat{\beta}_j|)$, and Stein's unbiased risk estimate (much cheaper: a single optimisation).

**Algorithm (Section 6).** Quadratic programming, introducing the violated sign constraints $\delta_i^T \beta \le t$ sequentially (out of the $2^p$ possible ones; in practice it converges in $0.5p$–$0.75p$ iterations). David Gay's alternative: write $\beta_j = \beta_j^+ - \beta_j^-$ with $2p$ variables and $2p+1$ constraints. **Historical note:** nobody solves it this way today — LARS (Efron et al. 2004) and above all coordinate descent (Friedman et al. 2007, the basis of `glmnet`) made it trivial.

## Main results

- **Prostate cancer data** (Stamey et al. 1989; $N=97$, 8 predictors, response `lpsa`): with $\hat{s} = 0.44$ chosen by GCV, the lasso retains `lcavol`, `lweight` and `svi` — the same subset as best subset selection, but with shrunken coefficients (Table 1, and Fig. 5 with the coefficient paths).
- **Simulations (Section 7), three regimes** that structure the conclusion:
  1. Few large effects → subset selection wins (and the garotte); the lasso stays close.
  2. A moderate number of moderate effects → **the lasso wins** (example 1: median MSE 1.93 with GCV against 2.79 for least squares and 3.21 for ridge).
  3. Many small effects → ridge wins clearly.
- GCV is consistently the best selector of $t$ across all the examples.
- **Extensions (Sections 8–9):** to GLMs via IRLS (demonstrated with logistic regression on the kyphosis data), to trees (shrinkage instead of pruning) and to MARS.

## Strengths and limitations

**Strengths:** it solves prediction and interpretability at once with a single convex idea; the geometric explanation is crystal clear; there is an uncommon empirical honesty — the paper itself delimits the regimes where the lasso loses; and the connections run deep (Bayes/Laplace, wavelet soft thresholding, the $L_q$ bridge family of Frank & Friedman where $q=1$ is the smallest convex exponent).

**Limitations (some visible only in hindsight):** the original QP algorithm is crude compared with LARS or coordinate descent; the standard error estimate (Eq. 7) gives variance 0 for precisely the coefficients that were annihilated; there is no selection consistency theory (that would come with Zhao & Yu 2006, Zou 2006 — the adaptive lasso); with strongly correlated predictors the lasso picks one arbitrarily (which motivated the elastic net, Zou & Hastie 2005); and it does not address the $p \gg N$ case that would later become its great niche.

## Implementation ideas

> **Later note (2026-08-01).** The [implementation](../implementations/1996-tibshirani-lasso/)
> is done, and it departed from idea 1 on this list: it uses the quadratic
> programming algorithm of the paper's own Section 6, not the coordinate descent of
> Friedman et al. (2007), which came eleven years later. Idea 4 (Table 3) runs but is
> not published: the ordering of methods comes out and the levels do not, and the
> diagnosis depends on details of the simulation setup that are not yet settled.
> Everything else is done and validated.

Everything central is reproducible in pure numpy:

1. ~~**Coordinate descent solver**~~ → the Sec. 6 one was written instead, which is the paper's. ✅
2. **Fig. 1**: the four shrinkage functions in an orthonormal design (subset, ridge, lasso, garotte). ✅
3. **Fig. 5 + Table 1**: coefficient paths and the model at $\hat{s}=0.44$ on the real prostate data (available from the *Elements of Statistical Learning* website). ✅
4. **Simulation example 1 (Table 3)**: compare the MSE of least squares, lasso-CV and ridge over 50 replicates of the model $\beta = (3, 1.5, 0, 0, 2, 0, 0, 0)$, correlation $\rho^{|i-j|}$ with $\rho = 0.5$, $\sigma = 3$. ⚠️ runs, not published.
5. Validate the solver against `sklearn.linear_model.Lasso`. ✅ to $8\times10^{-13}$ over the whole path, and against LARS with no conversion of conventions.

## What implementing it made clear

Things that only become clear after doing it, and that reading did not give:

**Shrinkage is the toll, not the goal.** The $L_1$ ball is exactly the convex hull of the 1-sparse points $\{\pm t\,e_j\}$: one takes the models one actually wants and convexifies in order to be able to solve. The vertices survive — hence the zeros — but the optimum can land in the interior of a face, and that is the whole of the bias. Shrinking is what is paid for convexity, not what is sought; the acronym puts *Shrinkage* ahead of *Selection* and misleads. The proof is that the later literature (garotte, adaptive lasso, SCAD) is devoted to removing that shrinkage without losing the selection.

**"Shrinkage" is not even well defined outside the orthonormal design.** Eq. 3 and the four curves of Fig. 1 are the portrait of a special case. With correlated predictors there exists no function $h$ with $\hat\beta_j = h(\hat\beta_j^{OLS})$: over 140 random designs, for $\hat\beta_j^{OLS}\approx 2$ the lasso spreads values between 0 and 2.84. And with $p\ge3$ a coefficient can **grow** as the budget tightens (checked: from 0.95 to 1.81), the lasso analogue of the ridge upturn at $\rho>1/2$. The only thing that shrinks is the scalar $\sum_j|\beta_j|$.

**What does survive is a conditional closed form.** On the active set $A$ with signs $s_A$, KKT gives $\hat\beta_A = \hat\beta^{\mathrm{ols}(A)} - \lambda (X_A^\top X_A)^{-1}s_A$: least squares refitted on $A$, displaced. It is not algebra that is missing, but **combinatorics** — which $A$ and which signs. From it also follows, now proved, the piecewise linearity of the paths, which is what LARS exploits. It is in section 14 of the [derivations](../implementations/1996-tibshirani-lasso/DERIVATIONS.md).

**What the paper proves and what it only asserts.** Selection and convexity are pure construction, and they hold. **Stability**, by contrast, is left to the simulations — when it could have been proved: the lasso is the projection of $\hat\beta^{OLS}$ onto a convex set in the $X^\top X$ metric, and projections onto convex sets are non-expansive, so it is *provably* at least as stable as least squares. Subset selection projects onto a union of subspaces, which is not convex, and that is why it jumps. Stability does not come from shrinking: it comes from the feasible set being convex, which is exactly what it shares with ridge.

**The structural half holds; the empirical half cracks.** We reproduce Table 1 exactly and the figures, but the GCV of Eq. 10 does not choose the paper's $\hat s = 0.44$ under any reasonable reading (we get 0.69), and in Table 3 there is no $\sigma$ that gives both the error levels and the structure of the selected models. Plus two errata found while deriving: Eq. 6 needs a lower limit the paper does not give, and the Stein risk formula prints `max` where `min` belongs. None of this touches the conclusion the paper actually draws, which is an **ordering** of methods, and that one does hold.

## Connections

- First review in the repo — no internal connections yet.
- Natural future ones from the [ROADMAP](../ROADMAP.md): Efron (1979, bootstrap — used here for the standard errors), Bottou et al. (2018, optimisation — the lasso as a non-differentiable convex problem), Blei et al. (2017, variational inference — the Bayesian reading with a Laplace prior), Friedman (2001, gradient boosting — the other route to implicit variable selection).
