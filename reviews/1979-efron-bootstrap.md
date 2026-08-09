# Bootstrap Methods: Another Look at the Jackknife

**Authors:** Bradley Efron · **Year:** 1979 · **Venue:** The Annals of Statistics, 7(1), 1–26 (1977 Rietz Lecture) · **Link/DOI:** [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552) · [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full)
**Field:** statistics / ML · **Read:** 2026-07-23

## TL;DR

Efron introduces the **bootstrap**: to estimate the sampling distribution of a statistic $R(\mathbf{X}, F)$ without analytic formulas, replace the unknown distribution $F$ by the **empirical distribution** $\hat{F}$ (mass $1/n$ at each observed datum) and study $R(\mathbf{X}^*, \hat{F})$, where $\mathbf{X}^*$ is resampled **with replacement** from the data. The paper presents it as a generalisation of the jackknife — indeed it proves that the jackknife is the **linear approximation (delta method)** of the bootstrap — and validates it on the mean, variance, median, error rates in discriminant analysis, the Wilcoxon statistic and regression. It is the founding paper of all resampling-based computational statistics.

## Context and motivation

The Quenouille–Tukey jackknife estimates the bias and variance of a statistic by recomputing it $n$ times, each time dropping one observation. It works well for "smooth" statistics (mean, correlation) but **fails for the median** (its variance estimator is not even consistent) and is confusing in unbalanced situations (two samples, regression). Efron looks for a more primitive and general method of which the jackknife is a special case. The answer: instead of approximating linearly by removing points, **resample directly** from $\hat{F}$ and let the computer compute the distribution by brute force.

## Methodology

**General problem.** Given a sample $\mathbf{X} = (X_1,\dots,X_n)$ with $X_i \sim_{\text{iid}} F$ (unknown) and a statistic $R(\mathbf{X}, F)$ — typically $R = t(\mathbf{X}) - \theta(F)$, the error of an estimator relative to the true parameter — estimate the **sampling distribution** of $R$ from the data.

**The bootstrap recipe (Sec. 2).** Three steps:

1. Build the empirical distribution $\hat{F}$: mass $1/n$ at each $x_i$.
2. Draw a **bootstrap sample** by resampling with replacement (Eq. 2.4):
$$X_i^* \sim_{\text{iid}} \hat{F}, \qquad \mathbf{X}^* = (X_1^*,\dots,X_n^*).$$
3. Approximate the distribution of $R(\mathbf{X}, F)$ by the **bootstrap distribution** of (Eq. 2.5):
$$R^* = R(\mathbf{X}^*, \hat{F}).$$

The justification is **Fisher consistency**: any reasonable estimator of the distribution of $R$ must be right when $F = \hat{F}$, and $\hat{F}$ is the centre of the class of plausible $F$ given that we have observed $\mathbf{X} = \mathbf{x}$.

**Minimal example — the mean (Eqs. 2.6–2.8).** With $R = \bar{X} - \theta(F)$, the bootstrap variance reproduces the classical formula:
$$\mathrm{Var}_*(\bar{X}^* - \bar{x}) = \frac{\bar{x}(1-\bar{x})}{n}.$$

**Three ways of computing the bootstrap distribution (Sec. 2, the key of the paper):**

- **Method 1 — direct theoretical computation.** When it can be done by hand (mean, variance, median).
- **Method 2 — Monte Carlo.** Generate $N$ bootstrap samples $\mathbf{x}^{*1},\dots,\mathbf{x}^{*N}$ and use the histogram of $R(\mathbf{x}^{*j}, \hat{F})$ as the approximation. **This is what is used universally today**; the rest of the modern bootstrap is this.
- **Method 3 — Taylor expansion** of $R$ about $\hat{F}$. It turns out to be **exactly the infinitesimal jackknife** (Sec. 5).

**Variants of $\hat{F}$ (Sec. 3).** The bootstrap does not force the use of the empirical distribution:
- **Smoothed bootstrap (Eq. 3.11):** instead of resampling discrete points, add noise: $X_i^* = \bar{x} + c\,[\,x_{I_i} - \bar{x} + \hat{\sigma} Z_i\,]$ with $Z_i$ of mean 0 and fixed variance. Equivalent to resampling from an $\hat{F}$ with a kernel window.
- **Symmetric bootstrap (Eq. 3.8):** if $F$ is assumed symmetric, reflect $\hat{F}$ about the median ($\hat{F}_{\text{SYM}}$).

In a Monte Carlo experiment with the median ($n=13$, $\mathcal{N}(0,1)$), the conclusion is sober: **the simplest bootstrap (3.6) does almost as well as the smoothed and symmetric versions** (Table 1). The bootstrap estimates $E_F R = 0.95$ reasonably with only $n=13$.

> **Later note.** Two numbers in that experiment do not survive checking. Eq. (3.12) as printed lacks a factor $\sqrt n$ — without it the tabulated quantity drifts to 0 with $n$ and Table 1 cannot be reproduced — and the true value is $E_F R = 0.9822 \pm 0.0012$, not $0.95$. Neither affects the conclusion, since every column of Table 1 moves together. See the [implementation](../implementations/1979-efron-bootstrap/).

**Relation to the jackknife (Sec. 5).** Writing $P_i^* = N_i^*/n$ (the proportion of times $x_i$ appears in the bootstrap sample) and expanding $R(\mathbf{P}^*)$ in a Taylor series about $\mathbf{P}^* = \mathbf{e}/n$:
$$R(\mathbf{P}^*) \doteq R(\mathbf{e}/n) + (\mathbf{P}^* - \mathbf{e}/n)\mathbf{U} + \tfrac{1}{2}(\mathbf{P}^* - \mathbf{e}/n)\mathbf{V}(\mathbf{P}^* - \mathbf{e}/n)'$$
one obtains (Eqs. 5.8–5.11) the expressions for bias and variance:
$$\mathrm{Bias}_F\,\theta(\hat{F}) \approx \frac{1}{2n}\bar{V}, \qquad \mathrm{Var}_F\,\theta(\hat{F}) \approx \sum_{i=1}^n U_i^2 / n^2.$$
These coincide with Jaeckel's **infinitesimal jackknife**; the ordinary jackknife replaces the derivatives $U_i = \partial R/\partial P_i$ by finite differences (Eq. 5.12): $\tilde{U}_i = (n-1)(\bar{R}^* - R^*_{(i)})$. **Moral: jackknife = linearised bootstrap.** That is why the jackknife fails for the median (it is not smooth: the quadratic extrapolation formulas do not hold, Remark J).

**Parametric bootstrap (Remark K).** If the family of $F$ is known (e.g. normal), use the **parametric MLE** as $\hat{F}$ instead of the empirical distribution. For the normal, the parametric bootstrap of a probability $\mathrm{Prob}\{\bar{X} \in [a,b]\}$ coincides with the Edgeworth approximation when $n \gtrsim 20$.

**Transformations and pivotal quantities (Remarks B, D–F).** The seed of modern confidence intervals. Any quantile of the bootstrap distribution of $R^*$ maps to the corresponding quantile of $S^* = g(R^* + \hat\theta) - g(\hat\theta)$ (Eq. 8.1): **the bootstrap is equivariant under monotone transformations**. Fig. 1 illustrates this with the correlation of 9 data pairs ($\hat\rho = 0.945$): the bootstrap distribution of $\hat\rho^* - \hat\rho$ is skewed, that of $\tanh^{-1}\hat\rho^* - \tanh^{-1}\hat\rho$ (Fisher's transformation) is almost symmetric and pivotal. **An important warning (Remark E):** the bootstrap gives *frequency* statements, not *likelihood* ones; inference problems remain that no amount of bootstrap precision resolves. The confidence interval methods (percentile, BCa) are **not in this paper** — they arrive in Efron (1981, 1987).

## Main results

- **Median (Sec. 3):** the bootstrap correctly estimates (asymptotically) the variance $n E_*(R^*)^2 \to 1/4f^2(\theta)$, **a case in which the jackknife is inconsistent** ($n\,\mathrm{Var}(R) \to \tfrac{1}{4f^2}\cdot[\chi_2^2/2]^2$, with mean 2 and variance 20). It is the flagship argument for the bootstrap. *(Later note: that limit law is not the right one. The jackknife variance of the median is driven by the **two** spacings flanking $x_{(m)}$, so the limit is $\tfrac{1}{4f^2}[\chi_4^2/4]^2$, of mean 1.5 and variance 5.25 — confirmed quantile by quantile in the [implementation](../implementations/1979-efron-bootstrap/). The inconsistency, which is the point, stands.)*
- **Error rates in discriminant analysis (Sec. 4, Table 2):** the bootstrap estimates both the bias ($E_* R^*$) and the standard deviation ($SD_*(R^*)$) of the classification error, and its estimator of $R$ has **~3× less variability than cross-validation / leave-one-out** (SD 0.078 vs 0.026 for the same bias).
- **Wilcoxon (Sec. 6):** the bootstrap reproduces the classical formula for the variance of the Wilcoxon statistic (Eq. 6.7).
- **Regression (Sec. 7):** resampling the **residuals** $\hat\epsilon_i$ gives $\mathrm{Cov}_*\hat\beta^* = \hat\sigma^2 G^{-1}$ (Eq. 7.7), the classical formula; the bootstrap "symmetrises" the data automatically, something the jackknife needs to be told to do by hand.
- **Cost (Remark A):** Method 2 costs ~$N$ times the original computation. In 1977, $N=1000$ for $n=20$ cost \$4 on Stanford's 370/168. Today it is free — hence the bootstrap's explosion in the 1980s.

## Strengths and limitations

**Strengths:** a simple and universal idea (replace $F$ by $\hat{F}$) that unifies the jackknife, the delta method and resampling in one framework; exemplary honesty (it delimits where the plain bootstrap suffices and where the smoothed version barely gains); an anticipatory vision of almost everything that would follow (parametric, smoothed, pivotality, transformations); and the asymptotic argument via the multinomial (Remarks G–H) is elegant and general.

**Limitations (some by design, others of their time):**
- It assumes **iid data**. There is no bootstrap for dependent data (time series): the *block bootstrap* arrives with Künsch (1989) and Hall. Nothing in the paper covers that case.
- It **does not develop confidence intervals.** Remarks D–F show that using $\hat\theta - \theta$ as a pivot is problematic, but the solution (percentile, $BC_a$) comes later (Efron 1981, 1987; Efron & Tibshirani 1993).
- The bootstrap **fails at the boundaries**: the maximum of a uniform, parameters on the boundary, heavy-tailed distributions without variance. The paper does not discuss this (it would be understood later).
- Method 3 (Taylor) "looks suspicious" because the dimension grows with $n$ (Remark H) — Efron justifies it but it remains technically delicate.
- The choice of $N$ (number of Monte Carlo replicates): the paper notes that going from 100 to 10000 barely improves the bias, but gives no systematic guidance (that would come with the $BC_a$ theory).

## Implementation ideas

> **Later note (2026-08-09).** The [implementation](../implementations/1979-efron-bootstrap/) is done and closed at four of the paper's eight sections. Ideas 1 and 4 are complete, and two things not on this list turned out to matter more than most of it: the **regression of Sec. 7**, where the paper's thesis about the jackknife is settled a second time and by a different route, and **Fig. 1** with Remark B, which is the only real data in the paper. Idea 3 (the variants of $\hat F$) is deliberately skipped: the paper's own Table 1 shows all seven columns agreeing, so reproducing them confirms a null result. Idea 5 became Remark D instead — the *diagnosis* is in the paper and is worth having; the cures (percentile, $BC_a$) are 1981–1987 and stay out. Ideas 2 and 6, and the rest of the post-1979 family, are mapped in the implementation's README with the hook each would attach to. Three errata of the paper turned up along the way, each noted above in its place.

The bootstrap is one of the most rewarding papers to implement: it is all Monte Carlo with numpy. What was asked for (several kinds of bootstrap) fits together as a family, **marking what is in the paper and what is a modern extension**:

1. **Non-parametric bootstrap (Method 2)** — the core of the paper. Resampling with replacement + histogram of $R^*$. ~15 lines. Validate by reproducing Eq. (2.8) for the mean and the median case (Sec. 3) where the jackknife fails.
2. **Parametric bootstrap (Remark K)** — fit a family (e.g. normal) and resample from it. Compare against the non-parametric one on normal data.
3. **Smoothed (Eq. 3.11)** and **symmetric (Eq. 3.8) bootstrap** — the paper's two variants of $\hat{F}$; reproduce Table 1 qualitatively (all three do about equally well for the median).
4. **Jackknife vs bootstrap (Sec. 5)** — implement the jackknife variance and **verify numerically that it is the linear approximation of the bootstrap**, and that it diverges for the median. It is the paper's central thesis turned into code.
5. **Confidence intervals** *(post-1979 extension, to be flagged in the README)*: **percentile** (the empirical quantiles of $R^*$; sketched in Remark D), **basic/pivotal** (reflecting the percentile), and **$BC_a$** (bias-corrected and accelerated, Efron 1987). Reproducing Fig. 1 (correlation of 9 pairs, Fisher transformation) is the natural vehicle.
6. **Block bootstrap** *(extension, Künsch 1989 — not in the paper)*: resample contiguous blocks for dependent data. Useful for the master's time series course and for quant work. Make clear in the README that it is a later addition.

Data: the correlation of Fig. 1 comes from the paper itself (9 pairs, Remark B). For the rest, synthetic simulations as in the paper.

## Connections

- **[Tibshirani (1996), Lasso](1996-tibshirani-lasso.md):** the lasso paper uses the bootstrap for the standard errors of the coefficients (its Eq. 7), and points out its failure — it gives variance 0 for precisely the annihilated coefficients. A direct connection: the bootstrap is the uncertainty tool that resists the lasso.
- **[Markowitz (1952), Portfolio Selection](1952-markowitz-portfolio-selection.md):** the bootstrap is the standard route to injecting uncertainty into the efficient frontier (resample returns → distribution of the optimal portfolios), which is notoriously unstable. A good bridge between the two.
- **Future ones from the [ROADMAP](../ROADMAP.md):** Breiman (2001, Random Forests — *bagging* is bootstrap + aggregation), Bottou et al. (2018, the resampling/optimisation connection), and the time series course (block bootstrap).
