# Portfolio Selection

**Authors:** Harry Markowitz (The Rand Corporation) · **Year:** 1952 · **Venue:** The Journal of Finance, 7(1), 77–91 · **Link/DOI:** [10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x) · [JSTOR 2975974](https://www.jstor.org/stable/2975974) · [PDF (HKUST course)](https://www.math.hkust.edu.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf)
**Field:** financial economics / optimisation · **Read:** 2026-07-29

## TL;DR

Markowitz founds **modern portfolio theory (MPT)** by turning the proverb "don't put all your eggs in one basket" into an optimisation problem. He rejects the rule of *maximising discounted expected return* (which never implies diversification: it puts everything in the single highest-valued asset) and proposes the **mean-variance (E-V) rule**: expected return $E$ desirable, variance $V$ undesirable, and the investor should choose an **efficient** portfolio — minimum $V$ for each $E$, maximum $E$ for each $V$. The key: portfolio risk is governed by the **covariances**, not by the individual variances; that is why diversification works, and why only "the right kind" works (among weakly correlated assets). The paper is deliberately **geometric**: it solves the 3- and 4-asset cases with isomean and isovariance curves, without giving the general algorithm. Nobel Prize in Economics, 1990.

## Context and motivation

Before 1952 there was no formal theory of *how to combine* assets. Markowitz separates the problem into **two stages** (p. 77):

1. From observation and experience one forms **beliefs** about the future behaviour of the assets ($\mu_i$, $\sigma_{ij}$).
2. From those beliefs one chooses the portfolio.

**This paper deals only with the second stage**; it says so on opening and repeats it on closing (p. 91). The first — how to estimate $\mu_i$ and $\sigma_{ij}$ — is explicitly left out.

**The attack on the discounted-value rule (pp. 77–78).** If $R = \sum_i X_i R_i$ with $R_i$ the discounted return of asset $i$, then $R$ is a weighted average of the $R_i$ with non-negative weights summing to 1. Maximising it requires $X_i = 1$ for the asset with the largest $R_i$ (and if there are ties, any split among them does equally well). Markowitz's conclusion: *"In no case is a diversified portfolio preferred to all non-diversified portfolios"*. Since diversification is both observed and sensible, the rule must be rejected **both as a descriptive hypothesis and as a normative maxim**.

**The attack on "diversify and trust the law of large numbers" (p. 79).** There is an intermediate rule — spread across the assets of maximum expected return, trusting that the LLN will bring the realised return close to the expected one — which also fails:

> *"This presumption, that the law of large numbers applies to a portfolio of securities, cannot be accepted. The returns from securities are too intercorrelated. Diversification cannot eliminate all variance."*

Correlation puts a **floor** on the risk reduction achievable by diversification. This is the crack through which all the later theory of systematic risk enters.

## Methodology

**The object (p. 81).** With $N$ assets, $X_i$ the fraction of wealth in asset $i$, $\mu_i = E(R_i)$, and $\sigma_{ij} = E[(R_i - \mu_i)(R_j - \mu_j)] = \rho_{ij}\sigma_i\sigma_j$ the covariance (with $\sigma_{ii}$ the variance). The portfolio return $R = \sum_i R_i X_i$ is a weighted sum of random variables, with:

$$
E = \sum_{i=1}^N X_i \,\mu_i, \qquad V = \sum_{i=1}^N \sum_{j=1}^N \sigma_{ij}\, X_i X_j.
$$

The $R_i$ are random; the $X_i$ are **not** — the investor sets them. Constraints: $\sum_i X_i = 1$ and **$X_i \ge 0$** (the paper explicitly excludes short selling).

**The E-V rule (p. 82, Fig. 1).** Of all attainable combinations $(E,V)$, the investor picks an **efficient** one: minimum $V$ for given $E$, or maximum $E$ for given $V$. Fig. 1 draws the attainable set as a region and marks its efficient frontier.

> ⚠️ **Watch the axes:** the paper plots $V$ vertically and $E$ horizontally (Figs. 1 and 6), the reverse of the modern convention ($\sigma$ horizontal, $E$ vertical). When reproducing the figures one has to decide whether to be faithful to the original or to translate.

**The 3-asset case (p. 83).** The model reduces to the paper's numbered equations:

$$
\text{1)}\ E = \sum_{i=1}^{3} X_i\mu_i \qquad \text{2)}\ V = \sum_{i=1}^{3}\sum_{j=1}^{3} X_iX_j\sigma_{ij} \qquad \text{3)}\ \sum_{i=1}^{3}X_i = 1 \qquad \text{4)}\ X_i \ge 0
$$

Substituting 3′) $X_3 = 1 - X_1 - X_2$ moves everything into **two-dimensional geometry** in $(X_1, X_2)$. In particular (Eq. 1′):

$$
E = \mu_3 + X_1(\mu_1 - \mu_3) + X_2(\mu_2 - \mu_3).
$$

The attainable set is the **triangle $abc$** (the simplex) of Fig. 2.

**Isomean and isovariance curves (p. 84).** Markowitz defines the *isomean curve* as the locus of portfolios with given $E$, and the *isovariance line* as that of portfolios with given $V$. From the formulas:

- The **isomeans are parallel straight lines** ($E$ is linear in $X$). Solving Eq. (1′):
$$
X_2 = \frac{E - \mu_3}{\mu_2 - \mu_3} - \frac{\mu_1 - \mu_3}{\mu_2 - \mu_3}X_1,
$$
whose **slope does not depend on $E$** (only the intercept changes) — hence the parallelism.
- The **isovariances are concentric ellipses**, centred at the point $\hat{X}$ that **minimises $V$**. The variance grows as one moves away from $\hat{X}$.

> A fine technical detail (footnote 12, p. 89): for the isovariances to be **ellipses** it is *necessary and sufficient* that no two distinct portfolios have perfectly correlated returns. If there are, the shape degenerates.

**The efficient set and the *critical line* (p. 85, Figs. 2–3).** For a given $E$, the best portfolio is the point where the **isomean line is tangent to an isovariance ellipse**; Markowitz calls it $\hat{X}(E)$. As $E$ varies, those points trace a curve which — he asserts, omitting the algebra — **is a straight line**: the ***critical line* $l$**, passing through $\hat{X}$.

The efficient set is built by following that logic **inside the triangle**:

- If $\hat{X}$ **falls inside** the attainable set (Fig. 2), $\hat{X}$ is efficient and the efficient set starts there, runs along the critical line until it hits an edge, and continues along the edge to the point of maximum $E$.
- If $\hat{X}$ **falls outside** (Fig. 3), one starts at the attainable point of minimum variance (on an edge), moves until crossing the critical line, follows it to another edge, and finishes at the vertex of maximum $E$.

The general result (p. 87, Fig. 4 for 4 assets in the tetrahedron): **the efficient set is always a polygonal chain — a series of connected segments**, with one end at the minimum-variance portfolio and the other at the maximum-expected-return one.

> **An important historical nuance:** the term *critical line* and its geometric role **are indeed in this 1952 paper** (including footnote 10, which sketches how to traverse the critical lines of the subspaces $X_i = 0$ in the general case). What is **not** there is the systematic algorithm: that comes in **Markowitz (1956)**, *"The optimization of a quadratic function subject to linear constraints"*. Today the problem is solved trivially as a **QP**.

**The frontier in $(E,V)$ space (p. 87, Figs. 5–6).** Over the plane of the $X$'s, $E$ is a **plane** and $V$ a **paraboloid**. Restricted to the efficient set (which is polygonal), the section of the plane gives straight segments and that of the paraboloid gives **parabolic arcs**. Therefore, plotting $V$ against $E$ for efficient portfolios yields **a series of connected parabolic segments** (Fig. 6) — not a single parabola, precisely because of the $X_i \ge 0$ constraints becoming active one by one.

**Why diversification works, and the "right kind" (p. 89).** Here is the most-cited conceptual contribution:

> *"Not only does the E-V hypothesis imply diversification, it implies the 'right kind' of diversification for the 'right reason'."*

With two assets, $V = X_1^2\sigma_1^2 + X_2^2\sigma_2^2 + 2X_1X_2\rho_{12}\sigma_1\sigma_2$: if $\rho_{12} < 1$, the portfolio variance falls below the weighted average of the individual ones. Hence the famous example: a portfolio of **sixty railway securities is not as well diversified** as one of the same size spread across railways, utilities, mining and manufacturing, because firms in the same sector tend to do badly at the same time. The adequacy of diversification **does not depend on the number of assets**, but on avoiding assets with **high covariances among themselves**.

**Robustness to the risk measure (p. 89).** If instead of $V$ the investor used the standard deviation $\sigma = \sqrt{V}$ or the coefficient of dispersion $\sigma/E$, **their choice would still lie in the same efficient set** (these are monotone transformations of $V$ at fixed $E$).

**Justification of the rule (pp. 90–91).** Markowitz does **not** derive E-V from expected utility axioms (no quadratic utility here — that is later, in his 1959 book). He defends it as *"a working hypothesis and a working maxim"* for institutions that regard return as good, risk as bad and gambling as to be avoided. And he bounds its scope with the **third moment** $M_3$ (footnote 13): if utility were $U(E,V,M_3)$ with $\partial U/\partial M_3 \neq 0$, the investor would accept some fair bets. That is why E-V describes the behaviour of **"investment"** and not of **"speculation"**.

## Main results

- **The maximum-discounted-return rule is discarded**: it never implies diversification, neither as description nor as norm. The E-V rule does imply it, and for the right reason (the covariances).
- **A complete geometric characterisation** of the efficient set for 3 and 4 assets via isomeans (parallel lines) and isovariances (concentric ellipses), with the *critical line* as the axis of the argument.
- **The efficient set is a polygonal chain** (connected segments) in portfolio space, and a **series of parabolic arcs** in the $(E,V)$ plane, for any number of assets.
- **The law of large numbers does not eliminate risk**: returns are too intercorrelated; diversification has a floor.
- **The relevant risk is covariance**, not individual variance — the direct seed of the CAPM's $\beta$ a decade later.

## Strengths and limitations

**Strengths:** it turns a qualitative intuition into a well-defined convex optimisation problem; it identifies **covariance** as the central object of risk (from which CAPM, APT and *factor investing* all derive); the geometric exposition is exceptionally clear and directly reproducible; it is honest about its scope — it says what it does not do (stage 1, the general $n$ case, dynamics) and announces the future general treatment; and it founds an entire field.

**Limitations (some acknowledged by the author, others visible in hindsight):**

- **Self-imposed and declared (p. 79):** *(1)* it does not derive results analytically for $N$ assets, only geometrically for 3 and 4; *(2)* it assumes **static probabilistic beliefs** — a single-period model, with no rebalancing or transaction costs.
- **It depends on $\mu_i$ and $\sigma_{ij}$ as inputs.** Markowitz **is aware** and says so at the end (p. 91): he *tentatively* suggests using observed past moments, but adds that *"better methods, which take into account more information, can be found"* and that a probabilistic reformulation of security analysis is needed — *"another story"*. The modern criticism is that optimisation **amplifies** estimation error (*error maximization*, Michaud 1989): extreme, unstable weights, poor out of sample.
- **Variance is symmetric**: it penalises the upside and the downside equally. The paper bounds this via $M_3$ (skewness), but the alternative Markowitz himself would later prefer — the **semivariance** — **does not appear here** (it is from 1959).
- **No risk-free asset.** Adding one (Tobin, 1958) gives the **two-fund separation theorem** and the *capital market line*; on that Sharpe (1964) builds the CAPM. This paper is the foundation, not the building.
- **No short selling** ($X_i \ge 0$), which is a reasonable modelling choice but restricts the space of portfolios and is exactly what makes the frontier piecewise polygonal.
- **Heavy tails and higher moments**: the $(E,V)$ structure ignores kurtosis and (apart from the note on $M_3$) skewness, which Mandelbrot and Fama would document as essential in real markets.

## Implementation ideas

The whole core is linear algebra reproducible with numpy/scipy. A proposal in pieces (in the style of the lasso):

1. **Efficient frontier in closed form** (relaxed case, only $\sum X_i = 1$, allowing shorts) via Lagrange multipliers. With $A = \mathbf 1^\top\Sigma^{-1}\mathbf 1$, $B = \mathbf 1^\top\Sigma^{-1}\boldsymbol\mu$, $C = \boldsymbol\mu^\top\Sigma^{-1}\boldsymbol\mu$, $D = AC - B^2$:
$$
V(E) = \frac{AE^2 - 2BE + C}{D}, \qquad \mathbf w_{\min} = \frac{\Sigma^{-1}\mathbf 1}{A}.
$$
It is a **parabola** in $(E,V)$ — consistent with the paper's Fig. 6, which without sign constraints would be a single arc. ~20 lines.
2. **Frontier with $X_i \ge 0$** as a **QP** (`scipy.optimize` or `cvxpy`): min $\mathbf w^\top\Sigma\mathbf w$ s.t. $\boldsymbol\mu^\top\mathbf w = E^\*$, $\sum w_i = 1$, $w_i \ge 0$; sweep $E^\*$. **Verify the paper's thesis**: that the optimal weights trace a **polygonal chain** and that $(E,V)$ gives **connected parabolic arcs**, detecting the kinks where some $w_i$ hits 0.
3. **Reproduce Fig. 2** (the pedagogical jewel): the simplex triangle in $(X_1,X_2)$, the **isomean lines**, the **isovariance ellipses**, the centre $\hat X$, the ***critical line*** and the efficient set in bold. And **Fig. 3** as a second case (with $\hat X$ outside the triangle).
4. **A demonstration of the diversification floor**: $V$ of the equally weighted portfolio against $N$, comparing uncorrelated assets ($V \to 0$) with correlated ones ($V \to$ the average covariance). It is the refutation of the LLN on p. 79, in one figure.
5. **The "sixty railways" example (p. 89)**: two portfolios of the same size, one within a sector (high correlation) and one across sectors, showing that the second dominates. The "*right kind of diversification*" thesis turned into code.
6. **Real data**: a handful of tickers, estimate $\boldsymbol\mu$ and $\Sigma$, trace the frontier, mark the minimum-variance and tangency portfolios.
7. **Resampled frontier** *(extension — links with the bootstrap)*: resample returns with the bootstrap, recompute the frontier $B$ times and visualise its **instability**. It is exactly the concern Markowitz leaves open on p. 91, with Efron's tool.
8. **Risk-free asset → capital market line** *(extension, Tobin 1958)*: tangency portfolio and capital allocation line; the antechamber of the CAPM.

Validate against `PyPortfolioOpt` or against the analytic minimum-variance solution.

## Connections

- **[Efron (1979), Bootstrap](1979-efron-bootstrap.md):** the bootstrap is the standard route to injecting uncertainty into the efficient frontier — notoriously unstable — by resampling returns and obtaining a distribution of optimal portfolios (Michaud). The bridge is exact: Markowitz closes the paper (p. 91) saying that estimating $\mu_i$ and $\sigma_{ij}$ from the past is only a tentative suggestion and that better methods are needed; the bootstrap is one of the answers.
- **[Tibshirani (1996), Lasso](1996-tibshirani-lasso.md):** penalising the weights with $L_1$ (Brodie et al. 2009, *"Sparse and stable Markowitz portfolios"*) produces **sparse and stable** portfolios, attacking the extreme-weights problem. The same $L_1$ diamond geometry that annihilates coefficients annihilates positions. A curious note: both problems are **constrained QPs**, and both have **piecewise linear** solutions in their parameter (Markowitz's *critical line* ↔ the lasso's *regularization path* via LARS).
- **Future ones from the [ROADMAP](../ROADMAP.md), quant track:** Tobin (1958, two-fund separation), Sharpe (1964, CAPM — the equilibrium built on this frontier), Black & Scholes (1973), Kalman (1960, sequential estimation of $\boldsymbol\mu$/$\Sigma$), and López de Prado (2016, *Hierarchical Risk Parity* — avoiding the inversion of $\Sigma$, attacking the same Achilles heel).
