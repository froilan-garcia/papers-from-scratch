# Mean-variance portfolio selection — Markowitz (1952)

Implementation of the E-V rule from *Portfolio Selection* (Markowitz, 1952).
See the [review](../../reviews/1952-markowitz-portfolio-selection.md) for the
paper's context and results.

Built incrementally, same as the [Lasso](../1996-tibshirani-lasso/). Current status:

| # | Piece | Status |
|---|-------|--------|
| 0 | Fundamentals — deriving the frontier by Lagrange (below) | ✅ notes |
| 1 | Closed-form efficient frontier (no sign constraint) | ⬜ pending |
| 2 | Fig. 2 — simplex triangle, isomean lines, isovariance ellipses, critical line | ⬜ pending |
| 3 | Fig. 6 — QP frontier with `w >= 0`; verify the connected parabola segments | ⬜ pending |
| 4 | The diversification floor — refuting the law of large numbers (p. 79) | ⬜ pending |
| 5 | Real data + bootstrap-resampled frontier (links to Efron 1979) | ⬜ pending |

## Fundamentals — the efficient frontier by Lagrange multipliers

The paper solves the 3- and 4-asset cases **geometrically** and says explicitly
(p. 79) that it does not derive the $n$-asset case analytically. That derivation —
standard today — is Piece 1. It is worth having at hand before writing any code,
because the result (a **parabola** in $(E,V)$) is exactly what Fig. 6 of the paper
draws piecewise.

**Statement.** With $\Sigma$ the covariance matrix ($p \times p$, symmetric positive
definite), $\boldsymbol\mu$ the vector of expected returns and $\mathbf 1$ the vector
of ones, we look for the weights $\mathbf w$ minimising the variance for a target
return $E$:

$$
\min_{\mathbf w} \ \tfrac{1}{2}\mathbf w^\top \Sigma \mathbf w
\quad \text{s.t.} \quad \boldsymbol\mu^\top \mathbf w = E, \quad \mathbf 1^\top \mathbf w = 1.
$$

> **Note:** the paper's $w_i \ge 0$ constraint is **relaxed** here. That is what makes
> the closed form possible; with the sign constraint one has to go to a QP (Piece 3)
> and the kinks of the polygonal chain appear. The $\tfrac12$ is cosmetic, so that the
> derivative comes out clean.

**Lagrangian** (here the constraints are *equalities*, so this is the ordinary
Lagrange — unlike the lasso, where $\lambda$ is set by hand):

$$
\mathcal L = \tfrac12 \mathbf w^\top\Sigma\mathbf w
- \lambda(\boldsymbol\mu^\top\mathbf w - E) - \gamma(\mathbf 1^\top\mathbf w - 1).
$$

Differentiating in $\mathbf w$ and setting to zero:

$$
\Sigma\mathbf w - \lambda\boldsymbol\mu - \gamma\mathbf 1 = \mathbf 0
\quad\Longrightarrow\quad
\boxed{\ \mathbf w = \Sigma^{-1}(\lambda\boldsymbol\mu + \gamma\mathbf 1)\ }
$$

The optimal weights are a **linear combination of two fixed portfolios**,
$\Sigma^{-1}\boldsymbol\mu$ and $\Sigma^{-1}\mathbf 1$. This is already, in embryo,
Tobin's (1958) **two-fund separation theorem**: every efficient portfolio is obtained
by mixing two.

**The four scalars.** Substituting into the constraints, the same three quantities
always appear:

$$
A = \mathbf 1^\top\Sigma^{-1}\mathbf 1, \qquad
B = \mathbf 1^\top\Sigma^{-1}\boldsymbol\mu, \qquad
C = \boldsymbol\mu^\top\Sigma^{-1}\boldsymbol\mu, \qquad D = AC - B^2.
$$

The two constraints become a $2\times 2$ system in $(\lambda, \gamma)$:

$$
\begin{pmatrix} C & B \\ B & A \end{pmatrix}
\begin{pmatrix} \lambda \\ \gamma \end{pmatrix} =
\begin{pmatrix} E \\ 1 \end{pmatrix}
\quad\Longrightarrow\quad
\lambda = \frac{AE - B}{D}, \qquad \gamma = \frac{C - BE}{D}.
$$

**The frontier.** The trick for the variance is to expand nothing: using the
first-order condition $\Sigma\mathbf w = \lambda\boldsymbol\mu + \gamma\mathbf 1$,

$$
V = \mathbf w^\top\Sigma\mathbf w = \mathbf w^\top(\lambda\boldsymbol\mu + \gamma\mathbf 1)
= \lambda\underbrace{\boldsymbol\mu^\top\mathbf w}_{=\,E} + \gamma\underbrace{\mathbf 1^\top\mathbf w}_{=\,1}
= \lambda E + \gamma.
$$

Substituting:

$$
\boxed{\ V(E) = \frac{AE^2 - 2BE + C}{D}\ }
$$

**A parabola in $(E,V)$** — and therefore a **hyperbola** in $(\sigma, E)$, which is
how it is drawn today. Consistent with Fig. 6 of the paper: there it comes out
piecewise *because* Markowitz does impose $w_i \ge 0$.

**Minimum-variance portfolio.** Differentiating: $V'(E) = (2AE - 2B)/D = 0$, hence

$$
E_{\min} = \frac{B}{A}, \qquad V_{\min} = \frac{1}{A}, \qquad
\mathbf w_{\min} = \frac{\Sigma^{-1}\mathbf 1}{A}.
$$

Note that $\mathbf w_{\min}$ **does not depend on $\boldsymbol\mu$**: only on $\Sigma$.
That is the practical reason why the minimum-variance portfolio is far more robust
than the tangency one — expected returns are precisely the worst-estimated input
(see Michaud's criticism in the ROADMAP, block Q2).

**Signs.** With $\Sigma$ positive definite, so is $\Sigma^{-1}$, hence $A > 0$ and
$C > 0$. And $D = AC - B^2 > 0$ by Cauchy–Schwarz in the inner product
$\langle \mathbf x, \mathbf y\rangle = \mathbf x^\top\Sigma^{-1}\mathbf y$, with
equality only if $\boldsymbol\mu \propto \mathbf 1$ (all assets with the same expected
return — the degenerate case of footnote 9 of the paper, where the isomeans cease to
be defined). The parabola therefore **opens upwards** and the minimum is genuine.

### Why covariance and not variance

The paper's conceptual thesis (p. 89), in one line of algebra. For the equally
weighted portfolio $w_i = 1/N$:

$$
V = \frac{1}{N^2}\sum_{i}\sigma_{ii} + \frac{1}{N^2}\sum_{i\neq j}\sigma_{ij}
= \frac{1}{N}\overline{\sigma^2} + \frac{N-1}{N}\overline{\sigma_{ij}}
\ \xrightarrow[N\to\infty]{}\ \overline{\sigma_{ij}}.
$$

The first term (own variances) is **diluted** as $1/N$; the second (average
covariance) is **not**. That limit $\overline{\sigma_{ij}}$ is the floor Markowitz
speaks of when rejecting the law of large numbers, and it is the risk that would
later be called *systematic*. With uncorrelated assets $\overline{\sigma_{ij}} = 0$
and zero is indeed reached; with assets from the same sector, it is not. **It is
Piece 4 in one formula.**

## Stack

`numpy`, `scipy` (the QP in Piece 3), `matplotlib`. `pandas` for the real data of
Piece 5. Nothing outside the repo's base stack.
