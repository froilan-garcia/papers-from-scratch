# Bootstrap — Efron (1979)

Implementation of *Bootstrap Methods: Another Look at the Jackknife*. Context and
results of the paper are in the [review](../../reviews/1979-efron-bootstrap.md).

> **Status: closed at four of the paper's eight sections and three of its eleven
> remarks (B, D, J), by choice.** What is here is
> the thread the paper is built on: what the bootstrap is and how to compute it, by
> Monte Carlo and by exact enumeration (Sec. 2); the median, where it is hard and it
> works (Sec. 3); what it is *relative to the jackknife*, verified twice by independent
> routes — for the median (Sec. 5, Remark J) and again for regression (Sec. 7); the
> equivariance that survives a monotone transformation (Remark B and **Figure 1**, the
> paper's only figure and its only real data); and the inferential trap the paper warns
> about and does not resolve (Remark D).
>
> **Deliberately out of scope**, each for a stated reason:
>
> - **The variants of $\hat F$** — smoothed (3.11) and symmetrized (3.8), and the
>   remaining columns of Table 1. The paper's own conclusion is negative: its seven
>   columns average 1.01, 1.00, 1.00, 1.01, 1.00, .99, .93, so the plain bootstrap does
>   as well as either variant. Reproducing them confirms a null result, and the part of
>   Table 1 that did turn out to be worth checking — the missing $\sqrt n$ in Eq. (3.12),
>   and the value of $E_F R$ — is checked below.
> - **The Wilcoxon statistic (Sec. 6).** A third instance of a thesis already verified
>   twice here, transposed to two samples.
> - **The discriminant analysis (Sec. 4) and its Table 2.** The omission most worth
>   revisiting: it is the one section solving a *different* problem — estimating an error
>   rate rather than a sampling distribution — and the ancestor of the .632 estimator and
>   of bagging. Left out for size, not for lack of interest.
> - **Remark K, the parametric bootstrap.**
>
> One thing here is not in the paper and not, as far as I have found, anywhere else:
> the limiting law of the jackknife variance of a sample median **depends on the parity
> of $n$**, because an odd sample has a middle observation that can itself be deleted and
> an even one does not. Both cases are derived and simulated —
> [the section](#the-limit-law-depends-on-the-parity-of-n).
>
> Confidence intervals and the block bootstrap are **not in this paper** at all — they
> are 1981–1989. See [what came next](#what-came-next-and-is-not-in-this-paper).
> [`DERIVATIONS.md`](DERIVATIONS.md) covers sections 1–7 of ten, with their figures: the
> thread from the mean to the jackknife is closed there, and what remains is the choice
> of coordinate, regression, and the inferential step.

## Order of implementation

Not the paper's order. The dependency is:

1. **`bootstrap.py`** — resampling from $\hat F$, Monte Carlo (Method 2) and exact
   enumeration (Method 1). Nothing else works without it.
2. **`median.py`** — Sec. 3, which needs 1 to be checked against anything.
3. **`jackknife.py`** — Sec. 5, which needs 2: the claim "the jackknife is inconsistent
   here" cannot be verified until the bootstrap value it disagrees with exists.
4. **`correlation.py`** — Remark B and Fig. 1, which need only 1, and are the first use
   of paired data: an observation is a point of the plane and a resample must take both
   coordinates together.
5. **`regression.py`** — Sec. 7, which needs 3: its whole argument is Eq. (5.10) applied
   to two different data sets, so it cannot be checked before the machinery of Sec. 5
   exists. It imports `simplex_derivatives` from `jackknife.py` unchanged.
6. **`intervals.py`** — Remark D, which needs 2: it is a statement about the bootstrap
   distribution of the median, and only makes its point once that distribution is exact.

The jackknife comes **after** the median even though the paper puts it four sections
later, for the reason in 3.

## Running it

```bash
python bootstrap.py
python median.py
python jackknife.py
python correlation.py
python regression.py
python intervals.py
python derivation_figures.py
```

Seconds each, except `regression.py` at about fifteen — almost all of it the 2000-trial
comparison of four estimators at the end. `numpy`, `scipy` and `matplotlib` only. No
`requirements.txt`: nothing outside the base stack.

> **If any of these dies without a traceback**, the environment's BLAS is at fault, not
> the code. In the `papers` conda env, MKL 2026.1 and `llvm-openmp` 22.1.8 fail to load
> together: `np.dot` on two matrices, `np.linalg.lstsq` and every matplotlib `savefig`
> abort with the Windows delay-load exception `0xc06d007f`. Setting
> `MKL_THREADING_LAYER=SEQUENTIAL` sidesteps the threaded MKL layer and all three work.
> The figures here were generated that way.

## What matches and what does not

| Claim of the paper | Result |
|---|---|
| $\mathrm{Var}_*(\bar X^* - \bar x) = \hat\sigma^2/n$, by enumeration | ✅ exact to machine precision (0.0e+00 at $n=3,5,10$) |
| Eq. (2.8), $\bar x(1-\bar x)/n$ for 0/1 data | ✅ to 12 decimals |
| Monte Carlo error is $O(N^{-1/2})$ | ✅ RMS $\times\sqrt N$ flat from $N=100$ to $6400$ |
| Sec. 2 lists the variance among what Method 1 does "by hand", and does not | ✅ done in `DERIVATIONS.md` §5: bias exactly $-\hat\sigma^2/n$, and the exact $\mathrm{Var}_*$, to $10^{-16}$ |
| Eq. (3.5) closed form vs. enumerating all $\binom{2n-1}{n}$ resamples | ✅ identical to 12 decimals, $n=5,7,9$ |
| Eq. (3.6), the six probabilities for $n=13$ | ✅ to $7\times10^{-5}$, the paper's printing precision |
| $n\,E_*(R^*)^2 \to 1/4f^2(\theta) = \pi/2$ | ✅ but slowly — see below |
| Table 1, column (3.6): AVE 1.01, S.D. .31 | ✅ AVE 1.011, S.D. .317 (20000 trials) |
| Eq. (3.12) as printed | ❌ **needs a factor $\sqrt n$** to produce Table 1 |
| Eq. (3.13), $E_F R = 0.95$ | ⚠️ ours is 0.9822 ± 0.0012 — see below |
| Eq. (5.7), $\mathbf{e}U = 0$, $\mathbf{e}V = -n\mathbf{U}'$, $\mathbf{e}V\mathbf{e}' = 0$ | ✅ to $10^{-6}$ by finite differences |
| Eq. (5.14), $U$ and $V$ for the ratio estimator | ✅ to $2\times10^{-7}$ |
| Eq. (5.8), $E_*R^* = R(\mathbf{e}/n) + \bar V/2n$ | ✅ lands on Eq. (5.15) to 8 decimals |
| Eq. (5.15) against the bootstrap it approximates | ✅ gap 5% at $n=10$, 0.3% at $n=160$ |
| Eq. (5.10) $=$ the bootstrap variance, for the mean | ✅ to $10^{-9}$; the ordinary jackknife is exactly $\tfrac{n}{n-1}$ of it |
| Eq. (5.13), $\tilde U_i / U_i = 1 + O(1/n)$ | ✅ $n(v_{\text{ord}}/v_{\text{inf}} - 1)$ stays near 1 from $n=10$ to $160$ |
| The jackknife is inconsistent for the median (Sec. 3) | ✅ its spread does not shrink with $n$ |
| that inconsistency's limit law, $[\chi^2_2/2]^2$, mean 2, variance 20 | ✅ for **even** $n$ — but Sec. 3 works with $n$ odd, where it is $[\chi^2_4/4]^2$; the law depends on the parity, which the paper does not say. **The one result here that is not in the paper** — see below |
| Remark J: deleting in groups of $g = O(\sqrt n)$ repairs the median | ✅ it converges, but slowly — see below |
| Remark B, $\hat\rho = .945$ for the nine pairs of Fig. 1 | ✅ 0.944848 |
| Fig. 1: $\hat\rho^*$ straggles left, $\tanh^{-1}\hat\rho^*$ straggles right | ✅ and the reason is not the transformation's shape — see below |
| Fig. 1: the median is above zero, but small against the spread | ✅ 4.0% and 3.3% of the $1/6$–$5/6$ spread |
| Eq. (8.1), the bootstrap is equivariant under monotone $g$ | ✅ exact, to every digit, at every $N$ |
| Eq. (7.7), $\mathrm{Cov}_*\hat\beta^* = \hat\sigma^2G^{-1}$ by resampling residuals | ✅ Monte Carlo converges to it at $N^{-1/2}$ |
| Eq. (7.8) is the infinitesimal jackknife on the rows | ✅ it is Eq. (5.10) exactly, to six digits |
| Eq. (7.8) "doesn't look at all like (7.7)" | ✅ 0.186 against 0.160 on the same data |
| the ordinary jackknife is "a quite similar expression" | ✅ it is Eq. (7.8) with residuals inflated by $1/(1-h_i)$, to machine precision |
| Eq. (7.9), symmetrizing recovers (7.7) | ✅ to $2\times10^{-11}$ over the whole matrix |
| Sec. 7: jackknife methods are consistent without (7.2), at low efficiency | ✅ both halves — see below |
| Eq. (8.3), $\mathrm{Prob}\{4\le\mathrm{Bi}(13,\tfrac12)\le 9\} = .908$ | ✅ 0.90771, and distribution-free in simulation |
| Eq. (8.4), the bootstrap's own statement, $.914$ | ✅ 0.91364, with the continuity correction |
| Remark D: Eq. (8.6) is wrong because $\hat\theta-\theta$ is no pivot | ✅ and by a lot — it claims .914 and covers .70–.75 |

### The missing $\sqrt n$ in Eq. (3.12)

Eq. (3.12) defines $R = |t(\mathbf{X}) - \theta(F)|/\sigma(F)$. Computed literally, its
bootstrap expectation over 200 trials averages **0.287**, while Table 1 tabulates
**1.01**. Multiplying by $\sqrt{13} = 3.606$ gives **1.016**. The ratio is not a
coincidence of one experiment: the absolute error of a median shrinks like $n^{-1/2}$,
so the unscaled quantity drifts to 0 as $n$ grows and has no stable value to tabulate.

It changes nothing about the paper's argument — $R^*$ is scale invariant either way, and
that invariance is the only property Sec. 3 uses — but it has to be put back to
reproduce a single number of Table 1.

### $E_F R$: ours is 0.982, the paper says 0.95

With the scaling settled, the true value can be simulated directly: $400\,000$ samples
of size 13 from $\mathcal{N}(0,1)$ give $0.9822 \pm 0.0012$. The asymptotic median
distribution $\mathcal{N}(0, \pi/2n)$ gives exactly 1, approached from below, so 0.95 is
not the asymptotic value either, and no other natural reading produces it: scaling by
$\sqrt{n-1}$ instead gives 0.943 but then Table 1's own column comes out at 0.971
against its printed 1.01, so no single convention reconciles both numbers.

**This is a discrepancy and not an erratum.** The gap is 0.03, and the paper does not say
how 0.95 was obtained; a Monte Carlo estimate over a few hundred trials — the scale of
everything else in Section 3 — carries a standard error of about 0.025, which places
0.95 comfortably within sampling error of 0.982. Our $\pm 0.0012$ is the precision of
*our* computation, not a measure of how far the paper is from the truth.

It matters for reading Table 1 either way: against 0.95 the plain bootstrap looks biased
upwards by 6%, and against 0.982 by 3%. The paper's conclusion — that the plain
bootstrap does as well as the smoothed and symmetrized versions — is untouched, since
all columns move together.

### Why the asymptotics are slow

$n\,E_*(R^*)^2$ does converge to $\pi/2$, but from far away:

| $n$ | truth $n\,E_F R^2$ | bootstrap (mean) | ratio |
|---|---|---|---|
| 13 | 1.542 | 2.054 | 1.33 |
| 51 | 1.538 | 1.874 | 1.22 |
| 201 | 1.574 | 1.742 | 1.11 |

The bootstrap overestimates the squared error of the median by a third at $n=13$. This
is consistent with the paper's own Table 1, where the plain bootstrap averages 1.09
against a truth of 0.95 — a 15% overestimate of the *absolute* error, which squares to
about 32%. The two numbers are the same phenomenon seen through $R$ and $R^2$, and it is
worth knowing before trusting a bootstrap standard error for a median in a small sample.

### The median breaks the jackknife twice, in different ways

Sec. 5 turns the bootstrap into a formula by expanding $R(\mathbf{P}^*)$ about the
observed sample. Both jackknives are that formula; they differ only in how the
derivatives $U_i = \partial R/\partial P_i$ are obtained. For the median neither survives,
and it is worth separating the two failures because only one of them is in the paper.

**Method 3 returns exactly zero.** $\hat F$ is discrete, so the weighted median is a
*step* function of the weights: it stays at $x_{(m)}$ until some weight has moved by
about $1/2n$, and then jumps. Its derivative at $\mathbf{P}^* = \mathbf{e}/n$ is
therefore not merely hard to estimate but identically $0$, and so is every second
derivative. The infinitesimal jackknife reports that the sample median has no
sampling variability whatsoever, for every sample and every $n$. This is not a
step-size artefact: the function is locally constant, so every small enough step gives
the same $0$. Remark J blames a derivative "too irregular for the jackknife's quadratic
extrapolation formulas to work"; at $\hat F$ it is not irregular but absent.

**The ordinary jackknife survives, on three data points.** Finite differences reach
the edge of the simplex, where the median has moved, so $\tilde U_i \ne 0$. But
deleting one observation from an odd sample $n = 2m-1$ leaves the median at one of
only three values, so with $a = x_{(m)} - x_{(m-1)}$ and $b = x_{(m+1)} - x_{(m)}$

$$v_{\text{jack}} = \frac{(m-1)^2}{2(2m-1)}\left[a^2 + b^2 - \frac{(m-1)(a-b)^2}{2m-1}\right]
\;\sim\; \frac{n\,(a+b)^2}{16},$$

a function of three order statistics however large $n$ is (two, if $n$ is even — the
parity matters, and the next section is about that). Since $n(a+b)$ converges in
*distribution* to $\Gamma(2,1)/f(\theta)$ rather than concentrating, $n\,v_{\text{jack}}$
never settles:

| $n$ | truth $n\,\mathrm{Var}_F$ | jackknife: mean (s.d.) | bootstrap: mean (s.d.) |
|---|---|---|---|
| 13 | 1.488 | 1.881 (2.42) | 1.893 (1.24) |
| 51 | 1.565 | 2.225 (3.29) | 1.777 (0.89) |
| 201 | 1.595 | 2.374 (3.58) | 1.687 (0.65) |
| 1001 | 1.537 | 2.375 (3.59) | 1.625 (0.42) |

The bootstrap column walks towards $1/4f^2 = \pi/2 = 1.571$ with a spread that shrinks;
the jackknife column walks to $1.5 \times \pi/2$ with a spread that does not.

### The limit law depends on the parity of $n$

The jackknife estimate of the variance of a sample median does not converge, and *what
it converges to instead* turns out to depend on whether the sample size is odd or even:

$$n\,\hat v_{\text{jack}} \;\xrightarrow{\;d\;}\; \frac{1}{4f^2(\theta)} \times
\begin{cases}
\left[\chi^2_2/2\right]^2, & n \text{ even} \quad (\text{mean } 2,\;\text{variance } 20)\\[4pt]
\left[\chi^2_4/4\right]^2, & n \text{ odd} \quad\;\; (\text{mean } 1.5,\;\text{variance } 5.25)
\end{cases}$$

The mechanism is one observation:

- **$n$ even.** Deleting leaves an odd sample with a single middle value, so the
  replicates take *two* values, $\hat v_{\text{jack}} = \frac{n-1}{4}(x_{(m+1)}-x_{(m)})^2$,
  and **one** spacing enters.
- **$n$ odd.** Deleting leaves an even sample, and there is now a middle observation that
  can itself be deleted — the case with no analogue above. The replicates take *three*
  values and **two** spacings enter, so the limit averages them and is correspondingly
  tamer.

Since $n\times(\text{a spacing at the median}) \to \mathrm{Exp}(f(\theta))$, one spacing
gives the square of an exponential and two give the square of their mean, which is the
whole of the difference. Simulating $40\,000$ samples at each of $n = 4000$ and
$n = 4001$ separates the two at every quantile:

| | mean | var | $q_{.10}$ | $q_{.25}$ | $q_{.50}$ | $q_{.75}$ | $q_{.90}$ | $q_{.99}$ |
|---|---|---|---|---|---|---|---|---|
| simulated, $n=4000$ (even) | 2.005 | 19.6 | 0.011 | 0.084 | 0.480 | 1.93 | 5.39 | 21.0 |
| $[\chi^2_2/2]^2$ | 2.002 | 19.8 | 0.011 | 0.082 | 0.481 | 1.92 | 5.31 | 21.2 |
| simulated, $n=4001$ (odd) | 1.510 | 5.42 | 0.070 | 0.228 | 0.702 | 1.80 | 3.84 | 11.2 |
| $[\chi^2_4/4]^2$ | 1.502 | 5.24 | 0.071 | 0.231 | 0.705 | 1.81 | 3.78 | 11.1 |

Two estimators one sample size apart, on the same distribution, converging to different
laws. It is a small thing, but it is the kind of small thing that a resampling method is
supposed to be immune to, and it is not visible from the formula.

**What this says about the paper.** Sec. 3 prints the $[\chi^2_2/2]^2$ law — the **even**
one — in a section that has assumed $n = 2m-1$ odd. The formula is correct; what the
sentence omits is that the parity decides which of two laws applies. In the odd setting
where the claim is made, the estimator is the milder of the two, biased by 50% rather
than 100% and a quarter as variable, so the argument is conservative exactly where Efron
needs it. His conclusion is identical in both parities: the limit is a random variable,
so no amount of data makes the jackknife settle.

**What it says about the code.** This implementation missed the parity for as long as
its closed form rejected even $n$ outright — a restriction inherited from Sec. 6, which
genuinely needs a unique middle order statistic, by a calculation that never did. Code
that refuses the inputs on which a claim would fail cannot be used to test that claim.
Both parities are implemented now, and both appear in the checks.

### Remark J's repair works, and is expensive

Remark J diagnoses the failure as an overdependence on $\mathbf{P}^*$ within $1/n$ of
$\mathbf{e}/n$ — Eq. (8.14) — where the median is locally constant, while the bootstrap
looks out at distance $n^{-1/2}$. The cure proposed is to delete observations in groups
of size $g$, with "$g = O(n^{1/2})$"; the paper asserts the result and gives no numbers.
These are the numbers, as mean (s.d.) of $n\hat v$ over 2000 trials, computed exactly —
the delete-$d$ distribution of the median is a negative hypergeometric, so the
$\binom{n}{d}$ deletions are summed rather than sampled, exactly as Eq. (3.5) does for
the bootstrap:

| $n$ | $d = 2$ | $d \sim \sqrt n$ | $d \sim n^{3/5}$ | bootstrap |
|---|---|---|---|---|
| 101 | 2.687 (3.82) | 2.058 (1.74) | 1.929 (1.39) | 1.719 (0.75) |
| 401 | 2.724 (4.02) | 1.975 (1.43) | 1.860 (1.14) | 1.644 (0.54) |
| 1601 | 2.843 (4.28) | 1.865 (1.12) | 1.779 (0.86) | 1.624 (0.37) |
| 6401 | 2.718 (4.10) | 1.750 (0.91) | 1.682 (0.68) | 1.586 (0.26) |

Grouping is not by itself the cure: at $d = 2$ the estimator is as lost as at $d = 1$,
since two deletions still move $\mathbf{P}^*$ by $O(1/n)$. What repairs it is letting $d$
**grow** — both growing rules walk towards $\pi/2$ with a spread that shrinks. So the
claim holds, with poor constants: at $n = 6401$ the $\sqrt n$ rule is still 11% high and
3.5 times as variable as the bootstrap, which needed neither a repair nor a
choice of $d$.

### Figure 1, and why the skew changes sides

![Fig. 1 reproduced: the same 1000 replications before and after the Fisher transformation](fig1_correlation.png)

The nine pairs printed in the caption of Fig. 1, resampled $N = 1000$ times as in the
paper, in both scales, with the $1/6$, $1/2$ and $5/6$ quantiles marked as in the
original. What the original could not draw is the grey distribution behind: $n = 9$
makes the exact bootstrap of Sec. 2 enumerable — all $\binom{17}{9} = 24310$ resamples —
so the histogram can be compared against the fixed object it approximates rather than
against another histogram. At $N = 1000$ the three marked quantiles are off by up to
0.003, a third of a bin; at $N = 10^5$, by 0.0002.

Remark B reads the two panels as straggling in opposite directions, and the reason is
the dashed line. $\hat\rho^*$ **cannot exceed 1**, which is only $1 - \hat\rho = 0.055$
above the observed value, while it can fall as far as $-1$; the upper tail is cut off by
construction, and the mass has nowhere to go but left. Fisher's transformation sends
that ceiling to $+\infty$, and the asymmetry changes sides. It is not that
$\tanh^{-1}$ reshapes the distribution: it removes the boundary that was shaping it.

| $q$ | $\hat\rho^* - \hat\rho$ | $\tanh^{-1}\hat\rho^* - \tanh^{-1}\hat\rho$ |
|---|---|---|
| 0.01 | $-0.3388$ | $-1.0787$ |
| 1/6 | $-0.0382$ | $-0.2729$ |
| 1/2 | $+0.0031$ | $+0.0300$ |
| 5/6 | $+0.0393$ | $+0.6328$ |
| 0.99 | $+0.0538$ | $+1.8785$ |
| min | $-1.9448$ | $-\infty$ |
| max | $+0.0552$ | $+\infty$ |

Those last two rows are exact, not overflow. A resample that happens to draw only two
distinct pairs is perfectly collinear, so $\hat\rho^* = \pm 1$; there are 288 such
resamples with total probability $36[(2/9)^9 - 2\cdot 9^{-9}] = 4.74\times10^{-5}$, and
nine more — one pair drawn nine times — where $\hat\rho^*$ is undefined altogether.
Expected occurrences in a run of 1000: **0.05**, which is why none of this is visible in
the paper's figure.

It decides something real, though. The bootstrap distribution of
$\tanh^{-1}\hat\rho^*$ has atoms at $\pm\infty$, so it **has no mean and no variance** —
while its quantiles are untouched, the offending mass being 5 in $10^5$. That is the
sharpest reason for what Fig. 1 does: it marks quantiles and not moments. And it is why
Eq. (8.1), which is a statement about quantiles alone, survives a transformation that
destroys every moment the distribution had.

### Regression: the same formula, one index apart

Sec. 7 contains the paper's only exclamation mark. Deleting one row at a time gives

$$\mathrm{Cov}\,\hat\beta \;\approx\; G^{-1}\Big[\textstyle\sum_i c_i'c_i\hat\epsilon_i^2\Big]G^{-1},
\qquad \text{Eq. (7.8),}$$

which, Efron says, "doesn't look at all like (7.7)" — the classical $\hat\sigma^2G^{-1}$
that resampling the residuals reproduces exactly. On the data here, 0.160 against 0.186
for the same coefficient. Both are the **same formula**: Eq. (5.10) of the jackknife
section, applied to two different data sets.

| data set | what Eq. (5.10) sums | result |
|---|---|---|
| the $n$ rows as observed | $\sum_i c_i'c_i\hat\epsilon_i^2$ | Eq. (7.8) |
| the $n^2$ symmetrized points of Eq. (7.9) | $\sum_{i,j} c_i'c_i\hat\epsilon_j^2 = G\cdot n\hat\sigma^2$ | Eq. (7.7) |

One index. The second sum **factorizes** and the first does not, and that factorization
is the entire content of assumption (7.2), that the errors are identically distributed.
Efron's remark that "the bootstrap implicitly does this at step (7.5)" is exactly right:
drawing $\epsilon_i^*$ from all $n$ residuals regardless of $i$ is what severs a row from
its own error. Verified to $2\times10^{-11}$, with `simplex_derivatives` imported from
`jackknife.py` unchanged — the same routine, a different statistic.

The ordinary jackknife is what Efron calls "a quite similar expression", and the code
pins down how similar: deleting row $i$ moves the fit by exactly
$-G^{-1}c_i'\hat\epsilon_i/(1-h_i)$, so it *is* Eq. (7.8) with each residual inflated by
its own leverage factor. That agrees with the deletion to machine precision, and the
inflation turns out not to be a detail.

### Which one is right, and what it costs

Efron notes that jackknife-style estimates are consistent "without assumption (7.2) …
The price of such complete generality is low efficiency." Both halves are measurable:
the design is fixed, so the truth is $G^{-1}[\sum_i c_i'c_i\sigma_i^2]G^{-1}$. Standard
error of $\beta_1$, $n = 20$, 2000 trials:

| estimator | homosk.: mean/truth | s.d. | heterosk.: mean/truth | s.d. |
|---|---|---|---|---|
| (7.7) residual bootstrap | 0.911 | **0.031** | 0.582 | 0.040 |
| (7.8) sandwich | 0.853 | 0.040 | 0.727 | 0.077 |
| jackknife on rows | 1.081 | 0.056 | **0.979** | 0.120 |
| pairs bootstrap *(not in the paper)* | 1.026 | 0.044 | 0.829 | 0.080 |

The 9% shortfall of (7.7) in the homoskedastic column is not an error: it is the $1/n$ in
$\hat\sigma^2$, since $E\hat\sigma^2 = \frac{n-p}{n}\sigma^2$ puts $\sqrt{17/20} = 0.922$
into the standard error, with Jensen's inequality on the square root accounting for the
rest. Its real advantage is the spread — 22% below the sandwich and 45% below the
jackknife. That is the efficiency being charged for.

Under heteroskedasticity (7.7) estimates a quantity that no longer exists, and no amount
of resampling repairs it: **the residual bootstrap inherits every assumption that went
into $\hat F$**, and (7.2) is one of them. What is worth noticing is which alternative
survives. Eq. (7.8) reaches only 0.73 of the truth at $n=20$, while the jackknife reaches
0.979 — the leverage factors are the small-sample correction, and they are the reason to
prefer Efron's "quite similar expression" to Eq. (7.8) itself.

### Where the formulas run out

Eq. (7.1) allows any $g_i(\beta)$, and there the paper says these methods "may really be
necessary". With $g_i(\beta) = \beta_0 e^{\beta_1 t_i}$ there is no $G$, no (7.7) and no
(7.8); what gets reported is $\hat\sigma^2 (J'J)^{-1}$, the same formula after
linearizing $g$ — the delta method again, in regression clothing.

| $\sigma$ | bootstrap s.e. | $\hat\sigma^2(J'J)^{-1}$ | ratio | bootstrap skew |
|---|---|---|---|---|
| 0.05 | 0.0223 | 0.0224 | 0.998 | $-0.01$ |
| 0.15 | 0.0412 | 0.0403 | 1.021 | $-0.08$ |
| 0.40 | 0.1536 | 0.1479 | 1.038 | $-0.24$ |
| 0.80 | 0.2313 | 0.2373 | 0.975 | $-0.37$ |
| 1.50 | 0.7083 | 0.5479 | 1.293 | $-2.86$ |

The linearization gets the **scale** right long after it has stopped getting the **shape**
right: at $\sigma = 0.8$ the ratio is still 0.975 while the skewness has reached $-0.37$.
A symmetric answer is all $\hat\sigma^2(J'J)^{-1}$ can ever give, being the covariance of
a linear map. The last row is a different failure: the decay stops being identifiable in
some resamples, the bootstrap distribution goes heavy-tailed, and its standard deviation
stops meaning much — an instability that is real, and that the Jacobian formula cannot
express at all.

### Remark D: the agreement that means nothing

The paper does not develop confidence intervals, and Remark D is why. Two statements
about the median of $n = 13$:

$$\mathrm{Prob}_F\{x_{(4)} < \theta < x_{(10)}\} = \mathrm{Prob}\{4\le\mathrm{Bi}(13,\tfrac12)\le 9\} = 0.908,$$

which is **exact and distribution-free** — the only random thing in it is how many
observations fall below $\theta$, and that is $\mathrm{Bi}(n,\frac12)$ whatever $F$ is —
against the bootstrap's own $\mathrm{Prob}_*\{x_{(4)} < \hat\theta^* < x_{(10)}\} = 0.914$,
read off Eq. (3.6). Six thousandths apart, which Efron calls striking. Then, since
$\hat\theta = x_{(7)}$, treating $\hat\theta^* - \hat\theta$ as a **pivot** turns that
into a statement about $\hat\theta - \theta$, and inverting it gives Eq. (8.6),

$$\mathrm{Prob}_F\{2x_{(7)} - x_{(10)} < \theta < 2x_{(7)} - x_{(4)}\} \approx 0.914,$$

the reflection of the correct interval about the sample median. The paper stops at the
exclamation mark; the numbers are worth having:

| $F$ | Eq. (8.3) covers | Eq. (8.6) covers | both claim |
|---|---|---|---|
| $\mathcal{N}(0,1)$ | 0.9080 | 0.7538 | 0.914 |
| $U(0,1)$ | 0.9082 | 0.6985 | 0.914 |
| $\mathrm{Exp}(1)$ | 0.9077 | 0.7265 | 0.914 |
| lognormal | 0.9079 | 0.7391 | 0.914 |

A reflection preserves length, so both intervals have **identical width** in every
sample: the whole loss is position. And the failure is not confined to skewed $F$, which
is the natural guess — it is as bad under the two symmetric ones, because the reflection
is about $x_{(7)}$, which is itself random.

One line beyond the paper, worth it for what it explains: read as *percentiles* of the
bootstrap distribution rather than as a pivot, the same $\hat\theta^*$ returns
$(x_{(4)}, x_{(10)})$ — Eq. (8.3) itself, the exactly correct interval. The disaster is
entirely in the inferential step, precisely where Remark D puts it, and this single
example is why intervals needed their own papers.

## What came next, and is not in this paper

Nothing below is implemented here, and none of it is Efron (1979) — the paper predates
all of it. The list is a map of which later idea closes which gap, with the piece of this
implementation each would start from.

| Extension | The gap it closes | Where it would hook in |
|---|---|---|
| **Percentile intervals** (Efron 1981) | Remark D: how to get an interval without a pivot | `intervals.py` already computes it for the median |
| **$BC_a$** (Efron 1987) | percentile intervals are wrong when $\hat\theta^*$ is biased or skewed | the acceleration is $a = \sum_i U_i^3 / 6(\sum_i U_i^2)^{3/2}$ — the **same $U_i$** `jackknife.py` computes |
| **Bootstrap-$t$** (Efron 1982; Hall 1988) | studentizing to build the pivot Remark D assumed into existence | `intervals.py` + a nested resampling loop |
| **Wild bootstrap** (Wu 1986; Mammen 1993) | resampling residuals *without* assuming (7.2) | directly the heteroskedastic column of `regression.py` |
| **Block bootstrap** (Künsch 1989; Politis–Romano 1994) | the iid assumption of Sec. 2, for time series | `resample_indices` becomes block indices |
| **$m$-out-of-$n$ and subsampling** (Politis–Romano 1994; Bickel et al. 1997) | the cases where the bootstrap simply fails: maximum of a uniform, parameters on a boundary | the delete-$d$ jackknife of Remark J is the same "resample less" idea, already in `jackknife.py` |
| **Bayesian bootstrap** (Rubin 1981) | the multinomial weights of Eq. (3.2) replaced by Dirichlet ones | the simplex of `ded_simplex.png`, with a continuous law on it |
| **.632 and .632+** (Efron 1983; Efron–Tibshirani 1997) | the optimism of the apparent error rate | Sec. 4, the section left out |
| **Bagging** (Breiman 1996) | using the resamples to *improve* the estimator, not to assess it | Sec. 4, and Breiman in the [roadmap](../../ROADMAP.md) |

The two clusters are worth separating. Everything in the first three rows is the same
unfinished business — Remark D's — and it took eight years to settle. The rows about
dependence and failure are a different matter: they are cases where $\hat F$ is not a
good enough stand-in for $F$, which is the one assumption the whole paper rests on and
the one it never examines.

## Notes on the code

- `bootstrap_exact` enumerates the $\binom{2n-1}{n}$ distinct resamples and weights each
  by its **multinomial** probability $n!/(N_1!\cdots N_n!)\,n^{-n}$. The distinct
  resamples are *not* equiprobable, and treating them as such is a different (wrong)
  scheme. Cost limits it to $n \lesssim 12$.
- `median_pmf(n)` depends on $n$ **alone**, not on the data — the data only set the
  support $x_{(l)} - x_{(m)}$. That is what makes the median tractable here.
- Paired data needed **no change** to `bootstrap.py`. Resampling is resampling of
  indices, and both `bootstrap_mc` and `bootstrap_exact` index whatever they are given,
  so a $(9,2)$ array of pairs works as it stands. That was the reason for making the
  index vector the primitive rather than the resampled values.
- `pearson` clips to $[-1,1]$ and returns `nan` where a resample has no variability. Both
  are load-bearing rather than defensive: the clip is what makes $\tanh^{-1}$ return
  $\pm\infty$ instead of `nan` just outside the domain, and the infinity is the correct
  answer.
- Bootstrap variances use the $1/n$ convention throughout: $\hat F$ is a fully specified
  distribution, not a sample, so there is no $-1$ to make.
- `jackknife.py` takes statistics as **functions of the weight vector**, $R(\mathbf{p})$,
  not of a sample. The bootstrap only ever evaluates $R$ at weights that are multiples of
  $1/n$; the expansion of Sec. 5 needs it at fractional weights, which is information a
  sample statistic does not carry. There is deliberately no generic wrapper from one to
  the other: the paper concedes the same on p. 13, that the interpolation "will be
  obvious in most specific cases, but a general recipe is difficult to provide".
- The homogeneous extension $R(\mathbf{p}) = R(\mathbf{p}/\sum_i p_i)$ of Eq. (5.6) is
  what lets the deletion $\mathbf{e}_{(i)}$ be written as a vector of ones with a zero in
  it, with no renormalisation and no separate notion of "the sample without $x_i$".
- `median_functional` drops zero-weight observations *before* reading off the median,
  rather than merely giving them no mass. It matters exactly once, and decisively: at an
  even weight total the two middle values are averaged, and the second of them must be
  the next one present. That is the replicate which deletes the median itself — the one
  of the $n$ that distinguishes the median from its two neighbours.
- `delete_d_median_pmf` is the same idea as `median_pmf`, one level up: keeping $r = n-d$
  observations at random, the position of the retained median is negative hypergeometric,
  so the delete-$d$ jackknife is an exact sum over all $\binom{n}{d}$ deletions and costs
  one dot product. It is checked against the ordinary jackknife at $d = 1$ and against a
  literal enumeration of the $\binom92 = 36$ deletions at $d = 2$.
- `bootstrap_median_coverage` counts **half** of each endpoint atom. The bootstrap
  distribution of the median sits *on* the order statistics, so "inside the interval" is
  ambiguous at the ends; Efron's continuity correction is what produces the .914 of
  Eq. (8.4), and dropping it gives .859. The convention is carrying real weight, not
  tidying a rounding.
- `influence_cov` carries `n_real` separately from the number of points fitted. They
  differ only for the symmetrized data set of Eq. (7.9), and keeping them apart is the
  correction Eq. (5.16) makes for an "artificially increased amount of data": divide by
  $n_{\text{real}}\cdot M$, not by $M^2$. With $M = n$ it is Eq. (5.10) unchanged.
- Eq. (5.12) reads $\tilde U_i = (n-1)(R^*_\cdot - R^*_{(i)})$, but the gloss below it
  prints $\hat\theta$ where it needs the average $\hat\theta_{(\cdot)}$. It has to be the
  average: the $\tilde U_i$ must sum to zero, mirroring $\mathbf{e}U = 0$ of Eq. (5.7),
  and it is that centring which makes $\sum_i \tilde U_i^2/[n(n-1)]$ Tukey's variance.
