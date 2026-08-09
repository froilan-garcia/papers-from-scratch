# The bootstrap, derived

Everything below is built in the order the pieces need each other, which is not the
order of Efron (1979). We start from the question the paper answers, show what stands in
the way of answering it, and let each obstacle dictate the next tool. The paper's
equation numbers appear at the end of a derivation, as a cross-reference, never as a
starting point.

One thread runs through all of it: **how smooth $R$ is as a function of the distribution
it is fed**. That single question decides which cases can be solved by hand, which need
simulation, and where the jackknife breaks. It is worth keeping in view from the first
section, where it will look like an idle remark.

## Contents

1. [The question, and why it has no direct answer](#1-the-question-and-why-it-has-no-direct-answer)
2. [Replacing what we do not know](#2-replacing-what-we-do-not-know)
3. [What a resample actually is](#3-what-a-resample-actually-is)
4. [The first case we can solve completely: the mean](#4-the-first-case-we-can-solve-completely-the-mean)

*(Sections 5–9 — the variance, the median, Monte Carlo, the jackknife, and the variants
of $\hat F$ — are still being written.)*

---

## 1. The question, and why it has no direct answer

We observe $\mathbf{X} = (X_1, \dots, X_n)$, independent and identically distributed
according to some $F$ on the real line, and we compute from them a statistic
$t(\mathbf{X})$ meant to estimate a quantity of interest $\theta(F)$. The estimate will
be wrong. The whole of what follows is about the size and shape of that error,

$$R(\mathbf{X}, F) \;=\; t(\mathbf{X}) - \theta(F).$$

Note what kind of object $R$ is: a random variable, because $\mathbf{X}$ is random, but
one whose definition also involves $F$ directly through $\theta(F)$. Both dependencies
matter and they are different in nature; keeping them apart is most of the conceptual
work in Section 2.

What we would like is the whole **sampling distribution** of $R$, because every question
we might ask is a functional of it:

$$\mathrm{bias} = E_F\,R, \qquad \mathrm{variance} = \mathrm{Var}_F\,R, \qquad
\text{interval endpoints} = \text{quantiles of } R .$$

Write $\Psi(F)$ for that distribution — the law of $R(\mathbf{X},F)$ when
$\mathbf{X} \sim F^{\otimes n}$. It is a perfectly well-defined object, and it is what we
are after. The difficulty is immediate and total: **$\Psi$ depends on $F$, which we do
not know.** Nothing else is in the way.

It is worth being precise about what "in the way" means here, because the obvious
reading is wrong. The obstacle is not that the calculation is hard. Suppose someone
handed us $F$. Then $\Psi(F)$ would be available in at least two ways: by analysis when
$R$ is simple enough, and otherwise by brute force, generating samples
$\mathbf{X}^{(1)}, \dots, \mathbf{X}^{(B)}$ from $F$, computing $R$ on each, and reading
off the histogram — as accurately as we care to, since $B$ is limited only by patience.

So the problem is not computational. It is that we are one object short, and that object
is $F$.

Two examples fix the scale of the difficulty. For the mean, $t(\mathbf{X}) = \bar X$ and
$\theta(F) = \int x \, dF$, classical theory gives
$\mathrm{Var}_F R = \sigma^2(F)/n$ — a complete answer, except that $\sigma^2(F)$ is
itself unknown and must be estimated. For the median the situation is worse: there is no
elementary closed form at all for finite $n$, only an asymptotic statement involving the
density of $F$ at the median, which is a far more delicate thing to estimate than a
variance.

The gap between those two examples is not an accident of which formulas happen to have
been worked out. It is the smoothness thread announced above, making its first
appearance: $\bar X$ depends on the data in the gentlest way imaginable, the median in a
way that ignores almost all of them.

## 2. Replacing what we do not know

If the only missing ingredient is $F$, the only possible move is to substitute something
for it. The question is what.

### 2.1 The empirical distribution

Having observed $\mathbf{X} = \mathbf{x} = (x_1, \dots, x_n)$, define $\hat F$ to be the
distribution placing mass $1/n$ on each observed value:

$$\hat F(t) \;=\; \frac{1}{n}\,\#\{\,i : x_i \le t\,\}.$$

This is not an arbitrary choice, and three separate arguments point at it.

**It is the nonparametric maximum likelihood estimate.** Consider any distribution $G$
as a candidate, and ask for the likelihood it assigns to the data. Any mass $G$ places
away from the observed points contributes nothing to that likelihood, so a maximiser
must put all its mass on $x_1, \dots, x_n$; say mass $p_i$ on $x_i$, with
$\sum_i p_i = 1$ and $p_i \ge 0$. The likelihood is then $\prod_{i=1}^n p_i$ (taking the
$x_i$ distinct for the moment), and by the arithmetic–geometric mean inequality

$$\prod_{i=1}^{n} p_i \;\le\; \left(\frac{1}{n}\sum_{i=1}^{n} p_i\right)^{\!n}
\;=\; \frac{1}{n^n},$$

with equality if and only if all the $p_i$ coincide, that is $p_i = 1/n$. So $\hat F$ is
the maximiser, and uniquely so. If some observations tie, the same argument applies to
the distinct values with their multiplicities and returns the same $\hat F$.

**It converges to $F$.** The Glivenko–Cantelli theorem gives
$\sup_t |\hat F(t) - F(t)| \to 0$ almost surely. Whatever we build on $\hat F$ is at
least built on something that approaches the right object.

**It is the centre of what is plausible.** Among all $F$ consistent with having observed
$\mathbf{x}$, none is better supported. This is the informal version of the first
argument, and it is the one Efron leans on.

### 2.2 The plug-in principle, applied to a distribution

Now the substitution. We wanted $\Psi(F)$; we compute instead

$$\widehat{\Psi} \;=\; \Psi(\hat F).$$

This is worth pausing on, because it is the whole idea of the paper and it is easy to
read past. Estimating $\theta(F)$ by $\theta(\hat F)$ is a reflex — it is what the sample
mean and the sample median already are. What is new here is applying that same reflex to
a functional whose **value is an entire distribution** rather than a number. There is no
new principle involved, only a new place to apply an old one.

Concretely, $\Psi(\hat F)$ is the law of

$$R^* \;=\; R(\mathbf{X}^*, \hat F) \;=\; t(\mathbf{X}^*) - \theta(\hat F),
\qquad X_1^*, \dots, X_n^* \;\overset{\text{iid}}{\sim}\; \hat F,$$

which is Efron's Eqs. (2.4)–(2.5). Sampling from $\hat F$ is sampling with replacement
from $\{x_1, \dots, x_n\}$, since $\hat F$ puts mass $1/n$ on each.

Two features of this expression are what make it computable, and both come from the same
place. First, $\theta(\hat F)$ is **known**: it is the functional evaluated at a
distribution we hold in our hands — for the mean it is $\bar x$, for the median the
sample median. The unknown constant that made $R$ inaccessible has become a number we
can print. Second, the law of $\mathbf{X}^*$ is known exactly, being a draw from a
finite distribution we specified ourselves.

### 2.3 Two levels of randomness

The notation now has to carry two different kinds of variability, and confusing them
makes everything that follows unintelligible.

- **Under $F$:** $\mathbf{X}$ is random, $\hat F$ is random (it is a function of
  $\mathbf{X}$), and $R$ is random. This is the level at which the original question
  lives.
- **Under $\hat F$, with $\mathbf{x}$ held fixed:** $\hat F$ is a *fixed, fully
  specified* distribution, and the only randomness is in the resampling. We write
  $E_*$, $\mathrm{Var}_*$, $\mathrm{Prob}_*$ for calculations at this level, following
  the paper.

The consequence deserves to be stated bluntly, because it contradicts the way the
bootstrap is usually described. **Given the data, the bootstrap distribution is not
random at all.** It is a deterministic function of $\mathbf{x}$ — a finite sum, as
Section 3 will make explicit. Simulation enters later, and only as a way of
approximating that sum; it is not part of the definition. A great deal of confusion
about what the bootstrap estimates dissolves once this is fixed.

### 2.4 Why this is the right kind of answer, and what it does not promise

The plug-in estimator $T(\hat F)$ of a quantity $\theta(F) = T(F)$ is called **Fisher
consistent** when $T(F) = \theta(F)$ for every $F$ in the model: fed the true
distribution, the recipe returns the true answer. Fisher (1922) introduced this in
preference to asymptotic consistency, which he found too weak a requirement — one can
degrade an estimator arbitrarily at any fixed $n$ without disturbing its limit.
Fisher consistency mentions no limit and no $n$; it is a statement about the recipe.

By construction, $\Psi(\hat F)$ is Fisher consistent for $\Psi(F)$. Were $\hat F$ equal
to $F$, the answer would be exactly right rather than approximately right. This is what
licenses the bootstrap as *the* natural answer rather than one heuristic among several:
any method that got the wrong answer when handed the true distribution would be
incoherent.

It is equally important that this guarantees nothing on its own. Fisher consistency is a
minimum standard, automatic for every plug-in estimator, and the bootstrap satisfies it
even in the cases where it is known to fail — the maximum of a uniform, parameters on
the boundary, distributions without a variance, dependent data. What fails there is not
$\Psi(F) = \Psi(F)$ but the **continuity** of $\Psi$: a small discrepancy between
$\hat F$ and $F$ produces a large discrepancy between the sampling distributions.

That continuity is the smoothness thread again, now in its sharpest form. We will meet
it as the difference between a statistic that can be linearised and one that cannot.

## 3. What a resample actually is

Section 2 left us with a finite object to compute. Here we describe it exactly, because
its structure governs both what can be enumerated and what has to be simulated.

### 3.1 The count vector

Drawing $X_i^*$ from $\hat F$ means drawing an index uniformly from $\{1, \dots, n\}$.
A resample is therefore fully described by an index vector, and — provided $t$ is
symmetric in its arguments, which every statistic we care about is — the order of those
indices is irrelevant. All that survives is how many times each observation was drawn:

$$N_i^* \;=\; \#\{\,j : X_j^* = x_i\,\}, \qquad \sum_{i=1}^n N_i^* = n .$$

Since the $n$ draws are independent and each lands in cell $i$ with probability $1/n$,

$$\mathbf{N}^* \;\sim\; \mathrm{Multinomial}\!\left(n; \tfrac{1}{n}, \dots, \tfrac{1}{n}\right),$$

which is the paper's Eq. (3.2). It is convenient to normalise and write
$\mathbf{P}^* = \mathbf{N}^*/n$, a point of the simplex
$\{\mathbf{p} : p_i \ge 0, \sum_i p_i = 1\}$. Then $\mathbf{P}^*$ *is* the resampled
empirical distribution, expressed in coordinates, and any symmetric statistic can be
written as a function $R(\mathbf{P}^*)$ of it alone.

The original sample sits at the centre of the simplex, $\mathbf{P}^* = \mathbf{e}/n$
with $\mathbf{e} = (1, \dots, 1)$. This is the vantage point from which everything in
Section 8 will be expanded, and it is why the multinomial parametrisation is worth
setting up now rather than when it becomes indispensable.

### 3.2 How many resamples, and with what probabilities

Two counts must be distinguished.

**Ordered draws.** There are $n^n$ index vectors, all equally likely. This is the honest
description of the sampling mechanism, and it is useless for enumeration: $n = 10$ gives
$10^{10}$.

**Distinct resamples.** Collapsing orderings, a resample is determined by the count
vector $\mathbf{N}^*$, so the number of distinct ones is the number of non-negative
integer solutions of $N_1 + \dots + N_n = n$. By the standard stars-and-bars argument —
arrange $n$ indistinguishable stars and $n-1$ bars in a row, each arrangement encoding
one solution — this is

$$\binom{2n-1}{\,n-1\,} \;=\; \binom{2n-1}{n}.$$

For $n = 10$ that is $92\,378$, against $10^{10}$ ordered draws: a reduction by five
orders of magnitude, and the difference between an enumeration that finishes and one
that does not.

**But they are not equally likely.** This is the trap, and it is worth stating as such
because the reduction just performed is so appealing that one is tempted to work with
the distinct resamples as if they formed a uniform sample space. The probability of a
given count vector is the multinomial one,

$$\mathrm{Prob}_*\{\mathbf{N}^* = (N_1, \dots, N_n)\}
\;=\; \frac{n!}{N_1!\,N_2!\cdots N_n!}\cdot\frac{1}{n^n},$$

the leading factor counting how many of the $n^n$ ordered draws collapse onto that
resample.

The spread is not a minor correction. Take $n = 3$: the resample $(x_1, x_2, x_3)$ that
uses each datum once has multiplicity $3!/(1!1!1!) = 6$ and probability $6/27$, while
$(x_1, x_1, x_1)$ has multiplicity $1$ and probability $1/27$. A sixfold difference at
$n=3$, and it grows quickly with $n$. Any enumeration that omits these weights is
computing a different — and wrong — resampling scheme.

![The ten distinct resamples of n = 3 on the simplex](ded_simplex.png)

Every distinct resample for $n=3$, positioned at its own $\mathbf{P}^*$, with area
proportional to the multinomial weight computed by `bootstrap_exact`. What to look at is
the spread: the centre — which *is* the observed sample, $\mathbf{P}^* = \mathbf{e}/n$ —
carries $6/27$, each corner $1/27$, the edges $3/27$. Ten points, three different
probabilities. Treating them as ten equally likely outcomes would put $2.7$ times too
little weight on the centre and $2.7$ times too much on the corners, which is to say it
would systematically overstate how wild a resample can be.

Monte Carlo, by contrast, gets the weights for free: drawing indices with replacement
produces each count vector with exactly its multinomial probability, without anyone
having to write the factorials down. This is the first of several places where
simulating is not merely easier than enumerating but structurally safer.

Both routes are implemented in [`bootstrap.py`](bootstrap.py) — `bootstrap_exact`
carries the weights explicitly, `bootstrap_mc` inherits them from the sampling — and
Section 6 will use their agreement as a check on both.

## 4. The first case we can solve completely: the mean

We now have a well-defined finite object. The natural next question is whether it can be
computed without simulation, and the answer depends entirely on $R$. We begin with the
case where everything works, both because it must be checked before anything harder and
because the way it works turns out to be the exception rather than the rule.

Take $t(\mathbf{X}) = \bar X$ and $\theta(F) = \int x\,dF$, so that
$R(\mathbf{X}, F) = \bar X - \theta(F)$. Since $\theta(\hat F) = \bar x$, the bootstrap
version is

$$R^* \;=\; \bar X^* - \bar x .$$

### 4.1 The bootstrap mean

Each $X_i^*$ is drawn from $\hat F$, so its expectation under $\hat F$ is the mean of
that distribution,

$$E_*\,X_i^* \;=\; \sum_{j=1}^n \frac{1}{n}\,x_j \;=\; \bar x,$$

and by linearity $E_*\bar X^* = \bar x$, giving

$$E_*\,R^* \;=\; 0 .$$

Exactly zero, for every sample, at every $n$. The bootstrap reports no bias here, which
is correct: $\bar X$ is unbiased for the mean. It is worth registering that this came out
*exactly* rather than approximately, since the cases to come will not be so obliging.

### 4.2 The bootstrap variance

The $X_i^*$ are independent under $\hat F$ — this is the point of resampling with
replacement, and it would fail without it — so variances add:

$$\mathrm{Var}_*\big(\bar X^*\big) \;=\; \frac{1}{n^2}\sum_{i=1}^n \mathrm{Var}_*\big(X_i^*\big)
\;=\; \frac{\mathrm{Var}_*\big(X_1^*\big)}{n}.$$

It remains to compute the variance of a single draw from $\hat F$. That is just the
variance **of the distribution $\hat F$**, which puts mass $1/n$ on each $x_j$ and has
mean $\bar x$:

$$\mathrm{Var}_*\big(X_1^*\big) \;=\; \sum_{j=1}^{n} \frac{1}{n}\,(x_j - \bar x)^2
\;=\; \frac{1}{n}\sum_{j=1}^{n}(x_j - \bar x)^2 \;=:\; \hat\sigma^2 .$$

Therefore

$$\boxed{\;\mathrm{Var}_*\,R^* \;=\; \frac{\hat\sigma^2}{n}
\;=\; \frac{1}{n^2}\sum_{i=1}^{n}(x_i - \bar x)^2\;}$$

which for data taking the values $0$ and $1$ collapses to $\bar x(1-\bar x)/n$, the
paper's Eq. (2.8), since then $\sum_i (x_i - \bar x)^2 = n\bar x(1-\bar x)$.

### 4.3 Why $1/n$ and not $1/(n-1)$

The divisor deserves an explanation, because every reflex trained on sample variances
says it should be $n-1$, and the reflex is wrong here for a reason worth internalising.

The $-1$ in the usual sample variance corrects a bias: $\bar x$ is estimated from the
same data, so the squared deviations are on average slightly too small, and $n-1$
compensates. **None of that applies here.** In the calculation above we were not
estimating anything. We were computing the exact variance of a completely specified
distribution, $\hat F$, whose mean is exactly $\bar x$ — not an estimate of its mean,
its actual mean. A distribution's variance is what it is, and there is no bias to correct.

The same conclusion arrives from the plug-in view of Section 2.4. The variance
functional is $T(G) = \int (x - \int u\,dG)^2 dG$, and $T(\hat F) = \hat\sigma^2$ with
the $1/n$. The version with $1/(n-1)$ is not $T(\hat F)$ for any functional $T$ at all:
the sample size appears in it explicitly, not merely through $\hat F$. It is therefore
not a plug-in estimator, and one should not expect the bootstrap — which is the plug-in
principle and nothing else — to produce it.

Every $1/n$ appearing throughout this implementation has this origin, and none of them
is an oversight.

### 4.4 What the answer is worth

It is instructive to compare with the truth. The quantity we actually wanted is
$\mathrm{Var}_F R = \sigma^2(F)/n$, and the bootstrap has returned $\hat\sigma^2/n$. The
two differ **only** in that $\sigma^2(F)$ has been replaced by $\hat\sigma^2$.

So in this case the bootstrap introduces no error of its own whatsoever: whatever it
gets wrong, it gets wrong solely because $\hat\sigma^2 \ne \sigma^2$. The resampling
apparatus has faithfully reproduced the classical formula and then inherited the one
estimation problem that formula already had.

That inheritance is not free of consequences. Since
$E_F[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$, the bootstrap variance of the mean is
biased low by a factor $(n-1)/n$ — a 10% underestimate at $n = 10$. This is the mirror
image of what we will find for the median, where the bootstrap errs substantially in the
*other* direction, and the contrast between the two is diagnostic rather than
coincidental.

The exactness of this section is a consequence of $R^*$ being **linear** in
$\mathbf{P}^*$: writing $\bar X^* = \sum_i P_i^* x_i$ makes the dependence on the
resample affine, and expectations and variances of affine functions are available in
closed form with no approximation anywhere. Everything difficult in what follows is a
departure from that linearity, and the sections are ordered by how far each case departs.

![Exact bootstrap distribution against Monte Carlo, and the two errors](ded_exact_vs_mc.png)

Both panels are about the claim of Section 2.3, that the bootstrap distribution is fixed
once the data are. On the left, the grey bars are the exact distribution of
$R^* = \bar X^* - \bar x$ — all $\binom{15}{8} = 6435$ resamples with their weights — and
the two outlines are Monte Carlo with $N = 200$ and $N = 20\,000$. The outlines chase the
bars; the bars do not move. On the right, the same thing quantitatively: the Monte Carlo
error falls along a line of slope $-1/2$ over more than two decades, while the red line —
the distance from $\hat\sigma^2/n$ to the true $\sigma^2/n$, available here only because
we chose $F$ — does not respond to $N$ at all. Spending more computation buys the left
gap and never the right one.

*Verified numerically in [`bootstrap.py`](bootstrap.py): enumeration of all
$\binom{2n-1}{n}$ resamples reproduces $\hat\sigma^2/n$ to machine precision at
$n = 3, 5, 8, 10$, and Eq. (2.8) to twelve decimals.*
