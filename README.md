# Paper Reviews & From-Scratch Implementations

Reading foundational papers in statistics, machine learning and quantitative
finance — then reimplementing their core results in Python from scratch and
validating against the numbers published in the paper.

The goal is understanding, not production code: every implementation references
the specific equation or section it implements, and every claim is checked
against a figure, table or numerical value from the original work. Where the
paper turns out to be wrong, that is documented in place with the numerical
evidence rather than smoothed over.

**Diego Froilán García Campos** — B.Sc. Mathematics & Physics (University of
Oviedo), M.Sc. Big Data Analytics (Universidad Carlos III de Madrid).

## How it works

Each paper follows the same pipeline:

1. **Read** the paper in full.
2. **Review** — a structured write-up in `reviews/`: context, methodology with the
   key equations, results, strengths *and* limitations, links to related reviews.
3. **Derive** — `DERIVATIONS.md`, the mathematics developed end to end in its own
   logical order, to be read as a chapter rather than consulted. Figures are
   computed by the solver, so each one checks the step it accompanies.
4. **Implement** — a self-contained folder in `implementations/`, built
   incrementally, each piece validated against the paper.

```
papers/           Original PDFs (not versioned — see papers/INDEX.md)
reviews/          One markdown review per paper
implementations/  One folder per paper, each with its own README
ROADMAP.md        Reading queue, organised by topic
```

## Implemented so far

### Tibshirani (1996) — *Regression Shrinkage and Selection via the Lasso*

[Review](reviews/1996-tibshirani-lasso.md) ·
[Implementation](implementations/1996-tibshirani-lasso/) ·
**[Derivations](implementations/1996-tibshirani-lasso/DERIVATIONS.md)**

- **The paper's own algorithm** (Sec. 6): a quadratic program over the $2^p$ sign
  constraints, introduced one at a time, with a hand-written primal active-set
  inner solve. Deliberately *not* coordinate descent, which is Friedman et al.
  (2007) — eleven years later.
- **Validated against the paper**: Eq. 3 in the orthonormal case to `1e-15`,
  Table 1 and the prostate-cancer coefficient paths to the printed 0.01, Eqs. 5–6
  for two correlated predictors, and Figures 1, 2, 4 and 5.
- **Cross-checked externally** against `sklearn.linear_model.Lasso` to `8e-13`
  over the whole path — after deriving the conversion between the two objectives,
  which differ by a factor of $2N$ — and against LARS with no conversion at all.
- **Four discrepancies with the paper, documented with the numbers**: the
  `lweight` decimal-point typo corrected in the data file after 1996; GCV
  minimising at $s=0.69$ rather than the paper's 0.44; a lower limit missing from
  Eq. 6; and `max` printed where `min` belongs in the Stein risk formula.

The [derivations](implementations/1996-tibshirani-lasso/DERIVATIONS.md) build the
method from the problem statement to the choice of the penalty in 20 sections.
Two results in there go beyond the paper: the closed form that survives outside
the orthonormal case,

$$\hat\beta_A = \hat\beta^{\,\mathrm{ols}(A)} - \lambda\,(X_A^\top X_A)^{-1}s_A,$$

from which the piecewise linearity of the coefficient paths follows as a theorem;
and a proof of the stability the abstract only asserts — the lasso is a projection
onto a convex set, projections onto convex sets are non-expansive, so the fit
never moves further than the data did.

![Coefficient paths on the prostate data](implementations/1996-tibshirani-lasso/fig5_prostate_paths.png)

### Markowitz (1952) — *Portfolio Selection* · Vaswani et al. (2017) — *Attention Is All You Need*

Reviews done ([Markowitz](reviews/1952-markowitz-portfolio-selection.md),
[Transformer](reviews/2017-vaswani-attention.md)), implementations in progress.

### Efron (1979) — *Bootstrap Methods: Another Look at the Jackknife*

[Review](reviews/1979-efron-bootstrap.md) · implementation planned

The paper that introduced the bootstrap and showed the jackknife to be its linear
approximation.

## Running the code

```bash
conda env create -f environment.yml
conda activate papers
python implementations/1996-tibshirani-lasso/lasso.py
```

Python ≥ 3.11 with NumPy, SciPy, matplotlib and pandas; scikit-learn only in
`sklearn_check.py`, as an external reference to check against — no solver uses it.
Each implementation's README lists the scripts and what each one runs.

Data files are not versioned. The lasso implementation needs
`data/prostate.data`, downloadable from the *Elements of Statistical Learning*
website; its README says so.

## What's next

The [roadmap](ROADMAP.md) tracks the reading queue: papers mapped to the master's
syllabus, a quantitative-finance track (portfolio theory, covariance estimation,
financial time series), and a deep-learning track running from RNNs through the
Transformer to modern LLMs and generative models. `papers/INDEX.md` is the full
inventory, with download links.
