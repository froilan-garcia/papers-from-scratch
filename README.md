# Paper Reviews & From-Scratch Implementations

Reading foundational papers in statistics, machine learning and quantitative
finance — then reimplementing their core results in Python from scratch and
validating against the numbers published in the paper.

The goal is understanding, not production code: every implementation references
the specific equation or section it implements, and every claim is checked
against a figure, table or numerical value from the original work.

**Diego Froilán García Campos** — B.Sc. Mathematics & Physics (University of
Oviedo), M.Sc. Big Data Analytics (Universidad Carlos III de Madrid).

## How it works

Each paper follows the same pipeline:

1. **Read** the paper in full.
2. **Review** — a structured markdown write-up in `reviews/`: context, methodology
   with the key equations in LaTeX, results, strengths *and* limitations, and
   links to related reviews already in the repo.
3. **Implement** — a self-contained folder in `implementations/`, built
   incrementally in pieces, each validated against the paper.

```
papers/           Original PDFs (not versioned — see papers/INDEX.md)
reviews/          One markdown review per paper
implementations/  One folder per paper, each with its own README
ROADMAP.md        Reading queue, organised by topic
```

## Implemented so far

### Tibshirani (1996) — *Regression Shrinkage and Selection via the Lasso*

[Review](reviews/1996-tibshirani-lasso.md) ·
[Implementation](implementations/1996-tibshirani-lasso/)

- **Coordinate-descent solver** with soft thresholding, written from scratch in
  NumPy. Validated to machine precision (`1e-15`) against the paper's closed-form
  solution for an orthonormal design (Eq. 3).
- **Figure 1 reproduced** — the four shrinkage functions (subset selection,
  ridge, lasso, garotte), which show geometrically *why* the L1 penalty produces
  exact zeros while ridge never does.

![Shrinkage functions](implementations/1996-tibshirani-lasso/fig1_shrinkage_functions.png)

### Efron (1979) — *Bootstrap Methods: Another Look at the Jackknife*

[Review](reviews/1979-efron-bootstrap.md) · implementation in progress

The paper that introduced the bootstrap and showed the jackknife to be its
linear approximation. Planned implementation covers the non-parametric,
parametric, smoothed and symmetric bootstrap, the jackknife comparison, and
(as clearly-labelled post-1979 extensions) percentile/BCa confidence intervals
and the block bootstrap for dependent data.

## Running the code

```bash
conda env create -f environment.yml
conda activate papers
python implementations/1996-tibshirani-lasso/lasso.py
```

Python ≥ 3.11 with NumPy, SciPy, matplotlib, pandas and scikit-learn. Scripts
are plain `.py` files with `# %%` cell markers, so they run either end-to-end or
cell-by-cell in an interactive window.

## What's next

The [roadmap](ROADMAP.md) tracks the reading queue: papers mapped to the
master's syllabus, a quantitative-finance track (Markowitz, Black–Scholes,
Kalman), and a deep-learning track running from RNNs through the Transformer to
modern LLMs and generative models. `papers/INDEX.md` is the full inventory with
download links.
