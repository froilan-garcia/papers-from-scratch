# Lasso via coordinate descent — Tibshirani (1996)

Implementation of the lasso from *Regression Shrinkage and Selection via the
Lasso* (Tibshirani, 1996). See the [review](../../reviews/1996-tibshirani-lasso.md)
for the paper's context and results.

Built incrementally. Current status:

| # | Piece | Status |
|---|-------|--------|
| 1 | Coordinate-descent solver + soft thresholding | ✅ `lasso.py` |
| 2 | Fig. 1 — the four shrinkage functions (subset / ridge / lasso / garotte) | ⬜ pending |
| 3 | Fig. 5 + Table 1 — coefficient paths on the prostate-cancer data | ⬜ pending |
| 4 | Table 3 — MSE comparison OLS / lasso-CV / ridge over 50 replicates | ⬜ pending |
| 5 | Cross-check the solver against `sklearn.linear_model.Lasso` | ⬜ pending |

## Piece 1 — the solver (`lasso.py`)

Solves the Lagrangian form of the lasso (Eq. 1) by cyclic coordinate descent
with soft thresholding — the modern standard (Sec. 6 note; Friedman et al. 2007),
not the quadratic-programming algorithm of the original paper.

    f(b) = (1 / 2N) * ||y - X b||^2  +  lam * sum_j |b_j|

Inputs follow the paper's convention (Eq. 1): columns of `X` standardized to
`(1/N) sum_i x_ij^2 = 1`, and `y` centered so the intercept drops out. A
`standardize()` helper is included.

### Run

```bash
python lasso.py
```

Requires only `numpy`.

### Validation — against the paper, not a library

In an orthonormal design (`X^T X / N = I`) the paper gives the closed-form
solution as soft thresholding of the OLS coefficients (**Eq. 3**). The script
builds such a design and checks that coordinate descent reproduces that closed
form. It does, to machine precision:

```
lam   max|CD - closed form|   #nonzero  coefficients
0.0   1.11e-15                8         [ 2.95  0.05  1.58 -0.08  0.06  1.93  0.03  0.  ]
0.25  1.78e-15                3         [ 2.7   0.    1.33 -0.    0.    1.68  0.    0.  ]
0.5   1.78e-15                3         [ 2.45  0.    1.08 -0.    0.    1.43  0.    0.  ]
1.0   1.55e-15                3         [ 1.95  0.    0.58 -0.    0.    0.93  0.    0.  ]
2.0   0.00e+00                1         [ 0.95  0.    0.   -0.    0.    0.    0.    0.  ]

Validation vs Eq. (3) closed form: PASS
```

With a sparse ground truth `beta = [3, 0, 1.5, 0, 0, 2, 0, 0]`, raising `lam`
first zeroes the five null coefficients (variable selection) and then shrinks
the survivors linearly toward zero — the signature of soft thresholding, versus
ridge's proportional shrinkage.
