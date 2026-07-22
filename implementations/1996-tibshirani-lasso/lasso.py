"""
Lasso via coordinate descent — Tibshirani (1996).

Piece 1 of the implementation: the core solver.

The paper poses the lasso as constrained least squares (Eq. 1):

    minimize  sum_i (y_i - a - sum_j x_ij b_j)^2   s.t.  sum_j |b_j| <= t

We solve the equivalent Lagrangian form with coordinate descent + soft
thresholding (the modern standard, Sec. 6 note; Friedman et al. 2007):

    f(b) = (1 / 2N) * ||y - X b||^2  +  lam * sum_j |b_j|

Inputs are assumed *standardized* as in the paper (Eq. 1, p. 268):
columns of X centered and scaled so that (1/N) * sum_i x_ij^2 = 1, and y
centered so the intercept drops out.

Validation (see __main__): in an orthonormal design (X^T X / N = I) the paper
gives the closed form (Eq. 3) as soft thresholding of the OLS coefficients, so
coordinate descent must reproduce it exactly.
"""

import numpy as np


def soft_threshold(z, gamma):
    """Soft-thresholding operator S(z, gamma) = sign(z) (|z| - gamma)^+.

    This is the one-dimensional lasso solution and the building block of every
    coordinate update (Eq. 3, Sec. 2.2). Contrast with ridge, which shrinks
    proportionally, and subset selection (hard thresholding).
    """
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def lasso_coordinate_descent(X, y, lam, max_iter=1000, tol=1e-8):
    """Fit the lasso by cyclic coordinate descent.

    Parameters
    ----------
    X : (N, p) array. Standardized: each column centered with (1/N) sum x_ij^2 = 1.
    y : (N,)   array. Centered (mean 0), so no intercept is needed.
    lam : float. L1 penalty (the Lagrange multiplier of the constraint in Eq. 1).
    max_iter : int. Maximum full sweeps over the p coordinates.
    tol : float. Stop when the largest coefficient change in a sweep < tol.

    Returns
    -------
    beta : (p,) array of estimated coefficients.

    Coordinate update
    -----------------
    Fixing every coordinate but j, the partial residual is
        r^(j) = y - sum_{k != j} x_k b_k .
    Because (1/N) ||x_j||^2 = 1, the unpenalized coordinate minimizer is
        rho_j = (1/N) x_j^T r^(j) ,
    and adding the L1 term gives  b_j = S(rho_j, lam)  (Eq. 3 applied per axis).
    """
    N, p = X.shape
    beta = np.zeros(p)

    # Full residual r = y - X beta, updated incrementally as coordinates change.
    residual = y - X @ beta

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            # Partial residual r^(j): add coordinate j back into the residual.
            # r^(j) = residual + x_j * beta_j
            beta_j_old = beta[j]
            r_partial = residual + X[:, j] * beta_j_old

            # rho_j = (1/N) x_j^T r^(j): coordinate-wise OLS fit on the residual.
            rho_j = (X[:, j] @ r_partial) / N

            # Soft-thresholded update (Eq. 3 per coordinate).
            beta[j] = soft_threshold(rho_j, lam)

            # Keep the full residual consistent with the new beta_j.
            residual = r_partial - X[:, j] * beta[j]

            max_change = max(max_change, abs(beta[j] - beta_j_old))

        if max_change < tol:
            break

    return beta


def standardize(X, y):
    """Center/scale X to the paper's convention and center y.

    Returns standardized (X_std, y_centered). Columns of X_std have mean 0 and
    (1/N) sum x_ij^2 = 1 (i.e. divided by the population std).
    """
    N = X.shape[0]
    X_centered = X - X.mean(axis=0)
    col_norm = np.sqrt((X_centered ** 2).sum(axis=0) / N)  # population std
    X_std = X_centered / col_norm
    y_centered = y - y.mean()
    return X_std, y_centered


# ---------------------------------------------------------------------------
# Validation against the paper's closed form (Eq. 3), orthonormal design.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, p = 200, 8

    # Build an orthonormal design: Q has orthonormal columns (Q^T Q = I_p);
    # scaling by sqrt(N) gives (1/N) X^T X = I, the paper's Eq. 3 setting.
    Q, _ = np.linalg.qr(rng.standard_normal((N, p)))
    X = Q * np.sqrt(N)

    # A sparse ground truth so the selection behaviour is visible: three real
    # effects, five null. (Same flavour as the simulation in Sec. 7.)
    beta_true = np.array([3.0, 0.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0])
    y = X @ beta_true + 1.0 * rng.standard_normal(N)
    y = y - y.mean()

    # In an orthonormal design the coordinate-wise OLS fit is beta_ols = X^T y / N,
    # and the paper's closed form (Eq. 3) is its soft-thresholded version.
    beta_ols = (X.T @ y) / N

    print(f"{'lam':<6}{'max|CD - closed form|':<24}{'#nonzero':<10}coefficients")
    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0]
    all_ok = True
    for lam in lambdas:
        beta_cd = lasso_coordinate_descent(X, y, lam)
        beta_closed = soft_threshold(beta_ols, lam)  # Eq. 3
        err = np.max(np.abs(beta_cd - beta_closed))
        all_ok &= err < 1e-8
        coefs = np.array2string(beta_cd, precision=2, suppress_small=True)
        print(f"{lam:<6}{err:<24.2e}{int(np.sum(beta_cd != 0)):<10}{coefs}")

    print("\nGround-truth coefficients:            ", beta_true)
    print("As lam grows, the null coefficients are zeroed first, then the")
    print("survivors shrink toward 0 (soft thresholding, Eq. 3).")
    print("\nValidation vs Eq. (3) closed form:", "PASS" if all_ok else "FAIL")
