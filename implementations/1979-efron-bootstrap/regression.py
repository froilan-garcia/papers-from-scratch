"""Regression — Efron (1979), Sec. 7.

The model is Eqs. (7.1)-(7.2): x_i = g_i(beta) + eps_i with the eps_i
independent draws from an unknown F centred at zero, fitted by least squares
(7.3).  What is wanted is the sampling distribution of beta-hat.

The bootstrap of Sec. 2 does not apply directly, because the x_i are not
identically distributed — each has its own mean g_i(beta).  What IS
identically distributed is the errors, so F-hat is built from the residuals
(7.4) and the resampling happens there (7.5), the fit being redone on each
artificial data set (7.6).  For the linear model this reproduces the
classical answer exactly, Cov_* beta-hat* = sigma-hat^2 G^{-1} (7.7).

The interesting part is what the jackknife does instead, and why.  Deleting
one row at a time gives Eq. (7.8),

    Cov beta-hat  ~  G^{-1} [ sum_i c_i' c_i eps-hat_i^2 ] G^{-1},

of which Efron says, in the paper's only exclamation mark, that it "doesn't
look at all like (7.7)".  Sec. 5 explains the discrepancy and this module
verifies it: BOTH expressions are the same formula, Eq. (5.10), applied to
two different data sets.  The difference is one index.

    raw data          sum_i   c_i' c_i eps-hat_i^2      each row keeps its
                                                        own residual
    symmetrised (7.9) sum_i,j c_i' c_i eps-hat_j^2  =  G * n sigma-hat^2

The second sum factorises and the first does not, and that factorisation is
the whole of assumption (7.2).  Efron's remark that "the bootstrap implicitly
does this at step (7.5)" is exactly right: drawing eps_i* from all n
residuals regardless of i is what breaks the association between a row and
its own error.

The last section leaves the linear model, where the paper says these methods
"may really be necessary".  There the linearisation is an approximation
again, and the two answers separate.

Read with the MKL note in the README: this module multiplies matrices, so it
is the one most likely to abort on a broken BLAS.
"""

import numpy as np
from scipy.optimize import least_squares

from jackknife import simplex_derivatives


# --------------------------------------------------------------------------
# The linear model and its two covariance formulas
# --------------------------------------------------------------------------
#
# G = C'C throughout, as in the paper.  The inverse is formed explicitly
# rather than solved for, because Eqs. (7.7) and (7.8) are statements about
# G^{-1} as an object and both need it twice.

def fit(C, x, weights=None):
    """Least squares (7.3) for g_i(beta) = c_i beta, optionally weighted.

    The weighted version is what Sec. 5 needs: a fit is a function of the
    weight vector, and the weights are the P* of Eq. (5.1).
    """
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    if weights is None:
        Cw = C
    else:
        Cw = C * np.asarray(weights, dtype=float)[:, None]
    return np.linalg.solve(Cw.T @ C, Cw.T @ x)


def residuals(C, x, beta=None):
    """eps-hat_i = x_i - c_i beta-hat, the atoms of F-hat in Eq. (7.4)."""
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    beta = fit(C, x) if beta is None else beta
    return x - C @ beta


def cov_classical(C, x):
    """Cov_* beta-hat* = sigma-hat^2 G^{-1} — Eq. (7.7).

    sigma-hat^2 = sum_i eps-hat_i^2 / n, with the 1/n of Sec. 2: F-hat is a
    distribution, not a sample.  The paper flags this as the one point where
    (7.7) departs from traditional theory, which would divide by n - p.
    """
    C = np.asarray(C, dtype=float)
    Ginv = np.linalg.inv(C.T @ C)
    return float(np.mean(residuals(C, x) ** 2)) * Ginv


def cov_sandwich(C, x):
    """G^{-1} [sum_i c_i' c_i eps-hat_i^2] G^{-1} — Eq. (7.8).

    Efron attributes it to Hinkley's infinitesimal jackknife on the rows.  It
    is, unchanged, the heteroskedasticity-robust covariance estimator in
    standard use today.
    """
    C = np.asarray(C, dtype=float)
    Ginv = np.linalg.inv(C.T @ C)
    eps = residuals(C, x)
    return Ginv @ (C.T * eps ** 2) @ C @ Ginv


# --------------------------------------------------------------------------
# The bootstrap of Sec. 7 — resample the residuals, not the rows
# --------------------------------------------------------------------------

def residual_bootstrap(C, x, n_boot, rng, beta=None):
    """beta-hat* replications — Eqs. (7.4)-(7.6).

    Step (7.5) is where assumption (7.2) enters the machinery: eps_i* is
    drawn from ALL the residuals, so the error attached to row i is no longer
    the one row i produced.  Nothing marks this as an assumption in the code,
    which is precisely why it is worth pointing at.
    """
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    n = C.shape[0]
    beta = fit(C, x) if beta is None else beta
    eps = x - C @ beta

    stars = (C @ beta)[None, :] + eps[rng.integers(0, n, size=(n_boot, n))]
    G = C.T @ C
    return np.linalg.solve(G, (stars @ C).T).T


def pairs_bootstrap(C, x, n_boot, rng):
    """beta-hat* resampling whole rows (c_i, x_i).

    NOT in the paper — Sec. 7 resamples residuals only — and included as the
    alternative that assumption (7.2) is hiding: it makes no use of the
    errors being identically distributed, at a price measured in __main__.
    A resample with fewer than p distinct rows would give a singular G*, the
    same degeneracy that leaves the correlation undefined in correlation.py;
    with n = 20 and p = 3 it does not occur.
    """
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    n = C.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    Cs, xs = C[idx], x[idx]
    G = np.einsum("bnp,bnq->bpq", Cs, Cs)
    b = np.einsum("bnp,bn->bp", Cs, xs)
    return np.linalg.solve(G, b[..., None])[..., 0]


# --------------------------------------------------------------------------
# The jackknife on rows, and the symmetrisation that repairs it
# --------------------------------------------------------------------------

def jackknife_rows_cov(C, x):
    """Ordinary jackknife covariance, deleting one row at a time.

    Tukey's formula on the n leave-one-row-out fits, computed by actually
    deleting rows and refitting.
    """
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    n = C.shape[0]
    betas = np.array([fit(np.delete(C, i, axis=0), np.delete(x, i))
                      for i in range(n)])
    d = betas - betas.mean(axis=0)
    return (n - 1) / n * (d.T @ d)


def jackknife_rows_cov_leverage(C, x):
    """The same, in closed form — Efron's "quite similar expression".

    Deleting row i moves the fit by exactly

        beta-hat_(i) - beta-hat = - G^{-1} c_i' eps-hat_i / (1 - h_i),
        h_i = c_i G^{-1} c_i',

    so the jackknife is Eq. (7.8) with every residual inflated by its own
    leverage factor 1/(1 - h_i), plus the centring.  That is how similar
    "quite similar" is, and it is checked against the deletion above in
    __main__.  The inflation is not decoration: h_i averages p/n, so the
    jackknife is systematically the larger of the two, which turns out to
    matter in small samples.
    """
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    n = C.shape[0]
    Ginv = np.linalg.inv(C.T @ C)
    eps = residuals(C, x)
    h = np.einsum("ip,pq,iq->i", C, Ginv, C)
    d = -(Ginv @ C.T) * (eps / (1 - h))          # columns: beta_(i) - beta-hat
    d = d - d.mean(axis=1, keepdims=True)
    return (n - 1) / n * (d @ d.T)


def influence_cov(C, x, n_real=None, step=1e-3):
    """Eq. (5.10) applied to a least-squares fit: sum_l U_l U_l' / (n_real*M).

    U_l is the derivative of beta-hat with respect to the weight on point l,
    obtained from `simplex_derivatives` exactly as in Sec. 5 — the same
    routine, on a different statistic.

    M is the number of points carried in the fit and `n_real` the number of
    genuine observations behind them.  They differ only for a symmetrised
    data set, and keeping them apart is the correction Eq. (5.16) makes for
    an "artificially increased amount of data": with M = n it is Eq. (5.10)
    unchanged.
    """
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    M, p = C.shape
    n_real = M if n_real is None else n_real
    U = np.array([simplex_derivatives(lambda w, k=k: fit(C, x, w)[k], M,
                                      step=step, second=False)[0]
                  for k in range(p)])
    return U @ U.T / (n_real * M)


def symmetrise(C, x):
    """The n^2 hypothetical points of Eq. (7.9): x_ij = c_i beta-hat + eps-hat_j.

    Every design row is paired with every residual, which is what "the errors
    all come from the same F" says when written as data.  Returns the
    enlarged (C, x).
    """
    C, x = np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    n = C.shape[0]
    beta = fit(C, x)
    eps = x - C @ beta
    return np.repeat(C, n, axis=0), np.repeat(C @ beta, n) + np.tile(eps, n)


# --------------------------------------------------------------------------
# The nonlinear model of Eq. (7.1), where no formula is available
# --------------------------------------------------------------------------

def nonlinear_fit(t, x, model, beta0):
    """beta-hat of Eq. (7.3) for a general g_i(beta)."""
    return least_squares(lambda b: x - model(t, b), beta0).x


def nonlinear_residual_bootstrap(t, x, model, beta_hat, n_boot, rng):
    """Eqs. (7.4)-(7.6) with the fit of (7.6) done numerically each time."""
    fitted = model(t, beta_hat)
    eps = x - fitted
    n = t.shape[0]
    out = np.empty((n_boot, beta_hat.shape[0]))
    for b in range(n_boot):
        star = fitted + eps[rng.integers(0, n, size=n)]
        out[b] = nonlinear_fit(t, star, model, beta_hat)
    return out


def gauss_newton_cov(t, x, model, beta_hat, h=1e-6):
    """sigma-hat^2 (J'J)^{-1}: Eq. (7.7) after linearising g about beta-hat.

    The delta method again, in its regression clothing — the same move that
    turned the bootstrap into the jackknife in Sec. 5, applied to the model
    instead of to the statistic.
    """
    J = np.empty((t.shape[0], beta_hat.shape[0]))
    for k in range(beta_hat.shape[0]):
        step = np.zeros_like(beta_hat)
        step[k] = h
        J[:, k] = (model(t, beta_hat + step) - model(t, beta_hat - step)) / (2 * h)
    s2 = float(np.mean((x - model(t, beta_hat)) ** 2))
    return s2 * np.linalg.inv(J.T @ J)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # A design and a data set.  The paper gives none for Sec. 7, so this is
    # simulated; it stays fixed for every experiment below, since everything
    # here is conditional on the design.
    n, p = 20, 3
    C = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, n)])
    beta_true = np.array([1.0, 2.0, -0.5])
    sigma = 0.7
    x = C @ beta_true + sigma * rng.normal(size=n)

    beta_hat = fit(C, x)
    print(f"Linear model, n = {n}, p = {p}:\n")
    print(f"  beta      {np.array2string(beta_true, precision=4)}")
    print(f"  beta-hat  {np.array2string(beta_hat, precision=4)}")
    print(f"  sum of residuals: {residuals(C, x).sum():.1e}"
          "  (the intercept forces it, and Sec. 7.9 will need it)")

    # --- Eq. (7.7): the bootstrap reproduces the classical formula ---------
    #
    # beta-hat* = G^{-1}C'X* with X* = C beta-hat + eps*, and the eps_i* are
    # independent with variance sigma-hat^2 under F-hat, so
    # Cov_* = sigma-hat^2 G^{-1}C'C G^{-1} = sigma-hat^2 G^{-1} exactly.
    # Monte Carlo can only approach it, at the usual N^{-1/2}.

    print("\nEq. (7.7), by resampling residuals:\n")
    print(f"{'N':>8}  {'max |Cov_* - (7.7)|':>20}  {'max |E_* beta* - beta-hat|':>27}")
    target = cov_classical(C, x)
    for n_boot in [1000, 10_000, 100_000]:
        stars = residual_bootstrap(C, x, n_boot, rng)
        cov = np.cov(stars.T, bias=True)
        print(f"{n_boot:8d}  {np.abs(cov - target).max():20.2e}"
              f"  {np.abs(stars.mean(0) - beta_hat).max():27.2e}")

    # --- Eq. (7.8): the jackknife's answer, which looks nothing like it ----

    print("\nEqs. (7.7) and (7.8), standard errors of each coefficient:\n")
    print(f"{'':>26}  " + "  ".join(f"{f'beta_{k}':>9}" for k in range(p)))
    for label, cov in [("(7.7)  sigma-hat^2 G^-1", cov_classical(C, x)),
                       ("(7.8)  sandwich", cov_sandwich(C, x)),
                       ("(5.10) on the raw data", influence_cov(C, x)),
                       ("jackknife, delete a row", jackknife_rows_cov(C, x)),
                       ("the same, by leverage", jackknife_rows_cov_leverage(C, x))]:
        se = np.sqrt(np.diag(cov))
        print(f"{label:>26}  " + "  ".join(f"{v:9.5f}" for v in se))
    print("  rows 2 and 3 agree to six digits: Eq. (7.8) IS Eq. (5.10) applied")
    print("  to the rows.  Rows 4 and 5 agree to machine precision, which is")
    print("  what makes the ordinary jackknife 'quite similar': it is Eq. (7.8)")
    print("  with each residual inflated by 1/(1 - h_i)")

    # --- Eq. (7.9): symmetrising the data recovers (7.7) --------------------
    #
    # The same Eq. (5.10), on the n^2 points of (7.9), with the (5.16)
    # correction for the fact that there are still only n real observations.

    Cs, xs = symmetrise(C, x)
    sym = influence_cov(Cs, xs, n_real=n)
    print(f"\nEq. (7.9): the same formula on the {Cs.shape[0]} symmetrised points:\n")
    print(f"{'':>26}  " + "  ".join(f"{f'beta_{k}':>9}" for k in range(p)))
    for label, cov in [("(5.16) on (7.9)", sym), ("(7.7)  sigma-hat^2 G^-1", target)]:
        print(f"{label:>26}  " + "  ".join(f"{v:9.5f}"
                                           for v in np.sqrt(np.diag(cov))))
    print(f"  max |difference| over the whole matrix: {np.abs(sym - target).max():.1e}")
    print("  one index apart: sum_i c_i'c_i e_i^2 does not factor and")
    print("  sum_ij c_i'c_i e_j^2 = G * n sigma-hat^2 does.  Resampling the")
    print("  residuals performs that symmetrisation without being told to")

    # --- Which of them is right, and when ----------------------------------
    #
    # Efron notes that the jackknife-style estimates are consistent "without
    # assumption (7.2) that the residuals are identically distributed.  The
    # price of such complete generality is low efficiency."  Both halves of
    # that sentence are measurable.  The design is fixed, so for a linear
    # model the truth is available in closed form:
    #
    #     Cov beta-hat = G^{-1} [sum_i c_i' c_i sigma_i^2] G^{-1},
    #
    # which reduces to sigma^2 G^{-1} only when the sigma_i agree.

    Ginv = np.linalg.inv(C.T @ C)
    n_trials = 2000
    estimators = [
        ("(7.7)  residual bootstrap",
         lambda C, xt: np.sqrt(cov_classical(C, xt)[1, 1])),
        ("(7.8)  sandwich", lambda C, xt: np.sqrt(cov_sandwich(C, xt)[1, 1])),
        ("jackknife on rows",
         lambda C, xt: np.sqrt(jackknife_rows_cov_leverage(C, xt)[1, 1])),
        ("pairs bootstrap (not in the paper)",
         lambda C, xt: pairs_bootstrap(C, xt, 400, rng)[:, 1].std()),
    ]

    print(f"\nWho is right: standard error of beta_1, {n_trials} trials\n")
    summary = {}
    for label, scale in [("homoskedastic", np.ones(n)),
                         ("heteroskedastic", 0.4 + 1.4 * np.abs(C[:, 1]))]:
        # Same average error power in both, so the rows are comparable.
        sig = sigma * scale / np.sqrt(np.mean(scale ** 2))
        truth = np.sqrt(np.diag(Ginv @ (C.T * sig ** 2) @ C @ Ginv))[1]
        values = {name: [] for name, _ in estimators}
        for _ in range(n_trials):
            xt = C @ beta_true + sig * rng.normal(size=n)
            for name, f in estimators:
                values[name].append(f(C, xt))
        print(f"  {label} errors, truth = {truth:.4f}:\n")
        print(f"{'':>38}  {'mean':>7}  {'/truth':>7}  {'s.d.':>7}")
        for name, _ in estimators:
            v = np.array(values[name])
            summary[(label, name)] = (v.mean() / truth, v.std())
            print(f"{name:>38}  {v.mean():7.4f}  {v.mean() / truth:7.3f}"
                  f"  {v.std():7.4f}")
        print()

    ratio, sd = summary[("homoskedastic", estimators[0][0])]
    sd_sandwich = summary[("homoskedastic", estimators[1][0])][1]
    sd_jack = summary[("homoskedastic", estimators[2][0])][1]
    print(f"  The {(1 - ratio) * 100:.0f}% shortfall of (7.7) under homoskedasticity is"
          " the 1/n")
    print("  in sigma-hat^2: E sigma-hat^2 = (n-p)/n sigma^2, so the standard")
    print(f"  error carries sqrt({n - p}/{n}) = {np.sqrt((n - p) / n):.3f},"
          " and Jensen's inequality")
    print("  on the square root accounts for the remainder.  Its real advantage")
    print(f"  is the last column: {1 - sd / sd_sandwich:.0%} less variable than the"
          " sandwich and")
    print(f"  {1 - sd / sd_jack:.0%} less than the jackknife.  That is the efficiency")
    print("  Efron is charging for.")
    print("\n  Under heteroskedasticity (7.7) estimates a quantity that no")
    print("  longer exists, and no amount of resampling repairs it: the")
    print("  residual bootstrap inherits every assumption that went into")
    print("  F-hat, and (7.2) is one of them.  What is worth noticing is")
    print(f"  which alternative survives.  Eq. (7.8) reaches only "
          f"{summary[('heteroskedastic', estimators[1][0])][0]:.2f} of the")
    print(f"  truth at n = 20, while the jackknife reaches "
          f"{summary[('heteroskedastic', estimators[2][0])][0]:.2f} -- the")
    print("  leverage factors 1/(1 - h_i) are not a rounding detail but the")
    print("  small-sample correction, and they are the reason to prefer")
    print("  Efron's 'quite similar expression' to Eq. (7.8) itself.  The")
    print("  price is again the spread: "
          f"{summary[('heteroskedastic', estimators[2][0])][1]:.4f}"
          f" against {summary[('heteroskedastic', estimators[0][0])][1]:.4f}.")

    # --- The nonlinear model, where the formulas run out -------------------
    #
    # Eq. (7.1) allows any g_i(beta).  With g nonlinear there is no G, no
    # (7.7) and no (7.8): what is usually reported is sigma-hat^2 (J'J)^{-1},
    # the same formula after linearising g about beta-hat.  That is the delta
    # method once more, and the bootstrap is again the thing it approximates.

    def model(t, b):
        return b[0] * np.exp(b[1] * t)

    t = np.linspace(0, 2, 24)
    beta_nl = np.array([3.0, -1.2])
    print("\nNonlinear g_i(beta) = b0 exp(b1 t), n = 24, beta = (3, -1.2):\n")
    print(f"{'sigma':>7}  {'beta-hat_1':>11}  {'bootstrap s.e.':>14}"
          f"  {'sigma^2(J J)^-1':>16}  {'ratio':>6}  {'boot. skew':>11}")
    sweep = []
    for s in [0.05, 0.15, 0.40, 0.80, 1.50]:
        x_nl = model(t, beta_nl) + s * rng.normal(size=t.size)
        bh_nl = nonlinear_fit(t, x_nl, model, np.array([1.0, -1.0]))
        stars = nonlinear_residual_bootstrap(t, x_nl, model, bh_nl, 1500, rng)
        gn = np.sqrt(np.diag(gauss_newton_cov(t, x_nl, model, bh_nl)))
        v = stars[:, 1]
        skew = float(((v - v.mean()) ** 3).mean() / v.std() ** 3)
        sweep.append((s, v.std(), gn[1], v.std() / gn[1], skew))
        print(f"{s:7.2f}  {bh_nl[1]:11.4f}  {v.std():14.4f}  {gn[1]:16.4f}"
              f"  {v.std() / gn[1]:6.3f}  {skew:11.3f}")

    mid = sweep[3]
    print("  the linearisation gets the SCALE right long after it has stopped")
    print(f"  getting the SHAPE right: at sigma = {mid[0]:.1f} the ratio is still"
          f" {mid[3]:.3f}")
    print(f"  while the skewness has reached {mid[4]:.2f}.  A symmetric answer is all")
    print("  sigma^2 (J'J)^-1 can ever give, being the covariance of a linear")
    print("  map; the asymmetry is the curvature of g that the Jacobian threw")
    print("  away, and only resampling recovers it.")
    print(f"\n  The last row is a different failure.  At sigma = {sweep[4][0]:.1f}"
          " the decay is")
    print("  no longer identifiable in every resample -- some replicates fit a")
    print("  nearly flat exponential and send b1 off -- so the bootstrap")
    print("  distribution is heavy-tailed and its standard deviation means")
    print(f"  little.  That instability is real and is information: (J'J)^-1")
    print(f"  reports a comfortable {sweep[4][2]:.2f} and cannot express it at all.")
