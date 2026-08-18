"""The example markets used throughout this implementation.

The paper carries **no numbers**: Figs. 1-6 are schematic, drawn to display a
geometry rather than a data set, and Markowitz says on closing (p. 91) that
where the beliefs mu_i and sigma_ij come from is "another story" -- his stage 1,
left outside the paper.  So the inputs here are ours, not his, and they are
written down as annual expected returns, annual volatilities and a correlation
matrix, which is how such beliefs are actually held.

The sector names are the paper's own (p. 89): the railway/utility/mining/
manufacturing example is what he uses to argue that diversification has to be
across covariances and not merely across names.  The correlations reflect that
argument -- rails and manufacturing move together, bonds barely move with any
of them.
"""

from collections import namedtuple

import numpy as np

Market = namedtuple("Market", "names mu Sigma")


def from_correlations(names, mu, sd, corr):
    """Build a market from the form beliefs actually take: returns, vols, correlations.

    Sigma = diag(sd) @ corr @ diag(sd), which is sigma_ij = rho_ij sigma_i sigma_j,
    the paper's definition on p. 81.
    """
    mu, sd, corr = np.asarray(mu, float), np.asarray(sd, float), np.asarray(corr, float)
    if not np.allclose(corr, corr.T):
        raise ValueError("the correlation matrix is not symmetric")
    Sigma = corr * np.outer(sd, sd)
    eig = np.linalg.eigvalsh(Sigma)
    if eig.min() <= 0:
        raise ValueError(f"Sigma is not positive definite (lambda_min = {eig.min():.3e})")
    return Market(list(names), mu, Sigma)


def sub(market, names):
    """The market restricted to a subset of assets, in the order given."""
    idx = [market.names.index(n) for n in names]
    return Market(list(names), market.mu[idx], market.Sigma[np.ix_(idx, idx)])


def sectors():
    """Five assets: the paper's four sectors plus bonds.

    Rails and manufacturing are the strongly linked pair (rho = 0.60); mining is
    the high-return, high-variance outlier; bonds are nearly uncorrelated with
    everything, which is what makes them the backbone of the minimum-variance
    portfolio.
    """
    names = ["Rails", "Utilities", "Mining", "Manufacturing", "Bonds"]
    mu = [0.070, 0.065, 0.110, 0.090, 0.035]
    sd = [0.160, 0.130, 0.280, 0.190, 0.060]
    corr = [
        [1.00, 0.55, 0.30, 0.60, 0.10],
        [0.55, 1.00, 0.20, 0.45, 0.25],
        [0.30, 0.20, 1.00, 0.35, -0.05],
        [0.60, 0.45, 0.35, 1.00, 0.05],
        [0.10, 0.25, -0.05, 0.05, 1.00],
    ]
    return from_correlations(names, mu, sd, corr)


def triple_inside():
    """Three assets whose minimum-variance portfolio lies INSIDE the triangle (Fig. 2).

    Rails, mining and manufacturing, straight out of sectors(): three equities
    with comparable variances, none of them a proxy for another, so the
    unconstrained minimiser X-hat = (0.648, 0.120, 0.232) is a genuine portfolio
    and the sign constraint is slack around it.
    """
    return sub(sectors(), ["Rails", "Mining", "Manufacturing"])


def triple_outside():
    """Three assets whose minimum-variance portfolio lies OUTSIDE it (Fig. 3).

    The same three sectors as sectors() but with one belief changed:
    manufacturing is now both more volatile (22% against 19%) and more tightly
    linked to rails (rho = 0.85 against 0.60), which makes it a high-variance
    proxy of an asset already in the market.  That is the whole mechanism behind
    Fig. 3 -- a dominated asset gets a NEGATIVE unconstrained weight
    (X-hat = (0.465, -0.163, 0.698)), so the minimiser leaves the triangle and
    the efficient set has to start somewhere else.
    """
    names = ["Rails", "Manufacturing", "Utilities"]
    mu = [0.070, 0.090, 0.065]
    sd = [0.160, 0.220, 0.130]
    corr = [
        [1.00, 0.85, 0.55],
        [0.85, 1.00, 0.45],
        [0.55, 0.45, 1.00],
    ]
    return from_correlations(names, mu, sd, corr)
