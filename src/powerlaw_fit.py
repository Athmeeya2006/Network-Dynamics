"""
powerlaw_fit.py
===============
Clauset-Shalizi-Newman power-law fitting for discrete (integer) data such as
network degree sequences.

Implements the methodology of Clauset, Shalizi & Newman (2009),
"Power-law distributions in empirical data", SIAM Review 51(4):661-703:

    - discrete maximum-likelihood estimate (MLE) of the exponent alpha for a
      given lower cutoff x_min, using the Hurwitz-zeta normalisation;
    - automatic selection of x_min by minimising the Kolmogorov-Smirnov (KS)
      distance between the empirical and fitted distributions;
    - a goodness-of-fit p-value via the semiparametric bootstrap.

It also provides a naive log-log OLS exponent estimator (``ols_exponent``) so
that scripts can display the OLS bias alongside the unbiased MLE. This is the
methodological upgrade over the companion ER repo's OLS-only approach.

Do NOT fit power laws with OLS for inference; OLS is included only to expose
its bias.
"""

from __future__ import annotations

import numpy as np
from scipy.special import zeta
from scipy.optimize import minimize_scalar


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRETE POWER-LAW MLE
# ═══════════════════════════════════════════════════════════════════════════════

def _neg_loglike(alpha: float, tail: np.ndarray, xmin: int) -> float:
    """Negative log-likelihood of a discrete power law on the tail >= xmin."""
    n = len(tail)
    # zeta(alpha, xmin) is the Hurwitz zeta sum_{k=0}^inf (k + xmin)^{-alpha}
    return n * np.log(zeta(alpha, xmin)) + alpha * np.sum(np.log(tail))


def discrete_mle_alpha(tail: np.ndarray, xmin: int,
                       bounds: tuple[float, float] = (1.01, 6.0)) -> float:
    """
    Maximum-likelihood exponent for discrete power-law data x >= xmin.

    Parameters
    ----------
    tail : np.ndarray
        Integer observations with all values >= xmin.
    xmin : int
        Lower cutoff.
    bounds : tuple
        Search interval for alpha.

    Returns
    -------
    alpha : float
        MLE exponent.
    """
    res = minimize_scalar(_neg_loglike, args=(tail, xmin),
                          bounds=bounds, method='bounded')
    return float(res.x)


def _discrete_cdf_model(x: np.ndarray, alpha: float, xmin: int) -> np.ndarray:
    """
    Theoretical CDF P(X <= x) for a discrete power law with cutoff xmin.

    Uses the CCDF P(X >= x) = zeta(alpha, x) / zeta(alpha, xmin).
    """
    ccdf = zeta(alpha, x) / zeta(alpha, xmin)
    return 1.0 - ccdf + (x ** (-alpha)) / zeta(alpha, xmin)  # P(X<=x)=1-P(X>=x+1)


def ks_distance(tail: np.ndarray, alpha: float, xmin: int) -> float:
    """
    Kolmogorov-Smirnov distance between the empirical CDF of the tail and the
    fitted discrete power-law CDF.
    """
    xs = np.sort(np.unique(tail))
    n = len(tail)
    # empirical CDF evaluated at each distinct value
    emp = np.searchsorted(np.sort(tail), xs, side='right') / n
    # theoretical CDF P(X <= x) = 1 - zeta(alpha, x+1)/zeta(alpha, xmin)
    theo = 1.0 - zeta(alpha, xs + 1) / zeta(alpha, xmin)
    return float(np.max(np.abs(emp - theo)))


def fit_powerlaw(data, xmin_max: int | None = None) -> dict:
    """
    Full Clauset fit: select x_min by KS minimisation, MLE the exponent.

    Parameters
    ----------
    data : array_like of int
        Observations (e.g. a degree sequence). Zeros are dropped.
    xmin_max : int or None
        Largest x_min candidate to consider. Defaults to the value leaving at
        least ~10% of the data (and at least 2 distinct tail values) in the
        tail.

    Returns
    -------
    dict with keys:
        'alpha'  : MLE exponent at the selected x_min
        'xmin'   : selected lower cutoff
        'D'      : KS distance at the optimum
        'n_tail' : number of observations in the tail
        'n'      : total number of (positive) observations
    """
    data = np.asarray(data)
    data = data[data > 0].astype(int)
    n = len(data)
    candidates = np.unique(data)
    candidates = candidates[candidates >= 1]
    if xmin_max is not None:
        candidates = candidates[candidates <= xmin_max]
    # need at least a handful of distinct tail values to fit
    candidates = candidates[candidates <= np.max(data) - 1]

    best = {'D': np.inf, 'alpha': np.nan, 'xmin': int(candidates[0]),
            'n_tail': 0, 'n': n}
    for xmin in candidates:
        tail = data[data >= xmin]
        if len(np.unique(tail)) < 2:
            continue
        alpha = discrete_mle_alpha(tail, int(xmin))
        D = ks_distance(tail, alpha, int(xmin))
        if D < best['D']:
            best = {'D': D, 'alpha': alpha, 'xmin': int(xmin),
                    'n_tail': len(tail), 'n': n}
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRETE POWER-LAW SAMPLER (for the bootstrap)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_discrete_powerlaw(alpha: float, xmin: int, size: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Draw integer samples from a discrete power law p(x) ~ x^-alpha, x >= xmin,
    by inverting the CCDF with a doubling bracket + binary search.
    """
    z_xmin = zeta(alpha, xmin)
    u = rng.random(size)                 # target CCDF values in (0, 1)
    out = np.empty(size, dtype=int)
    for i in range(size):
        target = u[i]
        # CCDF(x) = zeta(alpha, x)/z_xmin is decreasing; find x with
        # CCDF(x) >= target > CCDF(x+1)
        x1 = xmin
        x2 = 2 * xmin
        while zeta(alpha, x2) / z_xmin > target:
            x1 = x2
            x2 *= 2
        # binary search in [x1, x2]
        while x2 - x1 > 1:
            mid = (x1 + x2) // 2
            if zeta(alpha, mid) / z_xmin > target:
                x1 = mid
            else:
                x2 = mid
        out[i] = x1
    return out


def bootstrap_pvalue(data, fit: dict | None = None, n_boot: int = 200,
                     seed: int = 0, xmin_max: int | None = None) -> float:
    """
    Semiparametric bootstrap goodness-of-fit p-value (Clauset et al. 2009).

    Generates synthetic datasets that follow the fitted power law above x_min
    and resample the empirical data below x_min, refits each, and reports the
    fraction whose KS distance exceeds the observed one. p > 0.1 means the
    power law is a plausible fit; p <= 0.1 means it can be ruled out.

    Parameters
    ----------
    data : array_like of int
        The observations.
    fit : dict or None
        Output of ``fit_powerlaw``; recomputed if None.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        RNG seed.
    xmin_max : int or None
        Passed through to refits for speed.

    Returns
    -------
    p : float
        Bootstrap p-value.
    """
    data = np.asarray(data)
    data = data[data > 0].astype(int)
    if fit is None:
        fit = fit_powerlaw(data, xmin_max=xmin_max)
    alpha, xmin, D_obs = fit['alpha'], fit['xmin'], fit['D']
    n = len(data)
    below = data[data < xmin]
    p_tail = fit['n_tail'] / n
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_boot):
        # number drawn from the power-law tail vs the empirical body
        n_tail = rng.binomial(n, p_tail)
        tail_part = sample_discrete_powerlaw(alpha, xmin, n_tail, rng)
        if len(below) > 0 and n - n_tail > 0:
            body_part = rng.choice(below, size=n - n_tail, replace=True)
            synth = np.concatenate([tail_part, body_part])
        else:
            synth = tail_part
        f = fit_powerlaw(synth, xmin_max=xmin_max)
        if f['D'] >= D_obs:
            count += 1
    return count / n_boot


# ═══════════════════════════════════════════════════════════════════════════════
# NAIVE OLS EXPONENT (shown only to expose its bias)
# ═══════════════════════════════════════════════════════════════════════════════

def ols_exponent(data, xmin: int = 1) -> tuple[float, float, float]:
    """
    Naive exponent estimate from an ordinary-least-squares fit to the log-log
    complementary CDF (CCDF). This is the biased estimator the Clauset method
    replaces; it is provided only so scripts can display the discrepancy.

    Returns
    -------
    alpha_ols : float
        Estimated exponent (slope of CCDF is 1 - alpha, so alpha = 1 - slope).
    intercept : float
        Fit intercept (log space).
    slope : float
        Raw CCDF slope.
    """
    data = np.asarray(data)
    data = data[data >= xmin].astype(int)
    xs = np.sort(np.unique(data))
    n = len(data)
    # empirical CCDF P(X >= x)
    ccdf = 1.0 - (np.searchsorted(np.sort(data), xs, side='left') / n)
    mask = ccdf > 0
    logx = np.log(xs[mask])
    logy = np.log(ccdf[mask])
    slope, intercept = np.polyfit(logx, logy, 1)
    alpha_ols = 1.0 - slope            # CCDF exponent is (1 - alpha)
    return float(alpha_ols), float(intercept), float(slope)
