"""
continuation.py
===============
Numerical continuation and bifurcation analysis tools.

Provides:
    - find_fixed_point  : Newton's method for f(x*, args) = 0
    - track_branch      : natural parameter continuation with stability
    - classify_2d       : classify a 2D fixed point from its Jacobian
    - stability_1d      : stability of a 1D fixed point from f'(x*)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED-POINT FINDING
# ═══════════════════════════════════════════════════════════════════════════════

def find_fixed_point(f, x0, args=(), full_output=False):
    """
    Find a fixed point x* such that f(x*, *args) = 0.

    Parameters
    ----------
    f : callable
        Function f(x, *args) -> same shape as x.
    x0 : float or array_like
        Initial guess.
    args : tuple
        Additional arguments to f.
    full_output : bool
        If True, return (x*, info_dict).

    Returns
    -------
    x_star : float or np.ndarray
        Fixed point.
    """
    result = fsolve(f, x0, args=args, full_output=True)
    x_star, info, ier, msg = result
    if ier != 1 and not full_output:
        pass  # silently return best estimate
    if full_output:
        return x_star, {"converged": ier == 1, "fval": info["fvec"]}
    return x_star


def stability_1d(df, x_star, r):
    """
    Determine stability of a 1D fixed point.

    Parameters
    ----------
    df : callable
        Derivative df/dx evaluated at (x, r).
    x_star : float
        Fixed point.
    r : float
        Parameter value.

    Returns
    -------
    stable : bool
        True if f'(x*) < 0 (stable).
    eigenvalue : float
        The value of f'(x*).
    """
    lam = df(x_star, r)
    return lam < 0, lam


def track_branch(f, df, x0, r_values):
    """
    Track a branch of fixed points across a parameter range.

    Uses Newton continuation: the fixed point at r_{i} seeds the solver
    at r_{i+1}.

    Parameters
    ----------
    f : callable
        f(x, r) = 0 defines the fixed point.
    df : callable
        df/dx(x, r) for stability assessment.
    x0 : float
        Initial guess at r_values[0].
    r_values : array_like
        Parameter values to sweep.

    Returns
    -------
    x_branch : np.ndarray
        Fixed points at each r.
    stable : np.ndarray (bool)
        Stability at each r.
    """
    r_values = np.asarray(r_values)
    x_branch = np.full(len(r_values), np.nan)
    stable = np.full(len(r_values), False)
    x_current = x0

    for i, r in enumerate(r_values):
        try:
            x_star = fsolve(f, x_current, args=(r,), full_output=False)
            if np.isscalar(x_star):
                x_star = float(x_star)
            else:
                x_star = float(x_star[0])
            # Verify it's actually a root
            if abs(f(x_star, r)) < 1e-8:
                x_branch[i] = x_star
                s, _ = stability_1d(df, x_star, r)
                stable[i] = s
                x_current = x_star
        except Exception:
            pass

    return x_branch, stable


# ═══════════════════════════════════════════════════════════════════════════════
# 2D CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_2d(J):
    """
    Classify a 2D fixed point from its Jacobian matrix.

    Parameters
    ----------
    J : np.ndarray, shape (2, 2)
        Jacobian at the fixed point.

    Returns
    -------
    classification : str
        One of: 'stable node', 'unstable node', 'saddle point',
        'stable spiral', 'unstable spiral', 'center'.
    eigenvalues : np.ndarray
        Eigenvalues of J.
    """
    eigenvalues = np.linalg.eigvals(J)
    tr = np.trace(J)
    det = np.linalg.det(J)
    disc = tr**2 - 4 * det

    if det < 0:
        return "saddle point", eigenvalues

    if abs(tr) < 1e-12 and det > 0:
        return "center", eigenvalues

    if disc >= 0:
        # Real eigenvalues
        if tr < 0:
            return "stable node", eigenvalues
        else:
            return "unstable node", eigenvalues
    else:
        # Complex eigenvalues
        if tr < 0:
            return "stable spiral", eigenvalues
        else:
            return "unstable spiral", eigenvalues
