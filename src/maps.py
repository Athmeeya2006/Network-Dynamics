"""
maps.py
=======
Discrete-time maps and iteration utilities.

Provides:
    - logistic_map / logistic_map_derivative
    - iterate_map     : iterate and discard transient
    - cobweb_data     : (x, y) pairs for cobweb diagrams
    - find_period_doubling_thresholds : for Feigenbaum verification
"""

from __future__ import annotations

import numpy as np


def logistic_map(x, r):
    """Logistic map: f(x) = r * x * (1 - x)."""
    return r * x * (1.0 - x)


def logistic_map_derivative(x, r):
    """Derivative of the logistic map: f'(x) = r * (1 - 2x)."""
    return r * (1.0 - 2.0 * x)


def iterate_map(f, x0, r, n_iter, transient=0):
    """
    Iterate a 1D map and return the orbit after discarding transient.

    Parameters
    ----------
    f : callable
        Map function f(x, r).
    x0 : float
        Initial condition.
    r : float
        Parameter value.
    n_iter : int
        Total iterations.
    transient : int
        Iterations to discard.

    Returns
    -------
    orbit : np.ndarray
        Post-transient orbit values.
    """
    x = x0
    for _ in range(transient):
        x = f(x, r)
    orbit = np.empty(n_iter)
    for i in range(n_iter):
        orbit[i] = x
        x = f(x, r)
    return orbit


def cobweb_data(f, x0, r, n_steps):
    """
    Generate cobweb diagram data.

    Returns arrays of (x, y) points for plotting the cobweb.

    Parameters
    ----------
    f : callable
        Map function f(x, r).
    x0 : float
        Initial condition.
    r : float
        Parameter.
    n_steps : int
        Number of cobweb steps.

    Returns
    -------
    xs, ys : np.ndarray
        Cobweb line coordinates.
    """
    xs = [x0, x0]
    ys = [0.0, f(x0, r)]
    x = x0
    for _ in range(n_steps):
        y = f(x, r)
        xs.extend([x, y])
        ys.extend([y, y])
        x = y
    return np.array(xs), np.array(ys)


def find_superstable_r(f, r_low, r_high, period, x0=0.5, tol=1e-12):
    """
    Find the parameter r at which the logistic map has a superstable
    orbit of given period (critical point x=0.5 is periodic).

    Uses bisection on the condition f^period(0.5, r) = 0.5.
    """
    for _ in range(200):
        r_mid = 0.5 * (r_low + r_high)
        x = x0
        for __ in range(period):
            x = f(x, r_mid)
        if x < x0:
            r_low = r_mid
        else:
            r_high = r_mid
        if r_high - r_low < tol:
            break
    return 0.5 * (r_low + r_high)


def find_period_doubling_thresholds(f, r_start=2.9, n_doublings=8):
    """
    Find successive period-doubling bifurcation thresholds r_n.

    Uses the superstable orbit approach: for each period 2^n, find the
    superstable r, then bracket the bifurcation between successive
    superstable values.

    Returns
    -------
    r_thresholds : list[float]
        Period-doubling bifurcation points.
    """
    # Known approximate values for the logistic map
    # Period 1->2 at r1=3.0, 2->4 at r2~3.44949, 4->8 at r3~3.54409, ...
    # We use a more direct approach: detect when period doubles
    thresholds = []
    r_scan = np.linspace(r_start, 4.0, 100000)
    current_period = 1

    for target_period_exp in range(1, n_doublings + 1):
        target_period = 2 ** target_period_exp
        found = False
        for r in r_scan:
            x = 0.5
            for _ in range(2000):  # transient
                x = f(x, r)
            # Check period
            orbit = []
            for _ in range(target_period * 4):
                x = f(x, r)
                orbit.append(x)
            orbit = np.array(orbit)
            # Check if orbit has period exactly target_period
            unique_vals = len(set(np.round(orbit[-target_period * 2:], 8)))
            if unique_vals >= target_period and not found:
                thresholds.append(r)
                found = True
                # Start scanning from here for next doubling
                r_scan = np.linspace(r, 4.0, 100000)
                break
        if not found:
            break

    return thresholds
