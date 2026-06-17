"""
flows.py
========
Continuous-time ODE systems (chaotic attractors and canonical flows).

Provides:
    - lorenz          : Lorenz system RHS
    - lorenz_jacobian : analytical Jacobian
    - lorenz_fixed_points : C+, C-, origin with stability
    - rossler         : Rossler system RHS
    - rossler_jacobian : analytical Jacobian
"""

from __future__ import annotations

import numpy as np


def lorenz(state, t, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    """
    Lorenz system RHS.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    x, y, z = state[0], state[1], state[2]
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def lorenz_jacobian(state, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    """Analytical Jacobian of the Lorenz system."""
    x, y, z = state[0], state[1], state[2]
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta]
    ])


def lorenz_fixed_points(sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    """
    Compute fixed points of the Lorenz system.

    Returns
    -------
    fps : list of dict
        Each dict has 'point', 'name', 'eigenvalues', 'stable'.
    """
    fps = []

    # Origin
    J0 = lorenz_jacobian(np.array([0, 0, 0]), sigma, beta, rho)
    eigs0 = np.linalg.eigvals(J0)
    fps.append({
        'point': np.array([0.0, 0.0, 0.0]),
        'name': 'Origin',
        'eigenvalues': eigs0,
        'stable': np.all(np.real(eigs0) < 0),
    })

    if rho > 1:
        # C+ and C-
        s = np.sqrt(beta * (rho - 1))
        for sign, name in [(1, 'C+'), (-1, 'C-')]:
            pt = np.array([sign * s, sign * s, rho - 1])
            J = lorenz_jacobian(pt, sigma, beta, rho)
            eigs = np.linalg.eigvals(J)
            fps.append({
                'point': pt,
                'name': name,
                'eigenvalues': eigs,
                'stable': np.all(np.real(eigs) < 0),
            })

    return fps


def lorenz_hopf_rho(sigma=10.0, beta=8.0 / 3.0):
    """
    Theoretical Hopf bifurcation threshold for the Lorenz system.

    rho_H = sigma * (sigma + beta + 3) / (sigma - beta - 1)

    At this value, C+ and C- undergo a subcritical Hopf bifurcation.
    """
    return sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)


def rossler(state, t, a=0.2, b=0.2, c=5.7):
    """
    Rossler system RHS.

    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)
    """
    x, y, z = state[0], state[1], state[2]
    return np.array([
        -y - z,
        x + a * y,
        b + z * (x - c)
    ])


def rossler_jacobian(state, a=0.2, b=0.2, c=5.7):
    """Analytical Jacobian of the Rossler system."""
    x, y, z = state[0], state[1], state[2]
    return np.array([
        [0.0, -1.0, -1.0],
        [1.0, a, 0.0],
        [z, 0.0, x - c]
    ])
