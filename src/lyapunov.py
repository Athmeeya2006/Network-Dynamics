"""
lyapunov.py
===========
Lyapunov exponent computation for maps and flows.

Provides:
    - map_lyapunov       : Lyapunov exponent for 1D maps
    - benettin_largest   : largest Lyapunov exponent via rescaling
    - benettin_spectrum  : full spectrum via QR reorthonormalization
    - kaplan_yorke_dimension : from sorted exponents
"""

from __future__ import annotations

import numpy as np
from src.integrators import rk4_step


def map_lyapunov(f, df, x0, r, n_iter=10000, transient=1000):
    """
    Compute the Lyapunov exponent of a 1D map.

    lambda = lim_{N->inf} (1/N) sum_{n=0}^{N-1} log|f'(x_n)|

    Parameters
    ----------
    f : callable
        Map f(x, r).
    df : callable
        Derivative f'(x, r).
    x0 : float
        Initial condition.
    r : float
        Parameter.
    n_iter : int
        Iterations after transient.
    transient : int
        Iterations to discard.

    Returns
    -------
    lam : float
        Lyapunov exponent.
    """
    x = x0
    for _ in range(transient):
        x = f(x, r)

    log_sum = 0.0
    for _ in range(n_iter):
        deriv = abs(df(x, r))
        if deriv > 0:
            log_sum += np.log(deriv)
        else:
            log_sum += -100.0  # effectively -inf contribution
        x = f(x, r)

    return log_sum / n_iter


def benettin_largest(rhs, x0, t_total, dt, d0=1e-8, renorm_steps=10):
    """
    Largest Lyapunov exponent via Benettin's rescaling algorithm.

    Parameters
    ----------
    rhs : callable
        ODE right-hand side f(x, t).
    x0 : array_like
        Initial condition on the attractor.
    t_total : float
        Total integration time.
    dt : float
        Integration step.
    d0 : float
        Initial perturbation magnitude.
    renorm_steps : int
        Steps between renormalizations.

    Returns
    -------
    lam : float
        Largest Lyapunov exponent.
    """
    x0 = np.asarray(x0, dtype=float)
    dim = len(x0)

    # Initial perturbed trajectory
    delta = np.random.randn(dim)
    delta = delta / np.linalg.norm(delta) * d0

    x = x0.copy()
    x_pert = x + delta

    n_steps = int(t_total / dt)
    n_renorms = n_steps // renorm_steps
    lam_sum = 0.0
    t = 0.0

    for _ in range(n_renorms):
        for __ in range(renorm_steps):
            x = rk4_step(rhs, x, t, dt)
            x_pert = rk4_step(rhs, x_pert, t, dt)
            t += dt

        delta = x_pert - x
        dist = np.linalg.norm(delta)
        if dist > 0:
            lam_sum += np.log(dist / d0)
            x_pert = x + delta * (d0 / dist)

    return lam_sum / (n_renorms * renorm_steps * dt)


def benettin_spectrum(rhs, jacobian, x0, t_total, dt,
                      renorm_steps=10):
    """
    Full Lyapunov spectrum via Benettin's QR algorithm.

    Parameters
    ----------
    rhs : callable
        ODE right-hand side f(x, t).
    jacobian : callable
        Jacobian function J(x, sigma, beta, rho) or J(x).
    x0 : array_like
        Initial condition on the attractor.
    t_total : float
        Total integration time.
    dt : float
        Integration step.
    renorm_steps : int
        Steps between QR reorthonormalization.

    Returns
    -------
    spectrum : np.ndarray
        Lyapunov exponents (sorted descending).
    """
    x0 = np.asarray(x0, dtype=float)
    dim = len(x0)

    # Augmented system: state + flattened tangent vectors
    Q = np.eye(dim)
    x = x0.copy()

    n_steps = int(t_total / dt)
    n_renorms = n_steps // renorm_steps
    lam_sums = np.zeros(dim)

    t = 0.0
    for _ in range(n_renorms):
        for __ in range(renorm_steps):
            # Evolve reference trajectory
            x = rk4_step(rhs, x, t, dt)

            # Evolve tangent vectors using linearized equation
            J = jacobian(x)
            # Simple Euler step for tangent vectors (sufficient with
            # frequent reorthonormalization)
            Q = Q + dt * (J @ Q)
            t += dt

        # QR decomposition for reorthonormalization
        Q, R = np.linalg.qr(Q)
        # Accumulate log of diagonal elements
        for i in range(dim):
            lam_sums[i] += np.log(abs(R[i, i]))

    spectrum = lam_sums / (n_renorms * renorm_steps * dt)
    return np.sort(spectrum)[::-1]  # descending


def master_stability_function(rhs, jacobian, coupling_matrix, alphas,
                              x0, t_total=400.0, dt=0.01, transient=20.0,
                              renorm_steps=10):
    """
    Master Stability Function Lambda(alpha) for a diffusively coupled system.

    For identical oscillators x_dot = f(x) coupled as
        x_i_dot = f(x_i) - sigma sum_j L_ij E x_j,
    the stability of the synchronous manifold decouples into transverse modes
    governed by the variational equation along the synchronous trajectory s(t):
        xi_dot = [Df(s(t)) - alpha E] xi,   alpha = sigma * lambda_k,
    where lambda_k are Laplacian eigenvalues. Lambda(alpha) is the largest
    Lyapunov exponent of this linear system; the synchronous state is
    transversely stable iff Lambda(sigma * lambda_k) < 0 for all k >= 2.

    Parameters
    ----------
    rhs : callable
        Single-oscillator RHS f(x, t).
    jacobian : callable
        Single-oscillator Jacobian Df(x).
    coupling_matrix : np.ndarray, shape (d, d)
        The matrix E selecting which components couple (e.g. diag(1,0,0) for
        coupling through the first variable).
    alphas : array_like
        Values of the (generally real) parameter alpha at which to evaluate.
    x0 : array_like
        Initial condition; relaxed onto the attractor before measuring.
    t_total, dt, transient, renorm_steps : float/int
        Integration controls.

    Returns
    -------
    Lambda : np.ndarray
        Largest transverse Lyapunov exponent at each alpha.
    """
    x0 = np.asarray(x0, dtype=float)
    E = np.asarray(coupling_matrix, dtype=float)

    # Precompute one synchronous trajectory (shared across all alpha)
    x = x0.copy()
    n_trans = int(transient / dt)
    for _ in range(n_trans):
        x = rk4_step(rhs, x, 0.0, dt)
    n_steps = int(t_total / dt)
    traj = np.empty((n_steps, len(x0)))
    for i in range(n_steps):
        traj[i] = x
        x = rk4_step(rhs, x, 0.0, dt)

    # Jacobians along the trajectory
    Js = [jacobian(traj[i]) for i in range(n_steps)]

    Lambda = np.empty(len(alphas))
    for a_idx, alpha in enumerate(alphas):
        xi = np.ones(len(x0))
        xi /= np.linalg.norm(xi)
        lyap = 0.0
        for i in range(n_steps):
            M = Js[i] - alpha * E
            k1 = M @ xi
            k2 = M @ (xi + 0.5 * dt * k1)
            k3 = M @ (xi + 0.5 * dt * k2)
            k4 = M @ (xi + dt * k3)
            xi = xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if (i + 1) % renorm_steps == 0:
                nrm = np.linalg.norm(xi)
                lyap += np.log(nrm)
                xi /= nrm
        Lambda[a_idx] = lyap / (n_steps * dt)
    return Lambda


def kaplan_yorke_dimension(spectrum):
    """
    Compute the Kaplan-Yorke (Lyapunov) dimension.

    D_KY = j + sum_{i=1}^{j} lambda_i / |lambda_{j+1}|

    where j is the largest index such that the sum of the first j
    exponents is non-negative.

    Parameters
    ----------
    spectrum : array_like
        Lyapunov exponents sorted in descending order.

    Returns
    -------
    D_KY : float
        Kaplan-Yorke dimension.
    """
    spectrum = np.sort(np.asarray(spectrum))[::-1]
    cumsum = np.cumsum(spectrum)

    j = 0
    for i in range(len(spectrum)):
        if cumsum[i] >= 0:
            j = i + 1
        else:
            break

    if j == 0:
        return 0.0
    if j >= len(spectrum):
        return float(len(spectrum))

    return j + cumsum[j - 1] / abs(spectrum[j])
