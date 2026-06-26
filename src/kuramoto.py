"""
kuramoto.py
===========
Kuramoto phase oscillators, on the mean field and on arbitrary networks
(Module 5). Provides the order parameter, vectorised right-hand sides, an
adiabatic coupling sweep (forward/backward, for hysteresis), and the
critical-coupling predictions used to verify K_c.

The network right-hand side is written so that Modules 5 and 6 can drop N
oscillators on any graph from src/networks.py with a single coupling term:

    dtheta_i/dt = omega_i + K * sum_j A_ij sin(theta_j - theta_i)

The pairwise sum is evaluated in O(N^2) via the identity
    sum_j A_ij sin(theta_j - theta_i) = Im( e^{-i theta_i} (A e^{i theta})_i ).
"""

from __future__ import annotations

import numpy as np


# ORDER PARAMETER

def order_parameter(theta: np.ndarray) -> tuple[float, float]:
    """
    Kuramoto complex order parameter r e^{i psi} = (1/N) sum_j e^{i theta_j}.

    Returns
    -------
    r : float
        Coherence in [0, 1] (0 = incoherent, 1 = fully synchronised).
    psi : float
        Mean phase.
    """
    z = np.mean(np.exp(1j * theta))
    return float(np.abs(z)), float(np.angle(z))


# RIGHT-HAND SIDES

def mean_field_rhs(theta, t, omega, K):
    """
    All-to-all (mean-field) Kuramoto RHS:
        dtheta_i/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i)
                    = omega_i + K r sin(psi - theta_i).
    """
    z = np.exp(1j * theta)
    mean_z = z.mean()
    return omega + K * np.imag(mean_z * np.conj(z))


def network_rhs(theta, t, omega, K, A):
    """
    Network Kuramoto RHS:
        dtheta_i/dt = omega_i + K sum_j A_ij sin(theta_j - theta_i).

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Adjacency matrix.
    """
    z = np.exp(1j * theta)
    coupling = np.imag(np.conj(z) * (A @ z))
    return omega + K * coupling


# INTEGRATION

def run_kuramoto(rhs, theta0, omega, K, extra=(), T=40.0, dt=0.01,
                 transient_frac=0.5, record_trace=False):
    """
    Integrate a Kuramoto system with fixed-step RK4 and return the
    time-averaged order parameter over the post-transient window.

    Parameters
    ----------
    rhs : callable
        One of ``mean_field_rhs`` or ``network_rhs``.
    theta0 : np.ndarray
        Initial phases.
    omega : np.ndarray
        Natural frequencies.
    K : float
        Coupling strength.
    extra : tuple
        Extra args after K (e.g. (A,) for the network RHS).
    T, dt : float
        Integration horizon and step.
    transient_frac : float
        Fraction of the run discarded before averaging r.
    record_trace : bool
        If True, also return the full r(t) trace and final phases.

    Returns
    -------
    r_mean : float
        Time-averaged order parameter.
    (optional) trace, theta_final
    """
    n_steps = int(round(T / dt))
    theta = np.array(theta0, dtype=float)
    start = int(transient_frac * n_steps)
    r_acc = []
    trace = [] if record_trace else None

    for step in range(n_steps):
        t = step * dt
        k1 = rhs(theta, t, omega, K, *extra)
        k2 = rhs(theta + 0.5 * dt * k1, t + 0.5 * dt, omega, K, *extra)
        k3 = rhs(theta + 0.5 * dt * k2, t + 0.5 * dt, omega, K, *extra)
        k4 = rhs(theta + dt * k3, t + dt, omega, K, *extra)
        theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        r, _ = order_parameter(theta)
        if record_trace:
            trace.append(r)
        if step >= start:
            r_acc.append(r)

    r_mean = float(np.mean(r_acc))
    if record_trace:
        return r_mean, np.array(trace), theta
    return r_mean


def adiabatic_sweep(K_values, omega, rhs, extra=(), theta0=None,
                    T=40.0, dt=0.01, transient_frac=0.5, seed=0):
    """
    Adiabatic (continuation) sweep of the order parameter over K_values.

    The final state at each K seeds the next K, so a forward sweep
    (increasing K) and a backward sweep (decreasing K) can expose
    first-order transitions and hysteresis. Pass K_values already ordered
    in the desired sweep direction.

    Returns
    -------
    r_of_K : np.ndarray
        Time-averaged order parameter at each K.
    """
    rng = np.random.default_rng(seed)
    N = len(omega)
    theta = rng.uniform(-np.pi, np.pi, N) if theta0 is None else np.array(theta0, float)
    r_of_K = np.empty(len(K_values))
    for i, K in enumerate(K_values):
        r_mean, _, theta = run_kuramoto(
            rhs, theta, omega, K, extra=extra, T=T, dt=dt,
            transient_frac=transient_frac, record_trace=True)
        r_of_K[i] = r_mean
    return r_of_K


# CRITICAL COUPLING PREDICTIONS  (K_c = 2 / (pi g(0)))

def kc_lorentzian(gamma: float) -> float:
    """
    Critical coupling for a Lorentzian frequency distribution of half-width
    gamma: g(0) = 1/(pi gamma), so K_c = 2 / (pi g(0)) = 2 gamma.
    """
    return 2.0 * gamma


def kc_gaussian(sigma: float) -> float:
    """
    Critical coupling for a Gaussian frequency distribution of std sigma:
    g(0) = 1/(sigma sqrt(2 pi)), so K_c = 2 / (pi g(0)) = 2 sigma sqrt(2/pi).
    """
    return 2.0 * sigma * np.sqrt(2.0 / np.pi)


def sample_lorentzian(gamma: float, size: int, rng: np.random.Generator,
                      center: float = 0.0) -> np.ndarray:
    """Sample natural frequencies from a Lorentzian (Cauchy) distribution."""
    return center + gamma * np.tan(np.pi * (rng.random(size) - 0.5))
