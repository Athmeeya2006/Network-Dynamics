"""
integrators.py
==============
Numerical integration infrastructure for the Nonlinear Dynamics on Networks
repository.

Provides:
    - rk4_step   : single fourth-order Runge-Kutta step
    - rk4        : fixed-step RK4 integration (required for Lyapunov routines)
    - solve_ode  : scipy.integrate.solve_ivp wrapper for adaptive methods
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# FIXED-STEP RK4

def rk4_step(rhs, x, t, dt, *args):
    """
    Perform a single fourth-order Runge-Kutta step.

    Parameters
    ----------
    rhs : callable
        Right-hand side function f(x, t, *args) -> array.
    x : np.ndarray
        Current state vector.
    t : float
        Current time.
    dt : float
        Time step.
    *args : tuple
        Additional arguments passed to rhs.

    Returns
    -------
    x_new : np.ndarray
        State after one RK4 step.
    """
    k1 = np.asarray(rhs(x, t, *args), dtype=float)
    k2 = np.asarray(rhs(x + 0.5 * dt * k1, t + 0.5 * dt, *args), dtype=float)
    k3 = np.asarray(rhs(x + 0.5 * dt * k2, t + 0.5 * dt, *args), dtype=float)
    k4 = np.asarray(rhs(x + dt * k3, t + dt, *args), dtype=float)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4(rhs, x0, t_span, dt, *args):
    """
    Integrate an ODE system using fixed-step RK4.

    This is the workhorse for Lyapunov exponent computation, where
    fixed-step integration and synchronized reorthonormalization are
    required.

    Parameters
    ----------
    rhs : callable
        Right-hand side function f(x, t, *args) -> array.
    x0 : array_like
        Initial state vector.
    t_span : tuple (t0, tf)
        Integration interval.
    dt : float
        Fixed time step.
    *args : tuple
        Additional arguments passed to rhs.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time points.
    x : np.ndarray, shape (N, dim)
        State trajectory (each row is a state vector).
    """
    x0 = np.asarray(x0, dtype=float)
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)

    dim = x0.shape[0] if x0.ndim > 0 else 1
    x = np.zeros((n_steps + 1, dim))
    x[0] = x0

    for i in range(n_steps):
        x[i + 1] = rk4_step(rhs, x[i], t[i], dt, *args)

    return t, x


# SCIPY ADAPTIVE WRAPPER

def solve_ode(rhs, x0, t_span, t_eval=None, dt=None, method="RK45",
              rtol=1e-9, atol=1e-9, max_step=np.inf):
    """
    Solve an ODE system using scipy.integrate.solve_ivp.

    Wraps solve_ivp to accept rhs(x, t, ...) signature (note: solve_ivp
    uses f(t, x)), and returns arrays in a consistent format.

    Parameters
    ----------
    rhs : callable
        Right-hand side function f(x, t) -> array.
        NOTE: this uses the (x, t) convention, not scipy's (t, x).
    x0 : array_like
        Initial state vector.
    t_span : tuple (t0, tf)
        Integration interval.
    t_eval : np.ndarray or None
        Times at which to store the solution. If None and dt is given,
        uses np.arange(t0, tf, dt).
    dt : float or None
        If t_eval is None, sample at this interval.
    method : str
        Integration method (default: 'RK45').
    rtol, atol : float
        Tolerances.
    max_step : float
        Maximum step size.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time points.
    x : np.ndarray, shape (N, dim)
        State trajectory.
    """
    x0 = np.asarray(x0, dtype=float)

    if t_eval is None and dt is not None:
        t_eval = np.arange(t_span[0], t_span[1], dt)

    # Wrap rhs to scipy's (t, x) convention
    def _rhs_scipy(t, x):
        return rhs(x, t)

    sol = solve_ivp(_rhs_scipy, t_span, x0, method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol,
                    max_step=max_step, dense_output=False)

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return sol.t, sol.y.T  # shape (N, dim)
