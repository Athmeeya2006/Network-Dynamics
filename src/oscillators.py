"""
oscillators.py
==============
Network-ready oscillator classes for the Nonlinear Dynamics on Networks
repository.

PRIME DIRECTIVE (composability):
    Every oscillator exposes the interface:
        class Oscillator:
            def __init__(self, params): ...
            def rhs(self, state, t, coupling_input=None): ...
            @property
            def dim(self): ...

    Module 5 instantiates N of these on a graph, injecting diffusive
    coupling via the coupling_input argument. This interface is locked
    before Module 5 begins.

Classes:
    FitzHughNagumo  — excitable / oscillatory neuron model (dim=2)
    StuartLandau    — Hopf normal form (dim=2)
    VanDerPol       — canonical relaxation oscillator (dim=2)
"""

from __future__ import annotations

import numpy as np


def coupled_network_rhs(big_state, t, units, A, K, couple_dims=None):
    """
    Assemble the right-hand side of N coupled oscillator units on a network,
    using only the locked composability interface (each unit exposes
    ``rhs(state, t, coupling_input)`` and ``dim``).

    Diffusive coupling is applied component-wise:
        coupling_input_i = K * sum_j A_ij (state_j - state_i)
    restricted to the components listed in ``couple_dims`` (default: all).

    This is the bridge promised by the prime directive: Module 5 instantiates
    N Module-1 oscillators on a Module-4 graph with one coupling term.

    Parameters
    ----------
    big_state : np.ndarray, shape (N * dim,)
        Flattened state of all units (unit-major: [u0_d0, u0_d1, u1_d0, ...]).
    t : float
        Time.
    units : list
        Length-N list of oscillator objects sharing the same ``dim``.
    A : np.ndarray, shape (N, N)
        Adjacency matrix.
    K : float
        Coupling strength.
    couple_dims : sequence of int or None
        State components through which coupling acts (default: all).

    Returns
    -------
    dbig : np.ndarray, shape (N * dim,)
        Flattened time derivative.
    """
    n = len(units)
    dim = units[0].dim
    X = np.asarray(big_state, dtype=float).reshape(n, dim)

    if couple_dims is None:
        couple_dims = range(dim)

    # Diffusive coupling term:  K * (A @ X - deg * X), component-wise
    deg = A.sum(axis=1)
    coupling = np.zeros_like(X)
    AX = A @ X
    for d in couple_dims:
        coupling[:, d] = K * (AX[:, d] - deg * X[:, d])

    dX = np.empty_like(X)
    for i, unit in enumerate(units):
        dX[i] = unit.rhs(X[i], t, coupling[i])
    return dX.reshape(-1)


class FitzHughNagumo:
    """
    FitzHugh-Nagumo model.

        dv/dt = v - v^3/3 - w + I + coupling_v
        dw/dt = (v + a - b*w) / tau + coupling_w

    Parameters
    ----------
    a : float
        Recovery variable offset (default 0.7).
    b : float
        Recovery variable slope (default 0.8).
    tau : float
        Time-scale separation (default 12.5).
    I : float
        External current (default 0.5).
    """

    def __init__(self, a=0.7, b=0.8, tau=12.5, I=0.5):
        self.a = a
        self.b = b
        self.tau = tau
        self.I = I

    @property
    def dim(self):
        return 2

    def rhs(self, state, t, coupling_input=None):
        """
        Right-hand side of the FHN equations.

        Parameters
        ----------
        state : array_like, shape (2,)
            [v, w] state vector.
        t : float
            Current time (unused, autonomous system).
        coupling_input : array_like or None, shape (2,)
            External coupling [coupling_v, coupling_w].

        Returns
        -------
        dxdt : np.ndarray, shape (2,)
        """
        v, w = state[0], state[1]
        cv, cw = (0.0, 0.0) if coupling_input is None else (
            coupling_input[0], coupling_input[1])

        dv = v - v**3 / 3.0 - w + self.I + cv
        dw = (v + self.a - self.b * w) / self.tau + cw
        return np.array([dv, dw])

    def nullcline_v(self, v_arr):
        """v-nullcline: w = v - v^3/3 + I."""
        return v_arr - v_arr**3 / 3.0 + self.I

    def nullcline_w(self, v_arr):
        """w-nullcline: w = (v + a) / b."""
        return (v_arr + self.a) / self.b

    def jacobian(self, state):
        """Jacobian at a given state."""
        v, w = state[0], state[1]
        return np.array([
            [1.0 - v**2, -1.0],
            [1.0 / self.tau, -self.b / self.tau]
        ])


class StuartLandau:
    """
    Stuart-Landau oscillator (Hopf normal form in real coordinates).

        z_dot = (mu + i*omega) z - |z|^2 z

    In Cartesian (x, y):
        dx/dt = mu*x - omega*y - (x^2 + y^2)*x + coupling_x
        dy/dt = omega*x + mu*y - (x^2 + y^2)*y + coupling_y

    Parameters
    ----------
    mu : float
        Bifurcation parameter. Hopf at mu = 0.
        mu > 0: stable limit cycle of radius sqrt(mu).
        mu < 0: stable fixed point at origin.
    omega : float
        Natural frequency (default 1.0).
    """

    def __init__(self, mu=1.0, omega=1.0):
        self.mu = mu
        self.omega = omega

    @property
    def dim(self):
        return 2

    def rhs(self, state, t, coupling_input=None):
        """
        Right-hand side of the Stuart-Landau equations.

        Parameters
        ----------
        state : array_like, shape (2,)
            [x, y] state vector.
        t : float
            Current time.
        coupling_input : array_like or None, shape (2,)
            External coupling [coupling_x, coupling_y].
        """
        x, y = state[0], state[1]
        r2 = x**2 + y**2
        cx, cy = (0.0, 0.0) if coupling_input is None else (
            coupling_input[0], coupling_input[1])

        dx = self.mu * x - self.omega * y - r2 * x + cx
        dy = self.omega * x + self.mu * y - r2 * y + cy
        return np.array([dx, dy])

    def limit_cycle_radius(self):
        """Theoretical limit cycle radius: sqrt(mu) for mu > 0."""
        if self.mu > 0:
            return np.sqrt(self.mu)
        return 0.0

    def jacobian(self, state):
        """Jacobian at a given state."""
        x, y = state[0], state[1]
        r2 = x**2 + y**2
        return np.array([
            [self.mu - r2 - 2 * x**2, -self.omega - 2 * x * y],
            [self.omega - 2 * x * y, self.mu - r2 - 2 * y**2]
        ])


class VanDerPol:
    """
    Van der Pol oscillator.

        dx/dt = y + coupling_x
        dy/dt = mu * (1 - x^2) * y - x + coupling_y

    (Liénard form: x'' - mu(1 - x^2)x' + x = 0)

    Parameters
    ----------
    mu : float
        Nonlinearity / damping parameter.
        mu = 0: harmonic oscillator.
        mu > 0: limit cycle. Large mu → relaxation oscillations.
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    @property
    def dim(self):
        return 2

    def rhs(self, state, t, coupling_input=None):
        """
        Right-hand side of the van der Pol equations.

        Parameters
        ----------
        state : array_like, shape (2,)
            [x, y] state vector.
        t : float
            Current time.
        coupling_input : array_like or None, shape (2,)
            External coupling [coupling_x, coupling_y].
        """
        x, y = state[0], state[1]
        cx, cy = (0.0, 0.0) if coupling_input is None else (
            coupling_input[0], coupling_input[1])

        dx = y + cx
        dy = self.mu * (1.0 - x**2) * y - x + cy
        return np.array([dx, dy])

    def jacobian(self, state):
        """Jacobian at a given state."""
        x, y = state[0], state[1]
        return np.array([
            [0.0, 1.0],
            [-2.0 * self.mu * x * y - 1.0, self.mu * (1.0 - x**2)]
        ])
