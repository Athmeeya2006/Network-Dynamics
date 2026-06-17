"""
simplicial.py
=============
Higher-order interaction infrastructure (Module 6): clique complexes,
triangle (2-simplex) enumeration, the Skardal-Arenas higher-order Kuramoto
right-hand side, and the Iacopini simplicial-contagion update.

This module closes the loop with the companion Erdos-Renyi-Contagion repo:
m6c is the higher-order analogue of that repo's pairwise SIR/percolation work.

Provides:
    Complex     : triangles, simplex_counts, triangle_array,
                  random_simplicial_complex
    Sync        : higher_order_kuramoto_meanfield (Skardal-Arenas)
    Contagion   : simplicial_sis_step
"""

from __future__ import annotations

import numpy as np
import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════════
# CLIQUE COMPLEX
# ═══════════════════════════════════════════════════════════════════════════════

def triangles(G: nx.Graph) -> list[tuple[int, int, int]]:
    """
    All 2-simplices (triangles) of the clique complex of G, as sorted tuples.

    Uses the standard intersection rule: for each edge (u, v) the common
    neighbours w (with w > v > u) close a triangle.
    """
    adj = {n: set(G.neighbors(n)) for n in G.nodes()}
    tris = []
    for u, v in G.edges():
        if u > v:
            u, v = v, u
        for w in adj[u] & adj[v]:
            if w > v:
                tris.append((u, v, w))
    return tris


def simplex_counts(G: nx.Graph) -> dict[int, int]:
    """Counts of 0-, 1-, and 2-simplices (nodes, edges, triangles)."""
    return {0: G.number_of_nodes(),
            1: G.number_of_edges(),
            2: len(triangles(G))}


def triangle_array(G: nx.Graph) -> np.ndarray:
    """Triangles as an integer array of shape (T, 3) for fast indexing."""
    tris = triangles(G)
    if not tris:
        return np.empty((0, 3), dtype=int)
    return np.array(tris, dtype=int)


def random_simplicial_complex(n: int, k1: float, k_delta: float,
                              seed: int | None = None) -> tuple[nx.Graph, np.ndarray]:
    """
    Iacopini-style random simplicial complex with tunable structure.

    Independently adds edges to target mean degree k1 and 2-simplices
    (triangles) to target mean 2-simplex degree k_delta. Returns the 1-skeleton
    graph (with all triangle edges included) plus the explicit triangle array,
    so a node may participate in triangles that are denser than the pairwise
    layer alone.

    Parameters
    ----------
    n : int
        Number of nodes.
    k1 : float
        Target mean pairwise degree.
    k_delta : float
        Target mean number of triangles per node.
    seed : int or None
        RNG seed.

    Returns
    -------
    G : nx.Graph
        1-skeleton (includes edges implied by triangles).
    tris : np.ndarray, shape (T, 3)
        The 2-simplices.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Pairwise layer: Erdos-Renyi with p1 = k1 / (n - 1)
    p1 = k1 / (n - 1)
    iu = np.triu_indices(n, k=1)
    mask = rng.random(len(iu[0])) < p1
    G.add_edges_from(zip(iu[0][mask], iu[1][mask]))

    # Triangle layer: number of 2-simplices = n * k_delta / 3
    n_tri = int(round(n * k_delta / 3.0))
    tris = set()
    attempts = 0
    while len(tris) < n_tri and attempts < 50 * n_tri + 100:
        t = tuple(sorted(rng.choice(n, size=3, replace=False)))
        attempts += 1
        if len(set(t)) == 3:
            tris.add(t)
    tris = np.array(sorted(tris), dtype=int)
    # ensure all triangle edges exist in the 1-skeleton
    for a, b, c in tris:
        G.add_edge(a, b)
        G.add_edge(a, c)
        G.add_edge(b, c)
    return G, tris


# ═══════════════════════════════════════════════════════════════════════════════
# HIGHER-ORDER KURAMOTO  (Skardal & Arenas)
# ═══════════════════════════════════════════════════════════════════════════════

def higher_order_kuramoto_meanfield(theta, t, omega, K1, K2):
    """
    All-to-all higher-order Kuramoto RHS (Skardal & Arenas 2020):

        dtheta_i/dt = omega_i
                      + (K1/N)  sum_j        sin(theta_j - theta_i)
                      + (K2/N^2) sum_{j,k}   sin(2 theta_j - theta_k - theta_i).

    The pairwise term is K1 r1 sin(psi1 - theta_i); the triadic term reduces in
    the mean field to K2 Im(Z2 conj(Z1) e^{-i theta_i}) with Z1 = <e^{i theta}>,
    Z2 = <e^{2 i theta}>. The 2-simplex term makes the synchronisation
    transition abrupt and bistable even without a degree-frequency correlation.
    """
    z = np.exp(1j * theta)
    Z1 = z.mean()
    Z2 = (z * z).mean()
    pairwise = K1 * np.imag(Z1 * np.conj(z))
    triadic = K2 * np.imag(Z2 * np.conj(Z1) * np.conj(z))
    return omega + pairwise + triadic


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLICIAL CONTAGION  (Iacopini, Petri, Barrat & Latora 2019)
# ═══════════════════════════════════════════════════════════════════════════════

def simplicial_sis_step(state, A, tris, beta, beta_delta, mu, dt, rng):
    """
    One discrete-time step of simplicial SIS (Iacopini et al. 2019).

    Infection channels for a susceptible node i:
        - pairwise: each infected neighbour transmits at rate beta;
        - triadic : each 2-simplex {i, j, k} with BOTH j and k infected
                    transmits at rate beta_delta.
    Infected nodes recover at rate mu. Rates are converted to per-step
    probabilities via 1 - exp(-rate * dt).

    Parameters
    ----------
    state : np.ndarray of bool, shape (N,)
        True = infected.
    A : scipy.sparse matrix, shape (N, N)
        Pairwise adjacency (used as A @ infected for the infection pressure).
    tris : np.ndarray, shape (T, 3)
        2-simplices.
    beta, beta_delta, mu : float
        Pairwise infection, triadic infection, recovery rates.
    dt : float
        Time step.
    rng : np.random.Generator

    Returns
    -------
    new_state : np.ndarray of bool
    """
    N = len(state)
    inf = state.astype(np.int64)

    # Pairwise infection pressure: number of infected neighbours per node
    n_inf_nb = A.dot(inf)

    # Triadic infection pressure: count 2-simplices where the other two are infected
    n_inf_tri = np.zeros(N, dtype=np.int64)
    if len(tris):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        ia, ib, ic = inf[a], inf[b], inf[c]
        # for vertex a: both b,c infected
        np.add.at(n_inf_tri, a, ib & ic)
        np.add.at(n_inf_tri, b, ia & ic)
        np.add.at(n_inf_tri, c, ia & ib)

    # Per-node infection rate (only susceptibles can be infected)
    rate_inf = beta * n_inf_nb + beta_delta * n_inf_tri
    p_inf = 1.0 - np.exp(-rate_inf * dt)
    p_rec = 1.0 - np.exp(-mu * dt)

    new_state = state.copy()
    susc = ~state
    draws = rng.random(N)
    new_state[susc & (draws < p_inf)] = True
    new_state[state & (draws < p_rec)] = False
    return new_state
