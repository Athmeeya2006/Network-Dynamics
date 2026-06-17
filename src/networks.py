"""
networks.py
===========
Network generators, spectra, centralities, and structural metrics for the
Nonlinear Dynamics on Networks repository (Module 4, reused by Modules 5-6).

The Erdos-Renyi generator is the O(n + M) Batagelj-Brandes geometric-skip
algorithm ported from the companion Erdos-Renyi-Contagion repo (src/fast_er.py)
to keep the two-repo research arc consistent. Everything else is a thin,
clean wrapper over networkx so that Module 5 can write

    G = ba_graph(N, m)
    L = laplacian(G)

and drop oscillators on the nodes with one coupling term.

Provides:
    Generators : er_graph, ba_graph, config_model, watts_strogatz, ring_lattice
    Spectra    : adjacency_matrix, laplacian, normalized_laplacian,
                 laplacian_spectrum, eigenratio
    Centrality : centralities (degree/closeness/betweenness/eigenvector/Katz/PR)
    Metrics    : degree_sequence, mean_degree, clustering, avg_path_length
"""

from __future__ import annotations

import math
import random

import numpy as np
import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════════
# ER GENERATOR (Batagelj-Brandes, ported from Erdos-Renyi-Contagion/src/fast_er.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _index_to_pair(e: int) -> tuple[int, int]:
    """Decode a flat upper-triangle edge index e to a vertex pair (i, j), i<j."""
    i = int((1 + math.isqrt(1 + 8 * e)) // 2)
    j = e - i * (i - 1) // 2
    return i, j


def er_adjacency(n: int, p: float, seed: int | None = None) -> list[list[int]]:
    """
    Generate G(n, p) adjacency lists using the Batagelj-Brandes O(n + M)
    geometric-skip algorithm (ported verbatim in spirit from the ER repo).

    Parameters
    ----------
    n : int
        Number of vertices, labeled 0 .. n-1.
    p : float
        Edge probability in [0, 1].
    seed : int or None
        Seed for the module-level ``random`` generator.

    Returns
    -------
    adj : list[list[int]]
        Adjacency list, adj[v] = neighbours of v.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if seed is not None:
        random.seed(seed)

    adj: list[list[int]] = [[] for _ in range(n)]
    if p <= 0.0:
        return adj
    if p >= 1.0:
        for i in range(n):
            for j in range(i + 1, n):
                adj[i].append(j)
                adj[j].append(i)
        return adj

    max_edges = n * (n - 1) // 2
    log1mp = math.log(1.0 - p)
    e = -1
    while True:
        u = 1.0 - random.random()
        skip = int(math.floor(math.log(u) / log1mp))
        e += skip + 1
        if e >= max_edges:
            break
        i, j = _index_to_pair(e)
        adj[i].append(j)
        adj[j].append(i)
    return adj


def er_graph(n: int, p: float, seed: int | None = None) -> nx.Graph:
    """Erdos-Renyi G(n, p) as a networkx Graph via the Batagelj-Brandes core."""
    adj = er_adjacency(n, p, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if u < v:
                G.add_edge(u, v)
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# OTHER GENERATORS (thin networkx wrappers, consistent signatures)
# ═══════════════════════════════════════════════════════════════════════════════

def ba_graph(n: int, m: int, seed: int | None = None) -> nx.Graph:
    """Barabasi-Albert preferential-attachment graph (p(k) ~ k^-3)."""
    return nx.barabasi_albert_graph(n, m, seed=seed)


def config_model(degree_sequence, seed: int | None = None) -> nx.Graph:
    """
    Configuration model with a prescribed degree sequence, returned as a
    simple graph (self-loops and parallel edges removed).
    """
    G = nx.configuration_model(list(degree_sequence), seed=seed)
    G = nx.Graph(G)            # collapse parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def watts_strogatz(n: int, k: int, p: float, seed: int | None = None) -> nx.Graph:
    """Watts-Strogatz small-world graph: ring of degree k, rewired w.p. p."""
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def ring_lattice(n: int, k: int) -> nx.Graph:
    """Regular ring lattice (Watts-Strogatz at p=0)."""
    return nx.watts_strogatz_graph(n, k, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRA / LAPLACIANS
# ═══════════════════════════════════════════════════════════════════════════════

def giant_component(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a relabelled copy (0..n-1)."""
    if nx.is_connected(G):
        return nx.convert_node_labels_to_integers(G)
    giant = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(giant).copy())


def adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """Dense adjacency matrix in node-sorted order."""
    nodes = sorted(G.nodes())
    return nx.to_numpy_array(G, nodelist=nodes)


def laplacian(G: nx.Graph) -> np.ndarray:
    """Combinatorial graph Laplacian L = D - A (dense, node-sorted)."""
    A = adjacency_matrix(G)
    return np.diag(A.sum(axis=1)) - A


def normalized_laplacian(G: nx.Graph) -> np.ndarray:
    """Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    A = adjacency_matrix(G)
    deg = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt


def laplacian_spectrum(G: nx.Graph, normalized: bool = False) -> np.ndarray:
    """
    Sorted (ascending) eigenvalues of the (combinatorial or normalized)
    Laplacian. lambda_0 = 0 for a connected graph; lambda_1 is the algebraic
    connectivity.
    """
    L = normalized_laplacian(G) if normalized else laplacian(G)
    return np.sort(np.linalg.eigvalsh(L))


def eigenratio(G: nx.Graph, normalized: bool = False) -> float:
    """
    Synchronizability eigenratio lambda_N / lambda_2 (largest over smallest
    nonzero Laplacian eigenvalue). Smaller is more synchronizable in the
    Master Stability Function sense. Returns inf if disconnected.
    """
    ev = laplacian_spectrum(G, normalized=normalized)
    # smallest nonzero eigenvalue (algebraic connectivity)
    nonzero = ev[ev > 1e-9]
    if len(nonzero) == 0:
        return float('inf')
    return float(ev[-1] / nonzero[0])


# ═══════════════════════════════════════════════════════════════════════════════
# CENTRALITIES
# ═══════════════════════════════════════════════════════════════════════════════

def centralities(G: nx.Graph) -> dict[str, dict]:
    """
    Compute a standard suite of node centralities.

    Returns
    -------
    dict mapping centrality name -> {node: value}, with keys:
        'degree', 'closeness', 'betweenness', 'eigenvector', 'katz', 'pagerank'
    """
    out = {
        'degree': nx.degree_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'pagerank': nx.pagerank(G),
    }
    try:
        out['eigenvector'] = nx.eigenvector_centrality_numpy(G)
    except (nx.NetworkXException, np.linalg.LinAlgError):
        out['eigenvector'] = {v: float('nan') for v in G.nodes()}
    # Katz: alpha must be < 1 / lambda_max for convergence
    try:
        lam_max = max(abs(np.linalg.eigvals(adjacency_matrix(G))))
        alpha = 0.9 / lam_max if lam_max > 0 else 0.1
        out['katz'] = nx.katz_centrality_numpy(G, alpha=alpha)
    except (nx.NetworkXException, np.linalg.LinAlgError):
        out['katz'] = {v: float('nan') for v in G.nodes()}
    return out


def centrality_vector(cdict: dict, nodes=None) -> np.ndarray:
    """Convert a {node: value} centrality dict to an array in node order."""
    if nodes is None:
        nodes = sorted(cdict.keys())
    return np.array([cdict[v] for v in nodes])


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def degree_sequence(G: nx.Graph) -> np.ndarray:
    """Degree sequence as an integer array."""
    return np.array([d for _, d in G.degree()], dtype=int)


def mean_degree(G: nx.Graph) -> float:
    """Mean degree <k> = 2E / N."""
    return 2.0 * G.number_of_edges() / G.number_of_nodes()


def clustering(G: nx.Graph) -> float:
    """Average clustering coefficient."""
    return nx.average_clustering(G)


def avg_path_length(G: nx.Graph) -> float:
    """
    Average shortest-path length on the largest connected component
    (graceful for disconnected graphs).
    """
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    giant = max(nx.connected_components(G), key=len)
    return nx.average_shortest_path_length(G.subgraph(giant))
