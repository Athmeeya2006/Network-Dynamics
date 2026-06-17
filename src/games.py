"""
games.py
========
Game dynamics on simplex and networks.

Provides:
    - rps_payoff : returns RPS payoff matrix (neutral, stable, unstable)
    - replicator_rhs : RHS for replicator dynamics ODE
    - simplex_projection : projects 3D simplex coordinates to 2D Cartesian coordinates
    - spatial_rps_step : one step of spatial Rock-Paper-Scissors (Reichenbach-Mobilia-Frey)
    - network_rps_step : one step of imitation-based RPS on a network
"""

import numpy as np


def rps_payoff(epsilon=0.0):
    """
    Returns the Rock-Paper-Scissors payoff matrix.
    
    A = [
        [ 0,          -1 - epsilon,  1           ],  # Rock
        [ 1,           0,          -1 - epsilon  ],  # Paper
        [-1 - epsilon,  1,           0           ]   # Scissors
    ]
    
    If epsilon = 0: Neutral (closed orbits)
    If epsilon > 0: Spiral in (stable fixed point) - wait, let's check:
        For x_dot = x_i * (f_i - avg_f), if epsilon > 0, the interior fixed point
        is stable (spirals in). If epsilon < 0, it spirals out (unstable).
    """
    return np.array([
        [0.0, -1.0 - epsilon, 1.0],
        [1.0, 0.0, -1.0 - epsilon],
        [-1.0 - epsilon, 1.0, 0.0]
    ])


def replicator_rhs(x, t, payoff_matrix):
    """
    Replicator dynamics ODE right-hand side.
    
    dx_i/dt = x_i * ((A @ x)_i - x @ A @ x)
    
    Ensures state stays on the simplex (if initial condition is on the simplex).
    """
    # Force x to be a numpy array
    x = np.asarray(x)
    # Payoffs for each strategy
    f = payoff_matrix @ x
    # Average payoff
    avg_f = np.dot(x, f)
    # Replicator equation
    return x * (f - avg_f)


def simplex_projection(p):
    """
    Project 3D simplex coordinates (p1, p2, p3) where p1+p2+p3=1
    to 2D Cartesian coordinates (x, y) of an equilateral triangle.
    
    Vertex mapping:
    - (1, 0, 0) [Rock] -> (0, 0)
    - (0, 1, 0) [Paper] -> (1, 0)
    - (0, 0, 1) [Scissors] -> (0.5, sqrt(3)/2)
    
    Parameters
    ----------
    p : array_like, shape (..., 3)
        Points on the 3D simplex.
        
    Returns
    -------
    xy : np.ndarray, shape (..., 2)
        2D Cartesian coordinates.
    """
    p = np.asarray(p)
    if p.ndim == 1:
        p1, p2, p3 = p[0], p[1], p[2]
        x = p2 + 0.5 * p3
        y = (np.sqrt(3.0) / 2.0) * p3
        return np.array([x, y])
    else:
        p1 = p[..., 0]
        p2 = p[..., 1]
        p3 = p[..., 2]
        x = p2 + 0.5 * p3
        y = (np.sqrt(3.0) / 2.0) * p3
        return np.column_stack([x, y])


def spatial_rps_step(grid, r_selection, r_reproduction, r_mobility, rng):
    """
    One Monte Carlo step of spatial Rock-Paper-Scissors on a 2D lattice.
    Uses the Reichenbach-Mobilia-Frey (RMF) mobility model.
    
    Grid values:
        0: Empty
        1: Rock
        2: Paper
        3: Scissors
        
    Each interaction selects a random site and a random neighbor.
    With probabilities proportional to rates:
        - Mobility: exchange states of the two sites.
        - Selection: if one prey and one predator, prey becomes Empty (0).
        - Reproduction: if one empty (0) and one organism, empty becomes organism.
        
    We perform N = L * L updates to define one full Monte Carlo step (sweep).
    """
    L = grid.shape[0]
    total_rate = r_selection + r_reproduction + r_mobility
    p_select = r_selection / total_rate
    p_reprod = r_reproduction / total_rate
    # p_mobil = r_mobility / total_rate
    
    for _ in range(L * L):
        # Pick random site
        x = rng.integers(0, L)
        y = rng.integers(0, L)
        
        # Pick random neighbor (4-connectivity)
        direction = rng.integers(0, 4)
        nx, ny = x, y
        if direction == 0:
            nx = (x + 1) % L
        elif direction == 1:
            nx = (x - 1) % L
        elif direction == 2:
            ny = (y + 1) % L
        else:
            ny = (y - 1) % L
            
        s1 = grid[x, y]
        s2 = grid[nx, ny]
        
        if s1 == s2:
            continue
            
        # Draw reaction type
        r = rng.random()
        
        if r < p_select:
            # Selection (predation): Rock (1) beats Scissors (3), Scissors (3) beats Paper (2), Paper (2) beats Rock (1)
            # Predator becomes s1, prey becomes 0 (empty)
            if (s1 == 1 and s2 == 3) or (s1 == 3 and s2 == 2) or (s1 == 2 and s2 == 1):
                grid[nx, ny] = 0
            elif (s2 == 1 and s1 == 3) or (s2 == 3 and s1 == 2) or (s2 == 2 and s1 == 1):
                grid[x, y] = 0
                
        elif r < p_select + p_reprod:
            # Reproduction: X (1,2,3) + Empty (0) -> X + X
            if s1 != 0 and s2 == 0:
                grid[nx, ny] = s1
            elif s2 != 0 and s1 == 0:
                grid[x, y] = s2
                
        else:
            # Mobility: exchange
            grid[x, y], grid[nx, ny] = s2, s1


def spatial_structure(grid):
    """
    Nearest-neighbour spatial order parameter for a spatial RPS lattice.

    Measures the excess probability (over the well-mixed baseline) that two
    adjacent occupied sites carry the same species:

        S = P(same species | both occupied, adjacent) - sum_s rho_s^2

    where rho_s is the fraction of occupied sites of species s. S ~ 0 for a
    well-mixed (homogenised) lattice and S -> (1 - baseline) for fully phase-
    separated domains. S decreases monotonically as mobility homogenises the
    lattice, which is the precursor to biodiversity loss in the RMF model.

    Parameters
    ----------
    grid : np.ndarray, shape (L, L)
        Lattice with 0 = empty, 1/2/3 = species.

    Returns
    -------
    S : float
        Spatial structure order parameter in [0, ~1].
    """
    g = grid
    occ = g > 0
    # Right and down neighbours (periodic)
    right = np.roll(g, -1, axis=1)
    down = np.roll(g, -1, axis=0)
    occ_r = np.roll(occ, -1, axis=1)
    occ_d = np.roll(occ, -1, axis=0)

    same = 0
    both = 0
    for nb, occ_nb in ((right, occ_r), (down, occ_d)):
        mask = occ & occ_nb
        both += np.count_nonzero(mask)
        same += np.count_nonzero(mask & (g == nb))
    if both == 0:
        return 0.0
    p_same = same / both

    # Well-mixed baseline: sum_s rho_s^2 over occupied sites
    counts = np.array([np.count_nonzero(g == s) for s in (1, 2, 3)], dtype=float)
    n_occ = counts.sum()
    if n_occ == 0:
        return 0.0
    rho = counts / n_occ
    baseline = np.sum(rho ** 2)
    return float(p_same - baseline)


def network_rps_step(G, strategies, payoff_matrix, temp=0.5, rng=None):
    """
    One step of imitation-based RPS on a network.
    
    Each node plays RPS with all its neighbors, accumulating payoff.
    Then, each node chooses a random neighbor, and adopts their strategy
    according to the Fermi-Dirac rule:
        P(i -> j) = 1 / (1 + exp(-(payoff_j - payoff_i) / temp))
        
    Parameters
    ----------
    G : networkx.Graph
        The network topology.
    strategies : np.ndarray, shape (N,)
        Current strategies of nodes (0, 1, 2).
    payoff_matrix : np.ndarray, shape (3, 3)
        RPS payoff matrix.
    temp : float
        Imitation temperature (noise level).
    rng : np.random.Generator
        Random number generator.
        
    Returns
    -------
    new_strategies : np.ndarray, shape (N,)
        Strategies after one imitation sweep.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    N = len(strategies)
    # Compute payoffs
    payoffs = np.zeros(N)
    for u in G.nodes():
        u_strat = strategies[u]
        for v in G.neighbors(u):
            v_strat = strategies[v]
            payoffs[u] += payoff_matrix[u_strat, v_strat]
            
    new_strategies = strategies.copy()
    
    # Imitation sweep
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        if not neighbors:
            continue
        v = rng.choice(neighbors)
        
        diff = payoffs[v] - payoffs[u]
        p = 1.0 / (1.0 + np.exp(-diff / temp))
        if rng.random() < p:
            new_strategies[u] = strategies[v]
            
    return new_strategies
