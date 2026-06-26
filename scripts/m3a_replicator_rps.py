"""
m3a_replicator_rps.py
=====================
Module 3a: Replicator dynamics of Rock-Paper-Scissors.

Proof verified:
    For neutral RPS (epsilon = 0), the quantity V = x_1 * x_2 * x_3
    is a conserved Lyapunov-like function (invariant of motion).
    We verify V(t) conservation (std(V) ~ 0) for the neutral case,
    and show that V(t) decreases for the unstable case (epsilon < 0) and
    increases for the stable case (epsilon > 0).

Output: media/figures/m3a_replicator_rps.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.games import rps_payoff, replicator_rhs, simplex_projection
from src.integrators import solve_ode

np.random.seed(42)
setup_light_theme()

# Simplex Boundary Setup
def draw_simplex_boundary(ax):
    # Vertices in 3D simplex: (1,0,0), (0,1,0), (0,0,1)
    v3d = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    v2d = simplex_projection(v3d)
    ax.plot(v2d[:, 0], v2d[:, 1], '-', color=SLATE, lw=2)
    
    # Labels
    # Rock (1,0,0) -> (0,0)
    # Paper (0,1,0) -> (1,0)
    # Scissors (0,0,1) -> (0.5, sqrt(3)/2)
    ax.text(v2d[0, 0] - 0.05, v2d[0, 1] - 0.03, 'Rock', fontsize=12, color=NAVY, fontweight='bold', ha='right')
    ax.text(v2d[1, 0] + 0.05, v2d[1, 1] - 0.03, 'Paper', fontsize=12, color=NAVY, fontweight='bold', ha='left')
    ax.text(v2d[2, 0], v2d[2, 1] + 0.03, 'Scissors', fontsize=12, color=NAVY, fontweight='bold', ha='center', va='bottom')
    
    # Mark the symmetric Nash equilibrium (1/3, 1/3, 1/3)
    ne_2d = simplex_projection([1/3, 1/3, 1/3])
    ax.scatter([ne_2d[0]], [ne_2d[1]], color=GOLD, s=100, marker='*', zorder=10, edgecolors='white', linewidths=1.5)

# Simulations
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.patch.set_facecolor("#F8FAFC")

# Cases: (epsilon, title, color)
cases = [
    (0.0, 'Neutral RPS (Invariant Orbits)', TEAL),
    (0.2, 'Stable RPS (Spiral In)', PURPLE),
    (-0.2, 'Unstable RPS (Spiral Out)', RED)
]

initial_conditions = [
    [0.4, 0.3, 0.3],
    [0.35, 0.45, 0.2],
    [0.2, 0.3, 0.5]
]

conservation_results = {}

for idx, (eps, title, color) in enumerate(cases):
    ax = axes[idx]
    apply_axes_style(ax, grid=False)
    ax.axis('off')
    draw_simplex_boundary(ax)
    
    payoff = rps_payoff(eps)
    
    # Run multiple initial conditions to show different orbits
    for x0 in initial_conditions:
        t, traj = solve_ode(lambda x, t: replicator_rhs(x, t, payoff),
                            x0, (0, 80), dt=0.01)
        
        # Project to 2D
        traj_2d = simplex_projection(traj)
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], '-', color=color, lw=1.5, alpha=0.8)
        
        # Mark start
        start_2d = simplex_projection(x0)
        ax.scatter([start_2d[0]], [start_2d[1]], color=color, s=40, zorder=5, edgecolors='white', linewidths=0.8)
        
        # For the first initial condition, let's track the conserved quantity
        if x0 == initial_conditions[0]:
            V = traj[:, 0] * traj[:, 1] * traj[:, 2]
            conservation_results[eps] = V

    ax.set_title(title, fontsize=13, color=NAVY, fontweight='bold', pad=15)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.0)

fig.suptitle('Module 3a: Rock-Paper-Scissors Replicator Dynamics on the Simplex',
             fontsize=16, color=NAVY, fontweight='bold', y=0.98)
plt.tight_layout()

# VERIFY: Conservation of V = x1 * x2 * x3
V_neutral = conservation_results[0.0]
V_stable = conservation_results[0.2]
V_unstable = conservation_results[-0.2]

# Standard deviation along the neutral trajectory
std_neutral = np.nanstd(V_neutral)
# Change in V for stable (should increase towards 1/27 ~ 0.037)
delta_stable = V_stable[-1] - V_stable[0]
# Change in V for unstable (should decrease towards 0)
delta_unstable = V_unstable[-1] - V_unstable[0]

print("=" * 65)
print("VERIFY: Replicator RPS Conservation of V = x_1*x_2*x_3:")
print(f"  Neutral (epsilon = 0):")
print(f"    Initial V:       {V_neutral[0]:.6f}")
print(f"    Final V:         {V_neutral[-1]:.6f}")
print(f"    Std Dev of V(t): {std_neutral:.6e}  (Theory: 0.000000)")
print(f"    Status:          {'CONSERVED' if std_neutral < 1e-5 else 'FAILS'}")
print(f"  Stable (epsilon = 0.2):")
print(f"    Change in V:     {delta_stable:+.6f}  (Theory: > 0, spiraling in)")
print(f"  Unstable (epsilon = -0.2):")
print(f"    Change in V:     {delta_unstable:+.6f}  (Theory: < 0, spiraling out)")
print("=" * 65)

save_figure(fig, "m3a_replicator_rps")
plt.close()
