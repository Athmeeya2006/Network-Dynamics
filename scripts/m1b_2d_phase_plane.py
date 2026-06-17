"""
m1b_2d_phase_plane.py
=====================
Module 1b — Phase portraits and fixed-point classification in 2D.

Proof verified:
    The trace-determinant classification of 2D linear systems matches
    the eigenvalue-predicted behavior of simulated phase portraits.
    Each region of the (tr, det) plane produces the expected portrait:
    stable/unstable nodes, saddle points, stable/unstable spirals, centers.

Deliverable:
    Trace-determinant plane with embedded sample phase portraits for
    each classification region.

Output: media/figures/m1b_2d_phase_plane.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE, GREEN,
                     setup_light_theme, apply_axes_style, save_figure)
from src.continuation import classify_2d
from src.integrators import rk4

np.random.seed(42)
setup_light_theme()

# ── Linear systems for each classification ───────────────────────────────────
systems = {
    'Stable Node': {
        'A': np.array([[-2, 0], [0, -1]]),
        'color': TEAL,
        'pos': (-3, 1.5),
    },
    'Unstable Node': {
        'A': np.array([[2, 0], [0, 1]]),
        'color': RED,
        'pos': (3, 1.5),
    },
    'Saddle Point': {
        'A': np.array([[1, 0], [0, -2]]),
        'color': GOLD,
        'pos': (0, -1.5),
    },
    'Stable Spiral': {
        'A': np.array([[-0.5, 2], [-2, -0.5]]),
        'color': PURPLE,
        'pos': (-2, 5),
    },
    'Unstable Spiral': {
        'A': np.array([[0.5, 2], [-2, 0.5]]),
        'color': '#EA580C',
        'pos': (2, 5),
    },
    'Center': {
        'A': np.array([[0, 1], [-4, 0]]),
        'color': GREEN,
        'pos': (0, 5),
    },
}

# ── Figure layout: main tr-det plane + 6 inset portraits ────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#F8FAFC")

# Main axes: trace-determinant plane
ax_main = fig.add_axes([0.08, 0.08, 0.5, 0.84])
apply_axes_style(ax_main)

# Parabola: det = tr^2 / 4 (discriminant = 0)
tr_range = np.linspace(-6, 6, 500)
det_parabola = tr_range**2 / 4.0
ax_main.plot(tr_range, det_parabola, '-', color=NAVY, lw=2, alpha=0.6,
             label=r'$\Delta = 0$:  $\det = \mathrm{tr}^2/4$')
ax_main.fill_between(tr_range, det_parabola, 10, alpha=0.08, color=PURPLE)
ax_main.fill_between(tr_range, 0, det_parabola, alpha=0.05, color=TEAL,
                     where=(det_parabola >= 0))

# Axes
ax_main.axhline(0, color=SLATE, lw=1, ls='-', alpha=0.5)
ax_main.axvline(0, color=SLATE, lw=1, ls='-', alpha=0.5)

# Region labels
ax_main.text(-4, 7.5, 'Stable\nSpirals', fontsize=11, color=PURPLE,
             ha='center', fontstyle='italic')
ax_main.text(4, 7.5, 'Unstable\nSpirals', fontsize=11, color='#EA580C',
             ha='center', fontstyle='italic')
ax_main.text(0, 9, 'Centers\n(tr = 0)', fontsize=10, color=GREEN,
             ha='center', fontstyle='italic')
ax_main.text(-4, 2, 'Stable\nNodes', fontsize=11, color=TEAL,
             ha='center', fontstyle='italic')
ax_main.text(4, 2, 'Unstable\nNodes', fontsize=11, color=RED,
             ha='center', fontstyle='italic')
ax_main.text(0, -2, 'Saddle Points\n(det < 0)', fontsize=11, color=GOLD,
             ha='center', fontstyle='italic')

# Plot each system's (tr, det) on the plane
for name, sys_info in systems.items():
    A = sys_info['A']
    tr_val = np.trace(A)
    det_val = np.linalg.det(A)
    ax_main.scatter([tr_val], [det_val], color=sys_info['color'], s=120,
                    zorder=5, edgecolors='white', linewidths=2)

ax_main.set_xlabel(r'Trace  $\tau = \mathrm{tr}(J)$', fontsize=13, color=NAVY)
ax_main.set_ylabel(r'Determinant  $\Delta = \det(J)$', fontsize=13, color=NAVY)
ax_main.set_title('Trace-Determinant Classification\nof 2D Linear Systems',
                  fontsize=14, color=NAVY, fontweight='bold', pad=15)
ax_main.set_xlim(-6, 6)
ax_main.set_ylim(-4, 10)
ax_main.legend(fontsize=10, loc='lower right', framealpha=0.95,
               facecolor='white', edgecolor=SLATE)

# ── Phase portrait insets ────────────────────────────────────────────────────
inset_positions = [
    [0.62, 0.62, 0.16, 0.28],  # Stable Node
    [0.82, 0.62, 0.16, 0.28],  # Unstable Node
    [0.62, 0.34, 0.16, 0.28],  # Stable Spiral
    [0.82, 0.34, 0.16, 0.28],  # Unstable Spiral
    [0.62, 0.06, 0.16, 0.28],  # Center
    [0.82, 0.06, 0.16, 0.28],  # Saddle
]

order = ['Stable Node', 'Unstable Node', 'Stable Spiral',
         'Unstable Spiral', 'Center', 'Saddle Point']

print("=" * 65)
print("VERIFY — 2D fixed-point classification:")

for idx, name in enumerate(order):
    sys_info = systems[name]
    A = sys_info['A']

    ax_in = fig.add_axes(inset_positions[idx])
    ax_in.set_facecolor(LIGHT)
    ax_in.set_aspect('equal')

    # Streamplot
    lim = 2.5
    xg = np.linspace(-lim, lim, 25)
    yg = np.linspace(-lim, lim, 25)
    X, Y = np.meshgrid(xg, yg)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    speed = np.sqrt(U**2 + V**2)
    ax_in.streamplot(X, Y, U, V, color=speed, cmap='coolwarm',
                     density=1.2, linewidth=0.8, arrowsize=0.8)

    # Trajectories
    def rhs_linear(x, t):
        return A @ x

    for _ in range(8):
        x0 = np.random.uniform(-lim * 0.8, lim * 0.8, 2)
        _, traj = rk4(rhs_linear, x0, (0, 5), 0.02)
        mask = (np.abs(traj[:, 0]) < lim * 1.2) & (np.abs(traj[:, 1]) < lim * 1.2)
        ax_in.plot(traj[mask, 0], traj[mask, 1], '-', color=sys_info['color'],
                   lw=1.2, alpha=0.7)

    ax_in.set_xlim(-lim, lim)
    ax_in.set_ylim(-lim, lim)
    ax_in.set_title(name, fontsize=9, color=sys_info['color'],
                    fontweight='bold', pad=3)
    ax_in.tick_params(labelsize=6, colors=SLATE)
    for spine in ax_in.spines.values():
        spine.set_color(sys_info['color'])
        spine.set_linewidth(1.5)

    # VERIFY classification
    classification, eigenvalues = classify_2d(A)
    tr_val = np.trace(A)
    det_val = np.linalg.det(A)
    match = classification.lower() == name.lower()
    print(f"  {name:20s} | tr={tr_val:+.2f}, det={det_val:+.2f} | "
          f"classify_2d → {classification:20s} | "
          f"{'✓ MATCH' if match else '✗ MISMATCH'}")

print("=" * 65)

save_figure(fig, "m1b_2d_phase_plane")
plt.close()
