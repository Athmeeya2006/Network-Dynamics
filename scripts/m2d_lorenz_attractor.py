"""
m2d_lorenz_attractor.py
=======================
Module 2d - Lorenz attractor: fixed points, stability, and the
strange attractor.

Proof verified:
    Hopf threshold rho_H = sigma*(sigma + beta + 3)/(sigma - beta - 1)
    recovers rho_H ~ 24.74 for sigma=10, beta=8/3.

Output: media/figures/m2d_lorenz_attractor.png
        media/interactive/lorenz_3d.html
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure,
                     get_media_dir)
from src.flows import lorenz, lorenz_fixed_points, lorenz_hopf_rho
from src.integrators import solve_ode

np.random.seed(42)
setup_light_theme()

sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0

# ── Integrate Lorenz ─────────────────────────────────────────────────────────
x0 = np.array([1.0, 1.0, 1.0])
t, traj = solve_ode(lambda x, t: lorenz(x, t, sigma, beta, rho),
                    x0, (0, 50), dt=0.005)

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor("#F8FAFC")

# 3D attractor
ax1 = fig.add_subplot(121, projection='3d')
# Color by time
colors = plt.cm.coolwarm(np.linspace(0, 1, len(t)))
for i in range(len(t) - 1):
    ax1.plot(traj[i:i + 2, 0], traj[i:i + 2, 1], traj[i:i + 2, 2],
             color=colors[i], lw=0.3)

# Fixed points
fps = lorenz_fixed_points(sigma, beta, rho)
for fp in fps:
    pt = fp['point']
    color = TEAL if fp['stable'] else RED
    marker = 'o' if fp['stable'] else 'x'
    ax1.scatter(*pt, color=color, s=80, marker=marker, zorder=10)

ax1.set_xlabel('X', fontsize=10, color=NAVY)
ax1.set_ylabel('Y', fontsize=10, color=NAVY)
ax1.set_zlabel('Z', fontsize=10, color=NAVY)
ax1.set_title('Lorenz Strange Attractor\n'
              rf'$\sigma={sigma}, \beta={beta:.2f}, \rho={rho}$',
              fontsize=12, color=NAVY, fontweight='bold')

# Time series
ax2 = fig.add_subplot(222)
apply_axes_style(ax2)
ax2.plot(t, traj[:, 0], '-', color=TEAL, lw=0.5, label='x(t)')
ax2.plot(t, traj[:, 1], '-', color=RED, lw=0.5, alpha=0.7, label='y(t)')
ax2.set_xlabel('Time', fontsize=11, color=NAVY)
ax2.set_ylabel('State', fontsize=11, color=NAVY)
ax2.set_title('Time Series', fontsize=12, color=NAVY, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax2.set_xlim(0, 50)

# Fixed point info
ax3 = fig.add_subplot(224)
apply_axes_style(ax3, grid=False)
ax3.axis('off')
info = "Fixed Points and Stability\n" + "=" * 35 + "\n"
for fp in fps:
    eigs = fp['eigenvalues']
    eig_str = ', '.join([f"{e.real:+.2f}" + (f"{e.imag:+.2f}i"
                         if abs(e.imag) > 0.01 else "")
                         for e in eigs])
    info += f"\n{fp['name']}: ({fp['point'][0]:.2f}, "
    info += f"{fp['point'][1]:.2f}, {fp['point'][2]:.2f})\n"
    info += f"  {'Stable' if fp['stable'] else 'Unstable'}\n"
    info += f"  eigs: [{eig_str}]\n"
ax3.text(0.05, 0.95, info, transform=ax3.transAxes, fontsize=9,
         color=NAVY, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                   edgecolor=SLATE, alpha=0.9))

fig.suptitle('Module 2d - Lorenz System',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

# ── VERIFY: Hopf threshold ──────────────────────────────────────────────────
rho_H = lorenz_hopf_rho(sigma, beta)
print("=" * 65)
print("VERIFY - Lorenz Hopf bifurcation threshold:")
print(f"  Theory:  rho_H = sigma*(sigma+beta+3)/(sigma-beta-1)")
print(f"           rho_H = {sigma}*({sigma}+{beta:.4f}+3)/({sigma}-{beta:.4f}-1)")
print(f"           rho_H = {rho_H:.4f}")
print(f"  Known:   rho_H ~ 24.7368")
print(f"  Error:   {abs(rho_H - 24.7368):.4f}")
print("=" * 65)

save_figure(fig, "m2d_lorenz_attractor")
plt.close()

# ── Plotly interactive 3D ────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go

    trace = go.Scatter3d(
        x=traj[::3, 0], y=traj[::3, 1], z=traj[::3, 2],
        mode='lines',
        line=dict(color=t[::3], colorscale='Inferno', width=2),
        name='Trajectory'
    )

    layout = go.Layout(
        title=dict(text='Lorenz Attractor (Interactive)',
                   font=dict(size=16)),
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            bgcolor='#F8FAFC',
        ),
        paper_bgcolor='#F8FAFC',
        width=900, height=700,
    )

    fig_plotly = go.Figure(data=[trace], layout=layout)
    html_path = os.path.join(get_media_dir('interactive'), 'lorenz_3d.html')
    fig_plotly.write_html(html_path)
    print(f"Saved: {html_path}")
except ImportError:
    print("Plotly not available, skipping interactive HTML.")
