"""
m1e_van_der_pol.py
==================
Module 1e - Van der Pol oscillator: weak to relaxation oscillations.

Proof verified:
    In the relaxation limit (large mu), the period of the van der Pol
    oscillator grows approximately linearly: T ~ (3 - 2*ln(2))*mu
    for large mu. We verify this scaling.

Output: media/figures/m1e_van_der_pol.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE, GREEN,
                     CYCLE, setup_light_theme, apply_axes_style, save_figure)
from src.oscillators import VanDerPol
from src.integrators import rk4

np.random.seed(42)
setup_light_theme()

mu_values = [0.1, 0.5, 1.0, 3.0, 6.0, 10.0]
colors = CYCLE[:len(mu_values)]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor("#F8FAFC")

# ── Top-left: Phase portraits ───────────────────────────────────────────────
ax = axes[0, 0]
apply_axes_style(ax)
for mu, color in zip(mu_values, colors):
    vdp = VanDerPol(mu=mu)
    x0 = np.array([0.1, 0.0])
    t, traj = rk4(lambda x, t, o=vdp: o.rhs(x, t), x0, (0, 80), 0.005)
    n_late = len(t) // 2
    ax.plot(traj[n_late:, 0], traj[n_late:, 1], '-', color=color, lw=1.2,
            alpha=0.8, label=rf'$\mu = {mu}$')
ax.set_xlabel(r'$x$', fontsize=12, color=NAVY)
ax.set_ylabel(r'$y$', fontsize=12, color=NAVY)
ax.set_title('Phase Portraits (Limit Cycles)', fontsize=12, color=NAVY,
             fontweight='bold')
ax.legend(fontsize=8, framealpha=0.95, facecolor='white', edgecolor=SLATE,
          ncol=2)

# ── Top-right: Time series for select mu ─────────────────────────────────────
ax = axes[0, 1]
apply_axes_style(ax)
for mu, color in [(0.5, TEAL), (3.0, PURPLE), (10.0, RED)]:
    vdp = VanDerPol(mu=mu)
    x0 = np.array([0.1, 0.0])
    t, traj = rk4(lambda x, t, o=vdp: o.rhs(x, t), x0, (0, 80), 0.005)
    ax.plot(t, traj[:, 0], '-', color=color, lw=1.5,
            label=rf'$\mu = {mu}$')
ax.set_xlabel('Time', fontsize=12, color=NAVY)
ax.set_ylabel(r'$x(t)$', fontsize=12, color=NAVY)
ax.set_title('Time Series: Weak to Relaxation', fontsize=12, color=NAVY,
             fontweight='bold')
ax.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax.set_xlim(0, 80)

# ── Bottom-left: Period vs mu ────────────────────────────────────────────────
ax = axes[1, 0]
apply_axes_style(ax)

mu_sweep = np.concatenate([np.linspace(0.5, 5, 20), np.linspace(5, 30, 30)])
periods = []
for mu in mu_sweep:
    vdp = VanDerPol(mu=mu)
    x0 = np.array([0.1, 0.0])
    t, traj = rk4(lambda x, t, o=vdp: o.rhs(x, t), x0, (0, 500), 0.005)
    # Find period from zero crossings in last half
    n_late = len(t) // 2
    x_late = traj[n_late:, 0]
    t_late = t[n_late:]
    crossings = np.where(np.diff(np.sign(x_late)) > 0)[0]
    if len(crossings) >= 2:
        T = np.mean(np.diff(t_late[crossings]))
        periods.append(T)
    else:
        periods.append(np.nan)

periods = np.array(periods)

ax.plot(mu_sweep, periods, 'o-', color=TEAL, lw=2, markersize=5,
        markeredgecolor='white', markeredgewidth=0.5, label='Simulated $T$')

# Theory: T ~ (3 - 2*ln(2))*mu for large mu
mu_theory = np.linspace(5, 30, 100)
T_theory = (3.0 - 2.0 * np.log(2)) * mu_theory
ax.plot(mu_theory, T_theory, '--', color=RED, lw=2, alpha=0.7,
        label=r'Theory: $T \approx (3 - 2\ln 2)\,\mu$')

ax.set_xlabel(r'$\mu$', fontsize=12, color=NAVY)
ax.set_ylabel(r'Period $T$', fontsize=12, color=NAVY)
ax.set_title('Period Scaling in Relaxation Limit', fontsize=12,
             color=NAVY, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)

# ── Bottom-right: Relaxation detail ─────────────────────────────────────────
ax = axes[1, 1]
apply_axes_style(ax)
vdp = VanDerPol(mu=10.0)
x0 = np.array([0.1, 0.0])
t, traj = rk4(lambda x, t, o=vdp: o.rhs(x, t), x0, (0, 80), 0.002)
n_late = len(t) // 2
ax.plot(traj[n_late:, 0], traj[n_late:, 1], '-', color=PURPLE, lw=1.5)
# Cubic nullcline y = mu*(x - x^3/3)
x_nc = np.linspace(-3, 3, 500)
y_nc = 10.0 * (x_nc - x_nc**3 / 3.0)
# In our formulation dy/dt = mu*(1-x^2)*y - x, so nullcline is y = x/(mu*(1-x^2))
# But we plot the slow manifold
ax.plot(x_nc, -x_nc / (10.0 * (1 - x_nc**2 + 1e-10)), '--', color=RED,
        lw=1, alpha=0.5)
ax.set_xlabel(r'$x$', fontsize=12, color=NAVY)
ax.set_ylabel(r'$y$', fontsize=12, color=NAVY)
ax.set_title(r'Relaxation Oscillation ($\mu = 10$)', fontsize=12,
             color=NAVY, fontweight='bold')
ax.set_xlim(-3, 3)
ax.set_ylim(-15, 15)

fig.suptitle('Module 1e - Van der Pol Oscillator',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

# ── VERIFY ───────────────────────────────────────────────────────────────────
# Check period scaling at large mu
large_mu_mask = mu_sweep >= 10.0
mu_large = mu_sweep[large_mu_mask]
T_large = periods[large_mu_mask]
valid = ~np.isnan(T_large)
if np.sum(valid) >= 3:
    coeffs = np.polyfit(mu_large[valid], T_large[valid], 1)
    slope = coeffs[0]
    theory_slope = 3.0 - 2.0 * np.log(2)
    print("=" * 65)
    print("VERIFY - Van der Pol period scaling (relaxation limit):")
    print(f"  Theory:   slope = 3 - 2*ln(2) = {theory_slope:.4f}")
    print(f"  Measured: slope = {slope:.4f}")
    print(f"  Error:    {abs(slope - theory_slope):.4f}")
    print("=" * 65)

save_figure(fig, "m1e_van_der_pol")
plt.close()
