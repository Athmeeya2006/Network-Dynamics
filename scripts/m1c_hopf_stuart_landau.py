"""
m1c_hopf_stuart_landau.py
=========================
Module 1c: Supercritical Hopf bifurcation via the Stuart-Landau normal form.

Proof verified:
    The Stuart-Landau oscillator z_dot = (mu + i*omega) z - |z|^2 z
    undergoes a supercritical Hopf bifurcation at mu = 0.
    For mu > 0 the stable limit cycle has radius sqrt(mu).
    Fitting amplitude vs mu recovers the critical exponent beta = 0.5.

Deliverable:
    (Left)  Bifurcation diagram: limit-cycle amplitude vs mu.
    (Right) Sample trajectories for mu < 0, mu ≈ 0, mu > 0.

Output: media/figures/m1c_hopf_stuart_landau.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.oscillators import StuartLandau
from src.integrators import rk4

np.random.seed(42)
setup_light_theme()

# Bifurcation diagram: sweep mu
mu_values = np.linspace(-1.0, 2.0, 300)
theoretical_amplitude = np.where(mu_values > 0, np.sqrt(mu_values), 0.0)

# Numerical amplitudes via simulation
measured_amplitudes = np.zeros(len(mu_values))
for i, mu in enumerate(mu_values):
    osc = StuartLandau(mu=mu, omega=1.0)
    x0 = np.array([0.1, 0.1])
    t, traj = rk4(lambda x, t: osc.rhs(x, t), x0, (0, 100), 0.01)
    # Measure amplitude from last 20% of trajectory
    n_late = len(t) // 5
    r = np.sqrt(traj[-n_late:, 0]**2 + traj[-n_late:, 1]**2)
    measured_amplitudes[i] = np.max(r) if mu > 0.01 else np.mean(r)

# Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#F8FAFC")

# Left: Bifurcation diagram
apply_axes_style(ax1)
# Stable fixed point for mu < 0
ax1.plot(mu_values[mu_values <= 0], np.zeros(np.sum(mu_values <= 0)),
         '-', color=TEAL, lw=2.5, label='Stable fixed point')
# Unstable fixed point for mu > 0
ax1.plot(mu_values[mu_values >= 0], np.zeros(np.sum(mu_values >= 0)),
         '--', color=RED, lw=2.5, label='Unstable fixed point')
# Theoretical limit cycle amplitude
mu_pos = mu_values[mu_values > 0]
ax1.plot(mu_pos, np.sqrt(mu_pos), '-', color=PURPLE, lw=2.5,
         label=r'Limit cycle: $A = \sqrt{\mu}$')
ax1.plot(mu_pos, -np.sqrt(mu_pos), '-', color=PURPLE, lw=2.5)
# Numerical measurements
mu_meas = mu_values[mu_values > 0.05]
amp_meas = measured_amplitudes[mu_values > 0.05]
ax1.scatter(mu_meas[::5], amp_meas[::5], color=GOLD, s=30, zorder=5,
            edgecolors='white', linewidths=0.5, label='Simulated amplitude')

ax1.axvline(0, color=GOLD, ls='--', lw=1.5, alpha=0.7)
ax1.text(0.05, -0.15, r'$\mu_c = 0$', fontsize=11, color=GOLD,
         fontweight='bold')
ax1.scatter([0], [0], color=GOLD, s=100, zorder=6, edgecolors='white',
            linewidths=2, marker='D')

ax1.set_xlabel(r'Bifurcation parameter $\mu$', fontsize=12, color=NAVY)
ax1.set_ylabel(r'Amplitude $A$', fontsize=12, color=NAVY)
ax1.set_title('Supercritical Hopf Bifurcation\n'
              r'Stuart-Landau: $\dot{z} = (\mu + i\omega)z - |z|^2 z$',
              fontsize=13, color=NAVY, fontweight='bold', pad=10)
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE,
           loc='upper left')
ax1.set_xlim(-1, 2)
ax1.set_ylim(-1.8, 1.8)

# Right: Sample trajectories
apply_axes_style(ax2)

sample_mus = [
    (-0.5, RED, r'$\mu = -0.5$ (stable origin)'),
    (0.3, TEAL, r'$\mu = 0.3$ (small cycle)'),
    (1.0, PURPLE, r'$\mu = 1.0$ (large cycle)'),
    (1.8, GOLD, r'$\mu = 1.8$ (larger cycle)'),
]

for mu, color, label in sample_mus:
    osc = StuartLandau(mu=mu, omega=1.0)
    x0 = np.array([0.3, 0.0])
    t, traj = rk4(lambda x, t, o=osc: o.rhs(x, t), x0, (0, 40), 0.01)
    ax2.plot(traj[:, 0], traj[:, 1], '-', color=color, lw=1.5,
             alpha=0.8, label=label)
    # Mark start
    ax2.scatter([x0[0]], [x0[1]], color=color, s=40, zorder=5,
                edgecolors='white', linewidths=1)

# Theoretical circles
for mu, color, _ in sample_mus:
    if mu > 0:
        theta = np.linspace(0, 2 * np.pi, 200)
        r = np.sqrt(mu)
        ax2.plot(r * np.cos(theta), r * np.sin(theta), '--', color=color,
                 lw=1, alpha=0.5)

ax2.set_xlabel(r'$x$', fontsize=12, color=NAVY)
ax2.set_ylabel(r'$y$', fontsize=12, color=NAVY)
ax2.set_title('Phase Plane Trajectories\n'
              r'Initial condition $(0.3, 0)$',
              fontsize=13, color=NAVY, fontweight='bold', pad=10)
ax2.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE,
           loc='upper right')
ax2.set_aspect('equal')
lim = 2.0
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.axhline(0, color=SLATE, lw=0.5, ls=':')
ax2.axvline(0, color=SLATE, lw=0.5, ls=':')

fig.suptitle('Module 1c: Hopf Bifurcation and the Stuart-Landau Oscillator',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

# VERIFY: critical exponent
mu_fit = mu_values[(mu_values > 0.05) & (mu_values < 1.5)]
amp_fit = measured_amplitudes[(mu_values > 0.05) & (mu_values < 1.5)]
valid = amp_fit > 0.01
log_mu = np.log(mu_fit[valid])
log_amp = np.log(amp_fit[valid])
coeffs = np.polyfit(log_mu, log_amp, 1)
exponent = coeffs[0]

print("=" * 65)
print("VERIFY: Hopf bifurcation amplitude scaling:")
print(f"  Theory:   exponent = 0.5000  (amplitude ∝ sqrt(mu))")
print(f"  Measured: exponent = {exponent:.4f}")
print(f"  Error:    {abs(exponent - 0.5):.6f}")
print("=" * 65)

save_figure(fig, "m1c_hopf_stuart_landau")
plt.close()
