"""
m2e_lorenz_lyapunov.py
======================
Module 2e - Lorenz system Lyapunov spectrum and sensitive dependence.

Proof verified:
    - Largest Lyapunov exponent lambda_max ~ 0.906
    - Spectrum sum ~ -(sigma + 1 + beta) ~ -13.67
    - Kaplan-Yorke dimension ~ 2.06

Output: media/figures/m2e_lorenz_lyapunov.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.flows import lorenz, lorenz_jacobian
from src.lyapunov import benettin_largest, benettin_spectrum, kaplan_yorke_dimension
from src.integrators import rk4

np.random.seed(42)
setup_light_theme()

sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0

# ── Transient warmup ─────────────────────────────────────────────────────────
def lorenz_rhs(x, t):
    return lorenz(x, t, sigma, beta, rho)


def lorenz_jac(x):
    return lorenz_jacobian(x, sigma, beta, rho)


x0_warm = np.array([1.0, 1.0, 1.0])
_, traj_warm = rk4(lorenz_rhs, x0_warm, (0, 30), 0.01)
x0 = traj_warm[-1]

# ── Sensitive dependence ─────────────────────────────────────────────────────
eps = 1e-9
x0_pert = x0 + np.array([eps, 0, 0])
t_sens, traj1 = rk4(lorenz_rhs, x0, (0, 30), 0.01)
_, traj2 = rk4(lorenz_rhs, x0_pert, (0, 30), 0.01)
dist = np.linalg.norm(traj1 - traj2, axis=1)

# ── Largest Lyapunov exponent ────────────────────────────────────────────────
lam_max = benettin_largest(lorenz_rhs, x0, t_total=200, dt=0.01,
                           d0=1e-8, renorm_steps=10)

# ── Full spectrum ────────────────────────────────────────────────────────────
spectrum = benettin_spectrum(lorenz_rhs, lorenz_jac, x0,
                             t_total=500, dt=0.01, renorm_steps=10)
d_ky = kaplan_yorke_dimension(spectrum)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#F8FAFC")

# Sensitive dependence
ax = axes[0]
apply_axes_style(ax)
log_dist = np.log10(dist + 1e-20)
ax.plot(t_sens, log_dist, '-', color=RED, lw=1.5)
ax.axhline(np.log10(eps), color=SLATE, ls='--', lw=1, alpha=0.5)
ax.text(1, np.log10(eps) + 0.3, rf'$\epsilon_0 = 10^{{{int(np.log10(eps))}}}$',
        fontsize=10, color=SLATE)
ax.set_xlabel('Time', fontsize=12, color=NAVY)
ax.set_ylabel(r'$\log_{10}\|\delta(t)\|$', fontsize=12, color=NAVY)
ax.set_title('Sensitive Dependence on\nInitial Conditions',
             fontsize=12, color=NAVY, fontweight='bold')
ax.set_xlim(0, 30)

# Spectrum bar chart
ax = axes[1]
apply_axes_style(ax)
colors_bar = [RED if s > 0 else (TEAL if s < 0 else GOLD) for s in spectrum]
bars = ax.bar(range(len(spectrum)), spectrum, color=colors_bar,
              edgecolor='white', linewidth=1.5, width=0.6)
ax.axhline(0, color=NAVY, lw=1)
for i, s in enumerate(spectrum):
    ax.text(i, s + (0.3 if s >= 0 else -0.6), f'{s:.3f}',
            ha='center', fontsize=11, color=NAVY, fontweight='bold')
ax.set_xlabel('Exponent index', fontsize=12, color=NAVY)
ax.set_ylabel(r'$\lambda_i$', fontsize=12, color=NAVY)
ax.set_title('Lyapunov Spectrum', fontsize=12, color=NAVY,
             fontweight='bold')
ax.set_xticks(range(len(spectrum)))
ax.set_xticklabels([r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$'])

# Summary panel
ax = axes[2]
apply_axes_style(ax, grid=False)
ax.axis('off')

theory_sum = -(sigma + 1.0 + beta)
info = (
    f"Lorenz System Verification\n"
    f"{'=' * 35}\n\n"
    f"Parameters:\n"
    f"  sigma = {sigma}, beta = {beta:.4f}, rho = {rho}\n\n"
    f"Lyapunov Spectrum:\n"
    f"  lam_1 = {spectrum[0]:+.4f}\n"
    f"  lam_2 = {spectrum[1]:+.4f}\n"
    f"  lam_3 = {spectrum[2]:+.4f}\n\n"
    f"Sum = {np.sum(spectrum):.4f}\n"
    f"Theory: -(s+1+b) = {theory_sum:.4f}\n\n"
    f"Kaplan-Yorke Dim:\n"
    f"  D_KY = {d_ky:.4f}\n"
    f"  Theory ~ 2.06\n\n"
    f"Largest exponent:\n"
    f"  lam_max = {lam_max:.4f}\n"
    f"  Theory ~ 0.906"
)
ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10,
        color=NAVY, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor=SLATE, alpha=0.9))

fig.suptitle('Module 2e — Lorenz System: Lyapunov Analysis',
             fontsize=15, color=NAVY, fontweight='bold', y=1.02)
plt.tight_layout()

# ── VERIFY ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("VERIFY — Lorenz Lyapunov spectrum:")
print(f"  lambda_max:  measured = {spectrum[0]:.4f},  theory ~ 0.906")
print(f"  Sum:         measured = {np.sum(spectrum):.4f},  "
      f"theory = {theory_sum:.4f}")
print(f"  D_KY:        measured = {d_ky:.4f},  theory ~ 2.06")
print(f"  Benettin:    lam_max  = {lam_max:.4f}")
print("=" * 65)

save_figure(fig, "m2e_lorenz_lyapunov")
plt.close()
