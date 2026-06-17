"""
m2c_logistic_lyapunov.py
========================
Module 2c - Lyapunov exponent of the logistic map.

Proof verified:
    lambda(r) crosses zero at each period-doubling, is positive in
    chaotic regions, and dips negative inside periodic windows
    (period-3 window is the clearest).

Output: media/figures/m2c_logistic_lyapunov.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.maps import logistic_map, logistic_map_derivative, iterate_map
from src.lyapunov import map_lyapunov

np.random.seed(42)
setup_light_theme()

# ── Compute Lyapunov exponent and bifurcation diagram ────────────────────────
n_r = 3000
r_values = np.linspace(2.8, 4.0, n_r)
lyap = np.array([map_lyapunov(logistic_map, logistic_map_derivative,
                               0.1, r, 10000, 1000) for r in r_values])

# Bifurcation data
r_bif, x_bif = [], []
for r in np.linspace(2.8, 4.0, 2000):
    orbit = iterate_map(logistic_map, 0.1, r, 200, 500)
    r_bif.extend([r] * 200)
    x_bif.extend(orbit)

# ── Figure: two panels sharing x-axis ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1.5]})
fig.patch.set_facecolor("#F8FAFC")

# Top: Lyapunov exponent
apply_axes_style(ax1)
ax1.fill_between(r_values, lyap, 0, where=(lyap >= 0), color=RED,
                 alpha=0.15, label=r'$\lambda > 0$ (chaos)')
ax1.fill_between(r_values, lyap, 0, where=(lyap < 0), color=TEAL,
                 alpha=0.15, label=r'$\lambda < 0$ (periodic)')
ax1.plot(r_values, lyap, '-', color=NAVY, lw=0.8)
ax1.axhline(0, color=GOLD, ls='--', lw=1.5, alpha=0.7)
ax1.set_ylabel(r'Lyapunov exponent $\lambda$', fontsize=12, color=NAVY)
ax1.set_title('Lyapunov Exponent of the Logistic Map',
              fontsize=13, color=NAVY, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax1.set_ylim(-3, 1)

# Period-3 window annotation
ax1.annotate('Period-3\nwindow', xy=(3.83, -1.5),
             fontsize=9, color=PURPLE, ha='center',
             fontweight='bold')

# Bottom: bifurcation diagram
apply_axes_style(ax2, grid=False)
ax2.scatter(r_bif, x_bif, s=0.01, color=NAVY, alpha=0.3, rasterized=True)
ax2.set_xlabel(r'Parameter $r$', fontsize=13, color=NAVY)
ax2.set_ylabel(r'$x^*$', fontsize=13, color=NAVY)
ax2.set_title('Bifurcation Diagram', fontsize=13, color=NAVY,
              fontweight='bold')
ax2.set_xlim(2.8, 4.0)
ax2.set_ylim(0, 1)

fig.suptitle('Module 2c — Lyapunov Exponent Overlaid on Bifurcation Diagram',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

# ── VERIFY ───────────────────────────────────────────────────────────────────
# Check: lambda at r=3.0 (period-doubling) should be near 0
# lambda at r=4.0 should be ln(2) ~ 0.693
lam_r3 = map_lyapunov(logistic_map, logistic_map_derivative, 0.1, 3.0,
                       50000, 2000)
lam_r4 = map_lyapunov(logistic_map, logistic_map_derivative, 0.1, 4.0,
                       50000, 2000)
# Period-3 window: r ~ 3.83 should have lambda < 0
lam_p3 = map_lyapunov(logistic_map, logistic_map_derivative, 0.1, 3.83,
                       50000, 2000)

print("=" * 65)
print("VERIFY — Lyapunov exponent at key r values:")
print(f"  r = 3.0:  lambda = {lam_r3:+.6f}  "
      f"(theory: 0, period-doubling)")
print(f"  r = 4.0:  lambda = {lam_r4:+.6f}  "
      f"(theory: ln(2) = {np.log(2):.6f})")
print(f"  r = 3.83: lambda = {lam_p3:+.6f}  "
      f"(theory: < 0, period-3 window)")
print(f"\n  r=4.0 error: {abs(lam_r4 - np.log(2)):.6f}")
print("=" * 65)

save_figure(fig, "m2c_logistic_lyapunov")
plt.close()
