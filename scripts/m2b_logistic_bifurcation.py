"""
m2b_logistic_bifurcation.py
===========================
Module 2b: Logistic map bifurcation diagram with Feigenbaum verification.

Proof verified:
    1. Feigenbaum delta: ratio of successive period-doubling intervals
       converges to delta = 4.6692...
    2. Feigenbaum alpha: ratio of successive superstable orbit widths
       converges to alpha = 2.5029...

Output: media/figures/m2b_logistic_bifurcation.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.maps import logistic_map, iterate_map

np.random.seed(42)
setup_light_theme()

# High-resolution bifurcation diagram
n_r = 4000
r_values = np.linspace(2.8, 4.0, n_r)
n_iter = 300
transient = 500
x0 = 0.5

r_plot = []
x_plot = []

for r in r_values:
    orbit = iterate_map(logistic_map, x0, r, n_iter, transient)
    r_plot.extend([r] * n_iter)
    x_plot.extend(orbit)

r_plot = np.array(r_plot)
x_plot = np.array(x_plot)

# Figure
fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                         gridspec_kw={'width_ratios': [3, 1]})
fig.patch.set_facecolor("#F8FAFC")

# Main bifurcation diagram
ax = axes[0]
apply_axes_style(ax, grid=False)
ax.scatter(r_plot, x_plot, s=0.01, color=NAVY, alpha=0.3, rasterized=True)
ax.set_xlabel(r'Parameter $r$', fontsize=13, color=NAVY)
ax.set_ylabel(r'$x^*$ (attractor)', fontsize=13, color=NAVY)
ax.set_title('Logistic Map Bifurcation Diagram\n'
             r'$x_{n+1} = r\,x_n(1 - x_n)$',
             fontsize=14, color=NAVY, fontweight='bold', pad=10)
ax.set_xlim(2.8, 4.0)
ax.set_ylim(0, 1)

# Mark period-3 window
ax.axvspan(3.828, 3.857, alpha=0.15, color=GOLD)
ax.text(3.843, 0.05, 'P-3', fontsize=9, color=GOLD, ha='center',
        fontweight='bold')

# Zoom inset
ax_zoom = axes[1]
apply_axes_style(ax_zoom, grid=False)
mask = (r_plot >= 3.4) & (r_plot <= 3.6)
ax_zoom.scatter(r_plot[mask], x_plot[mask], s=0.05, color=PURPLE,
                alpha=0.4, rasterized=True)
ax_zoom.set_xlabel(r'$r$', fontsize=12, color=NAVY)
ax_zoom.set_ylabel(r'$x^*$', fontsize=12, color=NAVY)
ax_zoom.set_title('Zoom: Self-Similarity', fontsize=12, color=NAVY,
                  fontweight='bold')
ax_zoom.set_xlim(3.4, 3.6)

fig.suptitle('Module 2b: Period-Doubling Cascade and Feigenbaum Constants',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

# VERIFY: Feigenbaum delta
# Known period-doubling thresholds for the logistic map
r_thresholds = [
    3.0,              # 1 -> 2
    3.4494897,        # 2 -> 4
    3.5440903,        # 4 -> 8
    3.5644073,        # 8 -> 16
    3.5687594,        # 16 -> 32
    3.5696916,        # 32 -> 64
    3.5698913,        # 64 -> 128
]

print("=" * 65)
print("VERIFY: Feigenbaum delta (period-doubling ratio):")
print(f"  {'n':>3s}  {'r_n':>12s}  {'delta_n':>12s}")
deltas = []
for i in range(2, len(r_thresholds)):
    delta = (r_thresholds[i - 1] - r_thresholds[i - 2]) / \
            (r_thresholds[i] - r_thresholds[i - 1])
    deltas.append(delta)
    print(f"  {i:3d}  {r_thresholds[i]:12.7f}  {delta:12.4f}")
print(f"\n  Theory:   delta = 4.6692")
print(f"  Best n:   delta = {deltas[-1]:.4f}")
print(f"  Error:    {abs(deltas[-1] - 4.6692):.4f}")

# VERIFY: Feigenbaum alpha
# d_n = f^{2^{n-1}}(0.5, r_ss_n) - 0.5  at the superstable parameter
# alpha = d_n / d_{n+1} -> 2.5029
print("\nVERIFY: Feigenbaum alpha (branch spacing ratio):")
from src.maps import find_superstable_r

r_ss = []
brackets = [
    (2.0, 3.0),       # period-1 superstable
    (3.0, 3.5),       # period-2
    (3.44, 3.56),     # period-4
    (3.54, 3.57),     # period-8
    (3.564, 3.570),   # period-16
    (3.5687, 3.5698), # period-32
    (3.56989, 3.56994),  # period-64
]
for period_exp, (r_lo, r_hi) in enumerate(brackets):
    period = 2 ** period_exp
    r_ss.append(find_superstable_r(logistic_map, r_lo, r_hi, period))

# d_n = f^{2^{n-1}}(0.5) - 0.5 at the superstable r of the 2^n-cycle
d_values = []
for n_exp in range(len(r_ss)):
    r = r_ss[n_exp]
    period = 2 ** n_exp
    half_period = max(period // 2, 1)
    x = 0.5
    for _ in range(half_period):
        x = logistic_map(x, r)
    d_values.append(x - 0.5)

alphas = []
for i in range(1, len(d_values) - 1):
    if abs(d_values[i + 1]) > 1e-15:
        alpha = d_values[i] / d_values[i + 1]
        alphas.append(alpha)
        print(f"  n={i}: d_{i}={d_values[i]:+.8f}, "
              f"alpha = {alpha:.4f}")

if alphas:
    print(f"\n  Theory:   alpha = -2.5029  (negative by convention)")
    print(f"  Best |a|: |alpha| = {abs(alphas[-1]):.4f}")
    print(f"  Error:    {abs(abs(alphas[-1]) - 2.5029):.4f}")
print("=" * 65)

save_figure(fig, "m2b_logistic_bifurcation")
plt.close()
