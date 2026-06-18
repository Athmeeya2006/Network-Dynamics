"""
m2a_logistic_cobweb.py
======================
Module 2a - Logistic map cobweb diagrams.

Shows the transition from fixed point to period-2 to chaos via
cobweb diagrams at r = 2.8, 3.3, 3.5, 3.9.

Output: media/figures/m2a_logistic_cobweb.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.maps import logistic_map, cobweb_data

np.random.seed(42)
setup_light_theme()

r_values = [2.8, 3.3, 3.5, 3.9]
titles = ['Fixed Point', 'Period-2', 'Period-4 / Onset', 'Chaos']
colors = [TEAL, GOLD, PURPLE, RED]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor("#F8FAFC")

for ax, r, title, color in zip(axes.flat, r_values, titles, colors):
    apply_axes_style(ax)

    # Plot f(x) = r*x*(1-x) and y=x
    x_line = np.linspace(0, 1, 300)
    ax.plot(x_line, logistic_map(x_line, r), '-', color=color, lw=2.5,
            label=rf'$f(x) = {r}\,x(1-x)$')
    ax.plot(x_line, x_line, '-', color=SLATE, lw=1.5, alpha=0.7,
            label=r'$y = x$')

    # Cobweb
    xs, ys = cobweb_data(logistic_map, 0.1, r, 60)
    ax.plot(xs, ys, '-', color=NAVY, lw=0.8, alpha=0.6)

    # Start marker
    ax.scatter([0.1], [0], color=color, s=60, zorder=5,
               edgecolors='white', linewidths=1.5)

    ax.set_xlabel(r'$x_n$', fontsize=12, color=NAVY)
    ax.set_ylabel(r'$x_{n+1}$', fontsize=12, color=NAVY)
    ax.set_title(f'r = {r} - {title}', fontsize=12, color=NAVY,
                 fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.95, facecolor='white',
              edgecolor=SLATE, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

fig.suptitle('Module 2a - Logistic Map Cobweb Diagrams',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

print("=" * 65)
print("Cobweb diagrams generated for r = 2.8, 3.3, 3.5, 3.9")
print("  r=2.8: converges to fixed point x* = 1 - 1/r = 0.6429")
x_star = 1.0 - 1.0 / 2.8
print(f"  Theory: x* = {x_star:.4f}")
print("=" * 65)

save_figure(fig, "m2a_logistic_cobweb")
plt.close()
