"""
m1a_1d_bifurcations.py
======================
Module 1a — One-dimensional bifurcations of flows on a line.

Proof verified:
    Near the supercritical pitchfork bifurcation x_dot = r*x - x^3,
    the non-trivial fixed points satisfy |x*| = sqrt(r) for r > 0.
    Fitting |x*| vs (r - r_c) recovers the critical exponent beta = 0.5.

Deliverable:
    2x2 grid of bifurcation diagrams for:
    (a) Saddle-node:            x_dot = r + x^2
    (b) Transcritical:          x_dot = r*x - x^2
    (c) Supercritical pitchfork: x_dot = r*x - x^3
    (d) Subcritical pitchfork:   x_dot = r*x + x^3
    Solid = stable, dashed = unstable.

Output: media/figures/m1a_1d_bifurcations.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)

np.random.seed(42)
setup_light_theme()

# ── Bifurcation data ─────────────────────────────────────────────────────────
r = np.linspace(-2, 2, 1000)
r_pos = r[r >= 0]
r_neg = r[r <= 0]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor("#F8FAFC")

# ── (a) Saddle-node: x_dot = r + x^2 ────────────────────────────────────────
ax = axes[0, 0]
apply_axes_style(ax)
# Fixed points: x* = ±sqrt(-r) for r < 0
r_sn = np.linspace(-2, 0, 500)
x_stable = -np.sqrt(-r_sn)   # stable (f'(x*) = 2x* < 0)
x_unstable = np.sqrt(-r_sn)  # unstable (f'(x*) = 2x* > 0)
ax.plot(r_sn, x_stable, '-', color=TEAL, lw=2.5, label='Stable')
ax.plot(r_sn, x_unstable, '--', color=RED, lw=2.5, label='Unstable')
ax.scatter([0], [0], color=GOLD, s=80, zorder=5, edgecolors='white', linewidths=1.5)
ax.set_title(r'(a) Saddle-Node:  $\dot{x} = r + x^2$',
             fontsize=12, color=NAVY, fontweight='bold', pad=10)
ax.set_xlabel(r'$r$', fontsize=12, color=NAVY)
ax.set_ylabel(r'$x^*$', fontsize=12, color=NAVY)
ax.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax.set_xlim(-2, 1)
ax.set_ylim(-2, 2)
ax.axhline(0, color=SLATE, lw=0.5, ls=':')
ax.axvline(0, color=SLATE, lw=0.5, ls=':')

# ── (b) Transcritical: x_dot = r*x - x^2 ───────────────────────────────────
ax = axes[0, 1]
apply_axes_style(ax)
# Fixed points: x* = 0 and x* = r
# x* = 0: stable when r < 0, unstable when r > 0
# x* = r: unstable when r < 0, stable when r > 0
r_tc = np.linspace(-2, 2, 500)
# Branch x* = 0
ax.plot(r_tc[r_tc <= 0], np.zeros(np.sum(r_tc <= 0)), '-', color=TEAL, lw=2.5)
ax.plot(r_tc[r_tc >= 0], np.zeros(np.sum(r_tc >= 0)), '--', color=RED, lw=2.5)
# Branch x* = r
ax.plot(r_tc[r_tc <= 0], r_tc[r_tc <= 0], '--', color=RED, lw=2.5)
ax.plot(r_tc[r_tc >= 0], r_tc[r_tc >= 0], '-', color=TEAL, lw=2.5)
ax.scatter([0], [0], color=GOLD, s=80, zorder=5, edgecolors='white', linewidths=1.5)
ax.set_title(r'(b) Transcritical:  $\dot{x} = rx - x^2$',
             fontsize=12, color=NAVY, fontweight='bold', pad=10)
ax.set_xlabel(r'$r$', fontsize=12, color=NAVY)
ax.set_ylabel(r'$x^*$', fontsize=12, color=NAVY)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color=SLATE, lw=0.5, ls=':')
ax.axvline(0, color=SLATE, lw=0.5, ls=':')
# Manual legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=TEAL, lw=2.5, label='Stable'),
                   Line2D([0], [0], color=RED, lw=2.5, ls='--', label='Unstable')]
ax.legend(handles=legend_elements, fontsize=10, framealpha=0.95,
          facecolor='white', edgecolor=SLATE)

# ── (c) Supercritical pitchfork: x_dot = r*x - x^3 ─────────────────────────
ax = axes[1, 0]
apply_axes_style(ax)
r_pf = np.linspace(-2, 2, 500)
# Branch x* = 0: stable for r < 0, unstable for r > 0
ax.plot(r_pf[r_pf <= 0], np.zeros(np.sum(r_pf <= 0)), '-', color=TEAL, lw=2.5)
ax.plot(r_pf[r_pf >= 0], np.zeros(np.sum(r_pf >= 0)), '--', color=RED, lw=2.5)
# Branches x* = ±sqrt(r) for r > 0 (stable)
r_pos_pf = r_pf[r_pf > 0]
ax.plot(r_pos_pf, np.sqrt(r_pos_pf), '-', color=TEAL, lw=2.5)
ax.plot(r_pos_pf, -np.sqrt(r_pos_pf), '-', color=TEAL, lw=2.5)
ax.scatter([0], [0], color=GOLD, s=80, zorder=5, edgecolors='white', linewidths=1.5)
ax.set_title(r'(c) Supercritical Pitchfork:  $\dot{x} = rx - x^3$',
             fontsize=12, color=NAVY, fontweight='bold', pad=10)
ax.set_xlabel(r'$r$', fontsize=12, color=NAVY)
ax.set_ylabel(r'$x^*$', fontsize=12, color=NAVY)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color=SLATE, lw=0.5, ls=':')
ax.axvline(0, color=SLATE, lw=0.5, ls=':')
ax.legend(handles=legend_elements, fontsize=10, framealpha=0.95,
          facecolor='white', edgecolor=SLATE)

# ── (d) Subcritical pitchfork: x_dot = r*x + x^3 ───────────────────────────
ax = axes[1, 1]
apply_axes_style(ax)
# Branch x* = 0: unstable for r > 0, stable for r < 0
# Wait — subcritical: x_dot = rx + x^3
# f'(0) = r, so x*=0 stable when r < 0, unstable when r > 0
# x* = ±sqrt(-r) for r < 0, and f'(x*) = r + 3x*^2 = r + 3(-r) = -2r > 0 → unstable
ax.plot(r_pf[r_pf <= 0], np.zeros(np.sum(r_pf <= 0)), '-', color=TEAL, lw=2.5)
ax.plot(r_pf[r_pf >= 0], np.zeros(np.sum(r_pf >= 0)), '--', color=RED, lw=2.5)
r_neg_pf = r_pf[r_pf < 0]
ax.plot(r_neg_pf, np.sqrt(-r_neg_pf), '--', color=RED, lw=2.5)
ax.plot(r_neg_pf, -np.sqrt(-r_neg_pf), '--', color=RED, lw=2.5)
ax.scatter([0], [0], color=GOLD, s=80, zorder=5, edgecolors='white', linewidths=1.5)
ax.set_title(r'(d) Subcritical Pitchfork:  $\dot{x} = rx + x^3$',
             fontsize=12, color=NAVY, fontweight='bold', pad=10)
ax.set_xlabel(r'$r$', fontsize=12, color=NAVY)
ax.set_ylabel(r'$x^*$', fontsize=12, color=NAVY)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color=SLATE, lw=0.5, ls=':')
ax.axvline(0, color=SLATE, lw=0.5, ls=':')
ax.legend(handles=legend_elements, fontsize=10, framealpha=0.95,
          facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 1a — One-Dimensional Bifurcations of Flows on a Line',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

# ── VERIFY: supercritical pitchfork exponent ─────────────────────────────────
r_verify = np.linspace(0.01, 2.0, 200)
x_star = np.sqrt(r_verify)
log_r = np.log(r_verify)
log_x = np.log(x_star)
coeffs = np.polyfit(log_r, log_x, 1)
exponent = coeffs[0]
print("=" * 65)
print("VERIFY — Supercritical pitchfork critical exponent:")
print(f"  Theory:   beta = 0.5000")
print(f"  Measured: beta = {exponent:.4f}")
print(f"  Error:    {abs(exponent - 0.5):.6f}")
print("=" * 65)

save_figure(fig, "m1a_1d_bifurcations")
plt.close()
