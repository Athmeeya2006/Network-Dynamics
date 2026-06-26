"""
m5d_explosive_synchronization.py
================================
Module 5d: Explosive synchronisation.

Proof verified (Gomez-Gardenes, Gomez, Arenas & Moreno, PRL 2011):
    On a scale-free network, correlating each oscillator's natural frequency
    with its degree (omega_i = k_i) turns the continuous (second-order)
    Kuramoto transition into an ABRUPT (first-order) one with a HYSTERESIS
    loop: the forward (increasing K) and backward (decreasing K) sweeps of the
    order parameter trace different branches. Destroying the correlation by
    randomly permuting the same frequencies across nodes restores a smooth,
    history-independent (second-order) transition.

    VERIFY:
      (1) the correlated case has a large forward/backward hysteresis area and
          a discontinuous jump in r;
      (2) the shuffled (uncorrelated) case has negligible hysteresis.

Output: media/figures/m5d_explosive_synchronization.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.networks import ba_graph, adjacency_matrix, degree_sequence
from src.kuramoto import network_rhs, adiabatic_sweep

np.random.seed(42)
setup_light_theme()

# Scale-free substrate
N = 500
G = ba_graph(N, m=3, seed=42)
A = adjacency_matrix(G)
k = degree_sequence(G).astype(float)

omega_corr = k - k.mean()                       # degree-frequency correlation
rng = np.random.default_rng(1)
omega_shuf = rng.permutation(omega_corr)        # same frequencies, no correlation

K_values = np.linspace(0.0, 2.6, 40)
T, dt = 35.0, 0.02


def hysteresis(omega, label):
    print(f"  {label}: forward sweep...")
    r_up = adiabatic_sweep(K_values, omega, network_rhs, extra=(A,),
                           T=T, dt=dt, seed=2)
    print(f"  {label}: backward sweep...")
    r_dn = adiabatic_sweep(K_values[::-1], omega, network_rhs, extra=(A,),
                           T=T, dt=dt, seed=2)[::-1]
    area = np.trapezoid(np.abs(r_up - r_dn), K_values)
    max_jump = np.max(np.diff(r_up))
    return r_up, r_dn, area, max_jump


print("Explosive (omega_i = k_i):")
ru_c, rd_c, area_c, jump_c = hysteresis(omega_corr, "correlated")
print("Continuous (shuffled frequencies):")
ru_s, rd_s, area_s, jump_s = hysteresis(omega_shuf, "shuffled")

# Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
fig.patch.set_facecolor("#F8FAFC")

apply_axes_style(ax1)
ax1.plot(K_values, ru_c, 'o-', color=RED, lw=2, ms=5, mfc='white', mec=RED,
         mew=1.4, label='forward (increasing $K$)')
ax1.plot(K_values, rd_c, 's-', color=TEAL, lw=2, ms=5, mfc='white', mec=TEAL,
         mew=1.4, label='backward (decreasing $K$)')
ax1.fill_between(K_values, ru_c, rd_c, color=GOLD, alpha=0.25, label='hysteresis loop')
ax1.set_xlabel('Coupling $K$', color=NAVY)
ax1.set_ylabel('Order parameter $r$', color=NAVY)
ax1.set_title(rf'Correlated $\omega_i = k_i$ - FIRST ORDER'
              f'\nhysteresis area = {area_c:.3f}', color=NAVY, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='upper left')

apply_axes_style(ax2)
ax2.plot(K_values, ru_s, 'o-', color=RED, lw=2, ms=5, mfc='white', mec=RED,
         mew=1.4, label='forward')
ax2.plot(K_values, rd_s, 's-', color=TEAL, lw=2, ms=5, mfc='white', mec=TEAL,
         mew=1.4, label='backward')
ax2.fill_between(K_values, ru_s, rd_s, color=GOLD, alpha=0.25)
ax2.set_xlabel('Coupling $K$', color=NAVY)
ax2.set_title(rf'Shuffled $\omega$ - SECOND ORDER'
              f'\nhysteresis area = {area_s:.3f}', color=NAVY, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='upper left')
ax2.set_ylim(-0.02, 1.0)

fig.suptitle('Module 5d: Explosive synchronisation: degree-frequency correlation '
             'makes the transition first-order',
             fontsize=15, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout()

# VERIFY
ok = (area_c > 0.1) and (jump_c > 0.3) and (area_s < area_c / 3)
print("=" * 70)
print("VERIFY: explosive synchronisation (first-order + hysteresis):")
print(f"  correlated (omega=k):  hysteresis area = {area_c:.4f}, "
      f"max forward jump = {jump_c:.3f}")
print(f"  shuffled  (random):    hysteresis area = {area_s:.4f}, "
      f"max forward jump = {jump_s:.3f}")
print(f"  first-order + hysteresis only under correlation: "
      f"{'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m5d_explosive_synchronization")
plt.close()
