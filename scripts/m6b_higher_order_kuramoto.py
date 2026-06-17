"""
m6b_higher_order_kuramoto.py
============================
Module 6b - Higher-order (2-simplex) Kuramoto synchronisation.

Proof verified (Skardal & Arenas 2019/2020):
    Adding a triadic (2-simplex) coupling term to the Kuramoto model,
        dtheta_i = omega_i + (K1/N) sum_j sin(theta_j - theta_i)
                            + (K2/N^2) sum_{j,k} sin(2 theta_j - theta_k - theta_i),
    turns the continuous synchronisation transition into an ABRUPT
    (explosive) one with a bistable region and a hysteresis loop between the
    forward and backward sweeps of the coupling K1 — and crucially this
    happens with random frequencies, WITHOUT any degree-frequency correlation
    (contrast with Module 5d, where the correlation was required).

    VERIFY: the triadic model (K2 > 0) has a large forward/backward hysteresis
    area and a discontinuous jump in the order parameter, whereas the
    pairwise-only model (K2 = 0) transitions continuously with negligible
    hysteresis.

Output: media/figures/m6b_higher_order_kuramoto.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.simplicial import higher_order_kuramoto_meanfield as ho_rhs
from src.kuramoto import adiabatic_sweep

np.random.seed(42)
setup_light_theme()

# ── Lorentzian natural frequencies (deterministic quantiles) ─────────────────
N = 1000
gamma = 1.0
u = (np.arange(1, N + 1)) / (N + 1)
omega = gamma * np.tan(np.pi * (u - 0.5))
omega -= omega.mean()

K1_values = np.linspace(0.0, 4.0, 30)
T, dt = 32.0, 0.02


def hysteresis(K2, label):
    print(f"  {label} (K2={K2}): forward...")
    r_up = adiabatic_sweep(K1_values, omega, ho_rhs, extra=(K2,), T=T, dt=dt, seed=1)
    print(f"  {label} (K2={K2}): backward...")
    r_dn = adiabatic_sweep(K1_values[::-1], omega, ho_rhs, extra=(K2,),
                           T=T, dt=dt, seed=1)[::-1]
    area = np.trapezoid(np.abs(r_up - r_dn), K1_values)
    jump = np.max(np.diff(r_up))
    return r_up, r_dn, area, jump


print("Pairwise-only (K2=0):")
ru_p, rd_p, area_p, jump_p = hysteresis(0.0, "pairwise")
print("With triadic 2-simplex coupling (K2=10):")
ru_h, rd_h, area_h, jump_h = hysteresis(10.0, "triadic")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
fig.patch.set_facecolor("#F8FAFC")

apply_axes_style(ax1)
ax1.plot(K1_values, ru_p, 'o-', color=RED, lw=2, ms=5, mfc='white', mec=RED,
         mew=1.4, label='forward')
ax1.plot(K1_values, rd_p, 's-', color=TEAL, lw=2, ms=5, mfc='white', mec=TEAL,
         mew=1.4, label='backward')
ax1.fill_between(K1_values, ru_p, rd_p, color=GOLD, alpha=0.25)
ax1.set_xlabel('Pairwise coupling $K_1$', color=NAVY)
ax1.set_ylabel('Order parameter $r$', color=NAVY)
ax1.set_title(rf'Pairwise only ($K_2=0$) — CONTINUOUS'
              f'\nhysteresis area = {area_p:.3f}', color=NAVY, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='upper left')

apply_axes_style(ax2)
ax2.plot(K1_values, ru_h, 'o-', color=RED, lw=2, ms=5, mfc='white', mec=RED,
         mew=1.4, label='forward (increasing $K_1$)')
ax2.plot(K1_values, rd_h, 's-', color=TEAL, lw=2, ms=5, mfc='white', mec=TEAL,
         mew=1.4, label='backward (decreasing $K_1$)')
ax2.fill_between(K1_values, ru_h, rd_h, color=GOLD, alpha=0.25, label='bistable region')
ax2.set_xlabel('Pairwise coupling $K_1$', color=NAVY)
ax2.set_title(rf'With 2-simplex coupling ($K_2=10$) — EXPLOSIVE'
              f'\nhysteresis area = {area_h:.3f}', color=NAVY, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='upper left')
ax2.set_ylim(-0.02, 1.0)

fig.suptitle('Module 6b — Higher-order Kuramoto: triadic coupling makes synchronisation '
             'explosive (no degree-frequency correlation needed)',
             fontsize=14.5, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout()

# ── VERIFY ────────────────────────────────────────────────────────────────────
ok = (area_h > 0.3) and (jump_h > 0.3) and (area_p < area_h / 3)
print("=" * 70)
print("VERIFY — higher-order (triadic) coupling induces explosive sync:")
print(f"  pairwise (K2=0):  hysteresis area = {area_p:.4f}, max jump = {jump_p:.3f}")
print(f"  triadic  (K2=10): hysteresis area = {area_h:.4f}, max jump = {jump_h:.3f}")
print(f"  triadic term induces abrupt transition + bistability: "
      f"{'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m6b_higher_order_kuramoto")
plt.close()
