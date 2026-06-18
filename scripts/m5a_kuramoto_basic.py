"""
m5a_kuramoto_basic.py
=====================
Module 5a - Mean-field Kuramoto model and the synchronisation transition.

Proof verified:
    For globally coupled phase oscillators with a unimodal frequency
    distribution g(omega), synchronisation appears continuously above a
    critical coupling
        K_c = 2 / (pi g(0)).
    For a Lorentzian g of half-width gamma, g(0) = 1/(pi gamma) so K_c = 2 gamma,
    and the steady order parameter follows r(K) = sqrt(1 - K_c/K) for K > K_c.

    VERIFY: fitting the supercritical branch r^2 = 1 - K_c/K (linear in 1/K)
    recovers K_c to within 5% of the prediction 2 gamma.

Output: media/figures/m5a_kuramoto_basic.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.kuramoto import (mean_field_rhs, adiabatic_sweep, run_kuramoto,
                          kc_lorentzian)

np.random.seed(42)
setup_light_theme()

# ── Frequencies: deterministic Lorentzian quantiles (low finite-N noise) ─────
N = 1000
gamma = 1.0
u = (np.arange(1, N + 1)) / (N + 1)
omega = gamma * np.tan(np.pi * (u - 0.5))
omega -= omega.mean()
Kc_pred = kc_lorentzian(gamma)       # = 2 gamma = 2.0
g0 = 1.0 / (np.pi * gamma)
print(f"Lorentzian half-width gamma={gamma}, g(0)={g0:.4f}, "
      f"predicted K_c = 2/(pi g(0)) = {Kc_pred:.3f}")

# ── Forward coupling sweep ───────────────────────────────────────────────────
K_values = np.linspace(0.4, 4.0, 25)
print("Forward coupling sweep...")
r_of_K = adiabatic_sweep(K_values, omega, mean_field_rhs, T=40, dt=0.02, seed=1)

# ── Recover K_c by fitting r^2 = 1 - K_c/K on the supercritical branch ───────
mask = r_of_K > 0.25
inv_K = 1.0 / K_values[mask]
r2 = r_of_K[mask] ** 2
slope, intercept = np.polyfit(inv_K, r2, 1)   # r^2 = intercept + slope*(1/K)
Kc_fit = -slope                                # since r^2 = 1 - K_c (1/K)
err = abs(Kc_fit - Kc_pred) / Kc_pred

# ── Phase snapshots at three couplings ───────────────────────────────────────
snap_Ks = [1.0, 2.0, 3.5]
snaps = {}
rng = np.random.default_rng(3)
for K in snap_Ks:
    _, _, th = run_kuramoto(mean_field_rhs, rng.uniform(-np.pi, np.pi, N),
                            omega, K, T=50, dt=0.02, record_trace=True)
    snaps[K] = np.mod(th, 2 * np.pi)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)
fig.patch.set_facecolor("#F8FAFC")

ax = fig.add_subplot(gs[0, 0])
apply_axes_style(ax)
ax.plot(K_values, r_of_K, 'o-', color=TEAL, lw=2, ms=7, mfc='white',
        mec=TEAL, mew=1.6, label='simulation')
Kfine = np.linspace(Kc_pred, K_values.max(), 200)
ax.plot(Kfine, np.sqrt(1 - Kc_pred / Kfine), '-', color=NAVY, lw=2,
        label=r'theory $\sqrt{1 - K_c/K}$')
ax.axvline(Kc_pred, color=RED, ls='--', lw=1.6,
           label=rf'$K_c = 2\gamma = {Kc_pred:.2f}$')
ax.axvline(Kc_fit, color=GOLD, ls=':', lw=1.8,
           label=rf'fit $K_c = {Kc_fit:.2f}$')
ax.set_xlabel('Coupling $K$', color=NAVY)
ax.set_ylabel('Order parameter $r$', color=NAVY)
ax.set_title('Mean-field synchronisation transition', color=NAVY, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax.set_ylim(-0.02, 1.0)

ax = fig.add_subplot(gs[0, 1], projection='polar')
colors = [SLATE, PURPLE, RED]
for K, col in zip(snap_Ks, colors):
    th = snaps[K]
    radii = np.ones_like(th) + 0.06 * np.random.default_rng(int(K * 10)).standard_normal(len(th))
    ax.scatter(th, radii * (0.5 + 0.2 * snap_Ks.index(K)), s=6, color=col,
               alpha=0.5, label=rf'$K={K}$ ($r$={np.abs(np.mean(np.exp(1j*th))):.2f})')
ax.set_yticklabels([])
ax.set_title('Phase distributions', color=NAVY, fontweight='bold', pad=18)
ax.legend(fontsize=8.5, loc='upper right', bbox_to_anchor=(1.18, 1.12),
          framealpha=0.95, facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 5a - Kuramoto mean-field model: critical coupling $K_c = 2/(\\pi g(0))$',
             fontsize=15, color=NAVY, fontweight='bold', y=1.0)

# ── VERIFY ────────────────────────────────────────────────────────────────────
ok = err < 0.05
print("=" * 70)
print("VERIFY - Kuramoto critical coupling:")
print(f"  predicted K_c = 2/(pi g(0)) = 2 gamma = {Kc_pred:.4f}")
print(f"  fitted    K_c (from r^2 = 1 - K_c/K) = {Kc_fit:.4f}")
print(f"  relative error = {err * 100:.2f}%  (< 5%)")
print(f"  status: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m5a_kuramoto_basic")
plt.close()
