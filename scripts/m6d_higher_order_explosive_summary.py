"""
m6d_higher_order_explosive_summary.py
=====================================
Module 6d - Synthesis: pairwise vs higher-order transitions, across both
synchronisation (Module 5) and contagion (Module 6).

This is the capstone figure of the whole ladder. It places four transitions on
one 2x2 dashboard to make the central message unmistakable: higher-order
(group) interactions turn smooth, second-order transitions into abrupt,
first-order ones with hysteresis - and the same story holds for both
synchronisation and contagion.

    sync     : pairwise Kuramoto (K2=0)        -> continuous
               higher-order Kuramoto (K2>0)    -> explosive (hysteresis)
    contagion: pairwise SIS (beta_Delta=0)     -> continuous
               simplicial SIS (beta_Delta>0)   -> explosive (hysteresis)

    VERIFY: in both domains the higher-order hysteresis area greatly exceeds
    the pairwise one (explosive >> continuous).

Output: media/figures/m6d_higher_order_explosive_summary.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.simplicial import (higher_order_kuramoto_meanfield as ho_rhs,
                            random_simplicial_complex, simplicial_sis_step)
from src.kuramoto import adiabatic_sweep

np.random.seed(42)
setup_light_theme()

# ══════════════════════════════════════════════════════════════════════════════
# SYNCHRONISATION: pairwise (K2=0) vs higher-order (K2>0)
# ══════════════════════════════════════════════════════════════════════════════
N_sync = 600
u = (np.arange(1, N_sync + 1)) / (N_sync + 1)
omega = np.tan(np.pi * (u - 0.5))
omega -= omega.mean()
K1_vals = np.linspace(0.0, 4.0, 22)


def sync_loop(K2):
    up = adiabatic_sweep(K1_vals, omega, ho_rhs, extra=(K2,), T=25, dt=0.02, seed=1)
    dn = adiabatic_sweep(K1_vals[::-1], omega, ho_rhs, extra=(K2,), T=25, dt=0.02, seed=1)[::-1]
    return up, dn, np.trapezoid(np.abs(up - dn), K1_vals)


print("Synchronisation: pairwise vs higher-order...")
sup_c, sdn_c, sarea_c = sync_loop(0.0)
sup_e, sdn_e, sarea_e = sync_loop(10.0)

# ══════════════════════════════════════════════════════════════════════════════
# CONTAGION: pairwise (beta_Delta=0) vs simplicial (beta_Delta>0)
# ══════════════════════════════════════════════════════════════════════════════
N_cont = 1500
mu, dt = 1.0, 0.1
G, tris = random_simplicial_complex(N_cont, k1=20, k_delta=6, seed=3)
A = nx.to_scipy_sparse_array(G, nodelist=range(N_cont), format='csr')
betas = np.linspace(0.0, 0.06, 14)


def stationary(beta, bD, rho0, nsteps=300, avg=110, seed=0):
    rng = np.random.default_rng(seed)
    state = np.zeros(N_cont, dtype=bool)
    state[rng.choice(N_cont, int(rho0 * N_cont), replace=False)] = True
    rho = []
    for s in range(nsteps):
        state = simplicial_sis_step(state, A, tris, beta, bD, mu, dt, rng)
        if s >= nsteps - avg:
            rho.append(state.mean())
    return float(np.mean(rho))


def cont_loop(bD):
    up = np.array([stationary(b, bD, 0.01) for b in betas])
    dn = np.array([stationary(b, bD, 0.95) for b in betas])
    return up, dn, np.trapezoid(np.abs(up - dn), betas)


print("Contagion: pairwise vs simplicial...")
cup_c, cdn_c, carea_c = cont_loop(0.0)
cup_e, cdn_e, carea_e = cont_loop(0.4)

# ══════════════════════════════════════════════════════════════════════════════
# 2x2 dashboard
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.patch.set_facecolor("#F8FAFC")


def panel(ax, x, up, dn, xlabel, ylabel, title, shade):
    apply_axes_style(ax)
    ax.plot(x, up, 'o-', color=RED, lw=2, ms=4.5, mfc='white', mec=RED, mew=1.3,
            label='forward')
    ax.plot(x, dn, 's-', color=TEAL, lw=2, ms=4.5, mfc='white', mec=TEAL, mew=1.3,
            label='backward')
    if shade:
        ax.fill_between(x, up, dn, color=GOLD, alpha=0.28)
    ax.set_xlabel(xlabel, color=NAVY)
    ax.set_ylabel(ylabel, color=NAVY)
    ax.set_title(title, color=NAVY, fontweight='bold', fontsize=12.5)
    ax.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='upper left')


panel(axes[0, 0], K1_vals, sup_c, sdn_c, r'Coupling $K_1$', r'Sync $r$',
      f'Pairwise Kuramoto - CONTINUOUS\nhysteresis = {sarea_c:.3f}', shade=False)
panel(axes[0, 1], K1_vals, sup_e, sdn_e, r'Coupling $K_1$', r'Sync $r$',
      f'Higher-order Kuramoto - EXPLOSIVE\nhysteresis = {sarea_e:.3f}', shade=True)
panel(axes[1, 0], betas, cup_c, cdn_c, r'Infection $\beta$', r'Infected $\rho^*$',
      f'Pairwise SIS - CONTINUOUS\nhysteresis = {carea_c:.4f}', shade=False)
panel(axes[1, 1], betas, cup_e, cdn_e, r'Infection $\beta$', r'Infected $\rho^*$',
      f'Simplicial SIS - EXPLOSIVE\nhysteresis = {carea_e:.4f}', shade=True)

fig.text(0.5, 0.945, 'SYNCHRONISATION (Module 5)  |  CONTAGION (Module 6)',
         ha='center', fontsize=12, color=SLATE, style='italic')
fig.suptitle('Module 6d - Higher-order interactions make transitions explosive, '
             'across sync and contagion alike',
             fontsize=15, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.94])

# ── VERIFY ────────────────────────────────────────────────────────────────────
sync_ok = sarea_e > 5 * max(sarea_c, 1e-3)
cont_ok = carea_e > 5 * max(carea_c, 1e-4)
ok = sync_ok and cont_ok
print("=" * 70)
print("VERIFY - higher-order transitions are explosive in both domains:")
print(f"  sync:      pairwise hysteresis = {sarea_c:.4f}, higher-order = {sarea_e:.4f} "
      f"-> {'PASS' if sync_ok else 'FAIL'}")
print(f"  contagion: pairwise hysteresis = {carea_c:.4f}, simplicial   = {carea_e:.4f} "
      f"-> {'PASS' if cont_ok else 'FAIL'}")
print(f"  Overall: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m6d_higher_order_explosive_summary")
plt.close()
