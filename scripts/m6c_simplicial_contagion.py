"""
m6c_simplicial_contagion.py
===========================
Module 6c: Simplicial contagion (higher-order analogue of the ER SIR work).

This script connects to the companion Erdos-Renyi-Contagion repo:
that repo studied pairwise contagion (SIR / bond percolation) with its smooth,
second-order epidemic threshold. Here the SAME pairwise channel is augmented
with a triadic (2-simplex) infection channel.

Proof verified (Iacopini, Petri, Barrat & Latora, Nat. Commun. 2019):
    In simplicial SIS, a susceptible node is infected by infected neighbours at
    rate beta (pairwise) and additionally by 2-simplices in which both other
    members are infected at rate beta_Delta (triadic). When the triadic channel
    is strong enough, the continuous (second-order) pairwise epidemic
    transition becomes DISCONTINUOUS (first-order), with a bistable region and
    a hysteresis loop between a low-density-seed (forward) branch and a
    high-density-seed (backward) branch.

    VERIFY:
      (1) with beta_Delta = 0 (pairwise only, the ER-repo baseline) the forward
          and backward branches coincide -> continuous transition, ~zero
          hysteresis;
      (2) with beta_Delta > 0 the forward branch jumps discontinuously and a
          bistable region opens (large hysteresis area).

Output: media/figures/m6c_simplicial_contagion.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.simplicial import random_simplicial_complex, simplicial_sis_step

np.random.seed(42)
setup_light_theme()

# Build a random simplicial complex
N = 2000
mu = 1.0
dt = 0.1
G, tris = random_simplicial_complex(N, k1=20, k_delta=6, seed=3)
A = nx.to_scipy_sparse_array(G, nodelist=range(N), format='csr')
k_mean = A.sum() / N
beta_c = mu / k_mean       # pairwise mean-field epidemic threshold
print(f"<k>={k_mean:.1f}, #triangles={len(tris)}, pairwise beta_c ~ {beta_c:.4f}")


def stationary(beta, beta_delta, rho0, nsteps=350, avg=120, seed=0):
    """Stationary infected fraction from an initial seed fraction rho0."""
    rng = np.random.default_rng(seed)
    state = np.zeros(N, dtype=bool)
    state[rng.choice(N, int(rho0 * N), replace=False)] = True
    rho = []
    for s in range(nsteps):
        state = simplicial_sis_step(state, A, tris, beta, beta_delta, mu, dt, rng)
        if s >= nsteps - avg:
            rho.append(state.mean())
    return float(np.mean(rho))


betas = np.linspace(0.0, 0.06, 16)


def branches(beta_delta, label):
    print(f"  {label}: forward (small seed) and backward (large seed)...")
    up = np.array([stationary(b, beta_delta, 0.01) for b in betas])
    dn = np.array([stationary(b, beta_delta, 0.95) for b in betas])
    area = np.trapezoid(np.abs(up - dn), betas)
    jump = np.max(np.diff(up))
    return up, dn, area, jump


up_p, dn_p, area_p, jump_p = branches(0.0, "pairwise only (beta_Delta=0)")
up_h, dn_h, area_h, jump_h = branches(0.4, "with triadic (beta_Delta=0.4)")

# Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
fig.patch.set_facecolor("#F8FAFC")

apply_axes_style(ax1)
ax1.plot(betas, up_p, 'o-', color=RED, lw=2, ms=5, mfc='white', mec=RED, mew=1.4,
         label='forward (seed 1%)')
ax1.plot(betas, dn_p, 's-', color=TEAL, lw=2, ms=5, mfc='white', mec=TEAL, mew=1.4,
         label='backward (seed 95%)')
ax1.axvline(beta_c, color=SLATE, ls=':', lw=1.4, alpha=0.8, label=rf'$\beta_c\approx{beta_c:.3f}$')
ax1.set_xlabel(r'Pairwise infection rate $\beta$', color=NAVY)
ax1.set_ylabel(r'Stationary infected fraction $\rho^*$', color=NAVY)
ax1.set_title(rf'Pairwise only ($\beta_\Delta=0$) - CONTINUOUS'
              f'\nhysteresis area = {area_p:.4f}', color=NAVY, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='upper left')

apply_axes_style(ax2)
ax2.plot(betas, up_h, 'o-', color=RED, lw=2, ms=5, mfc='white', mec=RED, mew=1.4,
         label='forward (seed 1%)')
ax2.plot(betas, dn_h, 's-', color=TEAL, lw=2, ms=5, mfc='white', mec=TEAL, mew=1.4,
         label='backward (seed 95%)')
ax2.fill_between(betas, up_h, dn_h, color=GOLD, alpha=0.28, label='bistable region')
ax2.set_xlabel(r'Pairwise infection rate $\beta$', color=NAVY)
ax2.set_title(rf'With triadic channel ($\beta_\Delta=0.4$) - DISCONTINUOUS'
              f'\nhysteresis area = {area_h:.4f}', color=NAVY, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='upper left')
ax2.set_ylim(-0.02, 0.85)

fig.suptitle('Module 6c: Simplicial contagion: the triadic channel turns the epidemic '
             'transition first-order (cf. the ER repo pairwise SIR baseline)',
             fontsize=14, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout()

# VERIFY
ok = (jump_h > 0.3) and (area_h > 5 * max(area_p, 1e-4))
print("=" * 70)
print("VERIFY: simplicial contagion (first-order + bistability):")
print(f"  pairwise (beta_D=0):   hysteresis area = {area_p:.4f}, max jump = {jump_p:.3f}")
print(f"  triadic  (beta_D=0.4): hysteresis area = {area_h:.4f}, max jump = {jump_h:.3f}")
print(f"  triadic transition is discontinuous + bistable: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m6c_simplicial_contagion")
plt.close()
