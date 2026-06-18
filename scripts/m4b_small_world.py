"""
m4b_small_world.py
==================
Module 4b - The Watts-Strogatz small-world transition.

Proof verified:
    Interpolating a ring lattice toward a random graph by rewiring each edge
    with probability p, the characteristic path length L(p) collapses much
    faster than the clustering coefficient C(p). There is therefore a broad
    window of p in which L(p)/L(0) is already small (random-like short paths)
    while C(p)/C(0) is still near 1 (lattice-like high clustering): the
    small-world regime (Watts & Strogatz 1998).

    VERIFY: there exists p with L(p)/L(0) < 0.5 while C(p)/C(0) > 0.5
    (path length has collapsed but clustering is largely retained).

Output: media/figures/m4b_small_world.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.networks import watts_strogatz, clustering, avg_path_length

np.random.seed(42)
setup_light_theme()

# ── Parameters ───────────────────────────────────────────────────────────────
N = 1000
k = 10
n_realizations = 20
p_values = np.logspace(-4, 0, 18)

# Baselines at p = 0 (ring lattice)
G0 = watts_strogatz(N, k, 0.0, seed=0)
L0 = avg_path_length(G0)
C0 = clustering(G0)

print(f"Ring baseline: L(0) = {L0:.3f}, C(0) = {C0:.4f}")
print("Sweeping rewiring probability p...")

L_ratio = np.empty(len(p_values))
C_ratio = np.empty(len(p_values))
for i, p in enumerate(p_values):
    Ls, Cs = [], []
    for r in range(n_realizations):
        G = watts_strogatz(N, k, p, seed=1000 * i + r)
        Ls.append(avg_path_length(G))
        Cs.append(clustering(G))
    L_ratio[i] = np.mean(Ls) / L0
    C_ratio[i] = np.mean(Cs) / C0

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#F8FAFC")
apply_axes_style(ax)

ax.semilogx(p_values, L_ratio, 'o-', color=TEAL, lw=2.2, ms=8, mfc='white',
            mec=TEAL, mew=1.8, label=r'$L(p)\,/\,L(0)$  (path length)')
ax.semilogx(p_values, C_ratio, 's-', color=RED, lw=2.2, ms=8, mfc='white',
            mec=RED, mew=1.8, label=r'$C(p)\,/\,C(0)$  (clustering)')

# Highlight the small-world window
sw_mask = (L_ratio < 0.5) & (C_ratio > 0.5)
if np.any(sw_mask):
    ax.axvspan(p_values[sw_mask].min(), p_values[sw_mask].max(),
               color=GOLD, alpha=0.15, label='small-world window')

ax.set_xlabel('Rewiring probability $p$', color=NAVY)
ax.set_ylabel('Normalised quantity', color=NAVY)
ax.set_title('Module 4b - Watts-Strogatz small-world transition\n'
             rf'$N={N}$, $k={k}$, {n_realizations} realisations per $p$',
             color=NAVY, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.95, facecolor='white', edgecolor=SLATE, loc='center left')
ax.set_ylim(0, 1.05)

# ── VERIFY ────────────────────────────────────────────────────────────────────
ok = np.any(sw_mask)
if ok:
    idx = np.where(sw_mask)[0]
    p_lo, p_hi = p_values[idx.min()], p_values[idx.max()]
    sample = idx[len(idx) // 2]
print("=" * 70)
print("VERIFY - small-world window (L collapsed, C retained):")
if ok:
    print(f"  window: p in [{p_lo:.1e}, {p_hi:.1e}]")
    print(f"  e.g. p={p_values[sample]:.1e}: "
          f"L/L0={L_ratio[sample]:.3f} (<0.5), C/C0={C_ratio[sample]:.3f} (>0.5)")
print(f"  status: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m4b_small_world")
plt.close()
