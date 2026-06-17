"""
m5b_kuramoto_on_networks.py
===========================
Module 5b - Kuramoto oscillators on Erdos-Renyi vs Barabasi-Albert networks.

Proof verified:
    When phase oscillators are coupled through a network rather than all-to-all,
    the synchronisation onset depends on topology. Heavy-tailed (scale-free,
    BA) connectivity lowers the synchronisation threshold relative to a
    homogeneous (ER) graph of the same mean degree: hubs act as synchronisation
    seeds. Mean-field theory on networks gives K_c proportional to <k>/<k^2>,
    which is smaller for BA because <k^2> is large.

    VERIFY: the coupling K_50 at which r first exceeds 0.5 is smaller on BA
    than on ER (K_50^BA < K_50^ER).

Output: media/figures/m5b_kuramoto_on_networks.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, SLATE, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.networks import er_graph, ba_graph, adjacency_matrix, mean_degree, degree_sequence
from src.kuramoto import network_rhs, adiabatic_sweep, sample_lorentzian

np.random.seed(42)
setup_light_theme()

# ── Networks with matched mean degree ────────────────────────────────────────
N = 500
m = 3
G_er = er_graph(N, p=2 * m / (N - 1), seed=42)
G_ba = ba_graph(N, m=m, seed=42)
A_er = adjacency_matrix(G_er)
A_ba = adjacency_matrix(G_ba)
print(f"<k>: ER={mean_degree(G_er):.2f}, BA={mean_degree(G_ba):.2f}")

# Frequencies (shared draw so the only difference is topology)
rng = np.random.default_rng(7)
omega = sample_lorentzian(0.5, N, rng)
omega -= omega.mean()


def k2_ratio(G):
    k = degree_sequence(G).astype(float)
    return np.mean(k) / np.mean(k ** 2)


print(f"<k>/<k^2>: ER={k2_ratio(G_er):.4f}, BA={k2_ratio(G_ba):.4f} (smaller => lower K_c)")

# ── Coupling sweeps ──────────────────────────────────────────────────────────
K_values = np.linspace(0.0, 0.5, 26)
print("Sweeping ER...")
r_er = adiabatic_sweep(K_values, omega, network_rhs, extra=(A_er,), T=40, dt=0.02, seed=1)
print("Sweeping BA...")
r_ba = adiabatic_sweep(K_values, omega, network_rhs, extra=(A_ba,), T=40, dt=0.02, seed=1)


def k_at_threshold(K, r, thr=0.5):
    idx = np.where(r >= thr)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return K[0]
    # linear interpolation to r = thr
    return K[i - 1] + (thr - r[i - 1]) * (K[i] - K[i - 1]) / (r[i] - r[i - 1])


K50_er = k_at_threshold(K_values, r_er)
K50_ba = k_at_threshold(K_values, r_ba)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#F8FAFC")
apply_axes_style(ax)
ax.plot(K_values, r_er, 'o-', color=TEAL, lw=2, ms=6, mfc='white', mec=TEAL,
        mew=1.5, label='Erdos-Renyi')
ax.plot(K_values, r_ba, 's-', color=PURPLE, lw=2, ms=6, mfc='white', mec=PURPLE,
        mew=1.5, label='Barabasi-Albert')
ax.axhline(0.5, color=SLATE, ls='--', lw=1, alpha=0.7)
ax.axvline(K50_er, color=TEAL, ls=':', lw=1.5, alpha=0.8)
ax.axvline(K50_ba, color=PURPLE, ls=':', lw=1.5, alpha=0.8)
ax.annotate(rf'$K_{{50}}^{{ER}}={K50_er:.3f}$', (K50_er, 0.52), color=TEAL, fontsize=10)
ax.annotate(rf'$K_{{50}}^{{BA}}={K50_ba:.3f}$', (K50_ba, 0.6), color=PURPLE, fontsize=10)
ax.set_xlabel('Coupling $K$', color=NAVY)
ax.set_ylabel('Order parameter $r$', color=NAVY)
ax.set_title('Module 5b — Kuramoto on networks: scale-free topology lowers the sync onset\n'
             rf'$N={N}$, matched $\langle k\rangle\approx{mean_degree(G_ba):.1f}$',
             color=NAVY, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax.set_ylim(-0.02, 1.0)

# ── VERIFY ────────────────────────────────────────────────────────────────────
ok = K50_ba < K50_er
print("=" * 70)
print("VERIFY — synchronisation onset, ER vs BA:")
print(f"  K_50 (r crosses 0.5):  ER = {K50_er:.4f},  BA = {K50_ba:.4f}")
print(f"  BA onset below ER onset: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m5b_kuramoto_on_networks")
plt.close()
