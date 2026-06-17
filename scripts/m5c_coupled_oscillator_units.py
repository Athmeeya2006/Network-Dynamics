"""
m5c_coupled_oscillator_units.py
===============================
Module 5c - Module-1 oscillator units dropped onto a network (composability).

This is the end-to-end test of the prime directive: the FitzHughNagumo and
StuartLandau classes from src/oscillators.py are instantiated N times on a
graph and coupled with a single diffusive term via coupled_network_rhs, with
no rewriting of the unit dynamics.

Proof verified:
    (A) Diffusively coupled FitzHugh-Nagumo units on a 1D chain support a
        propagating excitation pulse (a travelling wave): a localised stimulus
        triggers a front that sweeps the chain at finite speed.
    (B) Diffusively coupled Stuart-Landau oscillators with a spread of natural
        frequencies exhibit AMPLITUDE DEATH: at intermediate coupling the
        collective state collapses to the (now stabilised) origin and the
        oscillations are quenched, even though each isolated unit oscillates.

    VERIFY: the time-averaged oscillation amplitude of the Stuart-Landau
    ensemble drops below 0.05 in an intermediate coupling window, while the
    uncoupled (K=0) amplitude exceeds 0.5 — i.e. an amplitude-death region
    exists.

Output: media/figures/m5c_coupled_oscillator_units.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.oscillators import FitzHughNagumo, StuartLandau, coupled_network_rhs
from src.integrators import rk4
from src.networks import adjacency_matrix

np.random.seed(42)
setup_light_theme()

# ══════════════════════════════════════════════════════════════════════════════
# (A) Coupled FitzHugh-Nagumo travelling wave on a 1D chain
# ══════════════════════════════════════════════════════════════════════════════
N_chain = 80
fhn_units = [FitzHughNagumo(a=0.7, b=0.8, tau=12.5, I=0.0) for _ in range(N_chain)]
A_chain = adjacency_matrix(nx.path_graph(N_chain))

vrest, wrest = -1.2, (-1.2 + 0.7) / 0.8
X0 = np.zeros((N_chain, 2))
X0[:, 0] = vrest
X0[:, 1] = wrest
X0[:3, 0] = 1.5            # localised stimulus at the left end
print("Integrating coupled FitzHugh-Nagumo chain (travelling wave)...")
t_fhn, traj = rk4(coupled_network_rhs, X0.reshape(-1), (0, 400), 0.05,
                  fhn_units, A_chain, 0.5, [0])     # couple only the v-component
V = traj.reshape(len(t_fhn), N_chain, 2)[:, :, 0]

peak_t = t_fhn[np.argmax(V, axis=0)]
front_monotone = np.all(np.diff(peak_t[5:-5]) >= -1.0)
speed = (N_chain - 10) / (peak_t[-5] - peak_t[5])   # nodes per time unit

# ══════════════════════════════════════════════════════════════════════════════
# (B) Coupled Stuart-Landau amplitude death (mean-field diffusive coupling)
# ══════════════════════════════════════════════════════════════════════════════
N_sl = 30
rng = np.random.default_rng(0)
omegas = rng.uniform(-3.0, 3.0, N_sl)               # frequency spread drives death
sl_units = [StuartLandau(mu=1.0, omega=w) for w in omegas]
A_full = adjacency_matrix(nx.complete_graph(N_sl)) / (N_sl - 1)   # mean-field
sl_x0 = rng.uniform(-1, 1, 2 * N_sl)

K_sl = np.linspace(0.0, 16.0, 22)
print("Sweeping Stuart-Landau coupling (amplitude death)...")
amp = np.empty(len(K_sl))
for i, K in enumerate(K_sl):
    ts, Xs = rk4(coupled_network_rhs, sl_x0, (0, 120), 0.01, sl_units, A_full, K)
    Xs = Xs.reshape(len(ts), N_sl, 2)
    half = len(ts) // 2
    amp[i] = Xs[half:, :, 0].std(axis=0).mean()

amp0 = amp[0]
death_mask = amp < 0.05
amp_min = amp.min()

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
fig.patch.set_facecolor("#F8FAFC")

# Space-time of the FHN wave
ax = fig.add_subplot(gs[0, :])
im = ax.imshow(V.T, aspect='auto', origin='lower', cmap='magma',
               extent=[0, t_fhn[-1], 0, N_chain])
ax.set_xlabel('Time', color=NAVY)
ax.set_ylabel('Chain position $i$', color=NAVY)
ax.set_title(f'(A) Coupled FitzHugh-Nagumo chain — travelling wave '
             f'(speed ~ {speed:.2f} nodes/time)', color=NAVY, fontweight='bold')
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label('$v$ (voltage)', color=NAVY)

# Voltage traces of a few nodes
ax = fig.add_subplot(gs[1, 0])
apply_axes_style(ax)
for node, col in zip([5, 25, 45, 65], [TEAL, GOLD, RED, PURPLE]):
    ax.plot(t_fhn, V[:, node], color=col, lw=1.4, label=f'node {node}')
ax.set_xlabel('Time', color=NAVY)
ax.set_ylabel('$v_i(t)$', color=NAVY)
ax.set_title('Pulse arrival is delayed down the chain', color=NAVY, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE, ncol=2)

# Stuart-Landau amplitude death
ax = fig.add_subplot(gs[1, 1])
apply_axes_style(ax)
ax.plot(K_sl, amp, 'o-', color=PURPLE, lw=2, ms=6, mfc='white', mec=PURPLE, mew=1.6)
ax.axhline(0.05, color=RED, ls='--', lw=1.2, alpha=0.8, label='death threshold 0.05')
if np.any(death_mask):
    ax.axvspan(K_sl[death_mask].min(), K_sl[death_mask].max(),
               color=SLATE, alpha=0.18, label='amplitude-death window')
ax.set_xlabel('Coupling $K$', color=NAVY)
ax.set_ylabel('Mean oscillation amplitude', color=NAVY)
ax.set_title('(B) Coupled Stuart-Landau — amplitude death', color=NAVY, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 5c — Module-1 units on a network: travelling waves & amplitude death',
             fontsize=16, color=NAVY, fontweight='bold', y=0.98)

# ── VERIFY ────────────────────────────────────────────────────────────────────
death_exists = np.any(death_mask) and amp0 > 0.5
ok = death_exists and front_monotone
print("=" * 70)
print("VERIFY — composability: coupled Module-1 units on a graph:")
print(f"  (A) FHN travelling wave: monotone front = {front_monotone}, "
      f"speed ~ {speed:.3f} nodes/time")
print(f"  (B) Stuart-Landau amplitude death:")
print(f"      uncoupled amp (K=0) = {amp0:.3f} (>0.5)")
print(f"      minimum amp         = {amp_min:.3f} at K = {K_sl[np.argmin(amp)]:.2f}")
if np.any(death_mask):
    print(f"      death window K in [{K_sl[death_mask].min():.2f}, "
          f"{K_sl[death_mask].max():.2f}] (amp < 0.05)")
print(f"  status: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m5c_coupled_oscillator_units")
plt.close()
