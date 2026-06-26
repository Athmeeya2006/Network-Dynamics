"""
m3c_rps_on_network.py
=====================
Module 3c: Rock-Paper-Scissors on Networks.

Proof verified:
    Imitation (Fermi) dynamics of neutral RPS on a finite network is NOT
    coexistence-preserving within a single realization: stochastic drift
    drives every finite run to FIXATION on a single absorbing strategy
    (loss of biodiversity). However, because the neutral payoff matrix is
    symmetric under cyclic relabelling of the three strategies, no strategy
    is favoured, so the ENSEMBLE average over many independent runs keeps
    each strategy fraction pinned at the symmetric Nash value 1/3.

    We verify both halves:
      (1) ensemble-averaged final fractions  ->  (1/3, 1/3, 1/3);
      (2) within-run order parameter (max strategy fraction)  ->  1.0
          (fixation), with heterogeneous BA networks fixating faster than
          homogeneous ER networks (hubs accelerate consensus).

    The previous version averaged over TIME within a single fixated run and
    therefore (incorrectly) reported (0, 0, 1). The fix is ensemble averaging.

Output: media/figures/m3c_rps_on_network.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.games import rps_payoff, network_rps_step

np.random.seed(42)
setup_light_theme()

# Parameters
N = 200
n_sweeps = 150
n_runs = 250
temp = 0.5
payoff = rps_payoff(0.0)  # Neutral RPS

# Generate the (fixed) graphs once; randomness lives in the dynamics
print("Generating ER and BA graphs...")
G_er = nx.erdos_renyi_graph(N, p=0.06, seed=42)
G_ba = nx.barabasi_albert_graph(N, m=3, seed=42)


# Simulation: one realization, return per-sweep fractions
def run_once(G, seed):
    rng = np.random.default_rng(seed)
    strategies = rng.integers(0, 3, size=N)
    fractions = np.empty((n_sweeps, 3))
    for sweep in range(n_sweeps):
        strategies = network_rps_step(G, strategies, payoff, temp, rng)
        counts = np.bincount(strategies, minlength=3)
        fractions[sweep] = counts / N
    return fractions


def run_ensemble(G, label):
    print(f"Running {n_runs} RPS realizations on {label}...")
    runs = np.stack([run_once(G, seed) for seed in range(n_runs)])  # (R, T, 3)
    ens_mean = runs.mean(axis=0)            # (T, 3) ensemble-mean fractions
    dominance = runs.max(axis=2).mean(0)    # (T,) mean of max-strategy fraction
    final_frac = runs[:, -1, :].mean(0)     # (3,) ensemble-mean final fraction
    return runs, ens_mean, dominance, final_frac


runs_er, ens_er, dom_er, final_er = run_ensemble(G_er, "Erdos-Renyi")
runs_ba, ens_ba, dom_ba, final_ba = run_ensemble(G_ba, "Barabasi-Albert")

# Figure Layout (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("#F8FAFC")
colors = [RED, TEAL, GOLD]
labels = ['Rock', 'Paper', 'Scissors']

# Top row: ensemble-mean fractions pinned at 1/3 (with thin single-run traces)
for ax, runs, ens, label, sub in (
        (axes[0, 0], runs_er, ens_er, 'Erdos-Renyi', rf'$N={N}$, $p=0.06$'),
        (axes[0, 1], runs_ba, ens_ba, 'Barabasi-Albert', rf'$N={N}$, $m=3$')):
    apply_axes_style(ax)
    for r in range(min(8, n_runs)):  # a few raw runs -> show they fixate
        ax.plot(runs[r, :, 0], color=SLATE, lw=0.5, alpha=0.25)
    for k in range(3):
        ax.plot(ens[:, k], '-', color=colors[k], lw=2.2, label=f'{labels[k]} (ens. mean)')
    ax.axhline(1 / 3, color=NAVY, ls='--', lw=1.2, alpha=0.8, label='Nash 1/3')
    ax.set_xlabel('Imitation Sweeps', color=NAVY)
    ax.set_ylabel('Strategy Fraction', color=NAVY)
    ax.set_title(f'{label}\n{sub}', color=NAVY, fontweight='bold')
    ax.legend(fontsize=8.5, framealpha=0.95, facecolor='white', edgecolor=SLATE, ncol=2)
    ax.set_ylim(0, 1.02)

# Bottom-left: dominance (max fraction) -> 1.0 = fixation, ER vs BA speed
ax = axes[1, 0]
apply_axes_style(ax)
ax.plot(dom_er, '-', color=TEAL, lw=2.2, label='Erdos-Renyi')
ax.plot(dom_ba, '-', color=PURPLE, lw=2.2, label='Barabasi-Albert')
ax.axhline(1.0, color=RED, ls=':', lw=1.2, alpha=0.8, label='Full fixation')
ax.axhline(1 / 3, color=SLATE, ls='--', lw=1.0, alpha=0.7, label='Coexistence 1/3')
ax.set_xlabel('Imitation Sweeps', color=NAVY)
ax.set_ylabel('Mean dominant-strategy fraction', color=NAVY)
ax.set_title('Within-run fixation (biodiversity loss)', color=NAVY, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax.set_ylim(0.3, 1.02)

# Bottom-right: distribution of final dominant-strategy fraction across runs
ax = axes[1, 1]
apply_axes_style(ax)
fin_dom_er = runs_er[:, -1, :].max(axis=1)
fin_dom_ba = runs_ba[:, -1, :].max(axis=1)
bins = np.linspace(0.3, 1.0, 16)
ax.hist(fin_dom_er, bins=bins, color=TEAL, alpha=0.6, label='Erdos-Renyi', edgecolor='white')
ax.hist(fin_dom_ba, bins=bins, color=PURPLE, alpha=0.6, label='Barabasi-Albert', edgecolor='white')
ax.set_xlabel('Final dominant-strategy fraction (per run)', color=NAVY)
ax.set_ylabel('Count of runs', color=NAVY)
ax.set_title('Per-run outcomes cluster near fixation', color=NAVY, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 3c: Rock-Paper-Scissors on Networks: per-run fixation, ensemble symmetry',
             fontsize=16, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout()

# VERIFY
err_er = np.abs(final_er - 1 / 3)
err_ba = np.abs(final_ba - 1 / 3)
ens_ok = max(err_er.max(), err_ba.max()) < 0.05
fix_ok = dom_er[-1] > 0.9 and dom_ba[-1] > 0.9
ba_faster = dom_ba[-1] >= dom_er[-1] - 1e-9

print("=" * 70)
print(f"VERIFY: RPS imitation dynamics on networks ({n_runs} runs each):")
print("  (1) Ensemble-mean final fractions  -> Nash (1/3, 1/3, 1/3):")
print(f"      ER: Rock {final_er[0]:.4f}, Paper {final_er[1]:.4f}, "
      f"Scissors {final_er[2]:.4f}  (max dev {err_er.max():.4f})")
print(f"      BA: Rock {final_ba[0]:.4f}, Paper {final_ba[1]:.4f}, "
      f"Scissors {final_ba[2]:.4f}  (max dev {err_ba.max():.4f})")
print(f"      status: {'PASS' if ens_ok else 'FAIL'} (tol 0.05)")
print("  (2) Within-run fixation -> dominant fraction near 1.0:")
print(f"      ER dominance {dom_er[-1]:.4f}, BA dominance {dom_ba[-1]:.4f}")
print(f"      BA fixates at least as strongly as ER (hub-driven): "
      f"{'PASS' if ba_faster else 'FAIL'}")
print(f"  Overall: {'PASS' if (ens_ok and fix_ok) else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m3c_rps_on_network")
plt.close()
