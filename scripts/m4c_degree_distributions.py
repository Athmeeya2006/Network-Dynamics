"""
m4c_degree_distributions.py
===========================
Module 4c - Degree distributions of canonical network models.

Proof verified:
    Erdos-Renyi graphs have a Poisson degree distribution (light tail);
    Barabasi-Albert preferential-attachment graphs have a scale-free degree
    distribution p(k) ~ k^-gamma with gamma = 3; the configuration model
    realises a prescribed degree sequence (here a power-law sequence).

    VERIFY (BA exponent): a Clauset MLE fit of the BA degree sequence recovers
    gamma ~ 3 (finite-size estimates fall slightly below 3).

Output: media/figures/m4c_degree_distributions.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.networks import er_graph, ba_graph, config_model, degree_sequence, mean_degree
from src.powerlaw_fit import fit_powerlaw, sample_discrete_powerlaw

np.random.seed(42)
setup_light_theme()

# ── Build large graphs for clean statistics ──────────────────────────────────
N = 20000
m = 3
print("Generating ER, BA, and configuration-model graphs...")
G_er = er_graph(N, p=2 * m / (N - 1), seed=42)
G_ba = ba_graph(N, m=m, seed=42)

# configuration model with a power-law degree sequence (gamma_in = 2.5)
rng = np.random.default_rng(42)
deg_pl = sample_discrete_powerlaw(2.5, 2, N, rng)
if deg_pl.sum() % 2 == 1:          # configuration model needs even degree sum
    deg_pl[0] += 1
G_cfg = config_model(deg_pl, seed=42)

k_er = degree_sequence(G_er)
k_ba = degree_sequence(G_ba)
k_cfg = degree_sequence(G_cfg)
print(f"<k>: ER={mean_degree(G_er):.2f}, BA={mean_degree(G_ba):.2f}, cfg={mean_degree(G_cfg):.2f}")


def ccdf(data):
    data = np.asarray(data)
    data = data[data > 0]
    xs = np.sort(np.unique(data))
    n = len(data)
    c = 1.0 - np.searchsorted(np.sort(data), xs, side='left') / n
    return xs, c


# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
fig.patch.set_facecolor("#F8FAFC")

# Left: PMF on log-log
apply_axes_style(ax1)
for k, color, label in [(k_er, TEAL, 'ER (Poisson)'),
                        (k_ba, RED, 'BA (scale-free)'),
                        (k_cfg, GOLD, 'config model')]:
    vals, counts = np.unique(k[k > 0], return_counts=True)
    ax1.loglog(vals, counts / counts.sum(), 'o', color=color, ms=5,
               alpha=0.7, label=label)
# guide line k^-3
kk = np.array([3, 200])
ax1.loglog(kk, 0.5 * kk ** (-3.0), '--', color=NAVY, lw=1.5, alpha=0.8,
           label=r'slope $-3$ guide')
ax1.set_xlabel('Degree $k$', color=NAVY)
ax1.set_ylabel('$p(k)$', color=NAVY)
ax1.set_title('Degree distributions (PMF, log-log)', color=NAVY, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)

# Right: CCDF
apply_axes_style(ax2)
for k, color, label in [(k_er, TEAL, 'ER (Poisson)'),
                        (k_ba, RED, 'BA (scale-free)'),
                        (k_cfg, GOLD, 'config model')]:
    xs, c = ccdf(k)
    ax2.loglog(xs, c, '-', color=color, lw=2.0, label=label)
ax2.set_xlabel('Degree $k$', color=NAVY)
ax2.set_ylabel(r'$P(K \geq k)$', color=NAVY)
ax2.set_title('Complementary CDF', color=NAVY, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 4c — Degree distributions: Poisson vs scale-free vs configuration model',
             fontsize=16, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout()

# ── VERIFY: BA exponent ~ 3 via Clauset MLE ──────────────────────────────────
fit_ba = fit_powerlaw(k_ba, xmin_max=40)
gamma = fit_ba['alpha']
ok = abs(gamma - 3.0) < 0.3
print("=" * 70)
print("VERIFY — BA degree distribution exponent (Clauset MLE):")
print(f"  gamma_MLE = {gamma:.3f}  (xmin = {fit_ba['xmin']}, n_tail = {fit_ba['n_tail']})")
print(f"  theory: gamma = 3.0   |gamma - 3| = {abs(gamma - 3.0):.3f} (< 0.3)")
print(f"  status: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m4c_degree_distributions")
plt.close()
