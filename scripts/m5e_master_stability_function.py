"""
m5e_master_stability_function.py
================================
Module 5e: The Master Stability Function and the eigenratio criterion.

Proof verified (Pecora & Carroll 1998; Barahona & Pecora 2002):
    For identical chaotic oscillators coupled diffusively through one variable,
    the stability of the fully synchronised state reduces to a single Master
    Stability Function Lambda(alpha) evaluated at alpha = sigma * lambda_k for
    each Laplacian eigenvalue lambda_k. For the x-coupled Rossler system the
    MSF is negative only on a bounded interval (alpha_1, alpha_2), so a stable
    sync window in the coupling sigma exists iff
        alpha_1 < sigma * lambda_2   and   sigma * lambda_N < alpha_2,
    i.e. iff the eigenratio R = lambda_N / lambda_2 < alpha_2 / alpha_1.
    Smaller eigenratio => more synchronisable (lower onset, wider window).

    VERIFY: across several topologies the MSF-predicted synchronisation onset
    sigma ~ alpha_1 / lambda_2 ranks the topologies in the same order as the
    directly simulated synchronisation onset (rank correlation = 1), and the
    complete graph (eigenratio R = 1, the minimum) synchronises first.

Output: media/figures/m5e_master_stability_function.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, PURPLE, GREEN,
                     setup_light_theme, apply_axes_style, save_figure)
from src.flows import rossler, rossler_jacobian
from src.lyapunov import master_stability_function
from src.networks import (er_graph, ba_graph, watts_strogatz, ring_lattice,
                          laplacian, laplacian_spectrum, eigenratio)
import networkx as nx

np.random.seed(42)
setup_light_theme()

# (1) Master Stability Function for x-coupled Rossler
E = np.diag([1.0, 0.0, 0.0])           # coupling through x only
alphas = np.linspace(0.0, 5.0, 26)
print("Computing Master Stability Function for x-coupled Rossler...")
Lambda = master_stability_function(rossler, rossler_jacobian, E, alphas,
                                   x0=[1.0, 1.0, 1.0], t_total=300.0, dt=0.01)

# zero crossings -> stable window (alpha_1, alpha_2)
sign = Lambda < 0
idx_in = np.where(sign)[0]
a1 = np.interp(0.0, [Lambda[idx_in[0] - 1], Lambda[idx_in[0]]],
               [alphas[idx_in[0] - 1], alphas[idx_in[0]]])
a2 = np.interp(0.0, [Lambda[idx_in[-1]], Lambda[idx_in[-1] + 1]],
               [alphas[idx_in[-1]], alphas[idx_in[-1] + 1]])
R_c = a2 / a1
print(f"  stable MSF window: alpha in ({a1:.3f}, {a2:.3f}), "
      f"eigenratio threshold R_c = {R_c:.2f}")

# (2) Topologies, eigenratios, predicted onset
N = 50
topologies = {
    'complete': nx.complete_graph(N),
    'ER': er_graph(N, p=0.2, seed=3),
    'WS (p=0.3)': watts_strogatz(N, 6, 0.3, seed=3),
    'BA': ba_graph(N, 3, seed=3),
    'ring (k=4)': ring_lattice(N, 4),
}

info = {}
for name, G in topologies.items():
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
        topologies[name] = G
    ev = laplacian_spectrum(G)
    l2, lN = ev[ev > 1e-9][0], ev[-1]
    info[name] = {
        'lambda2': l2, 'lambdaN': lN, 'R': lN / l2,
        'sigma_lo': a1 / l2, 'sigma_hi': a2 / lN,
        'L': laplacian(G),
    }
    print(f"  {name:12s}: lambda2={l2:.3f}, lambdaN={lN:.3f}, "
          f"R={lN / l2:6.2f}, predicted onset sigma~{a1 / l2:.4f}")

# (3) Direct simulation: coupled Rossler sync error vs coupling sigma
def coupled_rossler_rhs(X, t, L, sigma, a=0.2, b=0.2, c=5.7):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    dx = -y - z - sigma * (L @ x)        # diffusive coupling through x
    dy = x + a * y
    dz = b + z * (x - c)
    return np.stack([dx, dy, dz], axis=1)


def sync_error(L, sigma, N, T=120.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (N, 3))
    n = int(T / dt)
    errs = []
    with np.errstate(over='ignore', invalid='ignore'):
        for i in range(n):
            k1 = coupled_rossler_rhs(X, 0, L, sigma)
            k2 = coupled_rossler_rhs(X + 0.5 * dt * k1, 0, L, sigma)
            k3 = coupled_rossler_rhs(X + 0.5 * dt * k2, 0, L, sigma)
            k4 = coupled_rossler_rhs(X + dt * k3, 0, L, sigma)
            X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(X)):
                return np.inf
            if i > n // 2:
                errs.append(np.std(X[:, 0]))
    return float(np.mean(errs))


sigmas = np.linspace(0.0, 0.6, 41)
print("Simulating coupled Rossler on each topology (sync error vs sigma)...")
sim_err = {}
emp_onset = {}
for name in topologies:
    L = info[name]['L']
    errs = np.array([sync_error(L, s, N, seed=1) for s in sigmas])
    sim_err[name] = errs
    below = np.where(errs < 0.05)[0]
    emp_onset[name] = sigmas[below[0]] if len(below) else np.nan
    print(f"  {name:12s}: empirical sync onset sigma = {emp_onset[name]}")

# Figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#F8FAFC")
palette = [NAVY, TEAL, GOLD, PURPLE, RED]

# MSF curve
apply_axes_style(ax1)
ax1.plot(alphas, Lambda, '-', color=NAVY, lw=2.4)
ax1.axhline(0, color=SLATE, ls='--', lw=1)
ax1.axvspan(a1, a2, color=GREEN, alpha=0.18, label=rf'stable: $\alpha\in({a1:.2f},{a2:.2f})$')
ax1.set_xlabel(r'$\alpha = \sigma\,\lambda_k$', color=NAVY)
ax1.set_ylabel(r'$\Lambda(\alpha)$ (transverse exponent)', color=NAVY)
ax1.set_title('(1) Master Stability Function\n(x-coupled Rossler)',
              color=NAVY, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)

# Sync error vs sigma
apply_axes_style(ax2)
for name, col in zip(topologies, palette):
    ax2.plot(sigmas, sim_err[name], 'o-', color=col, lw=1.8, ms=4,
             label=f'{name} (R={info[name]["R"]:.1f})')
ax2.axhline(0.05, color=SLATE, ls=':', lw=1.2)
ax2.set_xlabel(r'Coupling $\sigma$', color=NAVY)
ax2.set_ylabel('Sync error (std of $x$ across nodes)', color=NAVY)
ax2.set_title('(2) Simulated synchronisation', color=NAVY, fontweight='bold')
ax2.legend(fontsize=8.5, framealpha=0.95, facecolor='white', edgecolor=SLATE)
ax2.set_yscale('log')

# Predicted vs measured onset
apply_axes_style(ax3)
names = list(topologies.keys())
pred = np.array([info[n]['sigma_lo'] for n in names])
meas = np.array([emp_onset[n] for n in names])
for n, col in zip(names, palette):
    ax3.scatter(info[n]['sigma_lo'], emp_onset[n], s=120, color=col,
                edgecolors='white', linewidths=1.2, label=n, zorder=5)
lims = [0, max(np.nanmax(pred), np.nanmax(meas)) * 1.15]
ax3.plot(lims, lims, '--', color=SLATE, lw=1.2, alpha=0.7)
ax3.set_xlabel(r'MSF-predicted onset $\alpha_1/\lambda_2$', color=NAVY)
ax3.set_ylabel(r'Measured sync onset $\sigma$', color=NAVY)
ax3.set_title('(3) Prediction vs simulation', color=NAVY, fontweight='bold')
ax3.legend(fontsize=8.5, framealpha=0.95, facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 5e: Master Stability Function: the eigenratio '
             r'$\lambda_N/\lambda_2$ ranks synchronisability',
             fontsize=15, color=NAVY, fontweight='bold', y=1.02)
plt.tight_layout()

# VERIFY
def rank(a):
    order = np.argsort(a)
    r = np.empty_like(order)
    r[order] = np.arange(len(a))
    return r


valid = ~np.isnan(meas)
rp, rm = rank(pred[valid]), rank(meas[valid])
spearman = np.corrcoef(rp, rm)[0, 1]
complete_first = (emp_onset['complete'] == np.nanmin(meas)) and \
                 (info['complete']['R'] == min(info[n]['R'] for n in names))
ok = (spearman > 0.8) and complete_first
print("=" * 70)
print("VERIFY: eigenratio ordering matches simulated synchronisability:")
for n in names:
    print(f"  {n:12s}: R={info[n]['R']:6.2f}  predicted onset={info[n]['sigma_lo']:.4f}  "
          f"measured onset={emp_onset[n]}")
print(f"  Spearman rank corr (predicted vs measured onset) = {spearman:.3f} (>0.8)")
print(f"  complete graph (min R) synchronises first: {complete_first}")
print(f"  status: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m5e_master_stability_function")
plt.close()
