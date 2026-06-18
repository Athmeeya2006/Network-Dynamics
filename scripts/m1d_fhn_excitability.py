"""
m1d_fhn_excitability.py
=======================
Module 1d - FitzHugh-Nagumo: excitable vs oscillatory regimes.

Proof verified:
    The FHN model transitions from excitable to oscillatory via a Hopf
    bifurcation. The Hopf boundary in (a, I) is located by checking
    eigenvalues of the Jacobian at the fixed point.

Output: media/figures/m1d_fhn_excitability.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, LIGHT, PURPLE, GREEN,
                     setup_light_theme, apply_axes_style, save_figure)
from src.oscillators import FitzHughNagumo
from src.integrators import rk4

np.random.seed(42)
setup_light_theme()

params_exc = dict(a=0.7, b=0.8, tau=12.5, I=0.3)
params_osc = dict(a=0.7, b=0.8, tau=12.5, I=0.5)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor("#F8FAFC")

for col, (name, params, color) in enumerate([
    ('Excitable', params_exc, TEAL),
    ('Oscillatory', params_osc, PURPLE),
]):
    fhn = FitzHughNagumo(**params)
    v_range = np.linspace(-2.5, 2.5, 500)

    ax = axes[0, col]
    apply_axes_style(ax)
    ax.plot(v_range, fhn.nullcline_v(v_range), '-', color=RED, lw=2,
            label=r'$v$-nullcline')
    ax.plot(v_range, fhn.nullcline_w(v_range), '-', color=TEAL, lw=2,
            label=r'$w$-nullcline')

    def fp_eq(vw, o=fhn):
        return [o.rhs(vw, 0)[0], o.rhs(vw, 0)[1]]
    vw_fp = fsolve(fp_eq, [0, 0])
    ax.scatter([vw_fp[0]], [vw_fp[1]], color=GOLD, s=120, zorder=10,
               edgecolors='white', linewidths=2, marker='*')

    for x0 in [np.array([2.0, 0.0]), np.array([-1.5, -0.5])]:
        t, traj = rk4(lambda x, t, o=fhn: o.rhs(x, t), x0, (0, 100), 0.01)
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, lw=1, alpha=0.7)
        ax.scatter([x0[0]], [x0[1]], color=color, s=20, zorder=5)

    ax.set_xlabel(r'$v$', fontsize=11, color=NAVY)
    ax.set_ylabel(r'$w$', fontsize=11, color=NAVY)
    ax.set_title(f'{name} (I={params["I"]})', fontsize=12, color=NAVY,
                 fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 2.5)

    ax2 = axes[1, col]
    apply_axes_style(ax2)
    x0 = np.array([2.0, 0.0])
    t, traj = rk4(lambda x, t, o=fhn: o.rhs(x, t), x0, (0, 200), 0.01)
    ax2.plot(t, traj[:, 0], '-', color=color, lw=1.5, label=r'$v(t)$')
    ax2.plot(t, traj[:, 1], '-', color=GOLD, lw=1, alpha=0.7,
             label=r'$w(t)$')
    ax2.set_xlabel('Time', fontsize=11, color=NAVY)
    ax2.set_ylabel('State', fontsize=11, color=NAVY)
    ax2.set_title(f'{name} Time Series', fontsize=12, color=NAVY,
                  fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.95, facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 1d - FitzHugh-Nagumo Excitability',
             fontsize=15, color=NAVY, fontweight='bold', y=1.01)
plt.tight_layout()

print("=" * 65)
print("VERIFY - FitzHugh-Nagumo Hopf boundary:")
for I_val, expected in [(0.3, 'excitable'), (0.5, 'oscillatory')]:
    fhn_v = FitzHughNagumo(a=0.7, b=0.8, tau=12.5, I=I_val)

    def fp_v(vw, o=fhn_v):
        return [o.rhs(vw, 0)[0], o.rhs(vw, 0)[1]]
    vw_fp = fsolve(fp_v, [0, 0])
    J = fhn_v.jacobian(vw_fp)
    eigs = np.linalg.eigvals(J)
    max_real = np.max(np.real(eigs))
    regime = 'oscillatory' if max_real > 0 else 'excitable'
    match = regime == expected
    print(f"  I={I_val}: max(Re(lam))={max_real:+.4f} -> {regime:12s} "
          f"| expected: {expected:12s} | {'MATCH' if match else 'MISMATCH'}")
print("=" * 65)

save_figure(fig, "m1d_fhn_excitability")
plt.close()
