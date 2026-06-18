"""
m4d_powerlaw_fitting.py
=======================
Module 4d - Clauset-Shalizi-Newman power-law fitting vs naive OLS.

Proof verified:
    The maximum-likelihood estimator (with KS-selected x_min) of Clauset,
    Shalizi & Newman (2009) recovers the generating exponent of a synthetic
    discrete power law, whereas the naive ordinary-least-squares fit to the
    log-log CCDF is systematically biased. The semiparametric bootstrap
    p-value confirms the power law is a plausible fit (p > 0.1).

    VERIFY:
      (1) |alpha_MLE - alpha_true| is small (within ~0.05 for n = 10000);
      (2) the OLS estimate deviates from alpha_true by more than the MLE does
          (OLS bias exposed);
      (3) bootstrap goodness-of-fit p > 0.1.

    This is the methodological upgrade over the ER repo's OLS-only fitting.

Output: media/figures/m4d_powerlaw_fitting.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.powerlaw_fit import (fit_powerlaw, sample_discrete_powerlaw,
                              ols_exponent, bootstrap_pvalue)

np.random.seed(42)
setup_light_theme()

# ── Synthetic discrete power law with KNOWN exponent ─────────────────────────
ALPHA_TRUE = 2.5
XMIN_TRUE = 4
N = 10000
rng = np.random.default_rng(42)
print(f"Sampling {N} points from a discrete power law "
      f"(alpha={ALPHA_TRUE}, xmin={XMIN_TRUE})...")
data = sample_discrete_powerlaw(ALPHA_TRUE, XMIN_TRUE, N, rng)

# ── Clauset MLE fit ───────────────────────────────────────────────────────────
fit = fit_powerlaw(data, xmin_max=20)
alpha_mle, xmin_hat = fit['alpha'], fit['xmin']
print(f"Clauset MLE: alpha={alpha_mle:.3f}, xmin={xmin_hat}, D={fit['D']:.4f}")

# ── Naive OLS on the log-log CCDF (above the fitted xmin) ─────────────────────
alpha_ols, intercept, slope = ols_exponent(data, xmin=xmin_hat)
print(f"Naive OLS:   alpha={alpha_ols:.3f}")

# ── Bootstrap goodness-of-fit p-value ────────────────────────────────────────
print("Running semiparametric bootstrap (this takes a moment)...")
pval = bootstrap_pvalue(data, fit, n_boot=200, seed=7, xmin_max=20)
print(f"Bootstrap p-value: {pval:.3f}")


# ── Empirical CCDF and fitted curves ─────────────────────────────────────────
def ccdf(d):
    d = np.asarray(d)
    xs = np.sort(np.unique(d))
    n = len(d)
    return xs, 1.0 - np.searchsorted(np.sort(d), xs, side='left') / n


xs, c = ccdf(data)
tail = xs >= xmin_hat
c_at_xmin = c[np.searchsorted(xs, xmin_hat)]

# MLE model CCDF (normalised to pass through the empirical CCDF at xmin)
x_model = np.arange(xmin_hat, xs.max() + 1)
ccdf_mle = zeta(alpha_mle, x_model) / zeta(alpha_mle, xmin_hat) * c_at_xmin

# OLS line in CCDF space
x_ols = xs[tail]
ccdf_ols = np.exp(intercept) * x_ols ** slope

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
fig.patch.set_facecolor("#F8FAFC")

apply_axes_style(ax1)
ax1.loglog(xs, c, 'o', color=SLATE, ms=5, alpha=0.55, label='empirical CCDF')
ax1.axvline(xmin_hat, color=GOLD, ls=':', lw=1.6, alpha=0.9,
            label=rf'$x_{{\min}}={xmin_hat}$')
ax1.loglog(x_model, ccdf_mle, '-', color=TEAL, lw=2.4,
           label=rf'Clauset MLE $\hat\alpha={alpha_mle:.2f}$')
ax1.loglog(x_ols, ccdf_ols, '--', color=RED, lw=2.2,
           label=rf'naive OLS $\hat\alpha={alpha_ols:.2f}$')
ax1.set_xlabel('$x$', color=NAVY)
ax1.set_ylabel(r'$P(X \geq x)$', color=NAVY)
ax1.set_title('CCDF fit: MLE vs OLS', color=NAVY, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)

# Right: estimator comparison bar
apply_axes_style(ax2)
err_mle = abs(alpha_mle - ALPHA_TRUE)
err_ols = abs(alpha_ols - ALPHA_TRUE)
bars = ax2.bar(['true', 'Clauset MLE', 'naive OLS'],
               [ALPHA_TRUE, alpha_mle, alpha_ols],
               color=[NAVY, TEAL, RED], alpha=0.85, edgecolor='white', width=0.6)
ax2.axhline(ALPHA_TRUE, color=NAVY, ls='--', lw=1.2, alpha=0.7)
ax2.set_ylabel(r'exponent $\alpha$', color=NAVY)
ax2.set_ylim(0, max(ALPHA_TRUE, alpha_mle, alpha_ols) * 1.25)
ax2.set_title(f'Estimator accuracy (bootstrap $p={pval:.2f}$)',
              color=NAVY, fontweight='bold')
for b, v, e in zip(bars, [ALPHA_TRUE, alpha_mle, alpha_ols], [0, err_mle, err_ols]):
    lbl = f'{v:.3f}' + (f'\nerr {e:.3f}' if e > 0 else '')
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.05, lbl, ha='center',
             va='bottom', fontsize=10, color=NAVY)

fig.suptitle('Module 4d - Clauset MLE recovers the exponent; OLS is biased',
             fontsize=16, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout()

# ── VERIFY ────────────────────────────────────────────────────────────────────
ok_mle = err_mle < 0.05
ok_bias = err_ols > err_mle
ok_p = pval > 0.1
print("=" * 70)
print("VERIFY - Clauset MLE vs OLS on synthetic power law:")
print(f"  true alpha          = {ALPHA_TRUE:.3f}")
print(f"  MLE  alpha          = {alpha_mle:.3f}  (err {err_mle:.3f}, <0.05) "
      f"{'PASS' if ok_mle else 'FAIL'}")
print(f"  OLS  alpha          = {alpha_ols:.3f}  (err {err_ols:.3f})")
print(f"  OLS more biased than MLE: {'PASS' if ok_bias else 'FAIL'}")
print(f"  bootstrap p-value   = {pval:.3f}  (>0.1) {'PASS' if ok_p else 'FAIL'}")
print(f"  Overall: {'PASS' if (ok_mle and ok_bias and ok_p) else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m4d_powerlaw_fitting")
plt.close()
