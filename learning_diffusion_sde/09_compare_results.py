"""
09_compare_results.py
=====================
PURPOSE
-------
Generate a side-by-side comparison of Ground Truth and Our Method,
matching the layout of Figure 2 (panels 1 and 2) from the paper.

Later, when scripts 10 and 11 are run, this script is superseded by
12_reproduce_figure2.py which adds the other two panels.

OUTPUT
------
plots/figure2_partial.png   — Ground Truth | Our Method (side by side)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import config
import utils

config.ensure_dirs()

print("=" * 60)
print("STEP 09 — Compare Ground Truth vs Our Method")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
gt  = np.load(os.path.join(config.DATA_DIR, "ground_truth.npz"))
our = np.load(os.path.join(config.DATA_DIR, "our_method.npz"))

surv_gt  = gt["survivors"]
surv_our = our["survivors"]

# Analytical solution
def analytical_pdf_cond(x_vals, x0=config.X0, L=config.DOMAIN_HI, t=config.T, N=100):
    ns  = np.arange(1, N + 1)
    dec = np.exp(-ns**2 * np.pi**2 * t / (2 * L**2))
    phi = (2.0 / L) * np.sum(
        np.sin(ns[None, :] * np.pi * x0 / L) *
        np.sin(np.outer(x_vals, ns * np.pi / L)) *
        dec[None, :], axis=1
    )
    phi = np.maximum(phi, 0.0)
    dx  = x_vals[1] - x_vals[0]
    return phi / (phi.sum() * dx)   # normalise to conditional

x_th = np.linspace(0.01, 5.99, 400)
p_th = analytical_pdf_cond(x_th)

# KS test
ks_stat, ks_p = ks_2samp(surv_gt, surv_our)
print(f"  GT survivors   : {len(surv_gt):,}  ({100*len(surv_gt)/config.N_PARTICLES:.2f}%)")
print(f"  Our survivors  : {len(surv_our):,}  ({100*len(surv_our)/config.N_PARTICLES:.2f}%)")
print(f"  KS stat        : {ks_stat:.4f}   p={ks_p:.4f}")
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = utils.make_fig(nrows=1, ncols=2, figsize=(14, 5))

n_bins = 80
xlim   = (config.DOMAIN_LO, config.DOMAIN_HI)

def panel(ax, survivors, title, color, label):
    ax.hist(survivors, bins=n_bins, range=xlim, density=True,
            color=color, alpha=0.70, label=label)
    ax.plot(x_th, p_th, "r-", lw=2.0, label="Analytical")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Position x at T=3", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95,
            f"n={len(survivors):,}\nsurvival={100*len(survivors)/config.N_PARTICLES:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(fc="white", alpha=0.8))

panel(axes[0], surv_gt,  "Ground Truth",  "#607D8B", f"GT  (n={len(surv_gt):,})")
panel(axes[1], surv_our, "Our Method",    "#1565C0", f"Our (n={len(surv_our):,})")

fig.suptitle(
    "Figure 2 (partial): Ground Truth vs Our Method\n"
    f"KS stat={ks_stat:.4f}  p={ks_p:.4f}  "
    f"({'indistinguishable' if ks_p > 0.05 else 'statistically different'})",
    fontsize=12, y=1.02,
)
fig.tight_layout()
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "figure2_partial.png"))

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
print("Distribution statistics:")
print(f"  GT  : mean={surv_gt.mean():.4f}  std={surv_gt.std():.4f}  "
      f"median={np.median(surv_gt):.4f}")
print(f"  Our : mean={surv_our.mean():.4f}  std={surv_our.std():.4f}  "
      f"median={np.median(surv_our):.4f}")
print()
print("Percentile comparison:")
for pct in [10, 25, 50, 75, 90]:
    g = np.percentile(surv_gt, pct)
    o = np.percentile(surv_our, pct)
    print(f"  {pct:3d}th: GT={g:.4f}  Our={o:.4f}  diff={o-g:+.4f}")
print()
print("FILES WRITTEN")
print("  plots/figure2_partial.png")
print()
print("INTERPRETATION")
print("  If KS p > 0.05: Our Method is statistically indistinguishable from GT.")
print("  If p is small : the distributions differ — check training data balance.")
print()
print("  The current limitation: training data is imbalanced —")
print(f"  only 0.7% of pairs near x∈[4.5,6.0] (right wall).")
print("  The network under-learns the right boundary effect.")
print("  Fix: use importance sampling or symmetric data augmentation.")
print()
print("Done.  Run 10_all_trajectories_trained.py next.")
