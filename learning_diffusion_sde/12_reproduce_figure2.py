"""
12_reproduce_figure2.py
=======================
PURPOSE
-------
Reproduce the EXACT 4-panel layout of Figure 2 from the paper:

    Ground Truth | Our Method | All Trajectories | Only Confined

Loads all four survivor arrays from data/ and plots them with the
analytical Fourier series overlaid on each panel.

WHAT EACH PANEL SHOWS
----------------------
1. GROUND TRUTH
   Empirical simulation of the SDE with N=200k particles.
   This is the reference distribution all other methods aim to reproduce.
   It matches the analytical Fourier series (red curve) extremely well.

2. OUR METHOD
   Neural network G(x,z) trained on CLEAN pairs (both endpoints inside).
   Captures the boundary-correcting bias from the training data.
   Should closely match the ground truth.

3. ALL TRAJECTORIES TRAINED
   Same network trained on ALL pairs (including boundary crossings).
   Loses the boundary correction → particles don't respect the walls properly.
   Distribution differs from ground truth (systematic bias).

4. ONLY CONFINED TRAINED
   Same network trained ONLY on particles that survived to T=3.
   Selection bias: confined particles sample a different step distribution.
   Distribution is too narrow / concentrated compared to ground truth.

OUTPUT
------
plots/figure2_final.png   — the 4-panel figure
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
print("STEP 12 — Reproduce Figure 2")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# Load all four survivor arrays
# ---------------------------------------------------------------------------
def load_survivors(filename, label):
    path = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {filename} not found — run the corresponding script first.")
        return None
    arr = np.load(path)["survivors"]
    print(f"  {label:30s}: {len(arr):,} survivors "
          f"({100*len(arr)/config.N_PARTICLES:.2f}%)")
    return arr

surv_gt   = load_survivors("ground_truth.npz",   "Ground Truth")
surv_our  = load_survivors("our_method.npz",      "Our Method")
surv_all  = load_survivors("all_trajectories.npz","All Trajectories")
surv_conf = load_survivors("only_confined.npz",   "Only Confined")
print()

# ---------------------------------------------------------------------------
# Analytical PDF
# ---------------------------------------------------------------------------
def analytical_pdf_cond(x_v, x0=config.X0, L=config.DOMAIN_HI, t=config.T, N=100):
    ns  = np.arange(1, N + 1)
    dec = np.exp(-ns**2 * np.pi**2 * t / (2 * L**2))
    p   = (2.0 / L) * np.sum(
        np.sin(ns[None, :] * np.pi * x0 / L) *
        np.sin(np.outer(x_v, ns * np.pi / L)) * dec[None, :],
        axis=1,
    )
    p = np.maximum(p, 0.0)
    return p / (p.sum() * (x_v[1] - x_v[0]))

x_th = np.linspace(0.01, 5.99, 400)
p_th = analytical_pdf_cond(x_th)

# ---------------------------------------------------------------------------
# 4-panel figure
# ---------------------------------------------------------------------------
panels = [
    (surv_gt,   "Ground Truth",           "#546E7A", True),
    (surv_our,  "Our Method",             "#1565C0", True),
    (surv_all,  "All Trajectories\nTrained","#E65100", True),
    (surv_conf, "Only Confined\nTrained", "#6A1B9A", True),
]

fig, axes = utils.make_fig(nrows=1, ncols=4, figsize=(20, 5))

n_bins = 80
xlim   = (config.DOMAIN_LO, config.DOMAIN_HI)

for ax, (survivors, title, color, show_analytical) in zip(axes, panels):
    if survivors is None:
        ax.text(0.5, 0.5, "Data not\navailable\nRun script first",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="red")
        ax.set_title(title, fontsize=11)
        continue

    ax.hist(survivors, bins=n_bins, range=xlim, density=True,
            color=color, alpha=0.75)

    if show_analytical:
        ax.plot(x_th, p_th, "r-", lw=2.0, label="Analytical")

    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Position x at T = 3", fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # KS test vs ground truth (skip for ground truth itself)
    if survivors is not surv_gt and surv_gt is not None:
        ks_stat, ks_p = ks_2samp(surv_gt, survivors)
        ax.text(0.97, 0.97,
                f"KS={ks_stat:.3f}\np={ks_p:.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", alpha=0.85, boxstyle="round"))

    ax.text(0.03, 0.97,
            f"n={len(survivors):,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(fc="white", alpha=0.7))

fig.suptitle(
    "Figure 2: Comparison of training data choices for stochastic flow map learning\n"
    r"$dX_t = dW_t$,  $X_0 = 1$,  absorbing boundaries at 0 and 6,  $T = 3$",
    fontsize=13, y=1.02,
)
fig.tight_layout()
out = os.path.join(config.PLOTS_DIR, "figure2_final.png")
utils.save_fig(fig, out)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("FIGURE 2 SUMMARY TABLE")
print("=" * 60)
print()
print(f"  {'Panel':<30} {'Survivors':>10} {'Surv%':>7} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*66}")
for survivors, title, _, _ in panels:
    if survivors is None:
        print(f"  {title:<30} {'N/A':>10}")
        continue
    title_clean = title.replace("\n", " ")
    print(f"  {title_clean:<30} {len(survivors):>10,} "
          f"{100*len(survivors)/config.N_PARTICLES:>6.2f}% "
          f"{survivors.mean():>8.4f} {survivors.std():>8.4f}")

print()
print("KS statistics vs Ground Truth:")
for survivors, title, _, _ in panels:
    if survivors is None or survivors is surv_gt:
        continue
    title_clean = title.replace("\n"," ")
    ks_s, ks_p = ks_2samp(surv_gt, survivors)
    match = "GOOD MATCH" if ks_p > 0.05 else "DIFFERENT"
    print(f"  {title_clean:<35} KS={ks_s:.4f}  p={ks_p:.2e}  → {match}")
print()
print("FILES WRITTEN")
print("  plots/figure2_final.png")
print()
print("INTERPRETATION")
print()
print("  Ground Truth   : reference distribution (Fourier series matches).")
print("  Our Method     : should closely match GT if training data is balanced.")
print("                   Current limitation: only 0.7% data near right wall.")
print("  All Trajectories: boundary effect lost → different distribution.")
print("  Only Confined  : selection bias → distribution too narrow/central.")
print()
print("  The paper shows that 'Our Method' with correct data cleaning")
print("  produces the BEST match to the ground truth.")
print()
print("Done. All Figure 2 panels generated.")
