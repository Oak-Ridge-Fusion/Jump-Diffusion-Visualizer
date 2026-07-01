"""
06_generate_labels.py
=====================
PURPOSE
-------
Generate the TRAINING LABELS for the neural network G(x, z) → ΔX.

This is the bridge between the diffusion model (scripts 03-05) and the
neural network (script 07).  It answers the question:

    "Given a noise vector z ~ N(0,1) and a starting position x,
     what increment ΔX should the particle take?"

WHY THE DIFFUSION MODEL IS NOT THE FINAL MODEL
-----------------------------------------------
The ODE+KNN approach from script 05 can generate samples but is:

  1. SLOW: ~1 sample/second (30 ODE steps × KDTree query each step)
  2. MEMORY-HEAVY: needs the full reference dataset in RAM
  3. NON-DIFFERENTIABLE: can't be embedded in a larger ML pipeline
  4. APPROXIMATE: KNN score estimate has variance, especially in tails

Solution: use the ODE OFFLINE to generate a dataset of (x, z) → ΔX labels,
then train a CHEAP neural network G that imitates this mapping.

At rollout time:
   z ~ N(0,1)
   ΔX = G(x_t, z)          ← fast neural network forward pass
   x_new = x_t + ΔX

HOW WE GENERATE THE LABELS (LOCAL REPARAMETERISATION)
------------------------------------------------------
Running the full ODE for every training pair is too expensive.
Instead we use a mathematically equivalent approach for nearly-Gaussian
conditionals:

  The conditional p(ΔX | x_t) is approximately Gaussian near the domain
  interior.  For a Gaussian with mean μ(x) and std σ(x):

      ΔX = μ(x) + σ(x) · z     ←→     z = (ΔX - μ(x)) / σ(x)

  This gives a bijection z ↔ ΔX that is EXACTLY what the diffusion ODE
  computes in the limit of infinitely many neighbours.

  We estimate μ(x) and σ(x) using K nearest neighbours in x-space.

NEAR BOUNDARIES, THE CONDITIONAL IS NOT GAUSSIAN
-------------------------------------------------
Near x = 0, the condition "must survive to x_new > 0" truncates the left
tail.  Similarly near x = 6.  The local KNN estimator captures this
automatically: when we find K neighbours of x ≈ 0.1, those neighbours'
ΔX values are all > -0.1 (because they survived).

The network G(x, z) learns this asymmetric mapping from the labels.

THE KEY SCIENTIFIC POINT
-------------------------
The quality of "Our Method" (Figure 2, panel 2) depends entirely on WHICH
PAIRS we include in the training set:

  "Our Method"         : only clean pairs (both endpoints inside domain)
  "All Trajectories"   : all pairs including those that crossed the boundary
  "Only Confined"      : only pairs from confined (long-surviving) particles

This label-generation script implements "Our Method".
Scripts 10 and 11 modify which pairs are included.

OUTPUT
------
data/labels.npz
    "x_t"  : (M,)  conditioning position
    "z"    : (M,)  standardised noise ≈ N(0,1)
    "dx"   : (M,)  increment label (what G should output given (x_t, z))
    "mu_x" : (M,)  local mean of ΔX | x_t  (diagnostic)
    "sig_x": (M,)  local std  of ΔX | x_t  (diagnostic)

plots/label_z_distribution.png
    Distribution of z labels — should be N(0,1).

plots/label_mu_sigma_vs_x.png
    How μ(x) and σ(x) change with position.

plots/label_dx_reconstruction.png
    Check: z * σ(x) + μ(x) should equal the original ΔX.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.stats import norm as scipy_norm

import config
import utils

# ---------------------------------------------------------------------------
# 0.  Setup
# ---------------------------------------------------------------------------
config.ensure_dirs()
rng = np.random.default_rng(config.SEED + 5)

print("=" * 60)
print("STEP 06 — Generate Training Labels")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1.  Load a manageable subset of the dataset
# ---------------------------------------------------------------------------
print("Loading dataset …")
data    = np.load(os.path.join(config.DATA_DIR, "dataset.npz"))
x_t_all = data["x_t"].astype(np.float64)
dx_all  = data["dx"].astype(np.float64)

# Use 500k pairs for label generation
N_LABEL = 500_000
idx     = rng.choice(len(dx_all), size=N_LABEL, replace=False)
x_t_sub = x_t_all[idx]
dx_sub  = dx_all[idx]

print(f"  Dataset size (for labelling): {N_LABEL:,} pairs")
utils.print_stats("x_t", x_t_sub)
utils.print_stats("dx",  dx_sub)
print()

# ---------------------------------------------------------------------------
# 2.  Build the FULL reference set for KNN mu/sigma estimation
# ---------------------------------------------------------------------------
#
# We keep a LARGER reference set for accurate mu/sigma estimation.
# This is separate from the N_LABEL set to avoid self-referencing.
#
N_REF  = min(1_000_000, len(dx_all))
idx_r  = rng.choice(len(dx_all), size=N_REF, replace=False)
x_ref  = x_t_all[idx_r]
dx_ref = dx_all[idx_r]

print(f"  Reference set for KNN: {N_REF:,} pairs")
print(f"  Building KDTree on x_ref …")
tree_x = cKDTree(x_ref.reshape(-1, 1))
print(f"  Done.")
print()

# ---------------------------------------------------------------------------
# 3.  Estimate local μ(x) and σ(x) for each query point
# ---------------------------------------------------------------------------
#
# For each x_t in our label set:
#   1. Find K nearest neighbours in x_ref (1-D KDTree query)
#   2. Compute μ_k = mean(dx of neighbours)
#   3. Compute σ_k = std(dx of neighbours)
#   4. Store (x_t, z = (dx - μ_k)/σ_k, dx)
#
# This "local standardisation" IS the output of the diffusion ODE for
# Gaussian conditionals.  Near boundaries it captures the truncation.
#
K_LOCAL = 200   # larger K = smoother μ/σ estimate; smaller = more local

print(f"Computing local μ(x), σ(x) with K={K_LOCAL} neighbours …")

mu_x  = np.zeros(N_LABEL, dtype=np.float64)
sig_x = np.zeros(N_LABEL, dtype=np.float64)

# Batch query for speed
BATCH = 50_000
n_batches = (N_LABEL + BATCH - 1) // BATCH

for b in tqdm(range(n_batches), desc="KNN batches", ncols=60):
    lo = b * BATCH
    hi = min((b + 1) * BATCH, N_LABEL)
    queries = x_t_sub[lo:hi].reshape(-1, 1)

    # Find K nearest neighbours in x-space
    _, idxs = tree_x.query(queries, k=K_LOCAL)   # shape (batch, K)

    # Gather dx values of neighbours
    dx_nbrs = dx_ref[idxs]    # shape (batch, K)

    mu_x[lo:hi]  = dx_nbrs.mean(axis=1)    # local conditional mean
    sig_x[lo:hi] = dx_nbrs.std(axis=1) + 1e-8   # local conditional std

print()
print("  Local statistics summary:")
utils.print_stats("μ(x)",  mu_x)
utils.print_stats("σ(x)",  sig_x)
print()

# ---------------------------------------------------------------------------
# 4.  Compute standardised z = (ΔX - μ(x)) / σ(x)
# ---------------------------------------------------------------------------
#
# This is the LATENT CODE that the neural network G will receive at
# rollout time.  It encodes "how unusual" the increment is relative to
# the local conditional distribution.
#
z_labels = (dx_sub - mu_x) / sig_x

print("  Standardised z labels:")
utils.print_stats("z", z_labels)
print()

# Check: z should be approximately N(0,1)
# If σ(x) is estimated accurately, (ΔX - μ)/σ ~ N(0,1).
z_mean = z_labels.mean()
z_std  = z_labels.std()
print(f"  Expected z: N(0,1)  →  mean={z_mean:.4f}  std={z_std:.4f}")
print(f"  Deviation from N(0,1): mean_err={abs(z_mean):.4f}  std_err={abs(z_std-1):.4f}")
print()

# ---------------------------------------------------------------------------
# 5.  Save labels
# ---------------------------------------------------------------------------
save_path = os.path.join(config.DATA_DIR, "labels.npz")
np.savez(
    save_path,
    x_t   = x_t_sub,
    z     = z_labels,
    dx    = dx_sub,
    mu_x  = mu_x,
    sig_x = sig_x,
)
print(f"Labels saved → {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)")
print()

# ---------------------------------------------------------------------------
# 6.  Plot: z distribution (should be N(0,1))
# ---------------------------------------------------------------------------
print("Generating label_z_distribution.png …")

# Subsample for plotting
n_plot = 200_000
idx_p  = rng.choice(N_LABEL, size=n_plot, replace=False)
z_plot = z_labels[idx_p]
x_plot_2 = x_t_sub[idx_p]

fig, axes = utils.make_fig(nrows=1, ncols=2, figsize=(14, 5))
ax_l, ax_r = axes

# Left: overall z distribution
ax_l.hist(z_plot, bins=100, density=True, color="#7B1FA2", alpha=0.7,
          range=(-4, 4), label=f"z = (ΔX - μ(x)) / σ(x)  (n={n_plot:,})")

z_range = np.linspace(-4, 4, 300)
ax_l.plot(z_range, scipy_norm.pdf(z_range), "r-", lw=2.5,
          label="Standard N(0,1)")
ax_l.set_xlabel("z", fontsize=12)
ax_l.set_ylabel("Density", fontsize=12)
ax_l.set_title(
    "Distribution of latent codes z = (ΔX - μ(x)) / σ(x)\n"
    "Should match N(0,1) — deviations reveal non-Gaussianity at boundaries",
    fontsize=10,
)
ax_l.set_xlim(-4, 4)
ax_l.legend(fontsize=9)

# Add quantile-quantile check as text
from scipy.stats import kstest
ks_stat, ks_pval = kstest(z_plot[:10_000], "norm")
ax_l.text(0.03, 0.95,
          f"KS test vs N(0,1):\nstat={ks_stat:.4f}  p={ks_pval:.4f}",
          transform=ax_l.transAxes, va="top", fontsize=9,
          bbox=dict(fc="white", alpha=0.8))

# Right: z stratified by x zone (to show boundary effects)
zones = [
    ([0.0, 1.0], "#E53935", "x ∈ [0,1)"),
    ([1.0, 3.0], "#43A047", "x ∈ [1,3)"),
    ([3.0, 5.0], "#1E88E5", "x ∈ [3,5)"),
    ([5.0, 6.0], "#FB8C00", "x ∈ [5,6]"),
]
for (lo, hi), color, label in zones:
    mask = (x_plot_2 >= lo) & (x_plot_2 < hi)
    if mask.sum() < 50:
        continue
    ax_r.hist(z_plot[mask], bins=80, density=True, color=color, alpha=0.55,
              range=(-4, 4), label=f"{label}  (n={mask.sum():,})")

ax_r.plot(z_range, scipy_norm.pdf(z_range), "k--", lw=1.5,
          label="N(0,1) reference")
ax_r.set_xlabel("z", fontsize=12)
ax_r.set_ylabel("Density", fontsize=12)
ax_r.set_title(
    "z distribution stratified by x position\n"
    "Near boundaries: z deviates from N(0,1) due to truncation",
    fontsize=10,
)
ax_r.set_xlim(-4, 4)
ax_r.legend(fontsize=8)

fig.suptitle("Label generation: latent z = (ΔX - μ(x)) / σ(x)", fontsize=12, y=1.01)
fig.tight_layout()
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "label_z_distribution.png"))

# ---------------------------------------------------------------------------
# 7.  Plot: μ(x) and σ(x) vs position
# ---------------------------------------------------------------------------
print("Generating label_mu_sigma_vs_x.png …")

# Compute smooth μ(x) and σ(x) on a grid for visualisation
x_grid   = np.linspace(0.05, 5.95, 200)
mu_grid  = np.zeros(200)
sig_grid = np.zeros(200)

for i, xq in enumerate(x_grid):
    _, idxs = tree_x.query([[xq]], k=K_LOCAL)
    nbrs = dx_ref[idxs[0]]
    mu_grid[i]  = nbrs.mean()
    sig_grid[i] = nbrs.std()

fig2, (ax2a, ax2b) = utils.make_fig(nrows=1, ncols=2, figsize=(14, 5))

# Left: μ(x)
ax2a.plot(x_grid, mu_grid, "b-", lw=2.5, label="μ(x) = E[ΔX | x_t=x]")
ax2a.axhline(0, color="gray", ls="--", lw=1, label="μ=0 (pure interior)")
ax2a.fill_between(x_grid,
                   mu_grid - sig_grid,
                   mu_grid + sig_grid,
                   alpha=0.2, color="blue", label="±σ(x) band")

# Annotate boundary effects
ax2a.annotate("Positive bias:\nleft wall repels",
               xy=(0.2, mu_grid[10]), xytext=(0.8, 0.002),
               arrowprops=dict(arrowstyle="->", color="black"),
               fontsize=8)

ax2a.set_xlabel("x_t (current SDE position)", fontsize=12)
ax2a.set_ylabel(r"$\mu(x) = E[\Delta X \mid x_t=x]$", fontsize=12)
ax2a.set_title("Conditional mean of ΔX given x_t\n"
               "Should be ≈0 in interior, biased near walls", fontsize=10)
ax2a.set_xlim(0, 6)
ax2a.legend(fontsize=9)

# Right: σ(x)
ax2b.plot(x_grid, sig_grid, "r-", lw=2.5, label="σ(x) = Std[ΔX | x_t=x]")
theory_sigma = np.sqrt(config.DT) * np.ones_like(x_grid)
ax2b.plot(x_grid, theory_sigma, "k--", lw=1.5,
          label=f"Theory: √dt = {np.sqrt(config.DT):.4f}")

ax2b.set_xlabel("x_t (current SDE position)", fontsize=12)
ax2b.set_ylabel(r"$\sigma(x) = \mathrm{Std}[\Delta X \mid x_t=x]$", fontsize=12)
ax2b.set_title("Conditional std of ΔX given x_t\n"
               "Should be ≈√dt in interior, reduced near walls (truncation)", fontsize=10)
ax2b.set_xlim(0, 6)
ax2b.legend(fontsize=9)

fig2.suptitle(
    f"Local conditional statistics  (K={K_LOCAL} neighbours)",
    fontsize=12, y=1.01,
)
fig2.tight_layout()
utils.save_fig(fig2, os.path.join(config.PLOTS_DIR, "label_mu_sigma_vs_x.png"))

# ---------------------------------------------------------------------------
# 8.  Plot: reconstruction check
# ---------------------------------------------------------------------------
#
# Verify: x_t + ΔX_reconstructed ≈ x_t + dx_sub
# where ΔX_reconstructed = μ(x_t) + σ(x_t) · z
#
print("Generating label_dx_reconstruction.png …")

dx_recon = mu_x[idx_p] + sig_x[idx_p] * z_plot   # should = dx_sub[idx_p]
dx_true  = dx_sub[idx_p]
recon_err = dx_recon - dx_true

fig3, axes3 = utils.make_fig(nrows=1, ncols=2, figsize=(14, 5))
ax3a, ax3b = axes3

# Scatter: true vs reconstructed
ax3a.scatter(dx_true[:5000], dx_recon[:5000], s=2, alpha=0.3, color="#1E88E5")
lim = max(abs(dx_true[:5000]).max(), abs(dx_recon[:5000]).max()) * 1.1
ax3a.plot([-lim, lim], [-lim, lim], "r-", lw=1.5, label="Perfect reconstruction")
ax3a.set_xlabel("True ΔX", fontsize=12)
ax3a.set_ylabel("Reconstructed ΔX = μ(x) + σ(x)·z", fontsize=12)
ax3a.set_title("Reconstruction check: True vs Recovered ΔX\n"
               "Points on diagonal = perfect; scatter = approximation error", fontsize=10)
ax3a.legend(fontsize=9)

# Reconstruction error histogram
ax3b.hist(recon_err, bins=80, density=True, color="#E53935", alpha=0.7)
ax3b.set_xlabel("Reconstruction error: ΔX_recon - ΔX_true", fontsize=12)
ax3b.set_ylabel("Density", fontsize=12)
ax3b.set_title(
    "Distribution of reconstruction error\n"
    "Error should be ≈ N(0, very_small) if K is large enough", fontsize=10,
)
err_std = recon_err.std()
ax3b.text(0.97, 0.95,
          f"Error std: {err_std:.6f}\nTrue dx std: {dx_true.std():.6f}",
          transform=ax3b.transAxes, ha="right", va="top", fontsize=9,
          bbox=dict(fc="white", alpha=0.8))

fig3.suptitle("Label quality: ΔX = μ(x) + σ(x)·z  (reparameterisation)",
              fontsize=12, y=1.01)
fig3.tight_layout()
utils.save_fig(fig3, os.path.join(config.PLOTS_DIR, "label_dx_reconstruction.png"))

# ---------------------------------------------------------------------------
# 9.  Diagnostics
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print()
print(f"  Labels generated : {N_LABEL:,}")
print(f"  KNN K used       : {K_LOCAL}")
print()
print("  Latent z quality (should be N(0,1)):")
print(f"    mean(z) = {z_labels.mean():.5f}  (expect 0)")
print(f"    std(z)  = {z_labels.std():.5f}   (expect 1)")
print(f"    KS stat vs N(0,1) = {ks_stat:.4f}  p={ks_pval:.4f}")
print(f"    (small stat and large p = good Gaussian fit)")
print()
print("  Local statistics:")
print(f"    μ(x) range: [{mu_grid.min():.5f}, {mu_grid.max():.5f}]")
print(f"    σ(x) range: [{sig_grid.min():.5f}, {sig_grid.max():.5f}]")
print(f"    Theory √dt: {np.sqrt(config.DT):.5f}")
print()
print("  Reconstruction error:")
print(f"    std(error)  = {recon_err.std():.6f}")
print(f"    std(dx_true)= {dx_true.std():.6f}")
print(f"    Relative    = {recon_err.std()/dx_true.std():.4f}")
print()
print("FILES WRITTEN")
print("  data/labels.npz")
print("  plots/label_z_distribution.png")
print("  plots/label_mu_sigma_vs_x.png")
print("  plots/label_dx_reconstruction.png")
print()
print("WHAT TO UNDERSTAND FROM THE PLOTS")
print()
print("  label_z_distribution.png:")
print("    Left: overall z should be N(0,1). KS p-value > 0.05 = good.")
print("    Right: near the walls (orange, red) z deviates from N(0,1).")
print("    This deviation encodes the BOUNDARY EFFECT the network must learn.")
print("    In the interior the Gaussian approximation is exact.")
print()
print("  label_mu_sigma_vs_x.png:")
print("    μ(x): near x=0 it's positive (wall prevents negative increments).")
print("    Near x=6 it's negative (wall prevents positive increments).")
print("    σ(x): smaller near walls (distribution is truncated, narrower).")
print("    These are the CONDITIONAL STATISTICS the network G must capture.")
print()
print("  label_dx_reconstruction.png:")
print("    The scatter should lie on the diagonal: perfect bijection.")
print("    Error std << dx std means the reparameterisation is accurate.")
print("    (Error = 0 exactly would mean ΔX = μ(x) + σ(x)·z perfectly.)")
print("    Residual error comes from using K=200 neighbours (finite K).")
print()
print("WHAT G(x,z) MUST LEARN")
print("  Input  : (x_t, z)  where x_t ∈ (0,6) and z ~ N(0,1)")
print("  Output : ΔX = label for that (x, z) pair")
print("  The mapping is approximately ΔX = μ(x) + σ(x)·z")
print("  but the network will learn the NON-LINEAR boundary corrections")
print("  from the actual labels, not from the Gaussian approximation.")
print()
print("Done.  Run 07_train_network.py next.")
