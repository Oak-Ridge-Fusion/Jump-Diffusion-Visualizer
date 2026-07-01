"""
03_forward_diffusion.py
=======================
PURPOSE
-------
Implement and visualise the "forward diffusion process" — the process that
gradually ADDS noise to the training data until it becomes indistinguishable
from a standard Gaussian.

This is the first half of a diffusion model (DDPM / score-based).
The second half (denoising / sampling) is in scripts 04 and 05.

THE BIG PICTURE
---------------
Diffusion models work in two phases:

  FORWARD (this script): data → noise
  ─────────────────────────────────────────────────────
  Z_0 = data sample (a ΔX increment from our dataset)
      ↓  add a little noise
  Z_{0.25}
      ↓  add more noise
  Z_{0.5}
      ↓  add more noise
  Z_{0.75}
      ↓  add more noise
  Z_1 ≈ N(0, 1)   (pure Gaussian noise)

  REVERSE (scripts 04-05): noise → data
  ─────────────────────────────────────────────────────
  Z_1 ~ N(0, 1)
      ↓  denoise  (requires knowing ∇ log p_t)
  Z_{0.75}
      ...
  Z_0 ≈ new sample of ΔX   ← this is what we want!

WHY NOT JUST SAMPLE FROM N(0, dt) DIRECTLY?
--------------------------------------------
For pure Brownian motion we *know* ΔX ~ N(0, dt) in the interior.
But near the boundaries the distribution is TRUNCATED and ASYMMETRIC
(as seen in script 02).  The generative model must learn to reproduce
this conditional structure.  A diffusion model is one principled way
to do so.

THE VP-SDE NOISE SCHEDULE
--------------------------
The paper uses a Variance-Preserving (VP) schedule:

    Z_t = α(t) · Z_0  +  σ(t) · ε,    ε ~ N(0, 1),   t ∈ [0, 1]

where

    β(t)     = β_min + t·(β_max - β_min)              (linear interpolation)
    log α(t) = -½ ∫₀ᵗ β(s) ds
             = -½ (β_min·t + ½·(β_max-β_min)·t²)
    α(t)     = exp(log α(t))                           (signal coefficient)
    σ(t)     = sqrt(1 - α(t)²)                         (noise coefficient)

KEY PROPERTIES
--------------
  - At t=0: α=1, σ=0  →  Z_0 = Z_0 (no noise added yet)
  - At t=1: α≈0, σ≈1  →  Z_1 ≈ ε ~ N(0,1) (nearly pure noise)
  - Var[Z_t] = α(t)² · Var[Z_0] + σ(t)² ≈ Var[Z_0] for all t
               (variance is PRESERVED, hence the name)

  This is useful because:
    1. The scale of Z_t stays bounded for all t.
    2. At t=1 the distribution is always standard Gaussian — easy to sample.
    3. The path from data to noise is smooth — no discontinuities.

SCORE FUNCTION
--------------
The reverse process requires the "score":

    s(z, t) = ∇_z log p_t(z)

where p_t is the marginal distribution of Z_t.

For the forward process above:
    p_t(z | z_0)  =  N(z ; α(t)·z_0,  σ(t)²)

The score conditioned on z_0 is:
    ∇_z log p_t(z | z_0)  =  -(z - α(t)·z_0) / σ(t)²
                           =  -ε / σ(t)          (where ε is the noise)

The UNCONDITIONAL score ∇_z log p_t(z) is estimated in script 04 (KNN).

OUTPUT
------
plots/forward_diffusion_slices.png
    Five panels showing the distribution of Z_t at t=0, 0.25, 0.5, 0.75, 1.
    You should see the distribution gradually becoming Gaussian.

plots/forward_diffusion_trajectories.png
    10 example Z_0 values tracked through the forward process.
    Shows how individual points drift toward 0 and spread out.

plots/alpha_sigma_schedule.png
    Shows α(t) and σ(t) across t ∈ [0,1].
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

import config
import utils

# ---------------------------------------------------------------------------
# 0.  Setup
# ---------------------------------------------------------------------------
config.ensure_dirs()
rng = np.random.default_rng(config.SEED + 2)

print("=" * 60)
print("STEP 03 — Forward Diffusion Process")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1.  Load a subset of the training data (the Z_0 samples)
# ---------------------------------------------------------------------------
print("Loading dataset …")
data   = np.load(os.path.join(config.DATA_DIR, "dataset.npz"))
x_t    = data["x_t"]    # current positions
dx     = data["dx"]     # increments — these are our Z_0

# Subsample to 500 000 for speed (the full 77M is overkill for visualisation)
N_DEMO = 500_000
idx    = rng.choice(len(dx), size=N_DEMO, replace=False)
z0     = dx[idx].astype(np.float64)   # Z_0 samples

print(f"  Z_0 samples loaded : {len(z0):,}")
print(f"  Z_0 mean           : {z0.mean():.6f}  (expect ≈ 0)")
print(f"  Z_0 std            : {z0.std():.6f}  (expect ≈ {np.sqrt(config.DT):.6f})")
print()

# ---------------------------------------------------------------------------
# 2.  Compute α(t) and σ(t) across the full schedule
# ---------------------------------------------------------------------------
t_grid  = np.linspace(0, 1, 1000)
alpha_t, sigma_t = utils.vp_alpha_sigma(t_grid, config.BETA_MIN, config.BETA_MAX)

print("VP-SDE schedule at key time points:")
print(f"  {'t':>5}  {'alpha':>10}  {'sigma':>10}")
for t_val in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    a, s = utils.vp_alpha_sigma(np.array([t_val]), config.BETA_MIN, config.BETA_MAX)
    print(f"  {t_val:>5.2f}  {a[0]:>10.6f}  {s[0]:>10.6f}")
print()

# ---------------------------------------------------------------------------
# 3.  Sample Z_t for 5 time slices
# ---------------------------------------------------------------------------
#
# For each t, compute:
#    Z_t = α(t) · Z_0  +  σ(t) · ε,    ε ~ N(0,1)
#
# The distribution of Z_t is:
#    p_t(z) = N(z ; α(t)·μ₀, α(t)²·σ₀² + σ(t)²)
#
# where μ₀ = mean(Z_0), σ₀² = var(Z_0).
#
t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
z_at_t   = {}     # t_val → array of Z_t samples

for t_val in t_slices:
    a, s = utils.vp_alpha_sigma(np.array([t_val]), config.BETA_MIN, config.BETA_MAX)
    a, s = float(a[0]), float(s[0])
    eps  = rng.standard_normal(size=len(z0))
    z_t  = a * z0 + s * eps
    z_at_t[t_val] = z_t

    print(f"  t={t_val:.2f}:  alpha={a:.4f}  sigma={s:.4f}  "
          f"Z_t mean={z_t.mean():.4f}  Z_t std={z_t.std():.4f}")

print()

# ---------------------------------------------------------------------------
# 4.  Plot: distribution at each time slice
# ---------------------------------------------------------------------------
print("Generating forward_diffusion_slices.png …")

colors = ["#1565C0", "#2E7D32", "#F9A825", "#AD1457", "#37474F"]
fig, axes = utils.make_fig(nrows=1, ncols=5, figsize=(20, 5))

for i, (t_val, color) in enumerate(zip(t_slices, colors)):
    ax  = axes[i]
    z_t = z_at_t[t_val]
    a, s = utils.vp_alpha_sigma(np.array([t_val]), config.BETA_MIN, config.BETA_MAX)
    a, s = float(a[0]), float(s[0])

    # Histogram
    ax.hist(
        z_t,
        bins   = 100,
        density= True,
        color  = color,
        alpha  = 0.7,
        range  = (-0.5, 0.5),
    )

    # Overlay the theoretical distribution at this t:
    # p_t(z) = N(z; α·μ₀, α²·σ₀² + σ²)
    mu_t    = a * z0.mean()
    var_t   = a**2 * z0.var() + s**2
    z_range = np.linspace(-0.5, 0.5, 300)
    p_theory = scipy_norm.pdf(z_range, loc=mu_t, scale=np.sqrt(var_t))
    ax.plot(z_range, p_theory, "r-", lw=2, label="Theory")

    ax.set_xlim(-0.5, 0.5)
    ax.set_title(
        f"t = {t_val:.2f}\n"
        f"α={a:.3f}  σ={s:.3f}\n"
        f"std(Z_t)={z_t.std():.4f}",
        fontsize=9,
    )
    ax.set_xlabel("z", fontsize=9)
    if i == 0:
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(ax.get_title() + "\n← DATA", fontsize=9)
    if i == 4:
        ax.set_title(ax.get_title() + "\n← PURE NOISE", fontsize=9)

fig.suptitle(
    "Forward Diffusion: Z_t = α(t)·Z_0 + σ(t)·ε\n"
    "Data (left) gradually becomes Gaussian noise (right)",
    fontsize=12, y=1.02,
)
fig.tight_layout()
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "forward_diffusion_slices.png"))

# ---------------------------------------------------------------------------
# 5.  Plot: trajectory of individual points through forward process
# ---------------------------------------------------------------------------
#
# Pick 10 specific Z_0 values and show how Z_t evolves as t increases.
# Each line is a *mean trajectory*: E[Z_t | Z_0] = α(t)·Z_0.
# We also show the ±1σ band.
#
print("Generating forward_diffusion_trajectories.png …")

n_example = 10
# Pick a spread of Z_0 values
z0_examples = np.linspace(z0.min() * 0.5, z0.max() * 0.5, n_example)

t_dense = np.linspace(0, 1, 200)
alpha_dense, sigma_dense = utils.vp_alpha_sigma(t_dense, config.BETA_MIN, config.BETA_MAX)

fig2, ax2 = utils.make_fig(figsize=(10, 6))

cmap = plt.cm.plasma
for k, z0_k in enumerate(z0_examples):
    color = cmap(k / (n_example - 1))

    # Mean trajectory: E[Z_t | Z_0 = z0_k] = alpha(t) * z0_k
    mean_traj = alpha_dense * float(z0_k)

    # ±1σ band: std[Z_t | Z_0] = sigma(t)  (doesn't depend on z0_k)
    ax2.fill_between(
        t_dense,
        mean_traj - sigma_dense,
        mean_traj + sigma_dense,
        color=color, alpha=0.08,
    )
    ax2.plot(t_dense, mean_traj, color=color, lw=1.8,
             label=f"z₀={z0_k:.4f}" if k % 3 == 0 else None)
    # Mark the starting point
    ax2.plot(0, z0_k, "o", color=color, markersize=6)

# Draw ±1σ of the PRIOR N(0,1) as horizontal reference at t=1
ax2.axhline(+1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax2.axhline(-1.0, color="gray", ls="--", lw=0.8, alpha=0.5, label="±1σ of N(0,1)")
ax2.axhline(0,    color="gray", ls="-",  lw=0.5, alpha=0.3)

ax2.set_xlabel("Diffusion time t", fontsize=12)
ax2.set_ylabel("Z_t", fontsize=12)
ax2.set_xlim(0, 1)
ax2.set_title(
    "Forward diffusion trajectories for individual Z_0 values\n"
    "Mean trajectory = α(t)·Z_0  (shrinks toward 0)\n"
    "Shaded band = ±1σ(t)  (grows from 0 to 1)",
    fontsize=10,
)
ax2.legend(fontsize=8, ncol=2)
utils.save_fig(fig2, os.path.join(config.PLOTS_DIR, "forward_diffusion_trajectories.png"))

# ---------------------------------------------------------------------------
# 6.  Plot: α(t) and σ(t) schedule
# ---------------------------------------------------------------------------
print("Generating alpha_sigma_schedule.png …")

fig3, (ax3a, ax3b) = utils.make_fig(nrows=1, ncols=2, figsize=(12, 5))

# Left: α and σ vs t
ax3a.plot(t_grid, alpha_t, "b-",  lw=2.5, label="α(t) — signal coefficient")
ax3a.plot(t_grid, sigma_t, "r-",  lw=2.5, label="σ(t) — noise coefficient")
ax3a.plot(t_grid, alpha_t**2 + sigma_t**2, "g--", lw=1.5,
          label="α²+σ² (should = 1)")
ax3a.set_xlabel("Diffusion time t", fontsize=12)
ax3a.set_ylabel("Value", fontsize=12)
ax3a.set_title("VP-SDE noise schedule\nα(t)↓ as noise added, σ(t)↑", fontsize=10)
ax3a.legend(fontsize=9)
ax3a.set_xlim(0, 1)
ax3a.set_ylim(0, 1.05)

# Right: signal-to-noise ratio (SNR) = α²/σ²
snr = alpha_t**2 / (sigma_t**2 + 1e-12)
ax3b.semilogy(t_grid, snr, "purple", lw=2.5)
ax3b.set_xlabel("Diffusion time t", fontsize=12)
ax3b.set_ylabel("SNR = α²(t) / σ²(t)  [log scale]", fontsize=12)
ax3b.set_title("Signal-to-Noise Ratio over diffusion time\n"
               "SNR → 0 means the data is totally hidden in noise", fontsize=10)
ax3b.set_xlim(0, 1)
ax3b.axhline(1, color="gray", ls="--", lw=1, label="SNR=1 crossover")
ax3b.legend(fontsize=9)

fig3.suptitle(
    f"VP-SDE schedule: β_min={config.BETA_MIN}, β_max={config.BETA_MAX}",
    fontsize=12, y=1.01,
)
fig3.tight_layout()
utils.save_fig(fig3, os.path.join(config.PLOTS_DIR, "alpha_sigma_schedule.png"))

# ---------------------------------------------------------------------------
# 7.  Save the schedule arrays (needed by scripts 04 and 05)
# ---------------------------------------------------------------------------
schedule_path = os.path.join(config.DATA_DIR, "vp_schedule.npz")
np.savez(
    schedule_path,
    t=t_grid, alpha=alpha_t, sigma=sigma_t,
    beta_min=np.array(config.BETA_MIN),
    beta_max=np.array(config.BETA_MAX),
)
print(f"Schedule saved → {schedule_path}")
print()

# ---------------------------------------------------------------------------
# 8.  Diagnostics
# ---------------------------------------------------------------------------
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print()
print("  Z_0 (data):")
utils.print_stats("Z_0", z0)
print()
print("  Z_t at each time slice:")
for t_val in t_slices:
    a, s = utils.vp_alpha_sigma(np.array([t_val]), config.BETA_MIN, config.BETA_MAX)
    z_t  = z_at_t[t_val]
    print(f"    t={t_val:.2f}  α={float(a[0]):.4f}  σ={float(s[0]):.4f}  "
          f"mean={z_t.mean():.5f}  std={z_t.std():.5f}")
print()
print("  At t=1, Z_1 should be N(0,1):")
z1 = z_at_t[1.0]
print(f"    mean(Z_1) = {z1.mean():.5f}  (expect ≈ 0)")
print(f"    std(Z_1)  = {z1.std():.5f}   (expect ≈ 1)")
print()
print("FILES WRITTEN")
print("  data/vp_schedule.npz")
print("  plots/forward_diffusion_slices.png")
print("  plots/forward_diffusion_trajectories.png")
print("  plots/alpha_sigma_schedule.png")
print()
print("WHAT TO UNDERSTAND FROM THE PLOTS")
print()
print("  forward_diffusion_slices.png:")
print("    At t=0 you see the narrow ΔX distribution (std≈0.022).")
print("    At t=1 you see a broad N(0,1) Gaussian (std≈1).")
print("    The red theory curve should match the histogram perfectly.")
print("    The transition from narrow-non-Gaussian to Gaussian is the")
print("    'forward process' — the reverse is what the model learns.")
print()
print("  forward_diffusion_trajectories.png:")
print("    Each coloured line is one Z_0 value's mean trajectory.")
print("    All lines converge to 0 as α(t)→0.")
print("    The shaded band widens as σ(t)→1.")
print("    At t=1 all trajectories are lost in the same noise cloud.")
print()
print("  alpha_sigma_schedule.png:")
print("    The left panel shows the cross-over: at some t, α=σ≈0.7.")
print("    The right panel (log scale) shows SNR dropping from ~2000")
print("    (at t=0, data dominates) to ~0 (at t=1, noise dominates).")
print("    The large β_max=20 means the schedule is FAST — data is")
print("    destroyed quickly.  This forces the model to denoise efficiently.")
print()
print("Done.  Run 04_knn_score_estimation.py next.")
