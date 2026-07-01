"""
08_rollout.py
=============
PURPOSE
-------
Use the trained network G(x, z) to roll out full SDE trajectories from
t=0 to t=T=3, starting at x_0=1, and compare the final distribution
against the ground truth from script 01.

This generates "Our Method" histogram — Figure 2, panel 2.

THE ROLLOUT ALGORITHM
---------------------
For each of N_PARTICLES particles:

    x = x_0 = 1.0      (initial position)

    For each step n = 0, 1, 2, ..., N_STEPS-1:
        z ~ N(0,1)                    ← sample fresh latent noise
        ΔX = G(x, z)                  ← neural network forward pass
        x_new = x + ΔX               ← update position

        If x_new ≤ 0 or x_new ≥ 6:   ← check absorbing boundary
            particle is absorbed; stop this trajectory

    survivors: particles still inside at step N_STEPS

WHY THIS WORKS
--------------
The network G(x, z) was trained on labels generated from CLEAN PAIRS —
transitions where the particle survived both before and after the step.

Therefore:
  - When z ~ N(0,1) is sampled and G(x, z) is evaluated, the OUTPUT
    distribution p(ΔX = G(x,z)) should match p(ΔX | particle survived one step)

  - Repeated application of G gives trajectories that mimic the
    SURVIVING BROWNIAN PATHS from the ground truth simulation

  - The final distribution of x at t=T should match the Fourier series
    solution from script 01

IMPORTANT: the explicit boundary check is STILL NECESSARY
----------------------------------------------------------
Even though G was trained on surviving pairs, it might occasionally output
a large increment that pushes the particle outside.  We still check and
absorb those particles.  This is fine — the network just occasionally
"fails" near the boundary, which is natural.

COMPARISON WITH GROUND TRUTH
-----------------------------
We generate:
  - our_method_histogram.png : the rollout histogram vs ground truth
  - comparison_trajectories.png : 500 example rollout paths

OUTPUT
------
data/our_method.npz          — survivors and trajectories from rollout
plots/our_method_histogram.png
plots/comparison_trajectories.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import config
import utils
from model import FlowNet

# ---------------------------------------------------------------------------
# 0.  Setup
# ---------------------------------------------------------------------------
config.ensure_dirs()
rng = np.random.default_rng(config.SEED + 7)

# Use CPU for rollout: MPS has kernel-launch overhead that dominates for
# step-by-step inference.  CPU is faster for sequential batch-forward passes
# over 6000 steps.
device = torch.device("cpu")

print("=" * 60)
print("STEP 08 — Rollout: Generate 'Our Method' Histogram")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1.  Load trained network
# ---------------------------------------------------------------------------
model = FlowNet().to(device)
model.load_state_dict(
    torch.load(os.path.join(config.MODELS_DIR, "flow_model.pt"),
               map_location=device)
)
model.eval()
print(f"Loaded flow_model.pt  (device: {device})")
print()

# ---------------------------------------------------------------------------
# 2.  Helper: batch predict ΔX from (x_array, z_array)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(x_arr, z_arr):
    """
    x_arr : (N,) numpy array of current positions
    z_arr : (N,) numpy array of latent noise values
    returns: (N,) numpy array of predicted ΔX
    """
    x_norm = torch.tensor(x_arr / config.DOMAIN_HI, dtype=torch.float32)
    z_t    = torch.tensor(z_arr, dtype=torch.float32)
    inp    = torch.stack([x_norm, z_t], dim=1).to(device)
    out    = model(inp).cpu().numpy().ravel()
    return out


# ---------------------------------------------------------------------------
# 3.  Rollout simulation
# ---------------------------------------------------------------------------
print(f"Running rollout simulation:")
print(f"  N_PARTICLES : {config.N_PARTICLES:,}")
print(f"  N_STEPS     : {config.N_STEPS:,}")
print(f"  x_0         : {config.X0}")
print()

x     = np.full(config.N_PARTICLES, config.X0, dtype=np.float64)
alive = np.ones(config.N_PARTICLES, dtype=bool)

# Store 500 probe trajectories
N_PROBE   = 500
probe_idx = rng.choice(config.N_PARTICLES, size=N_PROBE, replace=False)
traj_our  = np.full((N_PROBE, config.N_STEPS + 1), np.nan, dtype=np.float32)
traj_our[:, 0] = x[probe_idx]

for step in tqdm(range(config.N_STEPS), desc="Rollout", ncols=70):

    n_alive = alive.sum()
    if n_alive == 0:
        break

    # Sample latent z for alive particles
    z_arr = rng.standard_normal(n_alive).astype(np.float32)
    x_arr = x[alive].astype(np.float32)

    # Neural network prediction
    dx_pred = predict_batch(x_arr, z_arr)

    # Update positions
    x[alive] += dx_pred

    # Absorbing boundary check
    hit   = (x <= config.DOMAIN_LO) | (x >= config.DOMAIN_HI)
    alive &= ~hit

    # Record probe trajectories (vectorised — no Python loop)
    probe_still_alive = alive[probe_idx]           # shape (N_PROBE,)
    alive_j = np.where(probe_still_alive)[0]       # indices into probe array
    traj_our[alive_j, step + 1] = x[probe_idx[alive_j]].astype(np.float32)

    if (step + 1) % 1000 == 0:
        pct = 100 * alive.sum() / config.N_PARTICLES
        tqdm.write(f"    step {step+1:5d}/{config.N_STEPS}  "
                   f"alive: {alive.sum():,}  ({pct:.1f}%)")

print()

# ---------------------------------------------------------------------------
# 4.  Collect results
# ---------------------------------------------------------------------------
survivors_our = x[alive].copy()
print(f"Rollout complete.")
print(f"  Survivors: {len(survivors_our):,} / {config.N_PARTICLES:,}  "
      f"({100*len(survivors_our)/config.N_PARTICLES:.2f}%)")
utils.print_stats("our_method survivors", survivors_our)
print()

# Load ground truth for comparison
gt_data = np.load(os.path.join(config.DATA_DIR, "ground_truth.npz"))
survivors_gt = gt_data["survivors"]
print(f"Ground truth survivors: {len(survivors_gt):,}  "
      f"({100*len(survivors_gt)/config.N_PARTICLES:.2f}%)")
print()

# ---------------------------------------------------------------------------
# 5.  Save
# ---------------------------------------------------------------------------
save_path = os.path.join(config.DATA_DIR, "our_method.npz")
np.savez(save_path, survivors=survivors_our, trajectories=traj_our, alive=alive)
print(f"Saved → {save_path}")
print()

# ---------------------------------------------------------------------------
# 6.  Plot: histogram comparison
# ---------------------------------------------------------------------------
print("Generating our_method_histogram.png …")

def analytical_pdf(x_vals, x0=config.X0, L=config.DOMAIN_HI, t=config.T, N=100):
    ns  = np.arange(1, N + 1)
    dec = np.exp(-ns**2 * np.pi**2 * t / (2 * L**2))
    phi_x0 = np.sin(ns[None, :] * np.pi * x0 / L)
    phi_x  = np.sin(np.outer(x_vals, ns * np.pi / L))
    p = (2.0 / L) * np.sum(phi_x0 * phi_x * dec[None, :], axis=1)
    return np.maximum(p, 0.0)

x_theory = np.linspace(config.DOMAIN_LO, config.DOMAIN_HI, 400)
p_theory  = analytical_pdf(x_theory)
dx_t      = x_theory[1] - x_theory[0]
P_surv    = p_theory.sum() * dx_t
p_theory_cond = p_theory / P_surv   # normalise to conditional

fig, ax = utils.make_fig(figsize=(10, 6))
n_bins = 80

ax.hist(
    survivors_gt,
    bins   = n_bins,
    range  = (config.DOMAIN_LO, config.DOMAIN_HI),
    density= True,
    color  = "#90A4AE",
    alpha  = 0.5,
    label  = f"Ground Truth  (n={len(survivors_gt):,})",
)
ax.hist(
    survivors_our,
    bins   = n_bins,
    range  = (config.DOMAIN_LO, config.DOMAIN_HI),
    density= True,
    color  = "#1565C0",
    alpha  = 0.65,
    label  = f"Our Method  (n={len(survivors_our):,})",
)
ax.plot(x_theory, p_theory_cond, "r-", lw=2.5,
        label="Analytical (Fourier series)")

ax.set_xlim(config.DOMAIN_LO, config.DOMAIN_HI)
ax.set_xlabel("Position x at T = 3", fontsize=12)
ax.set_ylabel("Probability density", fontsize=12)
ax.set_title(
    "Our Method vs Ground Truth\n"
    r"$dX_t = dW_t$,  $X_0=1$,  absorbing boundaries at 0 and 6",
    fontsize=11,
)
ax.legend(fontsize=10)

# KL divergence estimate
from scipy.stats import ks_2samp
ks_stat, ks_pval = ks_2samp(survivors_gt, survivors_our)
ax.text(0.97, 0.95,
        f"KS test (our vs GT):\nstat={ks_stat:.4f}  p={ks_pval:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(fc="white", alpha=0.8))

utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "our_method_histogram.png"))

# ---------------------------------------------------------------------------
# 7.  Plot: trajectory comparison
# ---------------------------------------------------------------------------
print("Generating comparison_trajectories.png …")

time_axis = np.linspace(0, config.T, config.N_STEPS + 1)

fig2, axes2 = utils.make_fig(nrows=2, ncols=2, figsize=(16, 10))

# Load ground truth probe trajectories
traj_gt = gt_data["trajectories"]   # shape (500, N_STEPS+1)

# Panel 1: GT trajectories
ax1 = axes2[0, 0]
for j in range(min(150, N_PROBE)):
    path = traj_gt[j]
    vm   = ~np.isnan(path)
    color = "#1565C0" if vm[-1] else "#C62828"
    ax1.plot(time_axis[vm], path[vm], color=color, alpha=0.2, lw=0.5)
ax1.axhline(config.DOMAIN_LO, color="black", lw=1.5, ls="--")
ax1.axhline(config.DOMAIN_HI, color="black", lw=1.5, ls="-.")
ax1.set_title("Ground Truth trajectories", fontsize=11)
ax1.set_xlabel("Time t"); ax1.set_ylabel("X_t")
ax1.set_ylim(-0.3, 6.3)

# Panel 2: Our Method trajectories
ax2 = axes2[0, 1]
for j in range(min(150, N_PROBE)):
    path = traj_our[j]
    vm   = ~np.isnan(path)
    color = "#1565C0" if vm[-1] else "#C62828"
    ax2.plot(time_axis[vm], path[vm], color=color, alpha=0.2, lw=0.5)
ax2.axhline(config.DOMAIN_LO, color="black", lw=1.5, ls="--")
ax2.axhline(config.DOMAIN_HI, color="black", lw=1.5, ls="-.")
ax2.set_title("Our Method trajectories", fontsize=11)
ax2.set_xlabel("Time t"); ax2.set_ylabel("X_t")
ax2.set_ylim(-0.3, 6.3)

# Panel 3: GT histogram
ax3 = axes2[1, 0]
ax3.hist(survivors_gt, bins=80, density=True, color="#90A4AE", alpha=0.7,
         range=(0,6), label=f"GT (n={len(survivors_gt):,})")
ax3.plot(x_theory, p_theory_cond, "r-", lw=2)
ax3.set_title("Ground Truth: final distribution at T=3", fontsize=11)
ax3.set_xlabel("Position x"); ax3.set_ylabel("Density")
ax3.set_xlim(0, 6)
ax3.legend(fontsize=9)

# Panel 4: Our Method histogram
ax4 = axes2[1, 1]
ax4.hist(survivors_our, bins=80, density=True, color="#1565C0", alpha=0.7,
         range=(0,6), label=f"Our Method (n={len(survivors_our):,})")
ax4.plot(x_theory, p_theory_cond, "r-", lw=2)
ax4.set_title("Our Method: final distribution at T=3", fontsize=11)
ax4.set_xlabel("Position x"); ax4.set_ylabel("Density")
ax4.set_xlim(0, 6)
ax4.legend(fontsize=9)

fig2.suptitle(
    "Comparison: Ground Truth vs Our Method\n"
    "Both distributions should match the red analytical curve",
    fontsize=12, y=1.01,
)
fig2.tight_layout()
utils.save_fig(fig2, os.path.join(config.PLOTS_DIR, "comparison_trajectories.png"))

# ---------------------------------------------------------------------------
# 8.  Diagnostics
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print()
print(f"  Ground Truth survivors : {len(survivors_gt):,} ({100*len(survivors_gt)/config.N_PARTICLES:.2f}%)")
print(f"  Our Method survivors   : {len(survivors_our):,} ({100*len(survivors_our)/config.N_PARTICLES:.2f}%)")
print()
print(f"  KS test (Our Method vs Ground Truth):")
print(f"    stat = {ks_stat:.4f}")
print(f"    p    = {ks_pval:.4f}")
print(f"    (p > 0.05 = distributions are statistically indistinguishable)")
print()
print("  Distribution statistics:")
print(f"    GT  mean={survivors_gt.mean():.4f}  std={survivors_gt.std():.4f}")
print(f"    Our mean={survivors_our.mean():.4f}  std={survivors_our.std():.4f}")
print()
print("FILES WRITTEN")
print("  data/our_method.npz")
print("  plots/our_method_histogram.png")
print("  plots/comparison_trajectories.png")
print()
print("WHAT TO CHECK")
print()
print("  our_method_histogram.png:")
print("    Blue (Our Method) and gray (GT) bars should overlap closely.")
print("    Both should match the red analytical curve.")
print("    If the blue bars are taller near x=0 or x=6: G is not learning")
print("    the boundary effect correctly — increase training data or epochs.")
print()
print("  comparison_trajectories.png:")
print("    The trajectory panels (top) should look visually similar.")
print("    The histogram panels (bottom) should both match the red curve.")
print()
print("Done.  Run 09_compare_results.py next.")
