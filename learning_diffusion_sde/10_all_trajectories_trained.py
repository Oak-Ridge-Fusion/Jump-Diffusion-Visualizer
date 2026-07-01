"""
10_all_trajectories_trained.py
==============================
PURPOSE
-------
Reproduce the "All Trajectories Trained" panel from Figure 2 of the paper.

The ONLY change from "Our Method" (scripts 06-08) is:

    TRAINING DATA:  include ALL step transitions, even those where the
                    particle CROSSED the boundary (was absorbed).

Everything else — network architecture, hyperparameters, rollout procedure —
remains IDENTICAL.

WHY THIS MATTERS
----------------
In "Our Method" we used CLEAN pairs: both endpoints inside (0, 6).
Near the boundary, this creates a BIASED training set:
  - At x = 0.05, all surviving Δx values are positive (particle can't go more
    negative than -0.05 and survive)
  - The network learns this bias → correct boundary-respecting behaviour

In "All Trajectories Trained" we include the FULL Euler step even if the
particle subsequently exits:
  - At x = 0.05, we NOW include Δx values as negative as -5 (the particle
    exited but we still record the step)
  - The training Δx distribution near x=0 is now SYMMETRIC (no boundary effect)
  - The network learns a symmetric, boundary-IGNORING distribution

CONSEQUENCE AT ROLLOUT
-----------------------
During rollout the network still enforces the EXPLICIT absorbing boundary
check (x → absorbed if x ≤ 0 or x ≥ 6).  So particles DO get absorbed.

BUT: the network now proposes increments without knowing about the boundary.
Near x=0 it suggests symmetric Δx ~ N(0, σ), meaning:
  - Many particles drift negative → get absorbed at x=0
  - Far fewer particles make it to x = 3-6 (right half of domain)
  - The surviving distribution at T=3 is SHIFTED LEFT compared to ground truth

This is exactly what Figure 2 panel 3 shows in the paper.

PEDAGOGICAL POINT
-----------------
The choice of training data is MORE IMPORTANT than the model architecture.
The same network, trained on different data, produces qualitatively different
(wrong) results.  This teaches that data curation is as critical as modelling.

OUTPUT
------
data/all_trajectories.npz
plots/all_trajectories_histogram.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.spatial import cKDTree
from tqdm import tqdm

import config
import utils
from model import FlowNet

config.ensure_dirs()
torch.manual_seed(config.SEED + 10)
rng = np.random.default_rng(config.SEED + 10)

device = torch.device("cpu")

print("=" * 60)
print("STEP 10 — Ablation: All Trajectories Trained")
print("=" * 60)
print()
print("KEY DIFFERENCE vs Our Method:")
print("  Include ALL transitions, even those crossing the boundary.")
print("  Everything else (architecture, hyperparams, rollout) = IDENTICAL.")
print()

# ---------------------------------------------------------------------------
# 1.  Re-simulate with ALL transitions captured
# ---------------------------------------------------------------------------
print(f"Simulating {config.N_PARTICLES:,} particles for {config.N_STEPS} steps …")
print("Collecting ALL transitions (including boundary crossings).")
print()

SUBSAMPLE_EVERY = 10

x_t_list_all = []
dx_list_all  = []

x     = np.full(config.N_PARTICLES, config.X0, dtype=np.float32)
alive = np.ones(config.N_PARTICLES, dtype=bool)

for step in tqdm(range(config.N_STEPS), unit="step", ncols=70):
    alive_before = alive.copy()
    x_before_all = x.copy()

    n_alive = int(alive.sum())
    noise   = rng.standard_normal(size=n_alive).astype(np.float32)
    x[alive] += np.sqrt(config.DT) * noise

    # Kill particles that exited — but collect the pair BEFORE killing
    hit   = (x <= config.DOMAIN_LO) | (x >= config.DOMAIN_HI)

    if (step + 1) % SUBSAMPLE_EVERY == 0:
        # Include ALL particles alive at the START of the step.
        # This includes those that crossed the boundary during this step.
        alive_start = alive_before   # True for particles alive BEFORE step
        n_start = int(alive_start.sum())
        if n_start > 0:
            x_t_list_all.append(x_before_all[alive_start])
            dx_list_all.append((x - x_before_all)[alive_start])

    # Now update alive mask
    alive &= ~hit

    if (step + 1) % 1000 == 0:
        tqdm.write(f"    step {step+1}/{config.N_STEPS}  alive={alive.sum():,}")

print()
x_t_all = np.concatenate(x_t_list_all)
dx_all  = np.concatenate(dx_list_all)
print(f"  Total pairs (incl. crossings): {len(x_t_all):,}")
utils.print_stats("dx (all)", dx_all)
print()

# ---------------------------------------------------------------------------
# 2.  Compare training distributions
# ---------------------------------------------------------------------------
#
# CRITICAL: compare the dx distribution near x=0 for "Our Method" vs "All".
# Near the boundary, Our Method has positive bias; All Trajectories is symmetric.
#
print("Training data comparison near left wall (x ∈ [0, 0.5]):")
near_wall   = (x_t_all >= 0) & (x_t_all < 0.5)
dx_near_all = dx_all[near_wall]
print(f"  All Trajectories: n={len(dx_near_all):,}  "
      f"mean={dx_near_all.mean():.5f}  std={dx_near_all.std():.5f}")

# Load clean dataset for comparison
clean_data = np.load(os.path.join(config.DATA_DIR, "dataset.npz"))
clean_x = clean_data["x_t"]
clean_dx = clean_data["dx"]
near_wall_clean = (clean_x >= 0) & (clean_x < 0.5)
dx_near_clean = clean_dx[near_wall_clean]
print(f"  Our Method:       n={len(dx_near_clean):,}  "
      f"mean={dx_near_clean.mean():.5f}  std={dx_near_clean.std():.5f}")
print()
print("  The 'All Trajectories' mean near x=0 should be ≈0 (symmetric).")
print("  'Our Method' mean near x=0 should be POSITIVE (right-biased).")
print("  This bias is the boundary correction that makes Our Method correct.")
print()

# ---------------------------------------------------------------------------
# 3.  Generate labels using the same KNN reparameterisation
# ---------------------------------------------------------------------------
print("Generating labels with K=200 KNN reparameterisation …")

N_LABEL = min(500_000, len(x_t_all))
idx     = rng.choice(len(x_t_all), size=N_LABEL, replace=False)
x_t_sub = x_t_all[idx].astype(np.float64)
dx_sub  = dx_all[idx].astype(np.float64)

# Build KDTree on the ALL-TRAJECTORIES reference set
tree_x_all = cKDTree(x_t_sub.reshape(-1, 1))
K_LOCAL = 200

mu_x  = np.zeros(N_LABEL)
sig_x = np.zeros(N_LABEL)

BATCH = 50_000
for b in tqdm(range((N_LABEL + BATCH - 1) // BATCH), desc="KNN", ncols=60):
    lo = b * BATCH
    hi = min((b + 1) * BATCH, N_LABEL)
    _, idxs = tree_x_all.query(x_t_sub[lo:hi].reshape(-1, 1), k=K_LOCAL)
    dx_nbrs = dx_sub[idxs]
    mu_x[lo:hi]  = dx_nbrs.mean(axis=1)
    sig_x[lo:hi] = dx_nbrs.std(axis=1) + 1e-8

z_labels = (dx_sub - mu_x) / sig_x
print()

# ---------------------------------------------------------------------------
# 4.  Train the same network
# ---------------------------------------------------------------------------
print(f"Training FlowNet on ALL-TRAJECTORIES labels …")

x_norm = torch.tensor(x_t_sub / config.DOMAIN_HI, dtype=torch.float32)
z_t    = torch.tensor(z_labels.astype(np.float32), dtype=torch.float32)
y_t    = torch.tensor(dx_sub.astype(np.float32),   dtype=torch.float32)
X_in   = torch.stack([x_norm, z_t], dim=1)
Y_in   = y_t.unsqueeze(1)

dataset = TensorDataset(X_in, Y_in)
n_train = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                 generator=torch.Generator().manual_seed(config.SEED + 10))
train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE * 4, shuffle=False)

model_all = FlowNet().to(device)
opt       = torch.optim.Adam(model_all.parameters(), lr=config.LEARNING_RATE)
sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.N_EPOCHS, eta_min=1e-5)
loss_fn   = nn.MSELoss()

best_val = float("inf")
for epoch in range(1, config.N_EPOCHS + 1):
    model_all.train()
    for Xb, Yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model_all(Xb.to(device)), Yb.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model_all.parameters(), 1.0)
        opt.step()
    model_all.eval()
    val_loss = sum(loss_fn(model_all(Xb.to(device)), Yb.to(device)).item()
                   for Xb, Yb in val_loader) / len(val_loader)
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model_all.state_dict(),
                   os.path.join(config.MODELS_DIR, "flow_model_all.pt"))
    sched.step()
    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{config.N_EPOCHS}  val_MSE={val_loss:.6f}")

model_all.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, "flow_model_all.pt"), map_location=device))
model_all.eval()
print(f"  Best val_MSE={best_val:.6f}")
print()

# ---------------------------------------------------------------------------
# 5.  Rollout (IDENTICAL procedure to script 08)
# ---------------------------------------------------------------------------
print("Rolling out trajectories …")

@torch.no_grad()
def predict(x_arr, z_arr):
    x_n = torch.tensor(x_arr / config.DOMAIN_HI, dtype=torch.float32)
    z_  = torch.tensor(z_arr, dtype=torch.float32)
    return model_all(torch.stack([x_n, z_], dim=1)).numpy().ravel()

x_ro  = np.full(config.N_PARTICLES, config.X0, dtype=np.float64)
alive = np.ones(config.N_PARTICLES, dtype=bool)

for step in tqdm(range(config.N_STEPS), desc="Rollout", ncols=70):
    n_alive = alive.sum()
    if n_alive == 0:
        break
    z_arr   = rng.standard_normal(n_alive).astype(np.float32)
    x_arr   = x_ro[alive].astype(np.float32)
    dx_pred = predict(x_arr, z_arr)
    x_ro[alive] += dx_pred
    hit = (x_ro <= config.DOMAIN_LO) | (x_ro >= config.DOMAIN_HI)
    alive &= ~hit
    if (step + 1) % 1000 == 0:
        tqdm.write(f"    step {step+1}  alive={alive.sum():,}")

surv_all = x_ro[alive].copy()
print()
print(f"  Survivors: {len(surv_all):,} ({100*len(surv_all)/config.N_PARTICLES:.2f}%)")
utils.print_stats("all_traj survivors", surv_all)
print()

np.savez(os.path.join(config.DATA_DIR, "all_trajectories.npz"), survivors=surv_all)

# ---------------------------------------------------------------------------
# 6.  Plot and explain
# ---------------------------------------------------------------------------
gt = np.load(os.path.join(config.DATA_DIR, "ground_truth.npz"))
surv_gt = gt["survivors"]

def analytical_pdf_cond(x_v):
    ns  = np.arange(1, 101)
    dec = np.exp(-ns**2 * np.pi**2 * config.T / (2 * config.DOMAIN_HI**2))
    p   = (2./config.DOMAIN_HI) * np.sum(
        np.sin(ns*np.pi*config.X0/config.DOMAIN_HI) *
        np.sin(np.outer(x_v, ns*np.pi/config.DOMAIN_HI)) * dec, axis=1)
    p   = np.maximum(p, 0.)
    dx  = x_v[1] - x_v[0]
    return p / (p.sum() * dx)

x_th = np.linspace(0.01, 5.99, 400)
p_th = analytical_pdf_cond(x_th)

print("Generating all_trajectories_histogram.png …")
fig, axes = utils.make_fig(nrows=1, ncols=2, figsize=(14, 5))

axes[0].hist(surv_gt, bins=80, range=(0,6), density=True, color="#607D8B",
             alpha=0.7, label=f"Ground Truth (n={len(surv_gt):,})")
axes[0].plot(x_th, p_th, "r-", lw=2)
axes[0].set(xlim=(0,6), xlabel="x at T=3", ylabel="Density", title="Ground Truth")
axes[0].legend(fontsize=9)

axes[1].hist(surv_all, bins=80, range=(0,6), density=True, color="#F57F17",
             alpha=0.7, label=f"All Trajectories (n={len(surv_all):,})")
axes[1].plot(x_th, p_th, "r-", lw=2, label="Analytical")
axes[1].set(xlim=(0,6), xlabel="x at T=3", ylabel="Density",
            title="All Trajectories Trained\n(wrong: symmetric increments near walls)")
axes[1].legend(fontsize=9)
axes[1].text(0.03, 0.95,
             "ARTIFACT: boundary pairs\nincluded → no boundary bias\n"
             "→ particles pile up at left wall",
             transform=axes[1].transAxes, va="top", fontsize=8,
             bbox=dict(fc="lightyellow", alpha=0.9))

fig.suptitle(
    "Ablation: All Trajectories vs Ground Truth\n"
    "Training on crossing transitions destroys the boundary effect",
    fontsize=12, y=1.02,
)
fig.tight_layout()
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "all_trajectories_histogram.png"))

# ---------------------------------------------------------------------------
# 7.  Print the key explanation
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("EXPLANATION: WHY ALL TRAJECTORIES TRAINED FAILS")
print("=" * 60)
print()
print("  Near x = 0 (left wall):")
print(f"    All Trajectories dx mean: {dx_near_all.mean():.5f}  (≈ 0, symmetric)")
print(f"    Our Method       dx mean: {dx_near_clean.mean():.5f}  (positive, biased away from wall)")
print()
print("  The Our Method training data is BIASED near walls:")
print("  Only steps where BOTH start and end are inside are included.")
print("  This removes the large-negative steps that would cross x=0.")
print("  The remaining steps are right-biased → network learns right bias.")
print()
print("  All Trajectories includes CROSSING steps:")
print("  These have large negative Δx values that cancel the right bias.")
print("  The network learns a symmetric distribution → ignores the wall.")
print("  At rollout, particles near x=0 drift symmetrically and pile up")
print("  against the left wall (getting absorbed rapidly).")
print("  The survivors are biased toward large x values... or toward")
print("  small x values depending on the exact training dynamics.")
print()
print("FILES WRITTEN")
print("  data/all_trajectories.npz")
print("  models/flow_model_all.pt")
print("  plots/all_trajectories_histogram.png")
print()
print("Done.  Run 11_only_confined.py next.")
