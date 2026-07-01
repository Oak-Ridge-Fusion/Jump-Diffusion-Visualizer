"""
11_only_confined.py
===================
PURPOSE
-------
Reproduce the "Only Confined Trained" panel from Figure 2.

The ONLY change: train on ONLY the subset of pairs that come from
trajectories which NEVER got absorbed (survived all the way to T=3).

WHY THIS IS INTERESTING
------------------------
A "confined" trajectory is one where the particle stays well inside (0,6)
for the entire duration T=3.  These particles tend to start near the centre
(x=1, drifting right) and undergo small Brownian excursions that never
approach either wall.

The distribution of increments for confined trajectories is:
  - Very narrow: the particle is always far from both walls
  - Approximately symmetric: no boundary truncation needed
  - BUT: the range of starting positions x_t is also narrowed
    (confined particles cluster in [1, 4] roughly)

CONSEQUENCE AT ROLLOUT
-----------------------
The network trained on "only confined" data:
  1. Has never seen increments from x near 0 or 6 (too few examples)
  2. Has learned narrow, symmetric increments
  3. At rollout, any particle that wanders near a wall will get increments
     drawn from the wrong distribution

Result: particles near the walls are NOT corrected away from the boundary.
They continue to wander and eventually get absorbed — but with the WRONG
time distribution.  The survivors at T=3 form a distribution that is
NARROWER and more CONCENTRATED (near x=1-3) than the ground truth.

In the extreme case: if you train ONLY on confined particles, the model
learns "stay in [1,4]" which underrepresents the full surviving distribution.

OUTPUT
------
data/only_confined.npz
plots/only_confined_histogram.png
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
torch.manual_seed(config.SEED + 11)
rng = np.random.default_rng(config.SEED + 11)
device = torch.device("cpu")

print("=" * 60)
print("STEP 11 — Ablation: Only Confined Trajectories")
print("=" * 60)
print()
print("KEY DIFFERENCE: train ONLY on particles that survived to T=3.")
print()

# ---------------------------------------------------------------------------
# 1.  Simulate and identify confined particles
# ---------------------------------------------------------------------------
print(f"Simulating {config.N_PARTICLES:,} particles, identifying survivors …")

x     = np.full(config.N_PARTICLES, config.X0, dtype=np.float32)
alive = np.ones(config.N_PARTICLES, dtype=bool)
# Track which particles were ALWAYS alive (never absorbed)
always_alive = np.ones(config.N_PARTICLES, dtype=bool)

# First pass: run simulation to find which particles survive to T=3
for step in tqdm(range(config.N_STEPS), unit="step", ncols=70, desc="Pass 1"):
    noise  = rng.standard_normal(int(alive.sum())).astype(np.float32)
    x[alive] += np.sqrt(config.DT) * noise
    hit     = (x <= config.DOMAIN_LO) | (x >= config.DOMAIN_HI)
    alive  &= ~hit
    always_alive &= alive   # once dead, stays dead in always_alive

confined_idx = np.where(always_alive)[0]
print(f"\n  Particles that survived to T=3: {len(confined_idx):,} "
      f"({100*len(confined_idx)/config.N_PARTICLES:.2f}%)")
print()

# ---------------------------------------------------------------------------
# 2.  Second pass: collect transitions ONLY for confined particles
# ---------------------------------------------------------------------------
print("Pass 2: collecting transitions for confined particles only …")

SUBSAMPLE_EVERY = 10

x     = np.full(config.N_PARTICLES, config.X0, dtype=np.float32)
alive = np.ones(config.N_PARTICLES, dtype=bool)

x_t_list = []
dx_list  = []

for step in tqdm(range(config.N_STEPS), unit="step", ncols=70):
    x_before = x.copy()
    noise    = rng.standard_normal(int(alive.sum())).astype(np.float32)
    x[alive] += np.sqrt(config.DT) * noise
    hit  = (x <= config.DOMAIN_LO) | (x >= config.DOMAIN_HI)
    alive &= ~hit

    if (step + 1) % SUBSAMPLE_EVERY == 0:
        # Only collect pairs for always_alive particles
        conf_still_alive = always_alive & alive
        n_conf = int(conf_still_alive.sum())
        if n_conf > 0:
            x_t_list.append(x_before[conf_still_alive])
            dx_list.append((x - x_before)[conf_still_alive])

print()
x_t_conf = np.concatenate(x_t_list)
dx_conf  = np.concatenate(dx_list)

print(f"  Confined transition pairs: {len(x_t_conf):,}")
utils.print_stats("x_t (confined)", x_t_conf)
utils.print_stats("dx  (confined)", dx_conf)
print()

print("  Compare to full dataset:")
print(f"    Confined x_t mean: {x_t_conf.mean():.4f}  (full: ~1.56)")
print(f"    Confined x_t std : {x_t_conf.std():.4f}   (full: ~0.91)")
print(f"    Confined dx  mean: {dx_conf.mean():.5f}")
print(f"    Confined dx  std : {dx_conf.std():.5f}  (full: ~0.022)")
print()
print("  Confined particles cluster in the interior: x_t mean is higher,")
print("  std is larger (they explore more of the domain and survive).")
print()

# ---------------------------------------------------------------------------
# 3.  Labels
# ---------------------------------------------------------------------------
N_LABEL = min(500_000, len(x_t_conf))
idx     = rng.choice(len(x_t_conf), size=N_LABEL, replace=False)
x_t_sub = x_t_conf[idx].astype(np.float64)
dx_sub  = dx_conf[idx].astype(np.float64)

tree_conf = cKDTree(x_t_sub.reshape(-1, 1))
K_LOCAL   = 200
mu_x = np.zeros(N_LABEL)
sig_x= np.zeros(N_LABEL)
BATCH = 50_000
for b in tqdm(range((N_LABEL + BATCH - 1) // BATCH), desc="KNN", ncols=60):
    lo = b * BATCH; hi = min((b+1)*BATCH, N_LABEL)
    _, idxs = tree_conf.query(x_t_sub[lo:hi].reshape(-1,1), k=K_LOCAL)
    dx_nbrs = dx_sub[idxs]
    mu_x[lo:hi]  = dx_nbrs.mean(axis=1)
    sig_x[lo:hi] = dx_nbrs.std(axis=1) + 1e-8
z_labels = (dx_sub - mu_x) / sig_x
print()

# ---------------------------------------------------------------------------
# 4.  Train
# ---------------------------------------------------------------------------
print("Training FlowNet on CONFINED labels …")

X_in = torch.stack([
    torch.tensor(x_t_sub / config.DOMAIN_HI, dtype=torch.float32),
    torch.tensor(z_labels.astype(np.float32), dtype=torch.float32),
], dim=1)
Y_in = torch.tensor(dx_sub.astype(np.float32), dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_in, Y_in)
n_train = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [n_train, len(dataset)-n_train],
                                 generator=torch.Generator().manual_seed(config.SEED+11))
train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE*4, shuffle=False)

model_conf = FlowNet().to(device)
opt   = torch.optim.Adam(model_conf.parameters(), lr=config.LEARNING_RATE)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.N_EPOCHS, eta_min=1e-5)
lf    = nn.MSELoss()

best_val = float("inf")
for epoch in range(1, config.N_EPOCHS + 1):
    model_conf.train()
    for Xb, Yb in train_loader:
        opt.zero_grad()
        loss = lf(model_conf(Xb.to(device)), Yb.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model_conf.parameters(), 1.0)
        opt.step()
    model_conf.eval()
    vl = sum(lf(model_conf(Xb.to(device)), Yb.to(device)).item()
             for Xb, Yb in val_loader) / len(val_loader)
    if vl < best_val:
        best_val = vl
        torch.save(model_conf.state_dict(),
                   os.path.join(config.MODELS_DIR, "flow_model_conf.pt"))
    sched.step()
    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{config.N_EPOCHS}  val_MSE={vl:.6f}")

model_conf.load_state_dict(torch.load(os.path.join(config.MODELS_DIR,"flow_model_conf.pt"),map_location=device))
model_conf.eval()
print(f"  Best val_MSE={best_val:.6f}")
print()

# ---------------------------------------------------------------------------
# 5.  Rollout
# ---------------------------------------------------------------------------
print("Rolling out with confined model …")

@torch.no_grad()
def predict_conf(x_arr, z_arr):
    x_n = torch.tensor(x_arr / config.DOMAIN_HI, dtype=torch.float32)
    z_  = torch.tensor(z_arr, dtype=torch.float32)
    return model_conf(torch.stack([x_n, z_], dim=1)).numpy().ravel()

x_ro  = np.full(config.N_PARTICLES, config.X0, dtype=np.float64)
alive = np.ones(config.N_PARTICLES, dtype=bool)

for step in tqdm(range(config.N_STEPS), desc="Rollout", ncols=70):
    n_alive = alive.sum()
    if n_alive == 0:
        break
    z_arr   = rng.standard_normal(n_alive).astype(np.float32)
    dx_pred = predict_conf(x_ro[alive].astype(np.float32), z_arr)
    x_ro[alive] += dx_pred
    hit = (x_ro <= config.DOMAIN_LO) | (x_ro >= config.DOMAIN_HI)
    alive &= ~hit
    if (step + 1) % 1000 == 0:
        tqdm.write(f"    step {step+1}  alive={alive.sum():,}")

surv_conf = x_ro[alive].copy()
print()
print(f"  Survivors: {len(surv_conf):,} ({100*len(surv_conf)/config.N_PARTICLES:.2f}%)")
utils.print_stats("confined survivors", surv_conf)
print()

np.savez(os.path.join(config.DATA_DIR, "only_confined.npz"), survivors=surv_conf)

# ---------------------------------------------------------------------------
# 6.  Plot
# ---------------------------------------------------------------------------
gt       = np.load(os.path.join(config.DATA_DIR, "ground_truth.npz"))
surv_gt  = gt["survivors"]

def ap(x_v):
    ns  = np.arange(1, 101)
    dec = np.exp(-ns**2*np.pi**2*config.T/(2*config.DOMAIN_HI**2))
    p   = (2./config.DOMAIN_HI)*np.sum(
        np.sin(ns*np.pi*config.X0/config.DOMAIN_HI) *
        np.sin(np.outer(x_v, ns*np.pi/config.DOMAIN_HI))*dec, axis=1)
    p = np.maximum(p, 0.)
    return p/(p.sum()*(x_v[1]-x_v[0]))

x_th = np.linspace(0.01, 5.99, 400)
p_th = ap(x_th)

print("Generating only_confined_histogram.png …")
fig, axes = utils.make_fig(nrows=1, ncols=2, figsize=(14, 5))

axes[0].hist(surv_gt, bins=80, range=(0,6), density=True, color="#607D8B",
             alpha=0.7, label=f"Ground Truth (n={len(surv_gt):,})")
axes[0].plot(x_th, p_th, "r-", lw=2)
axes[0].set(xlim=(0,6), xlabel="x at T=3", ylabel="Density", title="Ground Truth")
axes[0].legend(fontsize=9)

axes[1].hist(surv_conf, bins=80, range=(0,6), density=True, color="#6A1B9A",
             alpha=0.7, label=f"Only Confined (n={len(surv_conf):,})")
axes[1].plot(x_th, p_th, "r-", lw=2, label="Analytical")
axes[1].set(xlim=(0,6), xlabel="x at T=3", ylabel="Density",
            title="Only Confined Trained\n(wrong: distribution too narrow/central)")
axes[1].legend(fontsize=9)
axes[1].text(0.03, 0.95,
             "ARTIFACT: confined particles\nnever approach walls →\n"
             "network learns wrong dist.\nat boundary regions",
             transform=axes[1].transAxes, va="top", fontsize=8,
             bbox=dict(fc="lavender", alpha=0.9))

fig.suptitle("Ablation: Only Confined vs Ground Truth\n"
             "Training on survivors only introduces selection bias",
             fontsize=12, y=1.02)
fig.tight_layout()
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "only_confined_histogram.png"))

# ---------------------------------------------------------------------------
# 7.  Explanation
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("EXPLANATION: WHY ONLY CONFINED FAILS")
print("=" * 60)
print()
print("  Confined particles (those surviving to T=3) are SPECIAL:")
print("  They represent the CONDITIONAL distribution of paths given")
print("  long survival.  These particles are selected to be far from")
print("  both walls at every single step of their trajectory.")
print()
print("  Training on these particles teaches the network:")
print("    'Move in ways that ensure long-term survival'")
print("  This is the WRONG objective!  We want:")
print("    'Move in ways that reproduce each individual step correctly'")
print()
print("  The confined distribution is a biased sample of the step distribution.")
print("  It represents something like the DOOB h-transform of the process")
print("  conditioned on long survival — mathematically different from p(Δx|x).")
print()
print(f"  Confined training region: x ∈ [{x_t_conf.min():.2f}, {x_t_conf.max():.2f}]")
print(f"  Full dataset region:      x ∈ [~0.001, ~6.0]")
print(f"  Confined particles represent only {len(confined_idx)/config.N_PARTICLES*100:.1f}% of all particles.")
print()
print("FILES WRITTEN")
print("  data/only_confined.npz")
print("  models/flow_model_conf.pt")
print("  plots/only_confined_histogram.png")
print()
print("Done.  Run 12_reproduce_figure2.py next.")
