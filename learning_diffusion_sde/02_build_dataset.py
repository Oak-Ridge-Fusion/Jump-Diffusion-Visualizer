"""
02_build_dataset.py
===================
PURPOSE
-------
Turn the raw SDE trajectories from script 01 into supervised training pairs
that a generative model (scripts 03-07) will learn to reproduce.

THE KEY QUESTION: What should we teach the network?
----------------------------------------------------
We have trajectories X_{t0}, X_{t0+dt}, X_{t0+2dt}, ...

We want to learn the *step map*:  given X at time t, predict X at time t+Δt.

Option A: learn  X_t → X_{t+Δt}         ("predict next position")
Option B: learn  X_t → ΔX = X_{t+Δt} - X_t   ("predict increment")

WHY OPTION B IS BETTER (and what the paper uses)
-------------------------------------------------
For Brownian motion, dX = dW.  Over one step of size Δt:

    ΔX = X_{t+Δt} - X_t  ≈  sqrt(Δt) * N(0,1)

The increment is MUCH SMALLER than the position (X ~ O(1), ΔX ~ O(sqrt(Δt))).
More importantly:

1. STATIONARITY: ΔX does not depend on t (Brownian increments are
   stationary).  The network only needs to learn one distribution, not
   one per time slice.

2. CONDITIONING: ΔX does depend on x (because of the boundary — if you are
   near x=0 the increment can only be positive to avoid absorption).  So
   the network learns G(x) ≈ conditional increment given current position.

3. VARIANCE: the increment variance is tiny (dt=5e-4, so σ≈0.022).
   This is much easier for a network to regress than absolute positions
   which span [0,6].

THE TRAINING PAIRS
------------------
For each step n of each trajectory that:
  (a) started inside the domain at step n
  (b) survived to step n+1 (was still inside AFTER the step)

we record:
    x_t    = X at step n          (current position)
    dx     = X at step n+1 - X_n  (increment, what we want to learn)

We only keep "clean" transitions: both endpoints inside (0, 6).
This is the KEY DIFFERENCE from the ablation conditions (scripts 10 and 11).

IMPORTANT NOTE ON SUBSAMPLING
------------------------------
We have 200,000 particles × 6,000 steps = 1.2 billion potential pairs.
Storing all of them would require ~9 GB.  Instead we subsample:
  - We pick every SUBSAMPLE_EVERY-th time step (= decimation in time).
  - This gives us N_PARTICLES * (N_STEPS / SUBSAMPLE_EVERY) pairs.

The step-to-step correlation in a Brownian path is zero (increments are
i.i.d.), so skipping steps loses no information about the increment
distribution.

OUTPUT
------
data/dataset.npz
    "x_t"  : shape (M,)  — position at start of step
    "dx"   : shape (M,)  — increment  (target for the network)

plots/increment_histogram.png
    Distribution of ΔX — should be N(0, dt) = N(0, 5e-4).

plots/scatter_x_vs_dx.png
    Scatter of (x_t, ΔX) — shows how the increment distribution
    changes near the boundaries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
import utils

# ---------------------------------------------------------------------------
# 0.  Setup
# ---------------------------------------------------------------------------
config.ensure_dirs()
rng = np.random.default_rng(config.SEED + 1)   # different seed from script 01

print("=" * 60)
print("STEP 02 — Build Training Dataset")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1.  Re-run a SHORTER simulation to build the dataset
# ---------------------------------------------------------------------------
#
# Strategy: we re-simulate but now record ALL transitions (not just 500
# probe trajectories).  We use a coarser time resolution (SUBSAMPLE_EVERY
# steps) to stay within memory limits.
#
# WHY NOT just use the saved trajectories from script 01?
# Because we only saved 500 probe trajectories there (to keep the file
# small).  For training we need millions of (x, dx) pairs.
#
# SUBSAMPLE_EVERY: take a transition every this many steps.
# At dt=5e-4, skipping 10 steps means we collect at interval 5e-3.
# The increment is then  ΔX = sum_{k=0}^{9} sqrt(dt)*Z_k  ~ N(0, 10*dt).
# But we still model it as a single-step increment — the network just
# learns the aggregate.  The paper uses individual steps; we subsample
# for memory reasons and note this in comments.
#
SUBSAMPLE_EVERY = 10   # collect a pair every 10 Euler steps
BIG_DT = config.DT * SUBSAMPLE_EVERY   # effective dt for one collected pair

print(f"Simulation parameters:")
print(f"  N_PARTICLES     : {config.N_PARTICLES:,}")
print(f"  N_STEPS         : {config.N_STEPS:,}")
print(f"  SUBSAMPLE_EVERY : {SUBSAMPLE_EVERY}")
print(f"  Collected pairs/particle (approx): {config.N_STEPS // SUBSAMPLE_EVERY}")
print(f"  Effective BIG_DT: {BIG_DT:.4f}")
print()

# Arrays to collect training data (pre-allocated with generous size)
max_pairs = config.N_PARTICLES * (config.N_STEPS // SUBSAMPLE_EVERY + 1)
x_t_buf = np.empty(max_pairs, dtype=np.float32)
dx_buf  = np.empty(max_pairs, dtype=np.float32)
pair_count = 0

# Current positions and alive flags
x     = np.full(config.N_PARTICLES, config.X0, dtype=np.float32)
alive = np.ones(config.N_PARTICLES, dtype=bool)

print("Running simulation and collecting pairs …")
for step in tqdm(range(config.N_STEPS), unit="step", ncols=70):

    # Record positions BEFORE the step (for alive particles)
    if step % SUBSAMPLE_EVERY == 0:
        x_before = x[alive].copy()

    # Euler-Maruyama step
    n_alive = int(alive.sum())
    noise   = rng.standard_normal(size=n_alive).astype(np.float32)
    x[alive] += np.sqrt(config.DT) * noise

    # Absorbing boundary
    hit = (x <= config.DOMAIN_LO) | (x >= config.DOMAIN_HI)
    alive &= ~hit

    # Collect pair every SUBSAMPLE_EVERY steps
    if (step + 1) % SUBSAMPLE_EVERY == 0:
        # We want pairs where BOTH start and end are inside.
        # Particles that were alive at the start of this block and
        # are still alive at the end give us clean interior-to-interior pairs.
        #
        # "alive_after_block" is the subset of x_before particles that
        # survived all SUBSAMPLE_EVERY sub-steps.
        #
        # We identify them by re-computing the alive mask for those
        # exact particles.  Since we stored x_before (positions before
        # the block) and we know alive is now updated, we need to track
        # them explicitly.
        #
        # Simpler approach: we track alive BEFORE and after.
        # alive_start = alive BEFORE the block → we stored x_before for them
        # alive_end   = alive AFTER  the block
        #
        # The intersection (started alive AND ended alive) gives clean pairs.
        # x_after for these is just x[originally_alive & still_alive].
        #
        # Because we already updated x[alive] in place we can read
        # x[alive] for the end positions.  But x_before was the FULL alive
        # mask at step, so we need the indices.
        #
        # SIMPLER IMPLEMENTATION: restart tracking each block.
        # (The code below uses a per-block snapshot.)
        pass

# ---------------------------------------------------------------------------
# Cleaner two-pass approach: sample pairs properly
# ---------------------------------------------------------------------------
# Re-simulate cleanly using a snapshot-based approach.
#
print()
print("Re-running with snapshot-based pair collection …")

x_t_list = []
dx_list  = []

x     = np.full(config.N_PARTICLES, config.X0, dtype=np.float32)
alive = np.ones(config.N_PARTICLES, dtype=bool)

for step in tqdm(range(config.N_STEPS), unit="step", ncols=70):

    # Snapshot of alive particles BEFORE this step
    alive_before = alive.copy()
    x_before_all = x.copy()

    # Euler-Maruyama step for alive particles
    n_alive = int(alive.sum())
    noise   = rng.standard_normal(size=n_alive).astype(np.float32)
    x[alive] += np.sqrt(config.DT) * noise

    # Kill particles that exited
    hit   = (x <= config.DOMAIN_LO) | (x >= config.DOMAIN_HI)
    alive &= ~hit

    # Collect at every SUBSAMPLE_EVERY step
    if (step + 1) % SUBSAMPLE_EVERY == 0:
        # "clean" = alive both before AND after the step
        clean = alive_before & alive   # shape (N,), boolean
        n_clean = int(clean.sum())
        if n_clean > 0:
            x_t_list.append(x_before_all[clean])       # position before step
            dx_list.append((x - x_before_all)[clean])  # increment

            if (step + 1) % 1000 == 0:
                tqdm.write(f"    step {step+1:5d}  alive={alive.sum():,}  "
                           f"clean pairs this block={n_clean:,}")

print()
print("Concatenating pairs …")
x_t_arr = np.concatenate(x_t_list)
dx_arr  = np.concatenate(dx_list)

print(f"  Total training pairs : {len(x_t_arr):,}")
print()
utils.print_stats("x_t (start position)", x_t_arr)
utils.print_stats("dx  (increment)",       dx_arr)
print()

# Theoretical properties of dx:
# Each collected dx is the sum of SUBSAMPLE_EVERY individual steps:
#   dx ~ N(0, SUBSAMPLE_EVERY * dt) = N(0, BIG_DT)
# Standard deviation should be sqrt(BIG_DT).
print(f"  Expected dx std (theory) : {np.sqrt(BIG_DT):.6f}")
print(f"  Measured dx std          : {dx_arr.std():.6f}")
print(f"  Ratio (should be ~1.0)   : {dx_arr.std() / np.sqrt(BIG_DT):.4f}")
print()

# ---------------------------------------------------------------------------
# 2.  Save dataset
# ---------------------------------------------------------------------------
save_path = os.path.join(config.DATA_DIR, "dataset.npz")
np.savez(save_path, x_t=x_t_arr, dx=dx_arr, big_dt=np.array(BIG_DT))
print(f"Dataset saved → {save_path}")
print(f"  File size: {os.path.getsize(save_path) / 1e6:.1f} MB")
print()

# ---------------------------------------------------------------------------
# 3.  Plot: increment histogram
# ---------------------------------------------------------------------------
#
# WHAT TO LOOK FOR:
#   - The histogram should be a very narrow Gaussian centred at 0.
#   - std ≈ sqrt(BIG_DT) ≈ 0.0707
#   - The distribution should be the SAME regardless of x_t (for interior
#     particles, Brownian increments are i.i.d.).
#   - Near the boundaries the distribution will be TRUNCATED (particles
#     that went outside were removed), so the tails are cut.
#
print("Generating increment_histogram.png …")

fig, axes = utils.make_fig(nrows=1, ncols=2, figsize=(14, 5))
ax_left, ax_right = axes

# --- Left: overall increment distribution ---
# Subsample for speed (full 100M pairs would be slow to histogram)
idx_sample = rng.choice(len(dx_arr), size=min(500_000, len(dx_arr)), replace=False)
dx_sample  = dx_arr[idx_sample]

n_bins = 120
ax_left.hist(
    dx_sample,
    bins   = n_bins,
    density= True,
    color  = "#7B1FA2",
    alpha  = 0.7,
    label  = f"Empirical  (n={len(dx_sample):,})",
)

# Overlay theoretical Gaussian N(0, BIG_DT)
dx_range = np.linspace(-4*np.sqrt(BIG_DT), 4*np.sqrt(BIG_DT), 400)
gaussian  = (1 / np.sqrt(2 * np.pi * BIG_DT)) * np.exp(-dx_range**2 / (2*BIG_DT))
ax_left.plot(dx_range, gaussian, "r-", lw=2.5,
             label=rf"Theory $N(0,\,{BIG_DT:.4f})$")

ax_left.set_xlabel(r"Increment $\Delta X$", fontsize=12)
ax_left.set_ylabel("Density", fontsize=12)
ax_left.set_title(r"Distribution of $\Delta X = X_{t+\Delta t} - X_t$" + "\n"
                  "Should match a Gaussian (interior particles)", fontsize=10)
ax_left.legend(fontsize=10)

# --- Right: stratified by x_t ---
# Show that the increment distribution changes near boundaries.
# We split into 4 zones:
#   Zone 1: x in [0.0, 1.5]   — near left wall
#   Zone 2: x in [1.5, 3.0]   — interior
#   Zone 3: x in [3.0, 4.5]   — interior
#   Zone 4: x in [4.5, 6.0]   — near right wall

zones = [
    ([0.0, 1.5], "#E53935", "x ∈ [0, 1.5) — near left wall"),
    ([1.5, 3.0], "#43A047", "x ∈ [1.5, 3.0)"),
    ([3.0, 4.5], "#1E88E5", "x ∈ [3.0, 4.5)"),
    ([4.5, 6.0], "#FB8C00", "x ∈ [4.5, 6.0] — near right wall"),
]

x_t_sample = x_t_arr[idx_sample]

for (lo, hi), color, label in zones:
    mask = (x_t_sample >= lo) & (x_t_sample < hi)
    if mask.sum() < 100:
        continue
    ax_right.hist(
        dx_sample[mask],
        bins   = n_bins,
        density= True,
        color  = color,
        alpha  = 0.55,
        label  = f"{label}  (n={mask.sum():,})",
        range  = (-0.5, 0.5),
    )

# Overlay unconditioned Gaussian for reference
ax_right.plot(dx_range, gaussian, "k--", lw=1.5, label="Theory (unconditioned)")
ax_right.set_xlabel(r"Increment $\Delta X$", fontsize=12)
ax_right.set_ylabel("Density", fontsize=12)
ax_right.set_title(r"$\Delta X$ conditioned on starting position $x_t$" + "\n"
                   "Tails are cut near walls (absorbed particles removed)",
                   fontsize=10)
ax_right.set_xlim(-0.5, 0.5)
ax_right.legend(fontsize=8)

fig.suptitle(
    "Script 02: Training Data — Increment Distribution Analysis\n"
    r"$\Delta X = X_{t+\Delta t} - X_t$,  Δt = " + f"{BIG_DT:.4f}",
    fontsize=12, y=1.01,
)
fig.tight_layout()
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "increment_histogram.png"))

# ---------------------------------------------------------------------------
# 4.  Plot: scatter of x_t vs dx
# ---------------------------------------------------------------------------
#
# WHAT TO LOOK FOR:
#   - The scatter cloud should be roughly horizontal (dx doesn't depend
#     on x in the interior — Brownian motion is position-independent).
#   - But NEAR x=0 the cloud should NOT extend below a certain negative
#     value (a large negative dx would push the particle below 0 and kill it).
#   - Similarly near x=6 there are no large positive dx values.
#   - This boundary effect is EXACTLY what the network must learn.
#
print("Generating scatter_x_vs_dx.png …")

# Subsample heavily for a clean scatter plot
n_scatter = 30_000
idx_sc    = rng.choice(len(dx_arr), size=n_scatter, replace=False)

fig2, ax2 = utils.make_fig(figsize=(10, 6))
sc = ax2.scatter(
    x_t_arr[idx_sc],
    dx_arr[idx_sc],
    c      = x_t_arr[idx_sc],
    cmap   = "viridis",
    s      = 1,
    alpha  = 0.4,
)
plt.colorbar(sc, ax=ax2, label="x_t (current position)")

# Draw ±2σ horizontal reference lines
sigma = np.sqrt(BIG_DT)
ax2.axhline(+2*sigma, color="red", ls="--", lw=1, label=f"+2σ = +{2*sigma:.4f}")
ax2.axhline(-2*sigma, color="red", ls="--", lw=1, label=f"-2σ = -{2*sigma:.4f}")
ax2.axhline(0,        color="white", ls="-",  lw=0.5, alpha=0.5)

# Overlay "forbidden zone" triangles
# Near x=0, particles with dx < -(x_t) would go negative → absorbed.
# So the minimum possible dx given x_t is approximately -x_t (hard boundary).
x_plot2 = np.linspace(0, 6, 300)
ax2.plot(x_plot2, -x_plot2, "cyan",  lw=1.5, label="Lower cutoff: dx > -x_t")
ax2.plot(x_plot2, 6 - x_plot2, "magenta", lw=1.5, label="Upper cutoff: dx < 6-x_t")

ax2.set_xlim(config.DOMAIN_LO, config.DOMAIN_HI)
ax2.set_ylim(-1.0, 1.0)
ax2.set_xlabel("Current position  x_t", fontsize=12)
ax2.set_ylabel(r"Increment  $\Delta X = X_{t+\Delta t} - X_t$", fontsize=12)
ax2.set_title(
    "Scatter: current position vs. increment\n"
    "Note how the cloud is clipped near the walls (absorbed pairs removed)\n"
    "This asymmetry is what the generative model must reproduce",
    fontsize=10,
)
ax2.legend(fontsize=9, loc="upper right")

utils.save_fig(fig2, os.path.join(config.PLOTS_DIR, "scatter_x_vs_dx.png"))

# ---------------------------------------------------------------------------
# 5.  Diagnostics summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print(f"  Training pairs          : {len(x_t_arr):,}")
print(f"  x_t range               : [{x_t_arr.min():.4f}, {x_t_arr.max():.4f}]")
print(f"  dx  mean (expect ≈ 0)   : {dx_arr.mean():.6f}")
print(f"  dx  std  (expect {np.sqrt(BIG_DT):.4f}) : {dx_arr.std():.6f}")
print()
print("  Fraction of pairs by x_t zone:")
for (lo, hi), _, label in zones:
    mask = (x_t_arr >= lo) & (x_t_arr < hi)
    print(f"    {label:40s}: {100*mask.sum()/len(x_t_arr):.1f}%")
print()
print("FILES WRITTEN")
print("  data/dataset.npz")
print("  plots/increment_histogram.png")
print("  plots/scatter_x_vs_dx.png")
print()
print("WHAT TO UNDERSTAND FROM THE PLOTS")
print()
print("  increment_histogram.png (left panel):")
print("    The overall ΔX distribution is nearly Gaussian.  The small")
print("    deviation from the red curve is because the tails are cut")
print("    (particles that would have gone outside are removed).")
print()
print("  increment_histogram.png (right panel):")
print("    Near the left wall (orange) the left tail is missing.")
print("    Near the right wall (blue) the right tail is missing.")
print("    Interior particles (green) match the Gaussian perfectly.")
print("    THIS asymmetry is the core difficulty: the model must learn")
print("    different effective increment distributions depending on x_t.")
print()
print("  scatter_x_vs_dx.png:")
print("    The 'forbidden triangles' in the corners show where absorption")
print("    removes pairs.  If x_t = 0.1, then dx < -0.1 is impossible")
print("    (particle would go to x < 0 and be absorbed).  The cyan line")
print("    marks this hard lower cutoff.  Magenta marks the upper cutoff.")
print()
print("Done.  Run 03_forward_diffusion.py next.")
