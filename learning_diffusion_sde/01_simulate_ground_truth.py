"""
01_simulate_ground_truth.py
===========================
PURPOSE
-------
Simulate the "ground truth" distribution that our generative model must
learn to reproduce.  Everything downstream is measured against this.

THE SDE
-------
We study the simplest possible SDE:

    dX_t = dW_t,    X_0 = 1,    0 < X_t < 6   (absorbing boundaries)

  - dW_t  is a Wiener increment: W_{t+dt} - W_t ~ N(0, dt)
  - "Absorbing boundary" means: if a particle exits [0, 6], it is
    permanently removed from the ensemble.  It does not bounce back.

WHY THIS SDE?
-------------
It is pure Brownian motion.  The drift is zero, the diffusion coefficient
is 1.  Despite this simplicity, the absorbing boundaries make the
*conditional* distribution at time T non-trivial.  The paper uses this as
a clean test case before tackling more complex SDEs.

EULER-MARUYAMA DISCRETISATION
------------------------------
We cannot integrate dW exactly (it is nowhere differentiable), so we use
the Euler-Maruyama scheme:

    X_{n+1} = X_n  +  sqrt(dt) * Z_n,    Z_n ~ N(0,1)

This is exact for additive noise SDEs (no discretisation error in the
distribution beyond floating-point precision).

    dt = 5e-4,  T = 3  →  N_steps = T / dt = 6000 steps

ABSORBING BOUNDARY HANDLING
-----------------------------
After every Euler step we check:

    alive[i]  =  (X_i > 0) AND (X_i < 6)

Particles that exit are flagged as dead and excluded from all future
updates.  At the end we collect the positions of the survivors.

OUTPUT FILES
------------
data/ground_truth.npz
    - "survivors"   : shape (n_alive,)   positions at t=T of alive particles
    - "all_final"   : shape (N_PARTICLES,) final positions (nan for dead)
    - "trajectories": shape (500, N_steps+1) — 500 randomly sampled paths
                      (nan after absorption)

plots/ground_truth_histogram.png
    Empirical PDF of survivors vs. the analytical solution (see below).

plots/trajectory_examples.png
    500 example paths.  Dead paths stop at their absorption time.

ANALYTICAL SOLUTION
-------------------
For 1-D Brownian motion with absorbing walls at 0 and L, the survival
PDF at time T, conditioned on survival, is a series of Fourier modes:

    p(x,T | x0, survived) ∝ sum_{n=1}^{inf} sin(n*pi*x0/L) * sin(n*pi*x/L)
                              * exp(-n^2 * pi^2 * T / (2*L^2))

(This is the Green's function of the heat equation with Dirichlet BCs.)
We overlay this on the histogram so you can see how well the simulation
matches theory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))   # find config, utils

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
rng = np.random.default_rng(config.SEED)

print("=" * 60)
print("STEP 01 — Simulate Ground Truth SDE")
print("=" * 60)
print(f"  Domain          : [{config.DOMAIN_LO}, {config.DOMAIN_HI}]")
print(f"  Start position  : x0 = {config.X0}")
print(f"  Total time T    : {config.T}")
print(f"  Step size dt    : {config.DT}")
print(f"  Steps           : {config.N_STEPS:,}")
print(f"  Particles       : {config.N_PARTICLES:,}")
print()

# ---------------------------------------------------------------------------
# 1.  Initialise all particles at X0
# ---------------------------------------------------------------------------
#
#   x   : current position of each particle         shape (N,)
#   alive : boolean mask — True while inside (0, L)  shape (N,)
#
x     = np.full(config.N_PARTICLES, config.X0, dtype=np.float32)
alive = np.ones(config.N_PARTICLES, dtype=bool)

# ---------------------------------------------------------------------------
# 2.  Choose 500 particles to record full trajectories
# ---------------------------------------------------------------------------
#
# Storing all 200 000 × 6 000 positions would cost ~4.8 GB.
# Instead we record only 500 paths.  We call them "probe" particles.
#
N_PROBE   = 500
probe_idx = rng.choice(config.N_PARTICLES, size=N_PROBE, replace=False)
probe_idx = np.sort(probe_idx)

# trajectories[i, step] = position of probe particle i at the given step
# We pre-fill with NaN; absorbed particles stay NaN from absorption onwards.
trajectories = np.full((N_PROBE, config.N_STEPS + 1), np.nan, dtype=np.float32)
trajectories[:, 0] = x[probe_idx]   # step 0 = initial position

# ---------------------------------------------------------------------------
# 3.  Euler-Maruyama loop
# ---------------------------------------------------------------------------
#
#   At each step:
#     (a) Generate noise only for alive particles (saves computation).
#     (b) Update positions:  X_{n+1} = X_n + sqrt(dt) * N(0,1)
#     (c) Kill any particle that left the domain.
#     (d) Record probe-particle positions.
#
print("Running Euler-Maruyama simulation …")

for step in tqdm(range(config.N_STEPS), unit="step", ncols=70):

    # (a) Draw Gaussian increments for ALL alive particles in one batch.
    #     sqrt(dt) * N(0,1) ~ N(0, dt)  which is the Wiener increment dW.
    noise = rng.standard_normal(size=int(alive.sum())).astype(np.float32)

    # (b) Update only alive particles.
    x[alive] += np.sqrt(config.DT) * noise

    # (c) Absorbing boundary check.
    #     Particles that crossed 0 or 6 become permanently dead.
    hit_boundary = (x <= config.DOMAIN_LO) | (x >= config.DOMAIN_HI)
    alive &= ~hit_boundary   # once dead, stays dead

    # (d) Record probe particles.
    #     Probe particles that are alive get their new position.
    #     Dead probe particles already have NaN in trajectory array.
    current_step = step + 1
    for j, pidx in enumerate(probe_idx):
        if alive[pidx]:
            trajectories[j, current_step] = x[pidx]

    # Print a progress update every 1000 steps.
    if (step + 1) % 1000 == 0:
        pct_alive = 100 * alive.sum() / config.N_PARTICLES
        tqdm.write(f"    step {step+1:5d}/{config.N_STEPS}  "
                   f"alive: {alive.sum():,} ({pct_alive:.1f}%)")

print()

# ---------------------------------------------------------------------------
# 4.  Collect surviving positions
# ---------------------------------------------------------------------------
survivors = x[alive].copy()     # positions at t=T of particles still inside

print(f"Simulation complete.")
print(f"  Total particles  : {config.N_PARTICLES:,}")
print(f"  Survivors at T=3 : {len(survivors):,}  "
      f"({100*len(survivors)/config.N_PARTICLES:.2f}% survival rate)")
print()
utils.print_stats("survivors", survivors)
print()

# ---------------------------------------------------------------------------
# 5.  Save data
# ---------------------------------------------------------------------------
save_path = os.path.join(config.DATA_DIR, "ground_truth.npz")
all_final = x.copy()
all_final[~alive] = np.nan   # mark dead particles as NaN

np.savez(
    save_path,
    survivors   = survivors,
    all_final   = all_final,
    trajectories= trajectories,
    alive_mask  = alive,
)
print(f"Data saved → {save_path}")
print()

# ---------------------------------------------------------------------------
# 6.  Analytical solution (Fourier series)
# ---------------------------------------------------------------------------
#
# The survival PDF p(x, T) for Brownian motion with Dirichlet BCs at 0 and L
# is derived by solving the Fokker-Planck (= heat) equation:
#
#     ∂p/∂t = (1/2) ∂²p/∂x²,    p(0,t) = p(L,t) = 0,   p(x,0) = δ(x - x0)
#
# Solution via separation of variables:
#
#     p(x, t) = (2/L) Σ_{n=1}^{∞}  sin(n π x0 / L) · sin(n π x / L)
#                                    · exp(-n² π² t / (2 L²))
#
# We truncate at N_TERMS=100 (higher modes decay exponentially fast).
#
def analytical_pdf(x_vals, x0, L, t, N_terms=100):
    """Evaluate the survival PDF at positions x_vals for BM with absorbing walls."""
    ns   = np.arange(1, N_terms + 1)         # n = 1, 2, ..., N_terms
    decay = np.exp(-ns**2 * np.pi**2 * t / (2 * L**2))   # shape (N_terms,)

    # Broadcasting: x_vals is (M,), ns is (N_terms,) → result is (M,)
    phi_x0 = np.sin(ns[None, :] * np.pi * x0 / L)   # (1, N_terms)
    phi_x  = np.sin(np.outer(x_vals, ns * np.pi / L))   # (M, N_terms)

    # Sum over modes
    p = (2.0 / L) * np.sum(phi_x0 * phi_x * decay[None, :], axis=1)
    return np.maximum(p, 0.0)   # PDF is non-negative


x_plot    = np.linspace(config.DOMAIN_LO, config.DOMAIN_HI, 400)
p_exact   = analytical_pdf(x_plot, config.X0, config.DOMAIN_HI, config.T)

# Normalise the exact PDF to match the conditional (survived) distribution.
# The integral of p_exact over [0,L] gives the survival probability P_surv.
dx_plot = x_plot[1] - x_plot[0]
P_surv  = p_exact.sum() * dx_plot
print(f"  Analytical survival probability P_surv = {P_surv:.4f}")
print(f"  Empirical  survival probability         = "
      f"{len(survivors)/config.N_PARTICLES:.4f}")
print()
# Normalise so it integrates to 1 (conditional on survival)
p_exact_cond = p_exact / P_surv

# ---------------------------------------------------------------------------
# 7.  Plot: ground truth histogram
# ---------------------------------------------------------------------------
print("Generating ground_truth_histogram.png …")

fig, ax = utils.make_fig(figsize=(8, 5))

# Empirical histogram of survivors
n_bins = 80
ax.hist(
    survivors,
    bins   = n_bins,
    range  = (config.DOMAIN_LO, config.DOMAIN_HI),
    density= True,
    color  = "#2196F3",
    alpha  = 0.65,
    label  = f"Simulation  (n={len(survivors):,})",
)

# Analytical conditional PDF
ax.plot(x_plot, p_exact_cond, "r-", linewidth=2.5,
        label="Analytical (Fourier series)")

# Cosmetics
ax.set_xlim(config.DOMAIN_LO, config.DOMAIN_HI)
ax.set_xlabel("Position x at T = 3", fontsize=12)
ax.set_ylabel("Probability density", fontsize=12)
ax.set_title("Ground Truth: Brownian motion with absorbing boundaries\n"
             r"$dX_t = dW_t$,  $X_0=1$,  $0 \leq X_t \leq 6$",
             fontsize=11)
ax.legend(fontsize=10)

# Annotate survival rate
ax.text(0.97, 0.95,
        f"Survival rate: {100*len(survivors)/config.N_PARTICLES:.1f}%\n"
        f"N={config.N_PARTICLES:,}  T={config.T}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "ground_truth_histogram.png"))

# ---------------------------------------------------------------------------
# 8.  Plot: trajectory examples
# ---------------------------------------------------------------------------
print("Generating trajectory_examples.png …")

time_axis = np.linspace(0, config.T, config.N_STEPS + 1)

fig, (ax_top, ax_bot) = utils.make_fig(nrows=2, figsize=(12, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})

# --- Top panel: trajectory spaghetti ---
n_show_alive = 0
n_show_dead  = 0
cmap_alive = plt.cm.Blues
cmap_dead  = plt.cm.Reds

for j in range(N_PROBE):
    path = trajectories[j]                        # shape (N_steps+1,)
    valid_mask = ~np.isnan(path)                  # where particle was alive
    t_valid = time_axis[valid_mask]
    x_valid = path[valid_mask]

    if valid_mask[-1]:   # survived to T
        ax_top.plot(t_valid, x_valid, color="#1565C0", alpha=0.25, lw=0.6)
        n_show_alive += 1
    else:                # absorbed before T
        ax_top.plot(t_valid, x_valid, color="#C62828", alpha=0.12, lw=0.5)
        # Mark absorption point with a small dot
        ax_top.plot(t_valid[-1], x_valid[-1], "r.", markersize=2, alpha=0.4)
        n_show_dead += 1

# Draw boundary lines
ax_top.axhline(config.DOMAIN_LO, color="black", lw=1.5, ls="--", label="Absorbing wall (x=0)")
ax_top.axhline(config.DOMAIN_HI, color="black", lw=1.5, ls="-.",  label="Absorbing wall (x=6)")
ax_top.axhline(config.X0,        color="green",  lw=1.0, ls=":",   label=f"Start x₀={config.X0}", alpha=0.7)

ax_top.set_xlim(0, config.T)
ax_top.set_ylim(-0.3, config.DOMAIN_HI + 0.3)
ax_top.set_xlabel("Time t", fontsize=11)
ax_top.set_ylabel("Position X_t", fontsize=11)
ax_top.set_title(f"500 example trajectories  "
                 f"(blue=survived, red=absorbed)\n"
                 f"survived: {n_show_alive}, absorbed: {n_show_dead}",
                 fontsize=11)
ax_top.legend(loc="upper right", fontsize=8)

# --- Bottom panel: survival fraction over time ---
# For each time step, count how many probe particles are still alive.
alive_count = np.sum(~np.isnan(trajectories), axis=0)   # shape (N_steps+1,)
ax_bot.plot(time_axis, alive_count / N_PROBE * 100, color="#1B5E20", lw=1.5)
ax_bot.set_xlim(0, config.T)
ax_bot.set_xlabel("Time t", fontsize=11)
ax_bot.set_ylabel("% alive", fontsize=11)
ax_bot.set_title("Fraction of probe particles still alive", fontsize=10)
ax_bot.set_ylim(0, 105)

fig.tight_layout(pad=2.0)
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "trajectory_examples.png"))

# ---------------------------------------------------------------------------
# 9.  Summary diagnostics
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print(f"  Survivors              : {len(survivors):,} / {config.N_PARTICLES:,}")
print(f"  Survival rate          : {100*len(survivors)/config.N_PARTICLES:.3f}%")
print(f"  Analytical P_surv      : {P_surv:.4f}")
print()
print("  Distribution of survivors:")
utils.print_stats("survivors", survivors)
print()
print("  Percentiles of survivor positions:")
for pct in [5, 25, 50, 75, 95]:
    print(f"    {pct:3d}th percentile : {np.percentile(survivors, pct):.4f}")
print()
print("  Probe trajectories:")
n_probe_survived = np.sum(~np.isnan(trajectories[:, -1]))
print(f"    Probe survived : {n_probe_survived} / {N_PROBE}")
print(f"    Probe absorbed : {N_PROBE - n_probe_survived} / {N_PROBE}")
print()
print("FILES WRITTEN")
print(f"  data/ground_truth.npz")
print(f"  plots/ground_truth_histogram.png")
print(f"  plots/trajectory_examples.png")
print()
print("WHAT TO CHECK IN THE PLOTS")
print("  ground_truth_histogram.png:")
print("    - Histogram should be roughly bell-shaped, centred near x=2-3.")
print("    - Particles near x=0 and x=6 should be scarce (absorbing walls).")
print("    - Red analytical curve should lie EXACTLY on top of the histogram.")
print("    - If they don't match, check DT (smaller = more accurate).")
print()
print("  trajectory_examples.png:")
print("    - Blue paths wander freely, red paths get absorbed at a wall.")
print("    - Survival fraction (bottom) should decrease monotonically.")
print("    - At t=3 the survival fraction should match the survival rate above.")
print()
print("Done.  Run 02_build_dataset.py next.")
