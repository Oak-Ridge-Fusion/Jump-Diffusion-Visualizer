"""
05_probability_flow_ode.py
==========================
PURPOSE
-------
Use the KNN score estimator (script 04) to reverse the forward diffusion
and generate NEW samples from the increment distribution p(Δx | x_t = x).

This is the "training-free" generative model of the paper.

THE PROBABILITY-FLOW ODE
-------------------------
Song et al. (2021) show that every diffusion SDE has a corresponding
deterministic ODE that has the SAME marginal distributions p_t(z).

For the VP-SDE:
    forward SDE:   dZ = -½ β(t) Z dt + √β(t) dW_t

The probability-flow ODE (PF-ODE) is:
    dZ/dt = -½ β(t) [Z  +  ∇_z log p_t(Z)]

We integrate BACKWARDS in t (from t=1 → t_min) to recover Z_0 from Z_1.

WHY THE ODE BECOMES STIFF NEAR t=0
-------------------------------------
As t→0 the distribution p_t(z) becomes very narrow (std → σ₀ ≈ 0.022).
The score ∇_z log p_t(z) = -(z-μ)/σ₀² becomes enormous (~O(1/σ₀²)=2000).
A fixed-step Euler integrator will blow up unless the step is tiny.

FIX: stop at t_min=0.1, then apply the TWEEDIE FORMULA:
    E[Z_0 | Z_t] = (Z_t + σ²(t) · s(Z_t, t)) / α(t)

This gives the MMSE estimate of Z_0 given Z_t without integrating
through the stiff region near t=0.

CRITICAL NORMALISATION NOTE
-----------------------------
The KDTree must be built in the NATURAL scale of (x_t, Z_t):
    x_scale = std(x_t_ref)         ≈ 0.9
    z_scale = std(Z_t_ref at t)    ≈ σ(t)  (depends on t!)

Using dx.std() as z_scale (a common mistake) inflates z coordinates by
~45×, making the tree useless and the score astronomically large.

COMPLETE SAMPLING ALGORITHM
-----------------------------
Given:
  - current SDE position x
  - training dataset {(xᵢ, ΔXᵢ)}

Do:
  1. Sample  Z_1 ~ N(0, 1)                            (start from noise)
  2. For t from 1 down to t_min in steps of -Δt:
       a. Compute β(t), α(t), σ(t)
       b. Forward-noise reference dx → Z_t using this t
       c. Build KDTree in normalised (x / x_scale, Z_t / σ(t)) space
       d. Find K nearest neighbours to (x, Z_curr) in this space
       e. Compute score in unnormalised Z_t space
       f. Euler step:  Z_{t-Δt} = Z_t + Δt · ½β(t)·[Z_t + score]
  3. Apply Tweedie at t=t_min:
       Z_0 ≈ (Z_{t_min} + σ²(t_min)·score) / α(t_min)

OUTPUT
------
plots/ode_reverse_trajectories.png
plots/ode_recovered_increments.png
plots/alpha_sigma_schedule.png
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

import config
import utils

# ---------------------------------------------------------------------------
# 0.  Setup
# ---------------------------------------------------------------------------
config.ensure_dirs()
rng = np.random.default_rng(config.SEED + 4)

print("=" * 60)
print("STEP 05 — Probability-Flow ODE (Reverse Diffusion)")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1.  Load training data
# ---------------------------------------------------------------------------
print("Loading dataset …")
data     = np.load(os.path.join(config.DATA_DIR, "dataset.npz"))
x_t_all  = data["x_t"].astype(np.float64)
dx_all   = data["dx"].astype(np.float64)

N_REF = 100_000
idx   = rng.choice(len(dx_all), size=N_REF, replace=False)
x_ref = x_t_all[idx]
dx_ref = dx_all[idx]

x_scale = x_ref.std()   # ≈ 0.9 (natural scale of SDE positions)
print(f"  Reference set : {N_REF:,} pairs")
print(f"  x_scale       : {x_scale:.4f}")
print()

# ---------------------------------------------------------------------------
# 2.  Helper functions
# ---------------------------------------------------------------------------

def beta_t(t):
    """VP-SDE β(t) = β_min + t·(β_max - β_min)."""
    return config.BETA_MIN + t * (config.BETA_MAX - config.BETA_MIN)


def knn_score(x_query, z_curr, x_ref, dx_ref, t, k, rng_local,
              score_clip=10.0):
    """
    Estimate ∇_z log p_t(z | x) at (x_query, z_curr) using KNN.

    Key design:
      - Build Z_t from dx_ref using the VP forward process
      - Normalise both x and Z_t by their natural std at this t
      - Find K nearest neighbours in this normalised joint space
      - Compute score in the UNNORMALISED Z_t space

    Parameters
    ----------
    x_query    : float — conditioning SDE position
    z_curr     : float — current Z_t value
    x_ref      : (N,) reference SDE positions
    dx_ref     : (N,) reference increments (Z_0 samples)
    t          : float in [0,1] — diffusion time
    k          : int — number of neighbours
    rng_local  : RNG for forward-noising the reference
    score_clip : float — clip score to [-score_clip, +score_clip]

    Returns
    -------
    score : float — estimated ∇_z log p_t(z | x)
    """
    a, s = utils.vp_alpha_sigma(np.array([t]), config.BETA_MIN, config.BETA_MAX)
    a, s = float(a[0]), float(s[0])

    # Forward-noise reference dx values to get Z_t samples
    eps    = rng_local.standard_normal(len(dx_ref))
    z_t_ref = a * dx_ref + s * eps     # shape (N,), std ≈ σ(t)

    # Natural scales at this diffusion time
    z_scale_t = max(z_t_ref.std(), 1e-6)   # ≈ σ(t)

    # Build KDTree in normalised joint space: (x_norm, z_norm)
    joint = np.column_stack([x_ref / x_scale,
                              z_t_ref / z_scale_t])
    tree  = cKDTree(joint)

    # Query in the same normalised space
    query = np.array([[x_query / x_scale,
                       z_curr  / z_scale_t]])
    _, idxs = tree.query(query, k=k)
    neighbours_unnorm = z_t_ref[idxs[0]]    # K nearest Z_t values (unnormalised)

    # Score in unnormalised Z_t space:
    #   s(z,t) = ∇_z log p_t(z) ≈ (mean_nbrs - z) / var_nbrs
    mu_n  = neighbours_unnorm.mean()
    var_n = neighbours_unnorm.var() + 1e-8
    score = (mu_n - z_curr) / var_n

    # Clip to prevent instability
    return float(np.clip(score, -score_clip, score_clip))


# ---------------------------------------------------------------------------
# 3.  Reverse ODE integrator with Tweedie correction at t_min
# ---------------------------------------------------------------------------

def reverse_ode(x_cond, z1_init, n_steps, k, x_ref, dx_ref, rng,
                t_min=0.1, score_clip=5.0, record_path=False):
    """
    Integrate the reverse PF-ODE from t=1 down to t=t_min,
    then apply the Tweedie MMSE estimator to get Z_0.

    Parameters
    ----------
    x_cond     : float — conditioning SDE position
    z1_init    : float — Z_1 ~ N(0,1) starting noise
    n_steps    : int   — Euler integration steps (from t=1 to t_min)
    k          : int   — KNN neighbours for score
    x_ref, dx_ref : reference dataset
    rng        : RNG
    t_min      : stop ODE here to avoid stiff region near t=0
    score_clip : clip score magnitude to this value
    record_path: if True return (t, z) path

    Returns
    -------
    z0_est  : float — estimated Z_0 ≈ ΔX
    path    : list of (t, z) if record_path else None
    """
    # Integration from t=1 down to t_min
    t_vals = np.linspace(1.0, t_min, n_steps + 1)
    dt     = (1.0 - t_min) / n_steps

    z_curr = z1_init
    path   = [(1.0, z_curr)] if record_path else None

    for step in range(n_steps):
        t_curr = t_vals[step]
        b      = beta_t(t_curr)

        # Score estimate (KNN with proper normalisation)
        sc = knn_score(x_cond, z_curr, x_ref, dx_ref, t_curr, k, rng,
                       score_clip=score_clip)

        # Euler step of reverse ODE:
        #   dZ/dt = -½β[Z + s]  →  Z_{t-dt} = Z_t + dt·½β[Z_t + s]
        drift  = 0.5 * b * (z_curr + sc)
        z_curr = z_curr + dt * drift

        if record_path:
            path.append((t_vals[step + 1], z_curr))

    # Tweedie MMSE at t_min:
    # E[Z_0 | Z_{t_min}] = (Z_{t_min} + σ²(t_min)·s(Z_{t_min}, t_min)) / α(t_min)
    #
    # This bypasses the stiff t→0 region by directly using the score
    # to denoise in one shot.
    a_f, s_f = utils.vp_alpha_sigma(np.array([t_min]),
                                     config.BETA_MIN, config.BETA_MAX)
    a_f, s_f = float(a_f[0]), float(s_f[0])
    sc_final = knn_score(x_cond, z_curr, x_ref, dx_ref, t_min, k, rng,
                         score_clip=score_clip)

    # Tweedie formula: z_0 = (z_t + σ²·score) / α
    z0_est = (z_curr + s_f**2 * sc_final) / max(a_f, 1e-6)

    # Clip to plausible increment range (domain knowledge)
    # A single Euler step can be at most sqrt(DT) * ~5 ≈ 0.11
    max_dx = config.DOMAIN_HI - config.DOMAIN_LO
    z0_est = float(np.clip(z0_est, -max_dx, max_dx))

    return (z0_est, path) if record_path else z0_est


# ---------------------------------------------------------------------------
# 4.  Show the α / σ schedule (for reference)
# ---------------------------------------------------------------------------
print("Generating alpha_sigma_schedule.png …")

t_grid  = np.linspace(0, 1, 500)
alpha_g, sigma_g = utils.vp_alpha_sigma(t_grid, config.BETA_MIN, config.BETA_MAX)

fig0, (ax0a, ax0b) = utils.make_fig(nrows=1, ncols=2, figsize=(12, 5))
ax0a.plot(t_grid, alpha_g, "b-", lw=2, label="α(t) — signal")
ax0a.plot(t_grid, sigma_g, "r-", lw=2, label="σ(t) — noise")
ax0a.axvline(0.1, color="orange", ls="--", lw=1.5, label="t_min=0.1 (stop ODE here)")
ax0a.set(xlabel="t", ylabel="value", title="VP-SDE schedule",
         xlim=(0,1), ylim=(0,1.05))
ax0a.legend(fontsize=9)
snr = alpha_g**2 / (sigma_g**2 + 1e-12)
ax0b.semilogy(t_grid, snr, "purple", lw=2)
ax0b.axvline(0.1, color="orange", ls="--", lw=1.5)
ax0b.set(xlabel="t", ylabel="SNR = α²/σ²  [log]",
         title="Signal-to-Noise Ratio\n(stiff region: SNR>>1 near t=0)",
         xlim=(0,1))
fig0.tight_layout()
utils.save_fig(fig0, os.path.join(config.PLOTS_DIR, "alpha_sigma_schedule.png"))

# ---------------------------------------------------------------------------
# 5.  Demo: record 10 ODE trajectories
# ---------------------------------------------------------------------------
N_ODE_STEPS = 30          # 30 Euler steps from t=1 to t=0.1
K_DEMO      = 40
T_MIN       = 0.1
X_DEMO      = 1.5

print(f"Demo parameters:")
print(f"  x_cond    : {X_DEMO}")
print(f"  ODE steps : {N_ODE_STEPS}  (t: 1.0 → {T_MIN})")
print(f"  KNN k     : {K_DEMO}")
print(f"  score_clip: 5.0")
print()

N_TRAJ = 10
print(f"Recording {N_TRAJ} example ODE trajectories …")
traj_data = []
for i in range(N_TRAJ):
    z1 = rng.standard_normal()
    z0_rec, path = reverse_ode(
        x_cond=X_DEMO, z1_init=z1,
        n_steps=N_ODE_STEPS, k=K_DEMO,
        x_ref=x_ref, dx_ref=dx_ref,
        rng=rng, t_min=T_MIN, record_path=True,
    )
    traj_data.append((z1, z0_rec, path))
    print(f"  traj {i+1:2d}: Z_1={z1:+.4f}  →  Z_0={z0_rec:+.6f}")

print()

# Plot trajectories
print("Generating ode_reverse_trajectories.png …")
fig, ax = utils.make_fig(figsize=(10, 6))
cmap = plt.cm.rainbow
true_dx_near = dx_ref[np.abs(x_ref - X_DEMO) < 0.3]

for k_idx, (z1, z0_rec, path) in enumerate(traj_data):
    color = cmap(k_idx / max(len(traj_data) - 1, 1))
    t_vals_p = [p[0] for p in path]
    z_vals_p = [p[1] for p in path]
    ax.plot(t_vals_p, z_vals_p, color=color, lw=1.8, alpha=0.8)
    ax.plot(1.0,   z1,    "o", color=color, markersize=8)
    ax.plot(T_MIN, z_vals_p[-1], "^", color=color, markersize=8)
    ax.plot(-0.02, z0_rec, "s", color=color, markersize=8)   # Tweedie estimate

if len(true_dx_near) > 10:
    p5, p95 = np.percentile(true_dx_near, [5, 95])
    ax.axhspan(p5, p95, alpha=0.15, color="green",
               label=f"True dx [5,95]pct near x={X_DEMO}")

ax.axvline(1.0,   color="gray", ls="--", lw=1, label="t=1 start")
ax.axvline(T_MIN, color="orange", ls="--", lw=1.5,
           label=f"t={T_MIN} Euler stop → Tweedie")
ax.axvline(0.0,   color="gray", ls="-",  lw=1, label="t=0 (final Z_0)")

ax.set_xlabel("Diffusion time t", fontsize=12)
ax.set_ylabel("Z_t", fontsize=12)
ax.set_title(
    f"Reverse ODE trajectories  (x_cond={X_DEMO},  {N_ODE_STEPS} steps)\n"
    "Euler integration stops at t=0.1, then Tweedie formula gives Z_0\n"
    "Squares at t=0 are the final recovered ΔX estimates",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.set_xlim(-0.05, 1.05)
utils.save_fig(fig, os.path.join(config.PLOTS_DIR, "ode_reverse_trajectories.png"))

# ---------------------------------------------------------------------------
# 6.  Generate many samples and compare to truth
# ---------------------------------------------------------------------------
N_SAMPLES = 200
print(f"Generating {N_SAMPLES} recovered samples …")

z0_samples = np.zeros(N_SAMPLES)
for i in tqdm(range(N_SAMPLES), desc="Reverse ODE", ncols=60):
    z1 = rng.standard_normal()
    z0_samples[i] = reverse_ode(
        x_cond=X_DEMO, z1_init=z1,
        n_steps=N_ODE_STEPS, k=K_DEMO,
        x_ref=x_ref, dx_ref=dx_ref,
        rng=rng, t_min=T_MIN,
    )

print()
utils.print_stats("ODE recovered Z_0", z0_samples)
utils.print_stats("True dx near x=1.5", true_dx_near)
print()

# Plot comparison
print("Generating ode_recovered_increments.png …")
fig2, ax2 = utils.make_fig(figsize=(9, 5))
if len(true_dx_near) > 10:
    ax2.hist(true_dx_near, bins=50, density=True,
             color="#43A047", alpha=0.6, label=f"True dx near x={X_DEMO}  (n={len(true_dx_near):,})")
ax2.hist(z0_samples, bins=25, density=True,
         color="#1E88E5", alpha=0.7,
         label=f"ODE recovered Z_0  (n={N_SAMPLES})")
ax2.set_xlabel(r"$\Delta X$", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title(
    f"Recovered increments vs. true distribution  (x≈{X_DEMO})\n"
    f"{N_SAMPLES} samples, {N_ODE_STEPS} ODE steps + Tweedie correction",
    fontsize=10,
)
ax2.legend(fontsize=10)
utils.save_fig(fig2, os.path.join(config.PLOTS_DIR, "ode_recovered_increments.png"))

# ---------------------------------------------------------------------------
# 7.  Diagnostics
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
z0_finite = z0_samples[np.isfinite(z0_samples)]
print(f"  Samples generated : {N_SAMPLES}")
print(f"  Finite samples    : {len(z0_finite)}")
print()
if len(true_dx_near) > 0:
    print(f"  True dx near x={X_DEMO}:")
    utils.print_stats("  true", true_dx_near)
print(f"  Recovered Z_0:")
utils.print_stats("  recovered", z0_finite if len(z0_finite) > 0 else z0_samples)
print()
print("  NOTE: The ODE+KNN approach is slow (~1 sample/s) and approximate.")
print("  Its purpose in this pipeline is to generate LABELS for the neural")
print("  network (script 06→07), not to be used directly at rollout time.")
print()
print("FILES WRITTEN")
print("  plots/alpha_sigma_schedule.png")
print("  plots/ode_reverse_trajectories.png")
print("  plots/ode_recovered_increments.png")
print()
print("WHAT TO UNDERSTAND FROM THE PLOTS")
print()
print("  ode_reverse_trajectories.png:")
print("    Circles (t=1): random noise starting points.")
print("    Triangles (t=0.1): where Euler integration stops.")
print("    Squares (t=0): Tweedie-corrected final ΔX estimate.")
print("    The trajectories should converge from spread-1 at t=1")
print("    toward the narrow green band (true dx range) at t=0.")
print()
print("  ode_recovered_increments.png:")
print("    Green = true distribution of ΔX near x=1.5.")
print("    Blue = what the ODE+KNN produces.")
print("    With 200 samples the match is rough but the center and")
print("    width should be approximately correct.")
print()
print("Done.  Run 06_generate_labels.py next.")
