"""
04_knn_score_estimation.py
==========================
PURPOSE
-------
Implement the training-free score estimator that is the central
contribution of the paper.

Instead of training a neural network to predict the score
s(z,t) = ∇_z log p_t(z), we approximate it using K-Nearest Neighbours.

WHY DO WE NEED THE SCORE?
--------------------------
To reverse the forward diffusion and generate new samples, we need to
solve the reverse-time SDE (or its ODE equivalent):

    dZ = [-½ β(t) Z  -  β(t) · ∇_z log p_t(Z)] dt  +  √β(t) dW̄   (reverse SDE)

or the probability-flow ODE (used in script 05):

    dZ/dt = -½ β(t) Z  -  ½ β(t) · ∇_z log p_t(Z)

Both require ∇_z log p_t(z) — the score of the marginal density at time t.

THE TRAINING-FREE IDEA
-----------------------
If we have a set of samples {z₁, z₂, ..., z_M} drawn from p_t, then we
can estimate the score WITHOUT fitting a network:

    ∇_z log p_t(z)  ≈  (1/K) Σᵢ∈kNN(z)  (zᵢ - z) / σ_KDE²

This is the score of a Kernel Density Estimate (KDE) with bandwidth σ_KDE.

DERIVATION
----------
The KDE density estimate is:

    p̂(z) = (1/M) Σᵢ K_σ(z - zᵢ)

where K_σ is a Gaussian kernel:  K_σ(u) = (1/√(2πσ²)) exp(-u²/(2σ²))

The score of p̂ is:

    ∇_z log p̂(z) = Σᵢ K_σ(z-zᵢ) · (zᵢ-z)/σ²
                    ─────────────────────────────
                         Σᵢ K_σ(z-zᵢ)

For the K nearest neighbours (i.e. the K zᵢ closest to z):
    - K_σ(z-zᵢ) is larger for closer neighbours
    - If σ is chosen well, distant neighbours contribute negligibly

If we use all K neighbours equally (flat kernel / approximate):

    ∇_z log p̂(z)  ≈  (1/K) Σᵢ∈kNN(z) (zᵢ - z) / σ²

where σ² can be estimated from the mean squared distance to the neighbours.

In 1-D this simplifies to:

    score(z) ≈ (mean_of_neighbours - z) / var_of_neighbours

This is what we implement below.

HOW THE PAPER USES THIS
------------------------
The paper conditions the score on the CURRENT POSITION x_t in the SDE:
We are estimating the score of p_t(Δx | x_t), the conditional distribution
of increments at diffusion time t, given starting position x_t.

So we need samples from p_t(Δx | x_t = x).
In script 03 we showed how to compute Z_t from Z_0 = Δx using
    Z_t = α(t) · Δx  +  σ(t) · ε

So for a fixed x_t = x, we:
1. Find all training pairs where x_start ≈ x  (same SDE position)
2. Compute Z_t for those pairs using the forward process
3. Estimate the score of p_t(z | x) using KNN on those Z_t values

OUTPUT
------
plots/knn_score_1d.png
    Score function at several (x_t, t) combinations.
    Shows s(z, t | x) = ∇_z log p_t(z | x).

plots/knn_neighbour_visualization.png
    A scatter of data points with the KNN neighbourhood highlighted
    for a query point.

plots/score_accuracy_check.png
    Compare KNN score against the theoretical score for a Gaussian.
    This validates that KNN works before we use it on real data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import config
import utils

# ---------------------------------------------------------------------------
# 0.  Setup
# ---------------------------------------------------------------------------
config.ensure_dirs()
rng = np.random.default_rng(config.SEED + 3)

print("=" * 60)
print("STEP 04 — KNN Score Estimation")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1.  Load a manageable subset of the dataset
# ---------------------------------------------------------------------------
print("Loading dataset …")
data    = np.load(os.path.join(config.DATA_DIR, "dataset.npz"))
x_t_all = data["x_t"].astype(np.float64)   # SDE positions
dx_all  = data["dx"].astype(np.float64)    # increments (= Z_0)

# Subsample for speed
N_WORK  = 200_000
idx     = rng.choice(len(dx_all), size=N_WORK, replace=False)
x_t_work = x_t_all[idx]
dx_work  = dx_all[idx]

print(f"  Working with {N_WORK:,} pairs")
utils.print_stats("x_t", x_t_work)
utils.print_stats("dx",  dx_work)
print()

# ---------------------------------------------------------------------------
# 2.  Validate KNN score on a known Gaussian
# ---------------------------------------------------------------------------
#
# Before using KNN on real data, verify it works on a Gaussian where
# we know the exact score analytically.
#
# For p(z) = N(0, σ²):
#   log p(z) = -z²/(2σ²) + const
#   ∇_z log p(z) = -z / σ²
#
print("=" * 40)
print("VALIDATION: KNN score on a known Gaussian")
print("=" * 40)

sigma_true = 0.5
z_gauss    = rng.normal(loc=0, scale=sigma_true, size=50_000)

# Query points
z_query = np.linspace(-2*sigma_true, 2*sigma_true, 50)
score_true  = -z_query / sigma_true**2

# KNN score estimator
def knn_score_1d(z_samples, z_query_pts, k):
    """
    Estimate ∇_z log p(z) at each query point using K nearest neighbours.

    Parameters
    ----------
    z_samples  : (N,)  samples from the distribution
    z_query_pts: (M,)  query locations
    k          : int   number of neighbours

    Returns
    -------
    scores : (M,) estimated score at each query point
    """
    # Build a 1-D KD-tree for fast nearest-neighbour lookup
    # We reshape to (N,1) because cKDTree expects 2-D input
    tree = cKDTree(z_samples.reshape(-1, 1))

    scores = np.zeros(len(z_query_pts))
    for i, zq in enumerate(z_query_pts):
        # Find the k nearest neighbours to zq
        dists, idxs = tree.query([[zq]], k=k)
        neighbours  = z_samples[idxs[0]]   # shape (k,)

        # KNN score estimate:
        #   s(z) ≈ (mean(neighbours) - z) / var(neighbours)
        #
        # Derivation: for a Gaussian KDE with bandwidth σ_KDE:
        #   score = Σᵢ K_σ(z-zᵢ)·(zᵢ-z)/σ² / Σᵢ K_σ(z-zᵢ)
        # With a flat (uniform) weight over the k neighbours:
        #   ≈ (1/k)Σᵢ (zᵢ-z) / σ_KDE²
        # The natural bandwidth is the local std of the neighbours:
        sigma_kde   = neighbours.std() + 1e-12   # avoid division by zero
        scores[i]   = (neighbours.mean() - zq) / sigma_kde**2

    return scores

print("  Computing KNN scores for Gaussian validation …")
k_vals = [5, 20, 50, 100]
score_knn = {}
for k in k_vals:
    score_knn[k] = knn_score_1d(z_gauss, z_query, k=k)
    mse = np.mean((score_knn[k] - score_true)**2)
    print(f"    k={k:3d}:  MSE vs theory = {mse:.4f}")

print()

# Plot validation
fig, ax = utils.make_fig(figsize=(10, 6))
ax.plot(z_query, score_true, "k-", lw=3, label=r"True score $-z/\sigma^2$")
colors_k = ["#E53935", "#43A047", "#1E88E5", "#FB8C00"]
for k, c in zip(k_vals, colors_k):
    ax.plot(z_query, score_knn[k], "--", color=c, lw=1.8, label=f"KNN k={k}")

ax.set_xlabel("z", fontsize=12)
ax.set_ylabel(r"Score $\nabla_z \log p(z)$", fontsize=12)
ax.set_title(
    f"KNN Score Validation on Gaussian N(0, {sigma_true}²)\n"
    "Larger k = smoother estimate; must balance bias and variance",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.axhline(0, color="gray", lw=0.5, ls="-")
utils.save_fig(ax.get_figure(),
               os.path.join(config.PLOTS_DIR, "score_accuracy_check.png"))

# ---------------------------------------------------------------------------
# 3.  Neighbour visualisation
# ---------------------------------------------------------------------------
#
# Show a 2-D view of the (x_t, Z_t) space at a fixed diffusion time t.
# Highlight the KNN neighbourhood of a query point.
#
print("Generating knn_neighbour_visualization.png …")

# We work in 2-D: the joint space (x_t, z_t) at diffusion time t=0.5
t_demo  = 0.5
a_demo, s_demo = utils.vp_alpha_sigma(np.array([t_demo]),
                                       config.BETA_MIN, config.BETA_MAX)
a_demo, s_demo = float(a_demo[0]), float(s_demo[0])

# Compute Z_t for a subset
N_VIS = 5_000
eps   = rng.standard_normal(N_VIS)
z_t_vis = a_demo * dx_work[:N_VIS] + s_demo * eps

# Joint array for KDTree: shape (N_VIS, 2)
joint = np.column_stack([x_t_work[:N_VIS], z_t_vis])

# Query point
x_query  = 1.5
z_query_demo = 0.0
query_2d = np.array([[x_query, z_query_demo]])

# Find KNN in joint space
k_demo = 50
tree2d = cKDTree(joint)
dists2d, idxs2d = tree2d.query(query_2d, k=k_demo)
neighbour_pts   = joint[idxs2d[0]]

fig2, ax2 = utils.make_fig(figsize=(10, 7))
sc = ax2.scatter(
    joint[:, 0], joint[:, 1],
    c=z_t_vis, cmap="coolwarm", s=3, alpha=0.4, vmin=-1.5, vmax=1.5,
)
plt.colorbar(sc, ax=ax2, label=r"$Z_t$ value")

ax2.scatter(
    neighbour_pts[:, 0], neighbour_pts[:, 1],
    color="yellow", s=25, alpha=0.9, edgecolors="black", lw=0.5,
    label=f"K={k_demo} nearest neighbours",
)
ax2.scatter([x_query], [z_query_demo], color="red", s=150, zorder=5,
            marker="*", label=f"Query point (x={x_query}, z={z_query_demo})")

ax2.set_xlabel("SDE position $x_t$", fontsize=12)
ax2.set_ylabel(f"Diffusion latent $Z_t$ at t={t_demo}", fontsize=12)
ax2.set_title(
    f"KNN in joint (x_t, Z_t) space at diffusion t={t_demo}\n"
    f"Score at query ≈ mean(neighbours) - query / bandwidth²",
    fontsize=10,
)
ax2.legend(fontsize=9)
utils.save_fig(fig2, os.path.join(config.PLOTS_DIR, "knn_neighbour_visualization.png"))

# ---------------------------------------------------------------------------
# 4.  Score function at multiple (x, t) conditions
# ---------------------------------------------------------------------------
#
# The paper conditions the score on x_t (SDE position).
# Let's estimate s(z, t | x) for several x values at t=0.5.
#
# Strategy:
#   For each x of interest:
#     1. Filter training pairs where |x_t - x| < tolerance
#     2. Compute Z_t for those pairs
#     3. Estimate score along z using KNN
#
print("Generating knn_score_1d.png …")

t_score  = 0.5
a_s, s_s = utils.vp_alpha_sigma(np.array([t_score]),
                                  config.BETA_MIN, config.BETA_MAX)
a_s, s_s = float(a_s[0]), float(s_s[0])

x_conditions = [0.5, 1.0, 2.0, 3.5, 5.0]
tolerance     = 0.3
z_query_range = np.linspace(-1.5, 1.5, 60)

fig3, axes3 = utils.make_fig(nrows=1, ncols=len(x_conditions),
                               figsize=(4*len(x_conditions), 5))

for ax3, x_cond in zip(axes3, x_conditions):
    # Filter pairs near this x condition
    near_mask = np.abs(x_t_work - x_cond) < tolerance
    n_near    = near_mask.sum()

    if n_near < config.KNN_K * 2:
        ax3.text(0.5, 0.5, f"Not enough\ndata near x={x_cond}",
                 transform=ax3.transAxes, ha="center")
        ax3.set_title(f"x≈{x_cond}")
        continue

    dx_near = dx_work[near_mask]
    eps_near = rng.standard_normal(n_near)
    z_t_near = a_s * dx_near + s_s * eps_near

    print(f"  x≈{x_cond:.1f}:  {n_near:,} pairs,  "
          f"Z_t mean={z_t_near.mean():.4f}  std={z_t_near.std():.4f}")

    # Estimate score using KNN
    score_est = knn_score_1d(z_t_near, z_query_range, k=config.KNN_K)

    # Also overlay: the score should look like  -(z - α·μ_dx) / (α²·σ_dx² + σ²)
    mu_cond  = dx_near.mean()
    var_cond = dx_near.var()
    mu_zt    = a_s * mu_cond
    var_zt   = a_s**2 * var_cond + s_s**2
    score_theory = -(z_query_range - mu_zt) / var_zt

    ax3.hist(z_t_near, bins=50, density=True, color="#B0BEC5",
             alpha=0.5, range=(-3, 3), label="Z_t dist")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(z_query_range, score_est,    "b-",  lw=2, label="KNN score")
    ax3_twin.plot(z_query_range, score_theory, "r--", lw=2, label="Theory (Gaussian approx)")
    ax3_twin.axhline(0, color="gray", lw=0.5)
    ax3_twin.set_ylim(-20, 20)
    ax3_twin.set_ylabel(r"Score $\nabla_z \log p_t$", fontsize=8)

    ax3.set_xlim(-3, 3)
    ax3.set_xlabel("z", fontsize=9)
    ax3.set_title(f"x ≈ {x_cond}\n(n={n_near:,})", fontsize=9)
    if ax3 == axes3[0]:
        ax3.set_ylabel("Density", fontsize=9)
    if ax3 == axes3[-1]:
        ax3_twin.legend(fontsize=7, loc="upper right")

fig3.suptitle(
    f"Conditional Score s(z, t={t_score} | x_t=x) estimated by KNN\n"
    "Each panel: different SDE position x",
    fontsize=11, y=1.01,
)
fig3.tight_layout()
utils.save_fig(fig3, os.path.join(config.PLOTS_DIR, "knn_score_1d.png"))

# ---------------------------------------------------------------------------
# 5.  Diagnostics
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print()
print("  KNN validation (Gaussian test):")
for k in k_vals:
    mse = np.mean((score_knn[k] - score_true)**2)
    print(f"    k={k:3d}  MSE={mse:.5f}  "
          f"(MSE near 0 = KNN matches theory well)")
print()
print("  Best k for 1-D score estimation:")
best_k  = k_vals[np.argmin([np.mean((score_knn[k] - score_true)**2) for k in k_vals])]
print(f"    k = {best_k}  (used in config.KNN_K = {config.KNN_K})")
print()
print("FILES WRITTEN")
print("  plots/score_accuracy_check.png")
print("  plots/knn_neighbour_visualization.png")
print("  plots/knn_score_1d.png")
print()
print("WHAT TO UNDERSTAND FROM THE PLOTS")
print()
print("  score_accuracy_check.png:")
print("    For a Gaussian, the score is a straight line: s(z) = -z/σ².")
print("    Small k (k=5) is noisy; large k (k=100) is smoother but")
print("    may have bias in tails.  k=50 (config.KNN_K) is the sweet spot.")
print()
print("  knn_neighbour_visualization.png:")
print("    The yellow dots are the 50 nearest neighbours to the red star.")
print("    Notice how they cluster in (x_t, Z_t) space — they are")
print("    similar both in SDE position AND in diffusion latent value.")
print("    The score is computed from these neighbours' centroid vs query.")
print()
print("  knn_score_1d.png:")
print("    Each panel shows the score at a different SDE position x.")
print("    Near the left wall (x≈0.5) the score is pushed rightward")
print("    (positive) because the increment distribution is left-truncated.")
print("    Near the right wall (x≈5.0) the opposite happens.")
print("    This conditional asymmetry is what makes the boundary problem hard.")
print()
print("Done.  Run 05_probability_flow_ode.py next.")
