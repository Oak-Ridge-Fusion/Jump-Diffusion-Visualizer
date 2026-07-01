"""
config.py
=========
Central configuration for the learning_diffusion_sde project.

Every script imports these constants so that changing one value here
ripples through the entire pipeline consistently.

Paper reference:
    "Generative AI Models for Learning Flow Maps of Stochastic Dynamical
    Systems in Bounded Domains"

The SDE we study:
    dX_t = dW_t,   X_0 = x0,   0 <= X_t <= L  (absorbing boundaries)

This is pure Brownian motion confined to [0, L].  When a particle hits
x=0 or x=L it is immediately "absorbed" (removed from the population).
"""

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
DOMAIN_LO  = 0.0   # left absorbing boundary
DOMAIN_HI  = 6.0   # right absorbing boundary  (L in the paper)
X0         = 1.0   # all particles start here

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
T          = 3.0    # total simulation time
DT         = 5e-4   # Euler-Maruyama step size
N_STEPS    = int(T / DT)   # = 6000 steps

# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
N_PARTICLES = 200_000   # number of independent SDE trajectories

# ---------------------------------------------------------------------------
# Diffusion model (DDPM / VP-SDE schedule)
# ---------------------------------------------------------------------------
# The paper uses a Variance-Preserving (VP) noise schedule.
# We parameterise it with beta_min / beta_max following Song et al. (2021).
BETA_MIN  = 0.1
BETA_MAX  = 20.0

# Number of discretisation steps for the probability-flow ODE integrator
N_DIFFUSION_STEPS = 1000

# ---------------------------------------------------------------------------
# KNN score estimator
# ---------------------------------------------------------------------------
KNN_K = 50   # number of neighbours used to approximate grad log p

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
HIDDEN_DIM   = 256
N_LAYERS     = 4
LEARNING_RATE = 1e-3
BATCH_SIZE   = 2048
N_EPOCHS     = 100

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Output directories  (created on demand by each script)
# ---------------------------------------------------------------------------
import os
ROOT        = os.path.dirname(__file__)
PLOTS_DIR   = os.path.join(ROOT, "plots")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")

def ensure_dirs():
    """Create output directories if they do not exist."""
    for d in [PLOTS_DIR, DATA_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)
