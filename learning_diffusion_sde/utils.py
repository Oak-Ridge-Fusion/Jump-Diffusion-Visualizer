"""
utils.py
========
Shared helpers used across all scripts in learning_diffusion_sde.

Design principle: every function is self-contained, documented with the
*why*, and returns plain NumPy arrays so scripts remain easy to inspect
in a REPL.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_fig(nrows=1, ncols=1, figsize=None, **kwargs):
    """Thin wrapper that sets a consistent style for every figure."""
    plt.rcParams.update({
        "font.family":  "monospace",
        "axes.grid":    True,
        "grid.alpha":   0.3,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def save_fig(fig, path, dpi=150):
    """Save figure and print confirmation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# SDE utilities
# ---------------------------------------------------------------------------

def check_boundary(x, lo, hi):
    """
    Return a boolean mask: True  = particle is still alive (inside domain)
                           False = particle has been absorbed.

    We use strict inequalities so a particle that lands exactly on the
    boundary counts as absorbed.
    """
    return (x > lo) & (x < hi)


def euler_maruyama_step(x, dt, rng):
    """
    One step of the Euler-Maruyama scheme for dX = dW.

    dX_t = dW_t
         = sqrt(dt) * N(0,1)

    Parameters
    ----------
    x   : ndarray, shape (N,) — current positions of N particles
    dt  : float               — time step
    rng : np.random.Generator — seeded RNG for reproducibility

    Returns
    -------
    x_new : ndarray, shape (N,) — positions after one step
    """
    noise = rng.standard_normal(size=x.shape)   # N(0,1)
    return x + np.sqrt(dt) * noise              # Euler step


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def empirical_pdf(samples, lo, hi, n_bins=80):
    """
    Compute a normalised histogram (empirical PDF) of `samples` on [lo, hi].

    Returns (bin_centers, density) so it can be plotted as a bar or line.
    """
    counts, edges = np.histogram(samples, bins=n_bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def print_stats(label, arr):
    """Print summary statistics for a 1-D array."""
    print(f"  [{label}]  n={len(arr):,}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")


# ---------------------------------------------------------------------------
# VP-SDE noise schedule  (used from script 03 onwards)
# ---------------------------------------------------------------------------

def vp_alpha_sigma(t, beta_min, beta_max):
    """
    Variance-Preserving (VP) noise schedule from Song et al. (2021).

    The continuous-time interpolation is:

        beta(t) = beta_min + t*(beta_max - beta_min)

        log_alpha(t) = -0.5 * integral_0^t  beta(s) ds
                     = -0.5 * (beta_min*t + 0.5*(beta_max-beta_min)*t^2)

        alpha(t) = exp(log_alpha(t))          # signal coefficient
        sigma(t) = sqrt(1 - alpha(t)^2)       # noise coefficient

    So the noised sample at time t is:
        Z_t = alpha(t) * Z_0  +  sigma(t) * eps,   eps ~ N(0,1)

    This keeps Var[Z_t] = alpha^2 * Var[Z_0] + sigma^2 = 1
    when Z_0 is unit-variance (variance-PRESERVING).

    Parameters
    ----------
    t        : float or ndarray in [0,1]
    beta_min : float
    beta_max : float

    Returns
    -------
    alpha : same shape as t
    sigma : same shape as t
    """
    log_alpha = -0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t**2)
    alpha     = np.exp(log_alpha)
    sigma     = np.sqrt(np.maximum(1.0 - alpha**2, 0.0))   # clamp for fp safety
    return alpha, sigma
