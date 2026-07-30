"""
fig_dynamics_bd_bsd.py
========================
Combined BD (Problem 1, constant coefficients) + BSD (Problem 2,
state-dependent coefficients) DYNAMICS figure: instead of the one-step
transition-density snapshot (B0/B1_onestep.png), show how the survivor
density actually evolves over coarse time t = 1, 3, 5 tau (dt=1 each), for
both models side by side, and -- the new piece -- contrast the BOUNDED
domain (absorbing walls at x_min/x_max, the real model) against the
UNBOUNDED analog of the exact same SDE (same drift/diffusion/jump law from
bounded_coeffs, just no wall to absorb at).

Why this is the right way to explain "bounded vs unbounded"
-------------------------------------------------------------
Both start every particle as a delta function at the same x0. If there
were no wall, total probability mass is conserved (unbounded curve's area
stays 1 for all t). With the wall, particles that cross x_min/x_max are
removed (absorbed) -- the bounded curve is a SUB-density whose area equals
the surviving fraction at that t. Plotting both on the same absolute
(not area-normalized) y-axis makes the absorbed mass visible directly as
the gap between the unbounded curve's full area and the bounded curve's
shrunken area.

BD (no restoring force, upward-biased jumps) shows this dramatically: the
free density just keeps drifting/spreading past x=6 while the bounded
density piles up against, then gets eaten by, the upper wall. BSD (mean-
reverting drift toward x*=2.5) shows the opposite lesson: because the
restoring force keeps most mass away from both walls anyway, the bounded
and unbounded densities stay ALMOST ON TOP OF EACH OTHER even at t=5 --
the walls matter far less here, consistent with BSD's much higher survival
fraction already reported in the poster (0.995 vs 0.960 at x0=1, one step).

The "unbounded" step reuses bounded_coeffs() (so BD and BSD share one code
path, same as the bounded model) but skips the absorption check -- this
mirrors simulate_bounded_step exactly minus the "if x leaves domain, mark
dead" block. We do NOT modify common_1.py; this is a new, additive script.

Run:  python fig_dynamics_bd_bsd.py [N]
Out:  figures/fig_dynamics_bd_bsd.png
"""

import math
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_1 import Config, bounded_coeffs, simulate_bounded_step, set_seed

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

X0 = 1.0                       # shared starting point (matches the poster's existing P1.1/P2.1 x0=1)
CHECKPOINTS = [1, 3, 5]        # coarse steps t = 1, 3, 5 (dt = 1 each)
K = max(CHECKPOINTS)


def simulate_free_step(x, cfg: Config, rng, n_sub=None):
    """Same drift/diffusion/jump law as simulate_bounded_step (via
    bounded_coeffs), but NO absorbing boundary -- the unbounded analog.
    lam/sigma are clipped at 0 as a safety net for BSD's linear-in-x rate
    and diffusion far outside [x_min, x_max] (never active for BD's
    constant coefficients, and not active for BSD either within the
    horizons used here -- mean reversion keeps x well inside safe range)."""
    n_sub = cfg.n_sub if n_sub is None else n_sub
    x = np.array(x, dtype=np.float64).copy()
    dt_sub = cfg.dt / n_sub
    sqrt_dt = math.sqrt(dt_sub)
    for _ in range(n_sub):
        bx, sx, lx, jmx, jsx = bounded_coeffs(x, cfg)
        lx = np.maximum(lx, 0.0)
        sx = np.maximum(sx, 1e-6)
        fired = rng.random(x.shape) < (1.0 - np.exp(-lx * dt_sub))
        if fired.any():
            x[fired] += rng.normal(jmx[fired], jsx[fired])
        x += bx * dt_sub + sx * sqrt_dt * rng.standard_normal(x.shape)
    return x


def run_model(model, N, rng):
    cfg = Config(model=model)

    # ---- bounded (absorbing walls): the real model ----
    xb = np.full(N, X0, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    bounded_snap = {}
    t0 = time.time()
    for k in range(1, K + 1):
        idx = np.where(alive)[0]
        if len(idx):
            xe, al = simulate_bounded_step(xb[idx], cfg, rng)
            alive[idx[~al]] = False
            xb[idx[al]] = xe[al]
        if k in CHECKPOINTS:
            bounded_snap[k] = xb[alive].copy()
        print(f"  [{model} bounded]   t={k}  survival={alive.mean():.3f}  "
              f"elapsed={time.time()-t0:.1f}s")

    # ---- unbounded (no wall): same SDE, mass-conserving ----
    xu = np.full(N, X0, dtype=np.float64)
    free_snap = {}
    t0 = time.time()
    for k in range(1, K + 1):
        xu = simulate_free_step(xu, cfg, rng)
        if k in CHECKPOINTS:
            free_snap[k] = xu.copy()
        print(f"  [{model} unbounded] t={k}  elapsed={time.time()-t0:.1f}s")

    return bounded_snap, free_snap


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    set_seed(0)
    rng = np.random.default_rng(0)

    models = [("bounded", "Problem 1 -- BD (constant coefficients)"),
              ("bounded_sd", "Problem 2 -- BSD (state-dependent coefficients)")]

    results = {}
    for model, label in models:
        print(f"=== {label} ===")
        results[model] = run_model(model, N, rng)

    x_lo, x_hi = -3.0, 11.0
    bins = np.linspace(x_lo, x_hi, 181)

    fig, axes = plt.subplots(2, len(CHECKPOINTS),
                             figsize=(5.0 * len(CHECKPOINTS), 7.6),
                             sharex=True)

    for row, (model, label) in enumerate(models):
        bounded_snap, free_snap = results[model]
        cfg = Config(model=model)
        for col, k in enumerate(CHECKPOINTS):
            ax = axes[row, col]
            xb_k = bounded_snap[k]
            xu_k = free_snap[k]
            surv = len(xb_k) / N

            ax.hist(xu_k, bins=bins, weights=np.ones_like(xu_k) / N,
                    histtype="step", color="C3", lw=1.8, ls="--",
                    label="unbounded (no walls)")
            ax.hist(xb_k, bins=bins, weights=np.ones_like(xb_k) / N,
                    color="0.55", alpha=0.55, edgecolor="none",
                    label="bounded (absorbing walls)")

            ax.axvline(cfg.x_min, color="k", lw=1.2)
            ax.axvline(cfg.x_max, color="k", lw=1.2)
            ax.set_xlim(x_lo, x_hi)
            ax.set_title(f"t={k}  (survival={surv:.3f})", fontsize=11)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel(f"{label}\nfraction of N per unit x", fontsize=9.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(r"Dynamics: survivor density (bounded) vs. free density (unbounded), "
                 f"from $x_0$={X0:g}\n"
                 r"gap between the dashed curve's full area and the filled curve's "
                 "area = probability absorbed by the walls", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(FIG_DIR, "fig_dynamics_bd_bsd.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
