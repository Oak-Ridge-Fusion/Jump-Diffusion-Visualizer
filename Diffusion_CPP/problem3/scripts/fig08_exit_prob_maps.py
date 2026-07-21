"""
fig08_exit_prob_maps.py
=========================
Figure 8: Monte Carlo one-step exit-probability maps
  P(thermal | p, xi) = P(exit through p_min during one coarse step dt)
  P(runaway | p, xi) = P(exit through p_max during one coarse step dt)
over the whole (p, xi) domain.

Why this is exactly the ML exit-classifier's target
-----------------------------------------------------
data_generation_2.py builds data_exit.npz by drawing random (p, xi) starts,
calling common_2.simulate_re2d_step ONCE (one coarse step, dt=cfg.dt, using
the n_sub=400 fine sub-stepped reference integrator internally), and
recording the 3-class outcome side in {0: stay, 1: exit p_min, 2: exit
p_max}. ExitNet is then trained by cross-entropy to predict exactly this
3-class distribution from (p, xi). This figure computes the SAME quantity
data_generation_2.py samples -- one call to simulate_re2d_step -- but with
Monte Carlo particles concentrated at every point of a regular grid instead
of scattered random starts, so the exit-probability SURFACE (the thing the
classifier must learn to approximate) can be seen directly, at the same
grid/MC resolution Config already reserves for it (b1_grid=36x18,
b1_mc=1500 particles/point -- see common_2.Config).

Why the maps are boundary-layer, not separatrix-shaped
---------------------------------------------------------
This is a ONE-STEP (dt=1 tau_c) probability, not a probability of EVENTUAL
exit. Figures 2 and 6 characterized where the drift eventually SENDS a
particle (the b_p=0 separatrix, defined by the sign of the drift alone,
with no notion of "how long it takes"). One coarse step is not long enough
for a particle to cross the whole domain even if its drift points the right
way -- unless a knock-on jump does most of the work in one shot, or the
particle starts close enough to a wall already. So we expect these maps to
be sharply concentrated in thin bands near p_min and p_max (consistent with
the exit_features log-wall-distance featurization and the stratified
data_exit_extra.npz sampling in data_generation_2.py, both of which exist
specifically to handle this steep-front structure), NOT filled in across
the whole separatrix-delimited region the way Figure 7's long-horizon fates
were.

Output: problem3/figures/fig08_exit_prob_maps.png
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, re2d_coeffs, simulate_re2d_step, set_seed

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    n_p, n_xi = cfg.b1_grid       # (36, 18) -- Config's own reserved sizes
    n_mc = cfg.b1_mc              # 1500 MC particles per grid point
    p_grid = np.linspace(cfg.p_min, cfg.p_max, n_p)
    xi_grid = np.linspace(-1.0, 1.0, n_xi)
    PP, XX = np.meshgrid(p_grid, xi_grid)             # (n_xi, n_p) each

    # one big vectorized call to the REFERENCE integrator -- exactly the
    # function data_generation_2.py uses -- covering every grid point at
    # once (n_p*n_xi*n_mc particles, one coarse step dt=cfg.dt each)
    p0 = np.repeat(PP.ravel(), n_mc)
    xi0 = np.repeat(XX.ravel(), n_mc)
    t0 = time.time()
    _, _, alive, side = simulate_re2d_step(p0, xi0, cfg, rng)
    dt_wall = time.time() - t0
    print(f"MC exit map: {len(p0)} particles "
          f"({n_p}x{n_xi} grid x {n_mc} MC), one coarse step dt={cfg.dt:g}, "
          f"{dt_wall:.1f}s")

    side_grid = side.reshape(n_xi * n_p, n_mc)
    P_therm = (side_grid == 1).mean(axis=1).reshape(n_xi, n_p)
    P_run = (side_grid == 2).mean(axis=1).reshape(n_xi, n_p)
    P_stay = 1.0 - P_therm - P_run

    print(f"domain-averaged: P(stay)={P_stay.mean():.4f}, "
          f"P(thermal)={P_therm.mean():.4f}, P(runaway)={P_run.mean():.4f}")
    print(f"max P(thermal)={P_therm.max():.4f} at "
          f"p={PP.ravel()[P_therm.argmax()]:.2f}, "
          f"max P(runaway)={P_run.max():.4f} at "
          f"p={PP.ravel()[P_run.argmax()]:.2f}")

    # ---- the Figure 2 / 6 deterministic separatrix, for overlay ----
    p_fine = np.linspace(cfg.p_min, cfg.p_max, 300)
    xi_fine = np.linspace(-1.0, 1.0, 300)
    PF, XF = np.meshgrid(p_fine, xi_fine)
    b_p_fine, _, _ = re2d_coeffs(PF, XF, cfg)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0)  # gamma<1: boosts small
                                                      # probabilities so the
                                                      # boundary layer isn't
                                                      # invisible next to
                                                      # the saturated walls
    for ax, Pmap, title, cmap in [
        (axA, P_therm, r"$P(\mathrm{thermal}\mid p,\xi)$", "Blues"),
        (axB, P_run, r"$P(\mathrm{runaway}\mid p,\xi)$", "Reds"),
    ]:
        im = ax.pcolormesh(PP, XX, Pmap, shading="gouraud", cmap=cmap,
                            norm=norm)
        ax.contour(PF, XF, b_p_fine, levels=[0.0], colors="black",
                   linewidths=1.8, linestyles="--")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(title, fontsize=10)
        cb.set_ticks([0, 0.05, 0.2, 0.5, 1.0])
        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$\xi=\cos\theta$")
        ax.set_xlim(cfg.p_min, cfg.p_max)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{title}  (one coarse step, $dt={cfg.dt:g}$)\n"
                     f"MC: {n_mc} particles/grid pt, "
                     f"{n_p}"f"$\\times${n_xi} grid; dashed = $b_p=0$",
                     fontsize=10.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig08_exit_prob_maps.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
