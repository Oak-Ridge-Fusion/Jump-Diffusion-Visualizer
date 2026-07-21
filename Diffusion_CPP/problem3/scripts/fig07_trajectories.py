"""
fig07_trajectories.py
=======================
Figure 7: ~20 full trajectories (drift + diffusion + knock-on jumps, all
physics ON -- the default Config) started from random points across the
whole (p, xi) domain, colored by their FINAL FATE:
  survivor    (green)  -- still inside [p_min, p_max] at the end of the
                           simulated window
  thermalized (blue)   -- exited through p_min (rejoined the thermal bulk)
  runaway     (red)    -- exited through p_max (confirmed runaway electron)

This is the first figure to show the complete, unmodified model (Figures 3
and 4 isolated the diffusion and the jumps separately; this combines them
exactly as common_2.simulate_re2d_step does, just run long enough and
recorded at every sub-step so the full path can be drawn). The three-way
fate split visualized here is exactly what data_generation_2.py's exit
labels (side in {0,1,2}) encode for the ML exit-classifier training set --
this figure is the "what does one training example's underlying trajectory
actually look like" companion to that data.

Output: problem3/figures/fig07_trajectories.png
"""

import os
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, re2d_coeffs, knock_rate, apply_knock_jump

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FATE_COLOR = {"survivor": "#55a868", "thermalized": "#4c72b0",
              "runaway": "#c44e52"}


def simulate_full_paths(p0, xi0, cfg, rng, n_steps, dt_sub):
    """Fine Euler-Maruyama integration with BOTH drift-diffusion and
    knock-on jumps (the complete, unmodified model) -- same per-substep
    order of operations as common_2.simulate_re2d_step, recording every
    sub-step so the full path can be drawn. Returns P, XI (n_steps+1, n)
    and exit_step (n,)."""
    n = p0.shape[0]
    P = np.empty((n_steps + 1, n))
    XI = np.empty((n_steps + 1, n))
    P[0], XI[0] = p0, xi0
    p, xi = p0.copy(), xi0.copy()
    alive = np.ones(n, dtype=bool)
    exit_step = np.full(n, n_steps, dtype=np.int64)
    sqrt_dt = math.sqrt(dt_sub)

    for k in range(n_steps):
        a = alive
        if not a.any():
            P[k + 1:] = p
            XI[k + 1:] = xi
            break
        b_p, b_xi, s_xi = re2d_coeffs(p, xi, cfg)

        lam = knock_rate(p, cfg)
        fired = a & (rng.random(p.shape) < (1.0 - np.exp(-lam * dt_sub)))
        if fired.any():
            idx = np.where(fired)[0]
            p[idx], xi[idx] = apply_knock_jump(p[idx], xi[idx], cfg, rng)

        p_new, xi_new = p.copy(), xi.copy()
        na = int(a.sum())
        p_new[a] = p[a] + b_p[a] * dt_sub
        xi_new[a] = (xi[a] + b_xi[a] * dt_sub
                     + s_xi[a] * sqrt_dt * rng.standard_normal(na))
        hi = a & (xi_new > 1.0)
        xi_new[hi] = 2.0 - xi_new[hi]
        lo = a & (xi_new < -1.0)
        xi_new[lo] = -2.0 - xi_new[lo]
        np.clip(xi_new, -1.0, 1.0, out=xi_new)

        out_lo = a & (p_new < cfg.p_min)
        out_hi = a & (p_new > cfg.p_max)
        newly_dead = out_lo | out_hi
        exit_step[newly_dead] = k + 1
        alive = alive & ~newly_dead

        p, xi = p_new, xi_new
        P[k + 1], XI[k + 1] = p, xi

    return P, XI, exit_step


def main():
    cfg = Config()  # full model: drift + diffusion + jumps, all defaults
    rng = np.random.default_rng(cfg.seed)

    n_particles = 20
    p0 = rng.uniform(cfg.p_min, cfg.p_max, n_particles)
    xi0 = rng.uniform(-1.0, 1.0, n_particles)

    dt_sub = cfg.dt / cfg.n_sub
    T_total = 6.0 * cfg.dt
    n_steps = int(round(T_total / dt_sub))

    P, XI, exit_step = simulate_full_paths(p0, xi0, cfg, rng, n_steps, dt_sub)
    t = np.arange(n_steps + 1) * dt_sub

    survivor = exit_step == n_steps
    idx_last = np.minimum(exit_step, n_steps)
    p_end = P[idx_last, np.arange(n_particles)]
    thermalized = (~survivor) & (p_end <= cfg.p_min + 1e-6)
    runaway = (~survivor) & (~thermalized)

    fate = np.array(["survivor"] * n_particles, dtype=object)
    fate[thermalized] = "thermalized"
    fate[runaway] = "runaway"

    print(f"{n_particles} particles, T={T_total:g} tau_c, "
          f"full model (jumps_on={cfg.jumps_on}):")
    print(f"  survivors  : {survivor.sum():2d}")
    print(f"  thermalized: {thermalized.sum():2d}")
    print(f"  runaway    : {runaway.sum():2d}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                    gridspec_kw={"width_ratios": [1.3, 1.0]})

    for i in range(n_particles):
        k_end = exit_step[i]
        color = FATE_COLOR[fate[i]]
        axL.plot(P[:k_end + 1, i], XI[:k_end + 1, i], color=color, lw=1.0,
                  alpha=0.75, zorder=2)
        axL.plot(p0[i], xi0[i], "o", color="black", ms=4.5, zorder=4)
        if fate[i] == "thermalized":
            axL.plot(P[k_end, i], XI[k_end, i], "x", color=color, ms=8,
                      mew=1.8, zorder=5)
        elif fate[i] == "runaway":
            axL.plot(P[k_end, i], XI[k_end, i], "+", color=color, ms=10,
                      mew=1.8, zorder=5)
        else:
            axL.plot(P[k_end, i], XI[k_end, i], "s", color=color, ms=6,
                      mec="black", mew=0.7, zorder=5)

    axL.axvline(cfg.p_min, color="0.4", lw=1.2, ls=":")
    axL.axvline(cfg.p_max, color="0.4", lw=1.2, ls=":")
    axL.axhline(1.0, color="0.4", lw=1.0, ls=":")
    axL.axhline(-1.0, color="0.4", lw=1.0, ls=":")
    axL.set_xlim(cfg.p_min - 0.3, cfg.p_max + 0.3)
    axL.set_ylim(-1.05, 1.05)
    axL.set_xlabel(r"$p$")
    axL.set_ylabel(r"$\xi=\cos\theta$")
    axL.set_title(f"{n_particles} full trajectories (drift+diffusion+jumps),"
                  f" $T={T_total:g}\\,\\tau_c$\n"
                  r"$\bullet$=start, "
                  r"$\times$=thermalized, $+$=runaway, "
                  r"$\blacksquare$=survivor", fontsize=10.5)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=FATE_COLOR["survivor"], lw=2,
               label=f"survivor  (n={survivor.sum()})"),
        Line2D([0], [0], color=FATE_COLOR["thermalized"], lw=2,
               label=f"thermalized  (n={thermalized.sum()})"),
        Line2D([0], [0], color=FATE_COLOR["runaway"], lw=2,
               label=f"runaway  (n={runaway.sum()})"),
    ]
    axL.legend(handles=legend_elems, loc="lower center",
               bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=9.5,
               frameon=False)

    # ---- right panel: p(t) for all 20 particles, same fate coloring ----
    for i in range(n_particles):
        k_end = exit_step[i]
        color = FATE_COLOR[fate[i]]
        axR.plot(t[:k_end + 1], P[:k_end + 1, i], color=color, lw=1.0,
                  alpha=0.75)
    axR.axhline(cfg.p_min, color="0.4", lw=1.2, ls=":")
    axR.axhline(cfg.p_max, color="0.4", lw=1.2, ls=":")
    axR.set_xlabel(r"$t$  (units of $\tau_c$)")
    axR.set_ylabel(r"$p(t)$")
    axR.set_title("Same particles: $p(t)$\n"
                  "(thermalized/runaway trend visibly to a wall; "
                  "survivors wander)", fontsize=10.5)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(FIG_DIR, "fig07_trajectories.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
