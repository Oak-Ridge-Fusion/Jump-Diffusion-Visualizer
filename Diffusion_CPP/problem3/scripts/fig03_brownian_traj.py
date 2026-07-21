"""
fig03_brownian_traj.py
=======================
Figure 3: pure drift-diffusion (Brownian) trajectories of the 2D
runaway-electron model, with the knock-on JUMP CHANNEL DISABLED
(cfg.jumps_on = False). This isolates the small-angle-collision Fokker-Planck
part of the dynamics -- Figure 4 will re-enable jumps and show the
qualitative difference (discontinuous vs. continuous paths).

Physics
-------
With jumps off, the SDE is pure drift-diffusion:
  dp  = b_p(p,xi)  dt                      <- NO noise term on p
  dxi = b_xi(p,xi) dt + s_xi(p,xi) dW       <- noise only on xi

This is a genuine, physically important asymmetry of the model (see
common_2.re2d_coeffs docstring): the Fokker-Planck part only diffuses the
PITCH ANGLE directly. p is not frozen -- it still evolves via b_p -- but with
jumps off, p's random component is entirely SECOND-HAND: it comes from p's
coupling to the stochastically diffusing xi through b_p(p, xi), not from any
direct noise on p itself.

Two things fall out of this that are worth confirming numerically, not just
assuming (see the right-hand panel):
  (a) For the default E_hat=2, at moderate p the field-torque term
      E(1-xi^2)/p in b_xi dominates the collisional damping -xi*nu_c, so
      almost the ENTIRE ensemble relaxes toward xi=1 (field-aligned pitch)
      within a few tau_c and then piles up against the reflecting wall --
      diffusion is still present (the paths keep jittering) but it is
      compressed against xi=1, so Var(xi) actually SATURATES at a small
      value rather than growing without bound.
  (b) Despite having no direct noise term, p's ensemble variance grows
      FASTER than xi's over the same window. p(t) is (to leading order) the
      time-integral of a term proportional to xi(t); integrating a
      diffusing, history-dependent process compounds its fluctuations, the
      same reason the variance of integrated Brownian motion grows like t^3
      while the underlying process only grows like t. So "no direct
      diffusion on p" does NOT mean "small spread in p" -- it is exactly the
      opposite here.

What this script does
----------------------
1. Take Config(), force jumps_on=False (dataclasses.replace -- common_2.py
   is not modified).
2. Integrate the SAME fine Euler-Maruyama sub-step used by the reference
   integrator (common_2.simulate_re2d_step), but WITHOUT the jump section,
   and recording every sub-step so we can draw continuous paths (the
   reference function only returns the final state of one coarse step).
3. Reflecting xi boundary and absorbing p boundaries are handled identically
   to the ground-truth integrator.
4. Three starting points are chosen relative to the b_p=0 separatrix found
   in Figure 2 (below / near / above it), so the resulting trajectory
   bundles connect directly to that figure's physics.

Output: problem3/figures/fig03_brownian_traj.png
"""

import os
import sys
from dataclasses import replace
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, re2d_coeffs

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def simulate_paths_nojump(p0, xi0, cfg, rng, n_steps, dt_sub):
    """Fine Euler-Maruyama integration of the drift-diffusion part ONLY
    (no knock-on jumps), recording the full path. Same boundary handling as
    common_2.simulate_re2d_step: reflect xi at +-1, absorb p at p_min/p_max.
    Returns P, XI of shape (n_steps+1, n_particles) and exit_step (index at
    which each particle was absorbed, or n_steps if it survived)."""
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
        b_p, b_xi, s_xi = re2d_coeffs(p, xi, cfg)
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
    cfg = Config()
    cfg_nj = replace(cfg, jumps_on=False)
    rng = np.random.default_rng(cfg.seed)

    dt_sub = cfg.dt / cfg.n_sub          # same fine resolution as common_2
    T_total = 3.0 * cfg.dt               # three coarse-step equivalents
    n_steps = int(round(T_total / dt_sub))

    # starting points chosen relative to Figure 2's b_p=0 separatrix:
    # A below it (thermalizing tendency), B astride it (uncertain fate),
    # C above it (runaway tendency).
    starts = [
        ("A: below separatrix", 3.0, 0.30, "#4c72b0"),
        ("B: near separatrix",  5.0, 0.55, "#55a868"),
        ("C: above separatrix", 7.0, 0.80, "#c44e52"),
    ]
    n_per_group = 30

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                    gridspec_kw={"width_ratios": [1.5, 1.0]})

    group_data = []  # (label, color, P, XI, exit_step)
    for label, p0v, xi0v, color in starts:
        p0 = np.full(n_per_group, p0v)
        xi0 = np.full(n_per_group, xi0v)
        P, XI, exit_step = simulate_paths_nojump(p0, xi0, cfg_nj, rng,
                                                  n_steps, dt_sub)
        group_data.append((label, color, P, XI, exit_step))

        for i in range(n_per_group):
            k_end = exit_step[i]
            axL.plot(P[:k_end + 1, i], XI[:k_end + 1, i], color=color,
                      lw=0.6, alpha=0.35, zorder=2)
        axL.plot(p0v, xi0v, "o", color="black", ms=6, zorder=4)

        surv = exit_step == n_steps
        therm = (~surv) & (P[exit_step, np.arange(n_per_group)] <= cfg.p_min + 1e-6)
        run = (~surv) & (~therm)
        n_surv, n_therm, n_run = surv.sum(), therm.sum(), run.sum()
        print(f"{label}: start (p={p0v}, xi={xi0v}) | "
              f"survived={n_surv}/{n_per_group}, "
              f"thermalized={n_therm}, runaway={n_run}")

        # mark absorbed endpoints
        idx_therm = np.where(therm)[0]
        idx_run = np.where(run)[0]
        for i in idx_therm:
            axL.plot(P[exit_step[i], i], XI[exit_step[i], i], "x",
                      color=color, ms=7, mew=1.6, zorder=5)
        for i in idx_run:
            axL.plot(P[exit_step[i], i], XI[exit_step[i], i], "+",
                      color=color, ms=9, mew=1.8, zorder=5)

    axL.axvline(cfg.p_min, color="0.4", lw=1.2, ls=":")
    axL.axvline(cfg.p_max, color="0.4", lw=1.2, ls=":")
    axL.axhline(1.0, color="0.4", lw=1.0, ls=":")
    axL.axhline(-1.0, color="0.4", lw=1.0, ls=":")
    axL.set_xlim(cfg.p_min - 0.3, cfg.p_max + 0.3)
    axL.set_ylim(-1.05, 1.05)
    axL.set_xlabel(r"$p$")
    axL.set_ylabel(r"$\xi=\cos\theta$")
    axL.set_title(f"{n_per_group} paths/start, jumps OFF, "
                  f"$T={T_total:g}\\,\\tau_c$\n"
                  r"($\times$ = thermalized, $+$ = runaway, "
                  "$\\bullet$ = start)", fontsize=10.5)

    # ---- right panel: ensemble variance of xi and p vs. time, group B
    # (astride the separatrix -- the most dynamically active start).
    # This is the quantitative version of "visualize the stochastic
    # diffusion": sigma^2(t) growth curves, computed only over particles
    # still alive at each time (so absorbed particles do not bias the
    # in-domain spread). ----
    label, color, P, XI, exit_step = group_data[1]
    snap_idx = np.unique(np.linspace(0, n_steps, 60).astype(int))
    t_snap = snap_idx * dt_sub
    var_xi = np.full(len(snap_idx), np.nan)
    var_p = np.full(len(snap_idx), np.nan)
    n_live = np.empty(len(snap_idx), dtype=int)
    for j, k in enumerate(snap_idx):
        live = exit_step > k
        n_live[j] = int(live.sum())
        if n_live[j] > 1:
            var_xi[j] = np.var(XI[k, live])
            var_p[j] = np.var(P[k, live])

    axR.plot(t_snap, var_xi, color="#55a868", lw=2.0, label=r"Var$(\xi)$")
    axR.set_xlabel(r"$t$  (units of $\tau_c$)")
    axR.set_ylabel(r"Var$(\xi)$", color="#55a868")
    axR.tick_params(axis="y", labelcolor="#55a868")
    axR2 = axR.twinx()
    axR2.plot(t_snap, var_p, color="#8172b2", lw=2.0, ls="--",
              label=r"Var$(p)$")
    axR2.set_ylabel(r"Var$(p)$", color="#8172b2")
    axR2.tick_params(axis="y", labelcolor="#8172b2")
    lines = axR.get_lines() + axR2.get_lines()
    axR.legend(lines, [l.get_label() for l in lines], fontsize=9,
               loc="upper left")
    axR.set_title("Group B: ensemble variance vs. time\n"
                  "(no direct noise on $p$, yet Var$(p)$ grows FASTER than "
                  "Var$(\\xi)$)", fontsize=10)

    k_mid = n_steps // 2
    live_mid = exit_step > k_mid
    var_xi_mid = np.var(XI[k_mid, live_mid]) if live_mid.sum() > 1 else np.nan
    var_p_mid = np.var(P[k_mid, live_mid]) if live_mid.sum() > 1 else np.nan
    var_xi_0, var_p_0 = np.var(XI[0]), np.var(P[0])
    print(f"\nGroup B variance growth by t={T_total/2:g} "
          f"(n_live={int(live_mid.sum())}/{n_per_group}):")
    print(f"  Var(xi): {var_xi_0:.5f} -> {var_xi_mid:.5f}  "
          f"(delta={var_xi_mid - var_xi_0:.5f})  "
          f"[direct dW term, but compressed against the xi=1 reflecting "
          f"wall by the field torque]")
    print(f"  Var(p) : {var_p_0:.5f} -> {var_p_mid:.5f}  "
          f"(delta={var_p_mid - var_p_0:.5f})  "
          f"[NO direct dW term -- spread is entirely inherited from the "
          f"time-integral of the diffusing xi through b_p]")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig03_brownian_traj.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
