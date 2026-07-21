"""
fig04_jump_traj.py
====================
Figure 4: trajectories with the knock-on JUMP CHANNEL ENABLED
(cfg.jumps_on = True, the model default). Same drift-diffusion physics as
Figure 3, plus the compound-Poisson large-angle "knock-on" collision process
(common_2.knock_rate / apply_knock_jump). Directly comparable to Figure 3:
there, all randomness in p was second-hand (inherited from xi through the
drift coupling); here p can change by an O(1) amount in a single fine
sub-step -- a genuine discontinuity, not a diffusive step.

Physics
-------
A knock-on event is a close ("large-angle") Moller collision with a thermal
electron, transferring kinetic energy eps ~ Sigma(eps) (heavy-tailed,
~1/eps^2, common_2.sample_eps) to the target. The propagating electron keeps
the more energetic outgoing branch:
  gamma' = gamma - eps            (so p always DECREASES at a jump: the
                                    primary always loses energy to the
                                    knocked-on secondary)
  cos(theta_d) = sqrt[(gamma'-1)(gamma+1) / ((gamma'+1)(gamma-1))]
with a uniformly random azimuth, so xi also changes discontinuously.

Jump TIMING is a Poisson process with rate lambda(p) = common_2.knock_rate:
per fine sub-step dt_sub, P(fire) = 1 - exp(-lambda dt_sub), evaluated with
lambda at the PRE-jump state, matching common_2.simulate_re2d_step exactly.
A quick check (see printed rates below) shows lambda ~ 0.8-1.0 per tau_c
across the whole domain at the default eps_min=0.02 -- knock-on events are
NOT rare here; a typical trajectory should show several per T=2 tau_c.

What this script does
----------------------
Re-implements the fine sub-stepped integrator of common_2.simulate_re2d_step
(same order of operations: coefficients at the pre-jump state -> jump
thinning -> drift/diffusion using the pre-jump coefficients -> reflect xi ->
absorb p) but keeps every sub-step so the discontinuities are visible, and
separately logs each fired jump's (step, p_before, xi_before, p_after,
xi_after) for explicit annotation.

Output: problem3/figures/fig04_jump_traj.png
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


def simulate_paths_jump(p0, xi0, cfg, rng, n_steps, dt_sub):
    """Fine Euler-Maruyama integration WITH knock-on jumps, recording the
    full path plus a log of every fired jump. Mirrors
    common_2.simulate_re2d_step's per-substep order of operations exactly
    (coefficients at the pre-jump state; jump applied; THEN the pre-jump
    coefficients are used for the drift/diffusion increment -- a
    deliberate, tiny-dt_sub approximation carried over unchanged from the
    reference integrator)."""
    n = p0.shape[0]
    P = np.empty((n_steps + 1, n))
    XI = np.empty((n_steps + 1, n))
    P[0], XI[0] = p0, xi0
    p, xi = p0.copy(), xi0.copy()
    alive = np.ones(n, dtype=bool)
    exit_step = np.full(n, n_steps, dtype=np.int64)
    sqrt_dt = math.sqrt(dt_sub)
    jump_log = []  # (particle_idx, step, p_before, xi_before, p_after, xi_after)

    for k in range(n_steps):
        a = alive
        b_p, b_xi, s_xi = re2d_coeffs(p, xi, cfg)

        lam = knock_rate(p, cfg)
        fired = a & (rng.random(p.shape) < (1.0 - np.exp(-lam * dt_sub)))
        if fired.any():
            idx = np.where(fired)[0]
            p_bef, xi_bef = p[idx].copy(), xi[idx].copy()
            p_aft, xi_aft = apply_knock_jump(p[idx], xi[idx], cfg, rng)
            p[idx], xi[idx] = p_aft, xi_aft
            for j, i_ in enumerate(idx):
                jump_log.append((int(i_), k, p_bef[j], xi_bef[j],
                                  p_aft[j], xi_aft[j]))

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

    return P, XI, exit_step, jump_log


def main():
    cfg = Config()  # jumps_on=True by default
    rng = np.random.default_rng(cfg.seed)

    dt_sub = cfg.dt / cfg.n_sub
    T_total = 2.0 * cfg.dt
    n_steps = int(round(T_total / dt_sub))

    p_probe = np.array([cfg.p_min, 2.0, 5.0, cfg.p_max])
    lam_probe = knock_rate(p_probe, cfg)
    print("knock-on rate lambda(p)  [tau_c^-1]:")
    for pv, lv in zip(p_probe, lam_probe):
        print(f"  p={pv:5.2f}:  lambda={lv:.4f}  "
              f"(expected jumps over T={T_total:g}: {lv * T_total:.3f})")

    # same three starting points as Figure 3, for direct comparison
    starts = [
        ("A: below separatrix", 3.0, 0.30, "#4c72b0"),
        ("B: near separatrix",  5.0, 0.55, "#55a868"),
        ("C: above separatrix", 7.0, 0.80, "#c44e52"),
    ]
    n_per_group = 12

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                    gridspec_kw={"width_ratios": [1.4, 1.0]})

    all_jump_log = []
    best_traj = None  # (n_jumps, group_offset, P, XI, jump_log_local, T_total)
    offset = 0
    for label, p0v, xi0v, color in starts:
        p0 = np.full(n_per_group, p0v)
        xi0 = np.full(n_per_group, xi0v)
        P, XI, exit_step, jlog = simulate_paths_jump(p0, xi0, cfg, rng,
                                                      n_steps, dt_sub)

        for i in range(n_per_group):
            k_end = exit_step[i]
            axL.plot(P[:k_end + 1, i], XI[:k_end + 1, i], color=color,
                      lw=0.7, alpha=0.45, zorder=2)
        axL.plot(p0v, xi0v, "o", color="black", ms=6, zorder=4)

        surv = exit_step == n_steps
        therm = (~surv) & (P[exit_step, np.arange(n_per_group)] <= cfg.p_min + 1e-6)
        run = (~surv) & (~therm)
        print(f"{label}: survived={surv.sum()}/{n_per_group}, "
              f"thermalized={therm.sum()}, runaway={run.sum()}, "
              f"jumps fired={len(jlog)}")
        for i in np.where(therm)[0]:
            axL.plot(P[exit_step[i], i], XI[exit_step[i], i], "x",
                      color=color, ms=7, mew=1.6, zorder=5)
        for i in np.where(run)[0]:
            axL.plot(P[exit_step[i], i], XI[exit_step[i], i], "+",
                      color=color, ms=9, mew=1.8, zorder=5)

        # highlight every jump segment: a short black connector between the
        # pre- and post-jump state, so the discontinuity is unmistakable
        # against the much smaller diffusive wiggles
        for (i_, k_, pb, xb, pa, xa) in jlog:
            axL.plot([pb, pa], [xb, xa], color="black", lw=1.1,
                      alpha=0.8, zorder=3)
        all_jump_log.append((label, jlog))

        # track the trajectory with the most jumps, for the right-hand panel
        counts = np.zeros(n_per_group, dtype=int)
        for (i_, *_ ) in jlog:
            counts[i_] += 1
        i_best = int(np.argmax(counts))
        if best_traj is None or counts[i_best] > best_traj[0]:
            local_log = [ev for ev in jlog if ev[0] == i_best]
            best_traj = (counts[i_best], label, P[:, i_best].copy(),
                         XI[:, i_best].copy(), local_log)

    axL.axvline(cfg.p_min, color="0.4", lw=1.2, ls=":")
    axL.axvline(cfg.p_max, color="0.4", lw=1.2, ls=":")
    axL.axhline(1.0, color="0.4", lw=1.0, ls=":")
    axL.axhline(-1.0, color="0.4", lw=1.0, ls=":")
    axL.set_xlim(cfg.p_min - 0.3, cfg.p_max + 0.3)
    axL.set_ylim(-1.05, 1.05)
    axL.set_xlabel(r"$p$")
    axL.set_ylabel(r"$\xi=\cos\theta$")
    axL.set_title(f"{n_per_group} paths/start, jumps ON, "
                  f"$T={T_total:g}\\,\\tau_c$\n"
                  "black segments = knock-on jumps (discontinuous)",
                  fontsize=10.5)

    # ---- right panel: p(t) and xi(t) for the single trajectory with the
    # most jump events -- the clearest possible view of a discontinuity ----
    n_j, label, Pb, XIb, jlog_b = best_traj
    t = np.arange(len(Pb)) * dt_sub
    axR2 = axR.twinx()
    axR.plot(t, Pb, color="#333333", lw=1.3, label=r"$p(t)$")
    axR2.plot(t, XIb, color="#c44e52", lw=1.1, alpha=0.75, label=r"$\xi(t)$")
    for (_, k_, pb, xb, pa, xa) in jlog_b:
        tj = k_ * dt_sub
        axR.plot([tj, tj], [pb, pa], color="black", lw=2.2, zorder=5)
        axR.plot(tj, pa, "*", color="black", ms=10, zorder=6)
    axR.set_xlabel(r"$t$  (units of $\tau_c$)")
    axR.set_ylabel(r"$p$", color="#333333")
    axR2.set_ylabel(r"$\xi$", color="#c44e52")
    axR2.tick_params(axis="y", labelcolor="#c44e52")
    axR.set_title(f"One representative trajectory ({label}, "
                  f"{n_j} jumps)\n"
                  r"vertical segments/$\star$ = knock-on jumps in $p(t)$",
                  fontsize=10.5)
    lines = axR.get_lines() + axR2.get_lines()
    axR.legend([lines[0], lines[-1]], [r"$p(t)$", r"$\xi(t)$"],
               fontsize=9, loc="best")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig04_jump_traj.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    total_jumps = sum(len(j) for _, j in all_jump_log)
    print(f"\ntotal jumps fired across all {3 * n_per_group} trajectories: "
          f"{total_jumps}")
    print(f"best single trajectory for the right panel: {label}, "
          f"{n_j} jumps")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
