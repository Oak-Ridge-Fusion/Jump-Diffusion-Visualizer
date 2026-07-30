"""
fig09_runaway_evolution.py
===========================
Figure 9: time evolution of the RUNAWAY probability P_RE(p, xi) = P(exit
through p_max by coarse step k), at three snapshots across the rollout
horizon T = K*dt -- the 2D analog of Yang et al. (JCP 444, 110564, 2021)
Figs. 6/7, which show the same quantity for their 3D (p, theta, r) model as
P_RE(theta, p) / P_RE(r, p) slices at t = 24, 60, 120 (i.e. at fractions
0.2, 0.5, 1.0 of their T_max=120).  We use the same fractions of our
rollout horizon T = cfg.rollout_K * cfg.dt.

Unlike that paper (which compares a coarse vs. a refined DETERMINISTIC grid
in r, Figs. 6 vs 7), our two columns compare the REFERENCE fine integrator
(common_2.simulate_re2d_step, ground truth) against the trained ML
surrogate (ExitNet + FN_Net, exit-then-propagate, same as inference.py's
rollout_ml) -- this project's actual axis of comparison is numerical vs.
learned, not coarse-grid vs. fine-grid.

Method: tile n_mc Monte Carlo particles at every point of the same
(p, xi) grid fig08 uses (cfg.b1_grid), then advance ALL grid points'
particles together, one coarse step at a time, recording the cumulative
per-grid-point runaway fraction at k in {checkpoints}. This is exactly
fig08's one-step map computation, just iterated k times with the running
alive/side state carried forward (so grid points don't need independent
loops -- everything is one vectorized particle array of size
n_p*n_xi*n_mc, same trick fig08 and inference.py's b1_exit_maps use).

Needs: <code_2d>/artifacts_re2d/{ckpt_flow.pt, ckpt_exit.pt}
Run:   python fig09_runaway_evolution.py [n_mc]
Out:   problem3/figures/fig09_runaway_evolution.png
"""

import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import (Config, CODE_2D_DIR, re2d_coeffs,
                           simulate_re2d_step, get_device, set_seed)
from flow_training import load_flow_map, flow_step
from inference import ExitHead

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def grid_runaway_evolution_numerical(cfg, PP, XX, n_mc, checkpoints, rng):
    n_xi, n_p = PP.shape
    p = np.repeat(PP.ravel(), n_mc)
    xi = np.repeat(XX.ravel(), n_mc)
    alive = np.ones(p.shape, dtype=bool)
    side = np.zeros(p.shape, dtype=np.int64)
    K = max(checkpoints)
    maps = {}
    t0 = time.time()
    for k in range(1, K + 1):
        idx = np.where(alive)[0]
        if len(idx):
            pe, xe, al, sd = simulate_re2d_step(p[idx], xi[idx], cfg, rng)
            p[idx], xi[idx], alive[idx] = pe, xe, al
            side[idx[~al]] = sd[~al]
        if k in checkpoints:
            side_grid = side.reshape(n_xi * n_p, n_mc)
            maps[k] = (side_grid == 2).mean(axis=1).reshape(n_xi, n_p)
        print(f"  [numerical] step {k}/{K}  alive={alive.mean():.3f}  "
              f"elapsed={time.time()-t0:.1f}s")
    return maps


def grid_runaway_evolution_ml(cfg, PP, XX, n_mc, checkpoints, net, ck, eh, dev, rng):
    n_xi, n_p = PP.shape
    p0 = np.repeat(PP.ravel(), n_mc)
    xi0 = np.repeat(XX.ravel(), n_mc)
    state = np.stack([p0, xi0], axis=1)
    alive = np.ones(len(state), dtype=bool)
    side = np.zeros(len(state), dtype=np.int64)
    K = max(checkpoints)
    maps = {}
    t0 = time.time()
    for k in range(1, K + 1):
        idx = np.where(alive)[0]
        if len(idx):
            probs = eh.probs(state[idx])          # (m,3) [stay, therm, run]
            u = rng.random(len(idx))
            cum0 = probs[:, 0]
            cum1 = cum0 + probs[:, 1]
            cls = np.zeros(len(idx), dtype=np.int64)
            cls[u >= cum0] = 1
            cls[u >= cum1] = 2

            exit_mask = cls != 0
            alive[idx[exit_mask]] = False
            side[idx[cls == 1]] = 1
            side[idx[cls == 2]] = 2

            keep = idx[~exit_mask]
            if len(keep):
                nxt = flow_step(state[keep], net, ck, dev, rng)
                oob_lo = nxt[:, 0] < cfg.p_min
                oob_hi = nxt[:, 0] > cfg.p_max
                if oob_lo.any():
                    alive[keep[oob_lo]] = False
                    side[keep[oob_lo]] = 1
                if oob_hi.any():
                    alive[keep[oob_hi]] = False
                    side[keep[oob_hi]] = 2
                ok = ~(oob_lo | oob_hi)
                state[keep[ok]] = nxt[ok]
        if k in checkpoints:
            side_grid = side.reshape(n_xi * n_p, n_mc)
            maps[k] = (side_grid == 2).mean(axis=1).reshape(n_xi, n_p)
        print(f"  [ml]        step {k}/{K}  alive={alive.mean():.3f}  "
              f"elapsed={time.time()-t0:.1f}s")
    return maps


def main():
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 400

    cfg = Config()
    cfg.data_dir = os.path.join(CODE_2D_DIR, cfg.data_dir)
    set_seed(cfg.seed + 5)
    dev = get_device(cfg)
    print("device:", dev, "| data_dir:", cfg.data_dir, "| n_mc:", n_mc)

    net, ck = load_flow_map(cfg, dev)
    eh = ExitHead(cfg, dev)

    n_p, n_xi = cfg.b1_grid
    p_grid = np.linspace(cfg.p_min, cfg.p_max, n_p)
    xi_grid = np.linspace(-1.0, 1.0, n_xi)
    PP, XX = np.meshgrid(p_grid, xi_grid)             # (n_xi, n_p)

    K = cfg.rollout_K
    checkpoints = sorted({max(1, round(0.2 * K)), max(1, round(0.5 * K)), K})
    print("checkpoints (coarse steps):", checkpoints,
          "-> T =", [k * cfg.dt for k in checkpoints])

    rng_num = np.random.default_rng(cfg.seed + 6)
    print("running numerical (reference fine integrator) rollout ...")
    num_maps = grid_runaway_evolution_numerical(cfg, PP, XX, n_mc, checkpoints, rng_num)

    rng_ml = np.random.default_rng(cfg.seed + 7)
    print("running ML (exit-then-propagate) rollout ...")
    ml_maps = grid_runaway_evolution_ml(cfg, PP, XX, n_mc, checkpoints, net, ck, eh, dev, rng_ml)

    # deterministic separatrix (b_p = 0), for overlay, as in fig02/fig06/fig08
    p_fine = np.linspace(cfg.p_min, cfg.p_max, 300)
    xi_fine = np.linspace(-1.0, 1.0, 300)
    PF, XF = np.meshgrid(p_fine, xi_fine)
    b_p_fine, _, _ = re2d_coeffs(PF, XF, cfg)

    fig, axes = plt.subplots(len(checkpoints), 2,
                             figsize=(10.5, 4.6 * len(checkpoints)),
                             squeeze=False)
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0)
    errs = {}
    for r, k in enumerate(checkpoints):
        T = k * cfg.dt
        err = float(np.abs(num_maps[k] - ml_maps[k]).mean())
        errs[k] = err
        for c, (maps, label) in enumerate([(num_maps, "Numerical (reference)"),
                                            (ml_maps, "ML (exit-then-propagate)")]):
            ax = axes[r, c]
            im = ax.pcolormesh(PP, XX, maps[k], shading="gouraud",
                               cmap="viridis", norm=norm)
            ax.contour(PF, XF, b_p_fine, levels=[0.0], colors="white",
                      linewidths=1.5, linestyles="--", alpha=0.85)
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_ticks([0, 0.05, 0.2, 0.5, 1.0])
            ax.set_xlabel(r"$p$")
            ax.set_ylabel(r"$\xi=\cos\theta$")
            ax.set_xlim(cfg.p_min, cfg.p_max)
            ax.set_ylim(-1.0, 1.0)
            title = f"{label}\n" + r"$P_{RE}(p,\xi)$" + f" at T={T:g} " r"$\tau_c$"
            if c == 1:
                title += f"  (mean |err|={err:.3f})"
            ax.set_title(title, fontsize=10.5)

    fig.suptitle(r"$P_{RE}(p,\xi)$: cumulative runaway probability by time $T$"
                 f"\n(grid {n_p}x{n_xi}, {n_mc} MC particles/pt, "
                 f"dashed/white = deterministic $b_p=0$ separatrix)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG_DIR, "fig09_runaway_evolution.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    print("mean abs error per checkpoint:", errs)


if __name__ == "__main__":
    main()
