"""
inference.py  --  Problem 3 (2D runaway electron): combine EXIT + FLOW -> B0/B1/B2
=====================================================================================
One coarse ML step is EXIT, THEN PROPAGATE (same architecture as the 1D
verification, generalized to a 3-way exit side):
  - the exit head samples a SIDE (stay / exit p_min / exit p_max) for every
    particle from its softmax P(side | p, xi);
  - particles that stay are advanced by the flow map (trained only on true
    survivor pairs, so it is only ever applied in-domain -- see
    flow_training.flow_step's docstring on why xi is reflected there but p
    is left unclipped).

Reproduces the three verification tests, 2D analogs of jump1d /
runaway_flowmap's B0/B1/B2:

  B0  one-step transition from cfg.transition_test_x: ML vs numerical
      SURVIVOR marginal densities in p and in xi, plus the full 3-class
      exit probabilities (stay/thermal/runaway), ML vs Monte Carlo.
  B1  exit-probability MAPS over the whole (p, xi) domain, at the same
      cfg.b1_grid x cfg.b1_mc resolution problem3/figures/fig08 used for
      the ground truth -- this is literally the surface exit_training.py
      was trained to approximate; compares ExitNet vs fresh MC directly,
      panel by panel, plus reports max/mean abs error.
  B2  rollout to T = K*dt: exit-then-propagate vs
      common_2.simulate_re2d_rollout (already implemented in common_2.py),
      comparing surviving/thermalized/runaway fraction per step and the
      terminal (p, xi) distribution (sliced W1, common_2.sliced_w1 -- a 2D
      generalization of the 1D W1 metric).

Ground truth throughout = the fine sub-stepped reference integrator
(common_2.simulate_re2d_step / simulate_re2d_rollout), unchanged.

Needs: <code_2d>/artifacts_re2d/{ckpt_flow.pt, ckpt_exit.pt}
       (flow_training.py, exit_training.py)

Run:  python inference.py
Out:  <code_2d>/artifacts_re2d/{B0_onestep.png, B1_exit_maps.png,
      B2_rollout.png, metrics.json}
"""

import json
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import (Config, CODE_2D_DIR, simulate_re2d_step,
                           simulate_re2d_rollout, exit_features, ExitNet,
                           get_device, set_seed, sliced_w1)
from flow_training import load_flow_map, flow_step


# ---------------------------------------------------------------------------
class ExitHead:
    """Loads the trained 3-class P(side | p, xi) classifier."""
    def __init__(self, cfg, dev):
        ck = torch.load(os.path.join(cfg.data_dir, "ckpt_exit.pt"),
                        map_location=dev, weights_only=False)
        self.net = ExitNet(cfg).to(dev)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.xm, self.xs = ck["x_mean"], ck["x_std"]
        self.dev = dev
        self.cfg = cfg

    @torch.no_grad()
    def probs(self, X):
        """X:(N,2)=(p,xi) -> (N,3) softmax [P(stay), P(thermal), P(runaway)]."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
        feat = exit_features(X, self.cfg)
        feat = (feat - self.xm) / self.xs
        feat = torch.tensor(feat, dtype=torch.float32, device=self.dev)
        logits = self.net(feat)
        return torch.softmax(logits, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
def b0_onestep(net, ck, eh, cfg, dev, rng, metrics):
    starts = list(cfg.transition_test_x)
    N = 120000
    p_bins = np.linspace(cfg.p_min, cfg.p_max, 80)
    xi_bins = np.linspace(-1.0, 1.0, 60)

    fig, ax = plt.subplots(len(starts), 2, figsize=(11.5, 4.4 * len(starts)))
    if len(starts) == 1:
        ax = ax[None, :]
    rows = []
    for r, (p0v, xi0v) in enumerate(starts):
        p0 = np.full(N, p0v)
        xi0 = np.full(N, xi0v)
        p_e, xi_e, alive, side = simulate_re2d_step(p0, xi0, cfg, rng)
        surv_num = float(alive.mean())
        therm_num = float((side == 1).mean())
        run_num = float((side == 2).mean())
        p_num, xi_num = p_e[alive], xi_e[alive]

        probs = eh.probs([[p0v, xi0v]])[0]      # [stay, thermal, runaway]
        n_surv = int(round(probs[0] * N))
        if n_surv > 0:
            state0 = np.tile([p0v, xi0v], (n_surv, 1))
            state1 = flow_step(state0, net, ck, dev, rng)
            in_dom = (state1[:, 0] >= cfg.p_min) & (state1[:, 0] <= cfg.p_max)
            surv_ml = float(probs[0] * in_dom.mean())
            p_ml, xi_ml = state1[in_dom, 0], state1[in_dom, 1]
            flow_escape = float((~in_dom).mean())
        else:
            surv_ml, p_ml, xi_ml, flow_escape = 0.0, np.array([]), np.array([]), 0.0

        rows.append({
            "p0": p0v, "xi0": xi0v,
            "surv_num": surv_num, "surv_ml": surv_ml,
            "exit_probs_num_stay_therm_run": [surv_num, therm_num, run_num],
            "exit_probs_ml_stay_therm_run": [float(probs[0]), float(probs[1]),
                                             float(probs[2])],
            "flow_map_escape_frac": flow_escape,
        })

        ax[r, 0].hist(p_num, bins=p_bins, density=True, alpha=0.35,
                      color="gray", label="numerical survivors")
        if len(p_ml):
            ax[r, 0].hist(p_ml, bins=p_bins, density=True, histtype="step",
                          color="C3", label="ML flow map")
        ax[r, 0].set_title(f"B0: p-marginal, start=({p0v:g},{xi0v:g})\n"
                           f"P(stay): num={surv_num:.3f}, ml={probs[0]:.3f}",
                           fontsize=10)
        ax[r, 0].set_xlabel("p"); ax[r, 0].legend(fontsize=8)

        ax[r, 1].hist(xi_num, bins=xi_bins, density=True, alpha=0.35,
                      color="gray", label="numerical survivors")
        if len(xi_ml):
            ax[r, 1].hist(xi_ml, bins=xi_bins, density=True, histtype="step",
                          color="C3", label="ML flow map")
        ax[r, 1].set_title(f"B0: xi-marginal, start=({p0v:g},{xi0v:g})\n"
                           f"P(therm): num={therm_num:.3f}, ml={probs[1]:.3f}  |  "
                           f"P(run): num={run_num:.3f}, ml={probs[2]:.3f}",
                           fontsize=10)
        ax[r, 1].set_xlabel("xi"); ax[r, 1].legend(fontsize=8)

    metrics["b0"] = rows
    fig.tight_layout()
    out = os.path.join(cfg.data_dir, "B0_onestep.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
def b1_exit_maps(eh, cfg, rng, metrics):
    n_p, n_xi = cfg.b1_grid
    n_mc = cfg.b1_mc
    p_grid = np.linspace(cfg.p_min, cfg.p_max, n_p)
    xi_grid = np.linspace(-1.0, 1.0, n_xi)
    PP, XX = np.meshgrid(p_grid, xi_grid)          # (n_xi, n_p)

    p0 = np.repeat(PP.ravel(), n_mc)
    xi0 = np.repeat(XX.ravel(), n_mc)
    _, _, alive, side = simulate_re2d_step(p0, xi0, cfg, rng)
    side_grid = side.reshape(n_xi * n_p, n_mc)
    P_therm_mc = (side_grid == 1).mean(axis=1).reshape(n_xi, n_p)
    P_run_mc = (side_grid == 2).mean(axis=1).reshape(n_xi, n_p)

    X_grid = np.stack([PP.ravel(), XX.ravel()], axis=1)
    probs_ml = eh.probs(X_grid)                    # (n_p*n_xi, 3)
    P_therm_ml = probs_ml[:, 1].reshape(n_xi, n_p)
    P_run_ml = probs_ml[:, 2].reshape(n_xi, n_p)

    err_therm = np.abs(P_therm_ml - P_therm_mc)
    err_run = np.abs(P_run_ml - P_run_mc)
    metrics["b1"] = {
        "grid": [int(n_p), int(n_xi)], "mc_per_point": int(n_mc),
        "max_abs_err_thermal": float(err_therm.max()),
        "max_abs_err_runaway": float(err_run.max()),
        "mean_abs_err_thermal": float(err_therm.mean()),
        "mean_abs_err_runaway": float(err_run.mean()),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0))
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0)
    panels = [
        (axes[0, 0], P_therm_mc, "MC truth: P(thermal)", "Blues"),
        (axes[0, 1], P_therm_ml, "ExitNet: P(thermal)", "Blues"),
        (axes[1, 0], P_run_mc, "MC truth: P(runaway)", "Reds"),
        (axes[1, 1], P_run_ml, "ExitNet: P(runaway)", "Reds"),
    ]
    for ax, Z, title, cmap in panels:
        im = ax.pcolormesh(PP, XX, Z, shading="gouraud", cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax, pad=0.02)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("p"); ax.set_ylabel("xi")
        ax.set_xlim(cfg.p_min, cfg.p_max); ax.set_ylim(-1.0, 1.0)

    fig.suptitle(f"B1: one-step exit-probability maps, ExitNet vs MC "
                 f"({n_mc} particles/pt, {n_p}x{n_xi} grid)\n"
                 f"max abs err: thermal={err_therm.max():.3f}, "
                 f"runaway={err_run.max():.3f}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(cfg.data_dir, "B1_exit_maps.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
def rollout_ml(net, ck, eh, cfg, p0v, xi0v, N, dev, rng):
    """exit-then-propagate, N particles started at (p0v, xi0v). Mirrors
    common_2.simulate_re2d_rollout's return signature (state, alive, side,
    survival, frac_lo, frac_hi) so the two are directly comparable."""
    state = np.tile([p0v, xi0v], (N, 1)).astype(np.float64)
    alive = np.ones(N, dtype=bool)
    side = np.zeros(N, dtype=np.int64)
    survival, frac_lo, frac_hi = [], [], []

    for _ in range(cfg.rollout_K):
        idx = np.where(alive)[0]
        if len(idx):
            probs = eh.probs(state[idx])          # (m,3): [stay, therm, run]
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
                # a "stay"-classified particle whose flow-map step still
                # lands outside [p_min, p_max] is a genuine flow-map error;
                # resolve it to the nearer wall's class rather than leaving
                # an out-of-domain particle "alive" (see flow_step's
                # docstring: p is intentionally left unclipped so this can
                # be caught here, not hidden).
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

        survival.append(float(alive.mean()))
        frac_lo.append(float((side == 1).mean()))
        frac_hi.append(float((side == 2).mean()))

    return state, alive, side, survival, frac_lo, frac_hi


def b2_rollout(net, ck, eh, cfg, dev, rng, metrics):
    starts = list(cfg.transition_test_x)
    N = cfg.n_ref_mc
    K = cfg.rollout_K
    T = K * cfg.dt

    fig, ax = plt.subplots(len(starts), 3, figsize=(15.5, 4.6 * len(starts)))
    if len(starts) == 1:
        ax = ax[None, :]
    rows = []
    t_ml = t_num = 0.0
    for r, (p0v, xi0v) in enumerate(starts):
        p0 = np.full(N, p0v)
        xi0 = np.full(N, xi0v)

        t0 = time.perf_counter()
        p_n, xi_n, al_n, sd_n, surv_n, lo_n, hi_n = simulate_re2d_rollout(
            p0, xi0, cfg, rng, K=K)
        t_num += time.perf_counter() - t0

        t0 = time.perf_counter()
        state_l, al_l, sd_l, surv_l, lo_l, hi_l = rollout_ml(
            net, ck, eh, cfg, p0v, xi0v, N, dev, rng)
        t_ml += time.perf_counter() - t0

        term_num = np.stack([p_n[al_n], xi_n[al_n]], axis=1)
        term_ml = np.stack([state_l[al_l, 0], state_l[al_l, 1]], axis=1)
        w1 = (sliced_w1(term_num, term_ml, rng)
              if len(term_num) > 1 and len(term_ml) > 1 else float("nan"))

        rows.append({
            "p0": p0v, "xi0": xi0v,
            "surv_final_num": surv_n[-1], "surv_final_ml": surv_l[-1],
            "thermal_final_num": lo_n[-1], "thermal_final_ml": lo_l[-1],
            "runaway_final_num": hi_n[-1], "runaway_final_ml": hi_l[-1],
            "terminal_sliced_W1": w1,
        })

        steps = np.arange(1, K + 1)
        ax[r, 0].plot(steps, surv_n, "k-o", ms=4, label="numerical survival")
        ax[r, 0].plot(steps, surv_l, "r--s", ms=4, label="ML survival")
        ax[r, 0].plot(steps, hi_n, "k-^", ms=4, alpha=0.5,
                      label="numerical runaway frac")
        ax[r, 0].plot(steps, hi_l, "r--^", ms=4, alpha=0.5,
                      label="ML runaway frac")
        ax[r, 0].set_title(f"B2: survival & runaway fraction\n"
                           f"(p0,xi0)=({p0v:g},{xi0v:g})", fontsize=10)
        ax[r, 0].set_xlabel("coarse step"); ax[r, 0].legend(fontsize=7.5)

        ax[r, 1].hist(term_num[:, 0], bins=60, range=(cfg.p_min, cfg.p_max),
                      density=True, alpha=0.35, color="gray", label="numerical")
        if len(term_ml):
            ax[r, 1].hist(term_ml[:, 0], bins=60, range=(cfg.p_min, cfg.p_max),
                          density=True, histtype="step", color="C3",
                          label="ML rollout")
        ax[r, 1].set_title(f"terminal p @ T={T:g}", fontsize=10)
        ax[r, 1].set_xlabel("p"); ax[r, 1].legend(fontsize=8)

        ax[r, 2].hist(term_num[:, 1], bins=50, range=(-1, 1), density=True,
                      alpha=0.35, color="gray", label="numerical")
        if len(term_ml):
            ax[r, 2].hist(term_ml[:, 1], bins=50, range=(-1, 1), density=True,
                          histtype="step", color="C3", label="ML rollout")
        ax[r, 2].set_title(f"terminal xi @ T={T:g}  (sliced W1={w1:.3f})",
                           fontsize=10)
        ax[r, 2].set_xlabel("xi"); ax[r, 2].legend(fontsize=8)

    metrics["b2"] = {
        "K": K, "T": T, "per_start": rows,
        "ml_time_s": t_ml, "numerical_time_s": t_num,
        "speedup": t_num / max(t_ml, 1e-9),
        "step_ratio": f"{K} vs {K * cfg.n_sub}",
        "note": "numerical CPU, ML on " + str(next(net.parameters()).device),
    }
    fig.suptitle(f"B2: ML {K} big steps vs numerical {K * cfg.n_sub} small "
                 f"steps (speedup {t_num / max(t_ml, 1e-9):.0f}x)", y=1.0)
    fig.tight_layout()
    out = os.path.join(cfg.data_dir, "B2_rollout.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
def main():
    cfg = Config()
    cfg.data_dir = os.path.join(CODE_2D_DIR, cfg.data_dir)
    set_seed(cfg.seed + 2)
    rng = np.random.default_rng(cfg.seed + 2)
    dev = get_device(cfg)
    print("device:", dev, "| data_dir:", cfg.data_dir)

    for needed in ("ckpt_flow.pt", "ckpt_exit.pt"):
        fp = os.path.join(cfg.data_dir, needed)
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"{fp} not found -- run flow_training.py / exit_training.py "
                f"first.")

    net, ck = load_flow_map(cfg, dev)
    eh = ExitHead(cfg, dev)
    metrics = {"dt": cfg.dt}

    print("B0: one-step transition ...")
    b0_onestep(net, ck, eh, cfg, dev, rng, metrics)
    print("B1: exit-probability maps ...")
    b1_exit_maps(eh, cfg, rng, metrics)
    print("B2: long rollout to T ...")
    b2_rollout(net, ck, eh, cfg, dev, rng, metrics)

    with open(os.path.join(cfg.data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(metrics, indent=2))
    print("\nWrote B0_onestep.png, B1_exit_maps.png, B2_rollout.png, "
          "metrics.json to", cfg.data_dir)


if __name__ == "__main__":
    main()
