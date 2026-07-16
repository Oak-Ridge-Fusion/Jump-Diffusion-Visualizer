"""
inference.py  --  combine Exit Classifier + Flow Map -> B0 / B1 / B2 (Problem 1)
================================================================================
One coarse step of the learned model is EXIT, THEN PROPAGATE:
    - the exit head E(x) removes particles that cross a boundary this step,
    - the flow map advances the survivors (trained on survivors only, so it is
      never evaluated out-of-domain).

Reproduces the three Problem-1 tests:
  B0  one-step transition from x0 = 1 and 5, as SUB-probability densities
      (normalized by particles STARTED, so exit mass shows as missing area),
      overlaid with the free-space jump-count mixture (closed form).
  B1  exit-probability curve P(exit|x) (head vs MC) + survivor densities at x=1.5,3.
  B2  rollout to T=4 from both starts: surviving fraction per step + terminal density.

Ground truth = the code_1d fine integrator (simulate_bounded_step).
Needs:  <data_dir>/ckpt_flow.pt   (from flow_training.py)
        <data_dir>/ckpt_exit.pt   (from exit_training.py)

Run:  python inference.py   ->  B0_onestep.png, B1_onestep.png, B2_rollout.png, metrics.json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "code_1d"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from common_1 import (Config, simulate_bounded_step, bounded_free_logpdf,
                      wasserstein1, get_device, set_seed)
from exit_net import ExitNet
from flow_training import load_flow_map, flow_step   # reuse the trained flow map


# ---------------------------------------------------------------------------
class ExitHead:
    """Loads the trained P(exit during dt | x) classifier (your ExitNet)."""
    def __init__(self, cfg, dev):
        ck = torch.load(os.path.join(cfg.data_dir, "ckpt_exit.pt"),
                        map_location=dev, weights_only=False)
        self.net = ExitNet().to(dev)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        # if the exit head was trained on raw x, these default to a no-op
        self.xm = ck.get("x_mean", 0.0)
        self.xs = ck.get("x_std", 1.0)
        self.dev = dev

    @torch.no_grad()
    def prob(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        xn = torch.tensor((x - self.xm) / self.xs, dtype=torch.float32, device=self.dev)
        return self.net(xn).cpu().numpy().ravel()


def subprob(samples_in_domain, surv_frac, bins):
    """Sub-probability density: a density=True histogram scaled by the survival
    fraction, so it integrates to surv_frac (exit mass = missing area)."""
    h, _ = np.histogram(samples_in_domain, bins=bins, density=True)
    return h * surv_frac


# ---------------------------------------------------------------------------
def b0_onestep(net, ck, eh, cfg, dev, rng, metrics):
    starts = list(cfg.transition_test_x)          # (1.0, 5.0)
    N = 120000
    bins = np.linspace(cfg.x_min, cfg.x_max, 90)
    centers = 0.5 * (bins[:-1] + bins[1:])
    dx = bins[1] - bins[0]

    fig, ax = plt.subplots(1, len(starts), figsize=(6.2 * len(starts), 4.4))
    rows = []
    for k, x0 in enumerate(starts):
        # --- numerical bounded ground truth (sub-probability) ---
        xe, al = simulate_bounded_step(np.full(N, x0), cfg, rng)
        sp_num = subprob(xe[al], al.mean(), bins)

        # --- ML: exit head removes p_exit, flow map propagates the rest ---
        pe = float(eh.prob([x0])[0])
        n_surv = int(round((1.0 - pe) * N))
        if n_surv > 0:
            xs = flow_step(np.full(n_surv, x0), net, ck, dev, rng)
            ind = (xs >= cfg.x_min) & (xs <= cfg.x_max)
            surv_frac_ml = (1.0 - pe) * ind.mean()
            sp_ml = subprob(xs[ind], surv_frac_ml, bins)
        else:
            surv_frac_ml, sp_ml = 0.0, np.zeros_like(centers)

        # --- free-space jump-count mixture (full density, for reference) ---
        p_free = np.exp(bounded_free_logpdf(centers, x0, cfg))

        ax[k].fill_between(centers, sp_num, alpha=0.35, color="gray",
                           label="numerical (sub-prob)")
        ax[k].plot(centers, sp_ml, "r-", lw=1.5, label="ML: exit head + flow")
        ax[k].plot(centers, p_free, "k--", lw=1.0, label="free-space mixture")
        ax[k].set_title(f"B0: one-step from x0={x0:g}\n"
                        f"survival  num={al.mean():.3f}  ML={surv_frac_ml:.3f}")
        ax[k].set_xlabel("x_next"); ax[k].legend(fontsize=8)
        rows.append({"x0": x0, "surv_num": float(al.mean()),
                     "surv_ml": float(surv_frac_ml),
                     "exit_head": pe, "exit_true": float((~al).mean())})
    metrics["b0"] = rows
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.data_dir, "B0_onestep.png"), dpi=130)
    plt.close(fig)


def b1(net, ck, eh, cfg, dev, rng, metrics):
    # exit-probability curve: head vs MC truth
    grid = np.linspace(cfg.x_min, cfg.x_max, 121)
    pe_pred = eh.prob(grid)
    pe_true = np.empty_like(grid)
    for i, xv in enumerate(grid):
        _, al = simulate_bounded_step(np.full(3000, xv), cfg, rng)  # bump for smoother curve
        pe_true[i] = np.mean(~al)
    metrics["b1_exit_max_abs_err"] = float(np.max(np.abs(pe_pred - pe_true)))
    i5 = int(np.argmin(np.abs(grid - 5.0)))
    metrics["b1_exit_at_x5"] = {"head": float(pe_pred[i5]), "true": float(pe_true[i5])}

    test_x = [1.5, 3.0]
    N = 80000
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(grid, pe_true, "k-", label="numerical (MC)")
    ax[0].plot(grid, pe_pred, "r--", label="exit head")
    ax[0].set_title("B1: exit probability  P(exit | x)")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("P(exit)"); ax[0].legend()

    rows = []
    bins = np.linspace(cfg.x_min, cfg.x_max, 70)
    for j, x0 in enumerate(test_x):
        xe, al = simulate_bounded_step(np.full(N, x0), cfg, rng)
        truth = xe[al]                                   # numerical survivors
        ml = flow_step(np.full(N, x0), net, ck, dev, rng)
        ml = ml[(ml >= cfg.x_min) & (ml <= cfg.x_max)]   # survivor flow map
        w1 = wasserstein1(truth, ml)
        rows.append({"x0": x0, "W1": w1})
        ax[1 + j].hist(truth, bins=bins, density=True, alpha=0.35, color="gray",
                       label="numerical survivors")
        ax[1 + j].hist(ml, bins=bins, density=True, histtype="step", color="C3",
                       label="flow map")
        ax[1 + j].set_title(f"B1: survivor density | x={x0:g}  (W1={w1:.3f})")
        ax[1 + j].set_xlabel("x_next"); ax[1 + j].legend(fontsize=8)
    metrics["b1_survivor_W1"] = rows
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.data_dir, "B1_onestep.png"), dpi=130)
    plt.close(fig)


def rollout_ml(net, ck, eh, cfg, x0, N, dev, rng):
    """exit, then propagate -- from N particles all started at x0."""
    x = np.full(N, x0, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    surv = []
    for _ in range(cfg.rollout_K):
        idx = np.where(alive)[0]
        if len(idx):
            ex = rng.random(len(idx)) < eh.prob(x[idx])
            alive[idx[ex]] = False
            keep = idx[~ex]
            if len(keep):
                xn = flow_step(x[keep], net, ck, dev, rng)
                oob = (xn < cfg.x_min) | (xn > cfg.x_max)
                alive[keep[oob]] = False
                x[keep[~oob]] = xn[~oob]
        surv.append(float(alive.mean()))
    return x, alive, surv


def rollout_num(cfg, x0, N, rng):
    x = np.full(N, x0, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    surv = []
    for _ in range(cfg.rollout_K):
        idx = np.where(alive)[0]
        if len(idx):
            xe, al = simulate_bounded_step(x[idx], cfg, rng)
            alive[idx[~al]] = False
            x[idx[al]] = xe[al]
        surv.append(float(alive.mean()))
    return x, alive, surv


def b2(net, ck, eh, cfg, dev, rng, metrics):
    starts = list(cfg.transition_test_x)
    N = cfg.n_ref_mc
    K = cfg.rollout_K
    T = K * cfg.dt

    fig, ax = plt.subplots(len(starts), 2, figsize=(11, 4.3 * len(starts)))
    if len(starts) == 1:
        ax = ax[None, :]
    rows = []
    t_ml, t_num = 0.0, 0.0
    for r, x0 in enumerate(starts):
        t0 = time.perf_counter()
        xl, al_l, surv_l = rollout_ml(net, ck, eh, cfg, x0, N, dev, rng)
        t_ml += time.perf_counter() - t0
        t0 = time.perf_counter()
        xn, al_n, surv_n = rollout_num(cfg, x0, N, rng)
        t_num += time.perf_counter() - t0

        term_num, term_ml = xn[al_n], xl[al_l]
        w1 = (wasserstein1(term_num, term_ml)
              if len(term_num) > 1 and len(term_ml) > 1 else float("nan"))
        rows.append({"x0": x0, "surv_final_num": surv_n[-1],
                     "surv_final_ml": surv_l[-1], "terminal_W1": w1})

        steps = np.arange(1, K + 1)
        ax[r, 0].plot(steps, surv_n, "k-o", label="numerical")
        ax[r, 0].plot(steps, surv_l, "r--s", label="ML rollout")
        ax[r, 0].set_title(f"B2: surviving fraction, x0={x0:g}")
        ax[r, 0].set_xlabel("coarse step"); ax[r, 0].set_ylabel("survival")
        ax[r, 0].legend()
        bins = np.linspace(cfg.x_min, cfg.x_max, 70)
        if len(term_num) > 1:
            ax[r, 1].hist(term_num, bins=bins, density=True, alpha=0.35,
                          color="gray", label="numerical")
        if len(term_ml) > 1:
            ax[r, 1].hist(term_ml, bins=bins, density=True, histtype="step",
                          color="C3", label="ML rollout")
        ax[r, 1].set_title(f"B2: terminal density @T={T:g}, x0={x0:g}  (W1={w1:.3f})")
        ax[r, 1].set_xlabel("x"); ax[r, 1].legend(fontsize=8)

    metrics["b2"] = {"K": K, "T": T, "per_start": rows,
                     "ml_time_s": t_ml, "numerical_time_s": t_num,
                     "speedup": t_num / max(t_ml, 1e-9),
                     "step_ratio": f"{K} vs {K * cfg.n_sub}",
                     "note": "numerical CPU, ML on " + str(dev)}
    fig.suptitle(f"B2: ML {K} big steps vs numerical {K * cfg.n_sub} small steps "
                 f"(speedup {t_num / max(t_ml, 1e-9):.0f}x)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.data_dir, "B2_rollout.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def resolve_data_dir(cfg):
    candidates = []
    if not os.path.isabs(cfg.data_dir):
        candidates.extend([
            os.path.join(SCRIPT_DIR, cfg.data_dir),
            os.path.join(CODE_DIR, cfg.data_dir),
            os.path.join(os.getcwd(), cfg.data_dir),
            os.path.join(os.getcwd(), "..", cfg.data_dir),
        ])
    candidates.append(cfg.data_dir)
    for path in candidates:
        if os.path.exists(os.path.join(path, "ckpt_flow.pt")) and os.path.exists(os.path.join(path, "ckpt_exit.pt")):
            return os.path.abspath(path)
    return os.path.abspath(os.path.join(CODE_DIR, cfg.data_dir))


def main():
    cfg = Config(model="bounded")            # Problem 1
    cfg.data_dir = resolve_data_dir(cfg)
    set_seed(cfg.seed + 2)
    rng = np.random.default_rng(cfg.seed + 2)
    dev = get_device(cfg)
    print("device:", dev, "| model:", cfg.model, "| data_dir:", cfg.data_dir)

    net, ck = load_flow_map(cfg, dev)
    eh = ExitHead(cfg, dev)
    metrics = {"model": cfg.model, "dt": cfg.dt}

    print("B0: one-step sub-probability density ...");  b0_onestep(net, ck, eh, cfg, dev, rng, metrics)
    print("B1: exit curve + survivor density ...");     b1(net, ck, eh, cfg, dev, rng, metrics)
    print("B2: long rollout to T ...");                 b2(net, ck, eh, cfg, dev, rng, metrics)

    with open(os.path.join(cfg.data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(metrics, indent=2))
    print("\nWrote B0_onestep.png, B1_onestep.png, B2_rollout.png, metrics.json to",
          cfg.data_dir)


if __name__ == "__main__":
    main()