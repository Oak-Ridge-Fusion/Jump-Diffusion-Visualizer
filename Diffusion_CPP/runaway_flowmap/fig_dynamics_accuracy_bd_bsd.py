"""
fig_dynamics_accuracy_bd_bsd.py
==================================
Combined BD (Problem 1) + BSD (Problem 2) DYNAMICAL-EVOLUTION ACCURACY
figure: at coarse rollout steps t = 1, 3, 5 (dt=1 each), compare the
trained ML surrogate's exit-then-propagate rollout against the reference
bounded numerical integrator -- both on the REAL bounded domain
[x_min, x_max] with absorbing walls (no unbounded/free-space comparison;
per mentor feedback, that axis isn't needed).

Convention (per mentor request): gray histogram = ML model, red curve =
ground truth (numerical). Both are SUB-probability densities (histogram
density * survival fraction at that step), so exit mass shows up directly
as missing area under each curve -- same convention as B0/B1_onestep.png's
subprob() helper, just repeated at 3 rollout checkpoints instead of one
step, and evaluated at (survival, W1) per checkpoint so ML accuracy over
the FULL dynamical evolution (not just the final step) is visible and
quantified.

Needs: <data_dir>/{ckpt_flow.pt, ckpt_exit.pt} for both "bounded" and
       "bounded_sd" (already trained; see artifacts_bd/, artifacts_bsd/).

Run:  python fig_dynamics_accuracy_bd_bsd.py [N]
Out:  figures/fig_dynamics_accuracy_bd_bsd.png
"""

import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scipy.interpolate import PchipInterpolator

from common_1 import Config, simulate_bounded_step, wasserstein1, get_device, set_seed
from exit_net import ExitNet
from flow_training import load_flow_map, flow_step

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

X0 = 1.0                       # shared starting point (matches transition_test_x[0])
CHECKPOINTS = [1, 3, 5]
K = max(CHECKPOINTS)


class ExitHead:
    def __init__(self, cfg, dev):
        ck = torch.load(os.path.join(cfg.data_dir, "ckpt_exit.pt"),
                        map_location=dev, weights_only=False)
        self.net = ExitNet().to(dev)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.xm = ck.get("x_mean", 0.0)
        self.xs = ck.get("x_std", 1.0)
        self.dev = dev

    @torch.no_grad()
    def prob(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        xn = torch.tensor((x - self.xm) / self.xs, dtype=torch.float32, device=self.dev)
        return self.net(xn).cpu().numpy().ravel()


class CalibratedExitHead:
    """Wraps ExitHead with the PCHIP boundary-layer correction from
    calibrate_exit_bsd.py (artifacts_bsd/exit_calibration.npz), if present."""
    def __init__(self, base_head, cfg):
        self.base = base_head
        cal_path = os.path.join(cfg.data_dir, "exit_calibration.npz")
        d = np.load(cal_path)
        self.pchip = PchipInterpolator(d["grid"], d["correction"])

    def prob(self, x):
        x = np.asarray(x, dtype=np.float64)
        return np.clip(self.base.prob(x) + self.pchip(x), 1e-4, 1.0 - 1e-4)


def subprob(samples_in_domain, surv_frac, bins):
    if len(samples_in_domain) == 0:
        return np.zeros(len(bins) - 1)
    h, _ = np.histogram(samples_in_domain, bins=bins, density=True)
    return h * surv_frac


def run_numerical(cfg, N, rng):
    x = np.full(N, X0, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    snaps = {}
    t0 = time.time()
    for k in range(1, K + 1):
        idx = np.where(alive)[0]
        if len(idx):
            xe, al = simulate_bounded_step(x[idx], cfg, rng)
            alive[idx[~al]] = False
            x[idx[al]] = xe[al]
        if k in CHECKPOINTS:
            snaps[k] = (x[alive].copy(), float(alive.mean()))
        print(f"  [{cfg.model} numerical] t={k}  survival={alive.mean():.3f}  "
              f"elapsed={time.time()-t0:.1f}s")
    return snaps


def run_ml(net, ck, eh, cfg, N, dev, rng):
    x = np.full(N, X0, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    snaps = {}
    t0 = time.time()
    for k in range(1, K + 1):
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
        if k in CHECKPOINTS:
            snaps[k] = (x[alive].copy(), float(alive.mean()))
        print(f"  [{cfg.model} ml]        t={k}  survival={alive.mean():.3f}  "
              f"elapsed={time.time()-t0:.1f}s")
    return snaps


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000

    models = [("bounded", "Problem 1 -- BD (constant coefficients)"),
              ("bounded_sd", "Problem 2 -- BSD (state-dependent coefficients)")]

    results = {}
    for model, label in models:
        print(f"=== {label} ===")
        cfg = Config(model=model)
        set_seed(0)
        dev = get_device(cfg)
        net, ck = load_flow_map(cfg, dev)
        eh = ExitHead(cfg, dev)
        if model == "bounded_sd" and os.path.exists(
                os.path.join(cfg.data_dir, "exit_calibration.npz")):
            print("  using calibrated BSD exit head (exit_calibration.npz)")
            eh = CalibratedExitHead(eh, cfg)

        rng_num = np.random.default_rng(0)
        num_snaps = run_numerical(cfg, N, rng_num)
        rng_ml = np.random.default_rng(1)
        ml_snaps = run_ml(net, ck, eh, cfg, N, dev, rng_ml)

        w1 = {}
        for k in CHECKPOINTS:
            xn, _ = num_snaps[k]
            xm, _ = ml_snaps[k]
            w1[k] = wasserstein1(xn, xm) if len(xn) > 1 and len(xm) > 1 else float("nan")
            print(f"  t={k}  W1(numerical, ML) = {w1[k]:.4f}")
        results[model] = (cfg, num_snaps, ml_snaps, w1)

    bins = np.linspace(0.0, 6.0, 90)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(2, len(CHECKPOINTS),
                             figsize=(5.0 * len(CHECKPOINTS), 7.6),
                             sharex=True)

    for row, (model, label) in enumerate(models):
        cfg, num_snaps, ml_snaps, w1 = results[model]
        for col, k in enumerate(CHECKPOINTS):
            ax = axes[row, col]
            xn, sn = num_snaps[k]
            xm, sm = ml_snaps[k]
            sp_num = subprob(xn, sn, bins)
            sp_ml = subprob(xm, sm, bins)

            ax.fill_between(centers, sp_ml, alpha=0.45, color="0.55",
                            label="ML model (exit-then-propagate)")
            ax.plot(centers, sp_num, "r-", lw=1.6, label="numerical (ground truth)")
            ax.axvline(cfg.x_min, color="k", lw=1.0, alpha=0.6)
            ax.axvline(cfg.x_max, color="k", lw=1.0, alpha=0.6)
            ax.set_xlim(-0.3, 6.3)
            ax.set_title(f"t={k}  survival: num={sn:.3f}, ml={sm:.3f}\n"
                        f"W1={w1[k]:.3f}", fontsize=10.5)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel(f"{label}\nsub-probability density", fontsize=9.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Dynamical-evolution accuracy: ML surrogate vs. numerical ground truth\n"
                 f"(both bounded, absorbing walls at x=0,6; $x_0$={X0:g}, N={N:,})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(FIG_DIR, "fig_dynamics_accuracy_bd_bsd.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
