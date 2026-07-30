"""
fig_dynamics_accuracy_bsd.py
===============================
DYNAMICAL-EVOLUTION ACCURACY figure, Problem 2 (BSD, state-dependent
coefficients) only -- per mentor feedback, one case is enough; BD is
dropped here (still available via fig_dynamics_accuracy_bd_bsd.py if
needed). At coarse rollout steps t = 1, 3, 5 (dt=1 each), compares the
trained ML surrogate's exit-then-propagate rollout against the reference
bounded numerical integrator, both on the real bounded domain
[x_min, x_max] with absorbing walls.

Convention: gray histogram = ML model, red curve = ground truth
(numerical). Both are SUB-probability densities (histogram density *
survival fraction at that step), so exit mass shows up directly as
missing area under each curve -- same convention as B0/B1_onestep.png's
subprob() helper, just repeated at 3 rollout checkpoints instead of one
step, so ML accuracy over the FULL dynamical evolution (not just the
final step) is visible and quantified (survival match + W1 per panel).

Uses the calibrated BSD exit head (artifacts_bsd/exit_calibration.npz,
built by calibrate_exit_bsd.py) automatically if present -- this removes
the small systematic survival-fraction bias the uncalibrated head showed
at t=3,5.

Needs: artifacts_bsd/{ckpt_flow.pt, ckpt_exit.pt[, exit_calibration.npz]}

Run:  python fig_dynamics_accuracy_bsd.py [N]
Out:  figures/fig_dynamics_accuracy_bsd.png
"""

import os
import sys
import time

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common_1 import Config, simulate_bounded_step, wasserstein1, get_device, set_seed
from exit_net import ExitNet
from flow_training import load_flow_map, flow_step

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL = "bounded_sd"
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
    calibrate_exit_bsd.py (artifacts_bsd/exit_calibration.npz)."""
    def __init__(self, base_head, cfg):
        self.base = base_head
        d = np.load(os.path.join(cfg.data_dir, "exit_calibration.npz"))
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
        print(f"  [numerical] t={k}  survival={alive.mean():.3f}  "
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
        print(f"  [ml]        t={k}  survival={alive.mean():.3f}  "
              f"elapsed={time.time()-t0:.1f}s")
    return snaps


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000

    cfg = Config(model=MODEL)
    set_seed(0)
    dev = get_device(cfg)
    net, ck = load_flow_map(cfg, dev)
    eh = ExitHead(cfg, dev)
    cal_path = os.path.join(cfg.data_dir, "exit_calibration.npz")
    if os.path.exists(cal_path):
        print("using calibrated BSD exit head (exit_calibration.npz)")
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

    bins = np.linspace(0.0, 6.0, 90)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, len(CHECKPOINTS), figsize=(5.4 * len(CHECKPOINTS), 4.6),
                             sharex=True, sharey=False)

    for col, k in enumerate(CHECKPOINTS):
        ax = axes[col]
        xn, sn = num_snaps[k]
        xm, sm = ml_snaps[k]
        sp_num = subprob(xn, sn, bins)
        sp_ml = subprob(xm, sm, bins)

        ax.fill_between(centers, sp_ml, alpha=0.45, color="0.55",
                        label="ML model (exit-then-propagate)")
        ax.plot(centers, sp_num, "r-", lw=1.8, label="numerical (ground truth)")
        ax.axvline(cfg.x_min, color="k", lw=1.0, alpha=0.6)
        ax.axvline(cfg.x_max, color="k", lw=1.0, alpha=0.6)
        ax.set_xlim(-0.3, 6.3)
        ax.set_title(f"t={k}  survival: num={sn:.3f}, ml={sm:.3f}\nW1={w1[k]:.3f}",
                    fontsize=11.5)
        ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("sub-probability density", fontsize=10.5)
            ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Problem 2 (BSD, state-dependent coefficients)\n"
                 "Dynamical-evolution accuracy: ML surrogate vs. numerical ground truth "
                 f"(bounded, absorbing walls at x=0,6; $x_0$={X0:g}, N={N:,})",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    out = os.path.join(FIG_DIR, "fig_dynamics_accuracy_bsd.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
