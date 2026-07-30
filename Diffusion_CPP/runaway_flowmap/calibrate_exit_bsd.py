"""
calibrate_exit_bsd.py
========================
Diagnose + fix the small systematic exit-probability bias in the trained
BSD ExitNet (visible in fig_dynamics_accuracy_bd_bsd.py's rollout: ML
survival runs ~0.008-0.016 too HIGH at t=3,5, i.e. the net slightly
under-predicts P(exit | x) somewhere in the domain, and that one-step bias
compounds over the multi-step rollout).

Method: since x is 1D, we can just measure the true one-step exit
probability on a fine grid with a large MC sample (cheap: one call to
simulate_bounded_step per grid point's batch), compare to the trained
net's prediction on the same grid, and store the RESIDUAL
  correction(x) = MC_true(x) - net_pred(x)
as a lookup table, interpolated with a shape-preserving PCHIP spline (the
same interpolation scheme the project's own reference paper -- Yang et al.
JCP 444, 110564 (2021), Sec. 3.1.2 -- uses for its boundary-layer exit
probability, chosen there specifically because it doesn't overshoot near a
sharp front the way a plain smoothed/linear interpolant can). At
inference, calibrated_prob(x) = clip(net_pred(x) + spline(x), 0, 1). This
is a location-dependent fix (not a global Platt/isotonic-on-p rescaling),
appropriate here because the known worst-case error (B1: max abs err
0.275) is concentrated right at the boundaries' steep exit fronts, not a
uniform over/under-confidence across the whole domain.

We do NOT retrain ExitNet -- this is a cheap post-hoc correction, saved
alongside the checkpoint so it can be reloaded without re-running this
script.

Run:  python calibrate_exit_bsd.py
Out:  artifacts_bsd/exit_calibration.npz  (grid, raw_pred, mc_true, correction)
      artifacts_bsd/exit_calibration_diagnostic.png
"""

import os
import sys

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common_1 import Config, simulate_bounded_step, get_device, set_seed
from exit_net import ExitNet

MODEL = "bounded_sd"
N_GRID = 121
N_MC = 20000     # low MC noise (~0.35%) so the PCHIP fit isn't chasing sampling noise


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


def main():
    cfg = Config(model=MODEL)
    set_seed(0)
    rng = np.random.default_rng(0)
    dev = get_device(cfg)
    eh = ExitHead(cfg, dev)

    grid = np.linspace(cfg.x_min, cfg.x_max, N_GRID)
    raw_pred = eh.prob(grid)

    mc_true = np.empty(N_GRID)
    for i, xv in enumerate(grid):
        _, al = simulate_bounded_step(np.full(N_MC, xv), cfg, rng)
        mc_true[i] = float((~al).mean())

    residual = mc_true - raw_pred
    pchip = PchipInterpolator(grid, residual)
    calibrated = np.clip(raw_pred + pchip(grid), 1e-4, 1.0 - 1e-4)

    max_err_before = float(np.max(np.abs(raw_pred - mc_true)))
    max_err_after = float(np.max(np.abs(calibrated - mc_true)))
    worst_x = float(grid[np.argmax(np.abs(raw_pred - mc_true))])
    print(f"max abs err before calibration: {max_err_before:.4f}  (at x={worst_x:.2f})")
    print(f"max abs err after  calibration: {max_err_after:.4f}")

    out_npz = os.path.join(cfg.data_dir, "exit_calibration.npz")
    np.savez(out_npz, grid=grid, raw_pred=raw_pred, mc_true=mc_true,
             correction=residual)   # rebuild with PchipInterpolator(grid, correction) at load time
    print(f"saved -> {out_npz}")

    # dense query grid to check the spline doesn't overshoot between MC points
    xq = np.linspace(cfg.x_min, cfg.x_max, 1000)
    calibrated_q = np.clip(eh.prob(xq) + pchip(xq), 1e-4, 1.0 - 1e-4)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(grid, mc_true, "ko", ms=3, label="MC truth (N=%d/pt)" % N_MC)
    ax[0].plot(xq, eh.prob(xq), "r--", lw=1.4, label="ExitNet (raw)")
    ax[0].plot(xq, calibrated_q, "C0-", lw=1.4, label="ExitNet + PCHIP correction")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("P(exit | x)")
    ax[0].set_title("BSD one-step exit probability: before/after calibration")
    ax[0].legend(fontsize=8)

    ax[1].plot(grid, raw_pred - mc_true, "r--", lw=1.4, label="raw error")
    ax[1].plot(grid, calibrated - mc_true, "C0-", lw=1.4, label="calibrated error")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("x"); ax[1].set_ylabel("prediction - MC truth")
    ax[1].set_title(f"error: max |err| {max_err_before:.3f} -> {max_err_after:.3f}")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    out_png = os.path.join(cfg.data_dir, "exit_calibration_diagnostic.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
