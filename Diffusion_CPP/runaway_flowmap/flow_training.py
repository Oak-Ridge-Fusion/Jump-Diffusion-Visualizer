"""
flow_training.py
================
Train the Flow Map for SURVIVING particles (Problem 1: bounded 1D jump-diffusion).

The flow map is a generator  G(x, z) -> Δx,  z ~ N(0, I),  so that
    x_next = x + G(x, z)
is distributed like  X_{t+Δt} | X_t = x   (survivors only, in-domain -> in-domain).

WHY THIS IS TWO STAGES, NOT ONE
-------------------------------
Do NOT fit  FlowNet(x, z) -> (y - x)  directly by MSE on the raw survivor pairs
with a fresh random z per pair.  In the data z is INDEPENDENT of the target, so
the MSE-optimal map is  E[y - x | x]  -- the network ignores z and collapses to
the conditional MEAN.  You'd get one narrow bump instead of the multi-modal,
jump-induced density.

The training-free diffusion label step fixes this:
  (1) LABELS: for each conditioning x, use KNN neighbours + the probability-flow
      ODE to transport each latent z into an actual SAMPLE of the conditional
      increment.  This is what couples z to the target.  (common_1.generate_labels)
  (2) DISTILL: now MSE-fit FlowNet on those (x, z) -> increment labels.

We reuse common_1's label engine so we stay aligned with the ground-truth
generator; only the distilled FlowNet is "ours".  Pass model="bounded_sd" on
the command line to get Problem 2.

Run:  python flow_training.py [model]     (model defaults to "bounded")
Out:  <cfg.data_dir>/ckpt_flow.pt   (artifacts_bd/ or artifacts_bsd/)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common_1 import Config, generate_labels, get_device, set_seed
from flow_net import FlowNet


def main():
    cfg = Config(model=sys.argv[1]) if len(sys.argv) > 1 else Config()
    cfg.data_dir = os.path.join(SCRIPT_DIR, cfg.data_dir)
    set_seed(cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 1)
    dev = get_device(cfg)
    print("device:", dev, "| model:", cfg.model, "| data_dir:", cfg.data_dir)

    # ---- survivor pairs (in-domain -> in-domain) from data_generation_1 ----
    d = np.load(os.path.join(cfg.data_dir, "data_pairs.npz"))
    x = d["x_sample"].astype(np.float64).reshape(-1, 1)
    y = d["y_sample"].astype(np.float64).reshape(-1, 1)
    target = (y - x) * cfg.diff_scale          # learn the SCALED increment

    # ---- STAGE 1: training-free diffusion labels  (x, z) -> increment ------
    print(f"stage 1: generating labels "
          f"({cfg.train_size_labels} pts x {cfg.ode_steps} ODE steps) ...")
    c0, zT, incr = generate_labels(x.astype(np.float32),
                                   target.astype(np.float32), cfg, rng, dev)
    xTrain = np.hstack([c0, zT]).astype(np.float32)   # (B, 2) = (x0, z)
    yTrain = incr.astype(np.float32)                  # (B, 1) = increment
    ok = np.isfinite(xTrain).all(1) & np.isfinite(yTrain).all(1)
    xTrain, yTrain = xTrain[ok], yTrain[ok]
    print("usable labels:", len(xTrain))

    # ---- standardize inputs/outputs (stats saved for inference) -----------
    xm, xs = xTrain.mean(0, keepdims=True), xTrain.std(0, keepdims=True) + 1e-8
    ym, ys = yTrain.mean(0, keepdims=True), yTrain.std(0, keepdims=True) + 1e-8
    Xn = torch.tensor((xTrain - xm) / xs, dtype=torch.float32, device=dev)
    Yn = torch.tensor((yTrain - ym) / ys, dtype=torch.float32, device=dev)

    perm = torch.randperm(len(Xn))
    Xn, Yn = Xn[perm], Yn[perm]
    ntr = int(0.9 * len(Xn))
    Xtr, Ytr, Xva, Yva = Xn[:ntr], Yn[:ntr], Xn[ntr:], Yn[ntr:]

    # ---- STAGE 2: distill FlowNet by MSE (best-on-validation) --------------
    net = FlowNet(input_dim=2, hidden_dim=cfg.hid_size, output_dim=1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossfn = nn.MSELoss()

    best, best_state = float("inf"), None
    for it in range(cfg.distill_iters):
        net.train()
        opt.zero_grad()
        # FlowNet.forward takes x and z as SEPARATE (N,1) tensors:
        pred = net(Xtr[:, :1], Xtr[:, 1:])
        loss = lossfn(pred, Ytr)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            v = lossfn(net(Xva[:, :1], Xva[:, 1:]), Yva).item()
        if v < best:
            best = v
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        if it % max(1, cfg.distill_iters // 5) == 0:
            print(f"  iter {it}  train={loss.item():.6f}  valid={v:.6f}")

    net.load_state_dict(best_state)

    torch.save({"state_dict": net.state_dict(),
                "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
                "diff_scale": cfg.diff_scale, "cfg": cfg.to_dict()},
               os.path.join(cfg.data_dir, "ckpt_flow.pt"))
    print(f"saved {cfg.data_dir}/ckpt_flow.pt  (best valid mse = {best:.6f})")


# ---------------------------------------------------------------------------
# Inference helper -- import these when you combine with the exit classifier
# to reproduce B0 / B1 / B2.  One call = one big Δt step for survivors.
# ---------------------------------------------------------------------------
def load_flow_map(cfg, dev):
    ck = torch.load(os.path.join(cfg.data_dir, "ckpt_flow.pt"),
                    map_location=dev, weights_only=False)
    net = FlowNet(input_dim=2, hidden_dim=cfg.hid_size, output_dim=1).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


@torch.no_grad()
def flow_step(x, net, ck, dev, rng):
    """x -> x + G(x, z)/diff_scale  (a single large-Δt transition for survivors)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    z = rng.standard_normal((len(x), 1))
    inp = ((np.hstack([x, z]) - ck["x_mean"]) / ck["x_std"]).astype(np.float32)
    inp = torch.tensor(inp, device=dev)
    out = net(inp[:, :1], inp[:, 1:]).cpu().numpy()
    incr = (out * ck["y_std"] + ck["y_mean"]) / ck["diff_scale"]
    return (x + incr).ravel()


if __name__ == "__main__":
    main()