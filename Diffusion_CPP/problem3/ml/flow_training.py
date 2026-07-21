"""
flow_training.py  --  Problem 3 (2D runaway electron): the SURVIVOR FLOW MAP
=============================================================================
Learns the large-Delta-t stochastic flow map for particles that stay inside
the domain during one coarse step: a generator

    G(p, xi, z1, z2) -> (dp, dxi),     z ~ N(0, I_2)

such that (p, xi) + G(p, xi, z)/diff_scale is distributed like
X_{t+dt} | X_t = (p, xi), conditioned on SURVIVAL (never evaluated
out-of-domain over a long rollout, since it is only ever trained on, and
only ever applied to, particles the exit head has judged to stay -- see
inference.py's exit-then-propagate architecture).

TWO STAGES, NOT ONE (same reasoning as the 1D verification, common_1.py /
runaway_flowmap)
------------------------------------------------------------------------
Fitting FlowNet(p, xi, z) -> increment directly by MSE against the raw
survivor pairs, with a FRESH random z per pair, would fail: z is
independent of the recorded target in the data, so the MSE-optimal network
just learns E[increment | p, xi] and collapses onto one smooth mean
increment -- destroying exactly the multimodal structure the knock-on jumps
create (see problem3 Figure 4). Two stages fix this:
  (1) LABELS -- common_2.generate_labels(): for each conditioning (p, xi),
      use a KNN neighbourhood of similar starts plus a probability-flow ODE
      to transport a given z into an ACTUAL SAMPLE of the conditional
      increment. This is what correlates z with the target.
  (2) DISTILL -- now a plain MSE fit of common_2.FN_Net on the (p, xi, z)
      -> increment labels reproduces that whole conditional distribution
      (feeding a fresh z at inference time samples a new increment, not
      the mean).

We reuse common_2.py's FN_Net and generate_labels UNCHANGED -- they are
already dimension-general (built for exactly this 2D case per the module
docstring); only the distillation training loop below is new.

Needs: <code_2d>/artifacts_re2d/data_pairs.npz
       (produced by `cd code_2d && python data_generation_2.py`)

Run:  python flow_training.py
Out:  <code_2d>/artifacts_re2d/ckpt_flow.pt
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, CODE_2D_DIR, generate_labels, FN_Net, get_device, set_seed


def main():
    cfg = Config()
    cfg.data_dir = os.path.join(CODE_2D_DIR, cfg.data_dir)
    set_seed(cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 1)
    dev = get_device(cfg)
    print("device:", dev, "| data_dir:", cfg.data_dir)

    data_path = os.path.join(cfg.data_dir, "data_pairs.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found -- run "
            f"`cd {CODE_2D_DIR} && python data_generation_2.py` first.")

    d = np.load(data_path)
    x = d["x_sample"].astype(np.float64)                # (N,2) = (p, xi)
    y = d["y_sample"].astype(np.float64)                 # (N,2) = (p', xi')
    scale = np.asarray(cfg.diff_scale, dtype=np.float64)
    target = (y - x) * scale                             # learn the SCALED increment
    print(f"survivor pairs: {len(x)}")

    # ---- STAGE 1: training-free diffusion labels, (p,xi,z) -> increment ----
    print(f"stage 1: generating labels "
          f"({cfg.train_size_labels} pts x {cfg.ode_steps} ODE steps) ...")
    c0, zT, incr = generate_labels(x.astype(np.float32),
                                   target.astype(np.float32), cfg, rng, dev)
    xTrain = np.hstack([c0, zT]).astype(np.float32)      # (B,4) = (p, xi, z1, z2)
    yTrain = incr.astype(np.float32)                      # (B,2) = scaled (dp, dxi)
    ok = np.isfinite(xTrain).all(1) & np.isfinite(yTrain).all(1)
    xTrain, yTrain = xTrain[ok], yTrain[ok]
    print("usable labels:", len(xTrain))

    # ---- standardize (stats saved into the checkpoint for inference) ----
    xm = xTrain.mean(0, keepdims=True)
    xs = xTrain.std(0, keepdims=True) + 1e-8
    ym = yTrain.mean(0, keepdims=True)
    ys = yTrain.std(0, keepdims=True) + 1e-8
    Xn = torch.tensor((xTrain - xm) / xs, dtype=torch.float32, device=dev)
    Yn = torch.tensor((yTrain - ym) / ys, dtype=torch.float32, device=dev)

    perm = torch.randperm(len(Xn))
    Xn, Yn = Xn[perm], Yn[perm]
    ntr = int(0.9 * len(Xn))
    Xtr, Ytr, Xva, Yva = Xn[:ntr], Yn[:ntr], Xn[ntr:], Yn[ntr:]

    # ---- STAGE 2: distill FN_Net by MSE (best-on-validation) ----
    net = FN_Net(input_dim=4, output_dim=2, hid_size=cfg.hid_size).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossfn = nn.MSELoss()

    best, best_state = float("inf"), None
    for it in range(cfg.distill_iters):
        net.train()
        opt.zero_grad()
        loss = lossfn(net(Xtr), Ytr)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            v = lossfn(net(Xva), Yva).item()
        if v < best:
            best = v
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        if it % max(1, cfg.distill_iters // 10) == 0:
            print(f"  iter {it:6d}  train={loss.item():.6f}  valid={v:.6f}")

    net.load_state_dict(best_state)

    ckpt_path = os.path.join(cfg.data_dir, "ckpt_flow.pt")
    torch.save({"state_dict": net.state_dict(),
                "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
                "diff_scale": cfg.diff_scale, "cfg": cfg.to_dict()},
               ckpt_path)
    print(f"saved {ckpt_path}  (best valid mse = {best:.6f})")


# -----------------------------------------------------------------------
# Inference helpers -- imported by exit_training.py's sibling inference.py
# to combine this flow map with the exit classifier (B0/B1/B2).
# -----------------------------------------------------------------------
def load_flow_map(cfg, dev):
    ck = torch.load(os.path.join(cfg.data_dir, "ckpt_flow.pt"),
                    map_location=dev, weights_only=False)
    net = FN_Net(input_dim=4, output_dim=2, hid_size=cfg.hid_size).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


@torch.no_grad()
def flow_step(state, net, ck, dev, rng):
    """state:(N,2)=(p,xi) -> state + increment: one large-Delta-t transition
    for SURVIVORS. xi is explicitly reflected at +-1 here: that boundary is
    physically REFLECTING (see common_2.py), not absorbing, so a small
    network overshoot past +-1 is corrected the same way the ground-truth
    integrator corrects it -- this is a legitimate physical prior, not a
    fudge. p is left UNCLIPPED: p_min/p_max are ABSORBING, so if the flow
    map (trained only on true survivors, which by construction never leave
    [p_min, p_max]) ever predicts a p outside that range, that is a genuine
    flow-map error and should be visible as one, not silently hidden by
    clamping. inference.py's B0/B2 report and exclude such cases as an
    explicit "flow-map escape" diagnostic rather than clamping them."""
    state = np.asarray(state, dtype=np.float64).reshape(-1, 2)
    n = len(state)
    z = rng.standard_normal((n, 2))
    inp = np.hstack([state, z]).astype(np.float32)
    inp = (inp - ck["x_mean"]) / ck["x_std"]
    inp = torch.tensor(inp, device=dev)
    out = net(inp).cpu().numpy()
    incr = (out * ck["y_std"] + ck["y_mean"]) / np.asarray(ck["diff_scale"])
    nxt = state + incr

    p_next, xi_next = nxt[:, 0].copy(), nxt[:, 1].copy()
    hi = xi_next > 1.0
    xi_next[hi] = 2.0 - xi_next[hi]
    lo = xi_next < -1.0
    xi_next[lo] = -2.0 - xi_next[lo]
    np.clip(xi_next, -1.0, 1.0, out=xi_next)
    return np.stack([p_next, xi_next], axis=1)


if __name__ == "__main__":
    main()
