"""
exit_training.py  --  Problem 3 (2D runaway electron): the 3-CLASS EXIT HEAD
=============================================================================
Trains E(p, xi) = softmax(stay, exit p_min, exit p_max), a 3-class
classifier for what happens to a particle at (p, xi) during ONE coarse step
dt (common_2.ExitNet). This is the direct ML target that problem3 Figure 8
visualized as ground truth (P(thermal | p, xi), P(runaway | p, xi)) -- this
script is what actually learns that surface.

Why 3 classes, not 2
--------------------
The 1D verification's domain has one kind of boundary event ("exit"); here
there are TWO physically distinct absorbing walls -- p_min (thermalization)
and p_max (confirmed runaway) -- so the label is which side, not just
whether. common_2.ExitNet already has a 3-way softmax readout for exactly
this (see its docstring); we only add the training loop and the input
featurization here.

Why exit_features, not raw (p, xi)
-----------------------------------
common_2.exit_features maps (p, xi) -> (p, xi, log(p-p_min), log(p_max-p)):
the one-step exit probability has a STEEP FRONT near each wall (confirmed
by Figure 8: it's a boundary layer, not a broad separatrix-shaped region),
and in log-wall-distance that front becomes a much gentler, more learnable
slope.

Data
----
data_exit.npz        : uniform random (p, xi) starts (data_generation_2.py)
data_exit_extra.npz  : cfg.n_exit_extra STRATIFIED starts concentrated in
                        the two boundary bands, because uniform sampling
                        under-covers the steep fronts above (see
                        data_generation_2.py's gen_exit_extra) -- both files
                        are concatenated into one training set below.

Needs: <code_2d>/artifacts_re2d/{data_exit.npz, data_exit_extra.npz}
       (produced by `cd code_2d && python data_generation_2.py`)

Run:  python exit_training.py
Out:  <code_2d>/artifacts_re2d/ckpt_exit.pt
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.physics import Config, CODE_2D_DIR, exit_features, ExitNet, get_device, set_seed


def main():
    cfg = Config()
    cfg.data_dir = os.path.join(CODE_2D_DIR, cfg.data_dir)
    set_seed(cfg.seed + 1)
    dev = get_device(cfg)
    print("device:", dev, "| data_dir:", cfg.data_dir)

    parts_x, parts_side = [], []
    for fname in ("data_exit.npz", "data_exit_extra.npz"):
        fpath = os.path.join(cfg.data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"{fpath} not found -- run "
                f"`cd {CODE_2D_DIR} && python data_generation_2.py` first.")
        d = np.load(fpath)
        parts_x.append(d["x"].astype(np.float64))
        parts_side.append(d["side"].astype(np.int64))
    X = np.concatenate(parts_x, axis=0)         # (N,2) = (p, xi)
    side = np.concatenate(parts_side, axis=0)   # (N,) in {0,1,2}
    print(f"exit labels: {len(X)} total  "
          f"(stay={np.mean(side == 0):.3f}, "
          f"thermal={np.mean(side == 1):.3f}, "
          f"runaway={np.mean(side == 2):.3f})")

    feat = exit_features(X, cfg).astype(np.float32)   # (N,4)
    xm = feat.mean(0, keepdims=True)
    xs = feat.std(0, keepdims=True) + 1e-8
    Xn = (feat - xm) / xs

    perm = np.random.default_rng(cfg.seed + 2).permutation(len(Xn))
    Xn, side = Xn[perm], side[perm]
    ntr = int(0.9 * len(Xn))
    Xtr = torch.tensor(Xn[:ntr], dtype=torch.float32)
    Ytr = torch.tensor(side[:ntr], dtype=torch.long)
    Xva = torch.tensor(Xn[ntr:], dtype=torch.float32, device=dev)
    Yva = torch.tensor(side[ntr:], dtype=torch.long, device=dev)

    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=4096, shuffle=True)

    net = ExitNet(cfg).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    # cosine LR decay + best-on-valid: Config.exit_epochs' own comment notes
    # the corrected-physics run still had validation CE falling at 120
    # epochs with a flat LR, hence the longer schedule + decay here.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.exit_epochs)
    lossfn = nn.CrossEntropyLoss()

    best, best_state = float("inf"), None
    for ep in range(cfg.exit_epochs):
        net.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossfn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sched.step()

        net.eval()
        with torch.no_grad():
            v = lossfn(net(Xva), Yva).item()
        if v < best:
            best = v
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        if (ep + 1) % max(1, cfg.exit_epochs // 10) == 0:
            print(f"  epoch {ep + 1:4d}/{cfg.exit_epochs}  "
                  f"train_ce={tot / len(Xtr):.5f}  valid_ce={v:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    net.load_state_dict(best_state)
    net.eval()

    ckpt_path = os.path.join(cfg.data_dir, "ckpt_exit.pt")
    torch.save({"state_dict": net.state_dict(), "x_mean": xm, "x_std": xs,
                "cfg": cfg.to_dict()}, ckpt_path)
    print(f"saved {ckpt_path}  (best valid CE = {best:.5f})")


if __name__ == "__main__":
    main()
