"""
exit_training.py
================
Train the Exit Classifier  E(x) = P(exit during dt | x)  on data_exit.npz and
SAVE a checkpoint that inference.py can load.

This is your exit_net.py's ExitNet, unchanged -- the only additions are:
  * input normalization (store x_mean / x_std for inference), and
  * saving ckpt_exit.pt  ({state_dict, x_mean, x_std}).

Run:  python exit_training.py [model]   ->  <cfg.data_dir>/ckpt_exit.pt
      (model defaults to "bounded"; pass "bounded_sd" for Problem 2)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common_1 import Config, get_device, set_seed
from exit_net import ExitNet


def main():
    cfg = Config(model=sys.argv[1]) if len(sys.argv) > 1 else Config()
    cfg.data_dir = os.path.join(SCRIPT_DIR, cfg.data_dir)
    set_seed(cfg.seed + 1)
    dev = get_device(cfg)

    d = np.load(os.path.join(cfg.data_dir, "data_exit.npz"))
    x = d["x"].astype(np.float32).reshape(-1, 1)
    y = d["exited"].astype(np.float32).reshape(-1, 1)
    xm, xs = float(x.mean()), float(x.std() + 1e-8)

    Xb = torch.tensor((x - xm) / xs, dtype=torch.float32)
    Yb = torch.tensor(y, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xb, Yb), batch_size=2048, shuffle=True)

    net = ExitNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    lossfn = nn.BCELoss()

    for ep in range(50):                     # 50 > your 10: the sharp upper-wall layer needs it
        net.train()
        tot = 0.0
        for xi, yi in dl:
            xi, yi = xi.to(dev), yi.to(dev)
            opt.zero_grad()
            loss = lossfn(net(xi), yi)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yi)
        if (ep + 1) % 10 == 0:
            print(f"  [exit] epoch {ep + 1}/50  bce={tot / len(Xb):.5f}")

    net.eval()
    torch.save({"state_dict": net.state_dict(), "x_mean": xm, "x_std": xs,
                "cfg": cfg.to_dict()},
               os.path.join(cfg.data_dir, "ckpt_exit.pt"))
    print("saved", os.path.join(cfg.data_dir, "ckpt_exit.pt"))


if __name__ == "__main__":
    main()