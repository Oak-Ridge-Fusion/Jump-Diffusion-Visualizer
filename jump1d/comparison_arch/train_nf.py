#!/usr/bin/env python
"""
train_nf.py
===========
Train the Conditional RealNVP normalizing flow on the ground-truth
``(X_t, X_{t+Delta t})`` transition pairs, maximising exact log-likelihood.

Run:
    python train_nf.py [--nf_epochs 200] [--batch_size 1024] [--lr 2e-4] ...

All hyperparameters in ``config.Config`` are exposed as CLI flags (see
``config.py``). Runs on CUDA automatically if available, with
``torch.cuda.amp`` mixed precision enabled by default.
"""

from __future__ import annotations

import os
import time

import torch

from config import parse_args
from dataset import build_dataloaders
from models.realnvp import ConditionalRealNVP
from utils import amp_enabled, get_device, save_checkpoint, set_seed, timed_gpu_block


def evaluate_nll(model: ConditionalRealNVP, loader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            total += model(yb, xb).sum().item()
            count += len(xb)
    model.train()
    return total / max(count, 1)


def main() -> None:
    cfg = parse_args("Train a Conditional RealNVP flow on jump-diffusion transition pairs.")
    set_seed(cfg.seed)
    device = get_device(cfg)
    use_amp = amp_enabled(cfg, device)
    print(f"[train_nf] device={device}  amp={use_amp}")
    print(f"[train_nf] data_path={cfg.data_path}")

    train_loader, val_loader, (x_norm, y_norm), meta, _ = build_dataloaders(cfg)
    print(f"[train_nf] n_train={meta['n_train']}  n_val={meta['n_val']}  "
          f"x_dim={meta['x_dim']}  y_dim={meta['y_dim']}")

    model = ConditionalRealNVP(
        dim=meta["y_dim"],
        cond_dim=meta["x_dim"],
        n_coupling=cfg.nf_n_coupling,
        hidden_dim=cfg.nf_hidden_dim,
        n_hidden_layers=cfg.nf_n_hidden_layers,
        context_dim=cfg.nf_context_dim,
        use_actnorm=cfg.nf_use_actnorm,
        coupling_type=cfg.nf_coupling_type,
        n_bins=cfg.nf_n_bins,
        tail_bound=cfg.nf_tail_bound,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"train_nll": [], "val_nll": [], "epoch_time_s": []}
    log_every = max(1, cfg.nf_epochs // 20)

    with timed_gpu_block(device) as timing:
        for epoch in range(cfg.nf_epochs):
            epoch_t0 = time.perf_counter()
            model.train()
            running, count = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = model(yb, xb).mean()
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running += loss.item() * len(xb)
                count += len(xb)

            train_nll = running / max(count, 1)
            val_nll = evaluate_nll(model, val_loader, device)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            history["epoch_time_s"].append(time.perf_counter() - epoch_t0)

            if epoch % log_every == 0 or epoch == cfg.nf_epochs - 1:
                print(f"[train_nf] epoch {epoch:4d}/{cfg.nf_epochs}  "
                      f"train_nll={train_nll:.4f}  val_nll={val_nll:.4f}")

    print(f"[train_nf] total training time: {timing['time_s']:.1f}s  "
          f"peak GPU mem: {timing['peak_gpu_mem_bytes'] / 1e6:.1f} MB")

    ckpt_path = os.path.join(cfg.checkpoint_dir, "realnvp.pt")
    save_checkpoint(
        ckpt_path,
        model_state_dict=model.state_dict(),
        model_config=dict(
            dim=meta["y_dim"], cond_dim=meta["x_dim"], n_coupling=cfg.nf_n_coupling,
            hidden_dim=cfg.nf_hidden_dim, n_hidden_layers=cfg.nf_n_hidden_layers,
            context_dim=cfg.nf_context_dim, use_actnorm=cfg.nf_use_actnorm,
            coupling_type=cfg.nf_coupling_type, n_bins=cfg.nf_n_bins, tail_bound=cfg.nf_tail_bound,
        ),
        x_norm=x_norm.to_dict(),
        y_norm=y_norm.to_dict(),
        history=history,
        meta=meta,
        training_time_s=timing["time_s"],
        peak_gpu_mem_bytes=timing["peak_gpu_mem_bytes"],
        final_val_nll=history["val_nll"][-1],
        cfg=cfg.to_dict(),
    )
    print(f"[train_nf] saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
