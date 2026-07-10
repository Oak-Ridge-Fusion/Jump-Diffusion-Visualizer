#!/usr/bin/env python
"""
train_diffusion.py
===================
Train the Conditional DDPM (cosine noise schedule, epsilon-prediction) on the
ground-truth ``(X_t, X_{t+Delta t})`` transition pairs.

Run:
    python train_diffusion.py [--diff_epochs 200] [--diff_n_timesteps 1000] ...

All hyperparameters in ``config.Config`` are exposed as CLI flags (see
``config.py``). Runs on CUDA automatically if available, with
``torch.cuda.amp`` mixed precision enabled by default. Keeps an EMA copy of
the denoiser's weights for sampling, as is standard practice for DDPMs.
"""

from __future__ import annotations

import os
import time

import torch

from config import parse_args
from dataset import build_dataloaders
from models.diffusion import ConditionalDenoiser, GaussianDiffusion
from utils import EMA, amp_enabled, get_device, save_checkpoint, set_seed, timed_gpu_block


def evaluate_loss(diffusion: GaussianDiffusion, loader, device: torch.device,
                   n_timesteps: int) -> float:
    diffusion.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            t = torch.randint(0, n_timesteps, (len(xb),), device=device)
            loss = diffusion.p_losses(yb, xb, t)
            total += loss.item() * len(xb)
            count += len(xb)
    diffusion.train()
    return total / max(count, 1)


def main() -> None:
    cfg = parse_args("Train a Conditional DDPM on jump-diffusion transition pairs.")
    set_seed(cfg.seed)
    device = get_device(cfg)
    use_amp = amp_enabled(cfg, device)
    print(f"[train_diffusion] device={device}  amp={use_amp}")
    print(f"[train_diffusion] data_path={cfg.data_path}")

    train_loader, val_loader, (x_norm, y_norm), meta, _ = build_dataloaders(cfg)
    print(f"[train_diffusion] n_train={meta['n_train']}  n_val={meta['n_val']}  "
          f"x_dim={meta['x_dim']}  y_dim={meta['y_dim']}")

    denoiser = ConditionalDenoiser(
        data_dim=meta["y_dim"],
        cond_dim=meta["x_dim"],
        hidden_dim=cfg.diff_hidden_dim,
        n_res_blocks=cfg.diff_n_res_blocks,
        time_embed_dim=cfg.diff_time_embed_dim,
        cond_embed_dim=cfg.diff_cond_embed_dim,
    )
    diffusion = GaussianDiffusion(
        denoiser, n_timesteps=cfg.diff_n_timesteps, schedule=cfg.diff_schedule
    ).to(device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(diffusion, decay=cfg.diff_ema_decay)

    history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
    log_every = max(1, cfg.diff_epochs // 20)

    with timed_gpu_block(device) as timing:
        for epoch in range(cfg.diff_epochs):
            epoch_t0 = time.perf_counter()
            diffusion.train()
            running, count = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                t = torch.randint(0, cfg.diff_n_timesteps, (len(xb),), device=device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = diffusion.p_losses(yb, xb, t)
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                ema.update(diffusion)

                running += loss.item() * len(xb)
                count += len(xb)

            train_loss = running / max(count, 1)
            val_loss = evaluate_loss(diffusion, val_loader, device, cfg.diff_n_timesteps)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["epoch_time_s"].append(time.perf_counter() - epoch_t0)

            if epoch % log_every == 0 or epoch == cfg.diff_epochs - 1:
                print(f"[train_diffusion] epoch {epoch:4d}/{cfg.diff_epochs}  "
                      f"train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

    print(f"[train_diffusion] total training time: {timing['time_s']:.1f}s  "
          f"peak GPU mem: {timing['peak_gpu_mem_bytes'] / 1e6:.1f} MB")

    ckpt_path = os.path.join(cfg.checkpoint_dir, "diffusion.pt")
    save_checkpoint(
        ckpt_path,
        model_state_dict=diffusion.state_dict(),
        ema_state_dict=ema.state_dict(),
        model_config=dict(
            data_dim=meta["y_dim"], cond_dim=meta["x_dim"], hidden_dim=cfg.diff_hidden_dim,
            n_res_blocks=cfg.diff_n_res_blocks, time_embed_dim=cfg.diff_time_embed_dim,
            cond_embed_dim=cfg.diff_cond_embed_dim, n_timesteps=cfg.diff_n_timesteps,
            schedule=cfg.diff_schedule,
        ),
        x_norm=x_norm.to_dict(),
        y_norm=y_norm.to_dict(),
        history=history,
        meta=meta,
        training_time_s=timing["time_s"],
        peak_gpu_mem_bytes=timing["peak_gpu_mem_bytes"],
        final_val_loss=history["val_loss"][-1],
        cfg=cfg.to_dict(),
    )
    print(f"[train_diffusion] saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
