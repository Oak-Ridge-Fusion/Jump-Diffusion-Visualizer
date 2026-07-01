#!/usr/bin/env python
"""
compare.py
==========
Final benchmark orchestrator: evaluates the trained Conditional RealNVP and
Conditional DDPM checkpoints against the Ground Truth validation split,
writes ``results/results.csv`` + ``results/results.json``, and generates all
nine publication-quality comparison figures (PNG + PDF) into
``figures/``.

Run (after ``train_nf.py`` and ``train_diffusion.py`` have produced
checkpoints):

    python compare.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

import visualize
from config import parse_args
from evaluate import evaluate_anchors, evaluate_model, evaluate_rollout
from utils import get_device


def _require_checkpoints(cfg) -> None:
    missing = [
        name for name in ("realnvp.pt", "diffusion.pt")
        if not os.path.exists(os.path.join(cfg.checkpoint_dir, name))
    ]
    if missing:
        print(f"[compare] missing checkpoint(s): {missing} in {cfg.checkpoint_dir}")
        print("[compare] run `python train_nf.py` and/or `python train_diffusion.py` first.")
        sys.exit(1)


def main() -> None:
    cfg = parse_args("Full benchmark: Ground Truth vs. Conditional RealNVP vs. Conditional DDPM.")
    _require_checkpoints(cfg)
    device = get_device(cfg)
    print(f"[compare] device={device}")

    print("[compare] evaluating Conditional RealNVP ...")
    nf_result = evaluate_model("nf", cfg, device)
    print("[compare] evaluating Conditional DDPM ...")
    diff_result = evaluate_model("diffusion", cfg, device)

    print("[compare] building conditional anchor slices ...")
    anchors = evaluate_anchors(cfg, device)

    print("[compare] running multi-step rollout comparison ...")
    rollout = evaluate_rollout(cfg, device)

    # ---- results table -----------------------------------------------------
    rows = []
    for r in (nf_result, diff_result):
        rows.append({
            "model": r["model"],
            "wasserstein": r["wasserstein"],
            "kl_divergence": r["kl_divergence"],
            "hellinger": r["hellinger"],
            "mmd": r["mmd"],
            "nll": r["nll"],
            "sampling_rate_samples_per_s": r["sampling_rate_samples_per_s"],
            "sampling_time_s": r["sampling_time_s"],
            "training_time_s": r["training_time_s"],
            "peak_gpu_mem_mb": r["peak_gpu_mem_bytes"] / 1e6,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.results_dir, "results.csv")
    json_path = os.path.join(cfg.results_dir, "results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=float)

    print("\n" + "=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)
    print(f"[compare] wrote {csv_path}")
    print(f"[compare] wrote {json_path}")

    # ---- figures -------------------------------------------------------------
    print("[compare] generating figures ...")
    visualize.fig_histograms(anchors, cfg.figures_dir)
    visualize.fig_pdf_overlay(anchors, cfg.figures_dir)
    visualize.fig_qq(nf_result["y_val"], nf_result["gen_val_samples"],
                      diff_result["gen_val_samples"], cfg.figures_dir)
    visualize.fig_cdf(nf_result["y_val"], nf_result["gen_val_samples"],
                       diff_result["gen_val_samples"], cfg.figures_dir)
    visualize.fig_wasserstein_bar(anchors, nf_result["wasserstein"], diff_result["wasserstein"],
                                   cfg.figures_dir)
    visualize.fig_training_loss(nf_result["history"], diff_result["history"], cfg.figures_dir)
    visualize.fig_sampling_speed(nf_result["sampling_rate_samples_per_s"],
                                  diff_result["sampling_rate_samples_per_s"], cfg.figures_dir)
    visualize.fig_gpu_memory(nf_result["peak_gpu_mem_bytes"], diff_result["peak_gpu_mem_bytes"],
                              cfg.figures_dir)
    visualize.fig_rollout(rollout, cfg.figures_dir)

    print(f"[compare] wrote 9 figures (PNG+PDF) -> {cfg.figures_dir}")
    print("[compare] done.")


if __name__ == "__main__":
    main()
