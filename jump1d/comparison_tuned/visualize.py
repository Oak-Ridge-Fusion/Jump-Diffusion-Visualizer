"""
visualize.py
============
Publication-quality figure generation for the benchmark. Every figure is
saved as both PNG (300 dpi, for quick viewing) and PDF (vector, for papers)
into ``config.figures_dir``.

Figures produced (used by ``compare.py``):
  1. ``01_histograms``          -- GT vs NF vs Diffusion histograms per anchor X_t
  2. ``02_pdf_overlay``         -- KDE-smoothed density overlay per anchor X_t
  3. ``03_qq_plot``             -- quantile-quantile plots, NF vs GT and Diffusion vs GT
  4. ``04_cdf_comparison``      -- empirical CDF overlay (aggregate)
  5. ``05_wasserstein_bar``     -- Wasserstein-to-GT, grouped by anchor + overall
  6. ``06_training_loss``       -- NF and Diffusion training/validation curves
  7. ``07_sampling_speed``      -- samples/sec bar chart
  8. ``08_gpu_memory``          -- peak training GPU memory bar chart
  9. ``09_rollout_comparison``  -- survival fraction + terminal density, multi-step rollout
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from metrics import wasserstein

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

COLOR_GT = "#4d4d4d"
COLOR_NF = "#1f77b4"
COLOR_DIFF = "#d62728"


def save_fig(fig: "plt.Figure", figures_dir: str, name: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Histograms per anchor
# ---------------------------------------------------------------------------
def fig_histograms(anchor_slices: list, figures_dir: str) -> None:
    n = len(anchor_slices)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False)
    axes = axes[0]
    for ax, s in zip(axes, anchor_slices):
        lo = min(s["gt"].min(), s["nf"].min(), s["diffusion"].min())
        hi = max(s["gt"].max(), s["nf"].max(), s["diffusion"].max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(s["gt"], bins=bins, density=True, alpha=0.4, color=COLOR_GT, label="Ground Truth")
        ax.hist(s["nf"], bins=bins, density=True, histtype="step", lw=1.8,
                 color=COLOR_NF, label="Cond. RealNVP")
        ax.hist(s["diffusion"], bins=bins, density=True, histtype="step", lw=1.8,
                 color=COLOR_DIFF, label="Cond. DDPM")
        ax.set_title(f"$X_t={s['anchor']:.2f}$")
        ax.set_xlabel(r"$X_{t+\Delta t}$")
        ax.legend()
    axes[0].set_ylabel("density")
    fig.suptitle("Ground Truth vs. Normalizing Flow vs. Diffusion — transition histograms")
    fig.tight_layout()
    save_fig(fig, figures_dir, "01_histograms")


# ---------------------------------------------------------------------------
# 2. KDE overlay
# ---------------------------------------------------------------------------
def fig_pdf_overlay(anchor_slices: list, figures_dir: str) -> None:
    n = len(anchor_slices)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False)
    axes = axes[0]
    for ax, s in zip(axes, anchor_slices):
        lo = min(s["gt"].min(), s["nf"].min(), s["diffusion"].min())
        hi = max(s["gt"].max(), s["nf"].max(), s["diffusion"].max())
        grid = np.linspace(lo, hi, 400)
        for key, color, label in [("gt", COLOR_GT, "Ground Truth"),
                                    ("nf", COLOR_NF, "Cond. RealNVP"),
                                    ("diffusion", COLOR_DIFF, "Cond. DDPM")]:
            kde = gaussian_kde(s[key])
            ax.plot(grid, kde(grid), color=color, lw=2.0, label=label)
        ax.set_title(f"$X_t={s['anchor']:.2f}$")
        ax.set_xlabel(r"$X_{t+\Delta t}$")
        ax.legend()
    axes[0].set_ylabel("density (KDE)")
    fig.suptitle("Overlaid PDFs (kernel density estimate)")
    fig.tight_layout()
    save_fig(fig, figures_dir, "02_pdf_overlay")


# ---------------------------------------------------------------------------
# 3. QQ plot (aggregate, over the whole validation set)
# ---------------------------------------------------------------------------
def fig_qq(y_val: np.ndarray, gen_nf: np.ndarray, gen_diff: np.ndarray, figures_dir: str) -> None:
    q = np.linspace(0.01, 0.99, 200)
    gt_q = np.quantile(y_val, q)
    nf_q = np.quantile(gen_nf, q)
    diff_q = np.quantile(gen_diff, q)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    lims = [min(gt_q.min(), nf_q.min(), diff_q.min()), max(gt_q.max(), nf_q.max(), diff_q.max())]

    axes[0].plot(gt_q, nf_q, "o", ms=3, color=COLOR_NF)
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set_title("QQ: Cond. RealNVP vs. Ground Truth")
    axes[0].set_xlabel("Ground Truth quantiles"); axes[0].set_ylabel("Model quantiles")

    axes[1].plot(gt_q, diff_q, "o", ms=3, color=COLOR_DIFF)
    axes[1].plot(lims, lims, "k--", lw=1)
    axes[1].set_title("QQ: Cond. DDPM vs. Ground Truth")
    axes[1].set_xlabel("Ground Truth quantiles"); axes[1].set_ylabel("Model quantiles")

    fig.tight_layout()
    save_fig(fig, figures_dir, "03_qq_plot")


# ---------------------------------------------------------------------------
# 4. CDF comparison (aggregate)
# ---------------------------------------------------------------------------
def fig_cdf(y_val: np.ndarray, gen_nf: np.ndarray, gen_diff: np.ndarray, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for samples, color, label in [(y_val, COLOR_GT, "Ground Truth"),
                                    (gen_nf, COLOR_NF, "Cond. RealNVP"),
                                    (gen_diff, COLOR_DIFF, "Cond. DDPM")]:
        xs = np.sort(samples.ravel())
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, lw=1.8, label=label)
    ax.set_xlabel(r"$X_{t+\Delta t}$"); ax.set_ylabel("empirical CDF")
    ax.set_title("CDF comparison (aggregate over validation set)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, figures_dir, "04_cdf_comparison")


# ---------------------------------------------------------------------------
# 5. Wasserstein comparison bar chart
# ---------------------------------------------------------------------------
def fig_wasserstein_bar(anchor_slices: list, overall_nf: float, overall_diff: float,
                          figures_dir: str) -> None:
    labels = [f"$X_t={s['anchor']:.2f}$" for s in anchor_slices] + ["overall"]
    nf_vals = [wasserstein(s["gt"], s["nf"]) for s in anchor_slices] + [overall_nf]
    diff_vals = [wasserstein(s["gt"], s["diffusion"]) for s in anchor_slices] + [overall_diff]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4.6))
    ax.bar(x - width / 2, nf_vals, width, color=COLOR_NF, label="Cond. RealNVP")
    ax.bar(x + width / 2, diff_vals, width, color=COLOR_DIFF, label="Cond. DDPM")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Wasserstein-1 distance to Ground Truth")
    ax.set_title("Wasserstein distance comparison")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, figures_dir, "05_wasserstein_bar")


# ---------------------------------------------------------------------------
# 6. Training loss curves
# ---------------------------------------------------------------------------
def fig_training_loss(history_nf: dict, history_diff: dict, figures_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    axes[0].plot(history_nf["train_nll"], color=COLOR_NF, lw=1.6, label="train NLL")
    axes[0].plot(history_nf["val_nll"], color=COLOR_NF, lw=1.6, ls="--", label="val NLL")
    axes[0].set_title("Conditional RealNVP training curve")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("negative log-likelihood (nats)")
    axes[0].legend()

    axes[1].plot(history_diff["train_loss"], color=COLOR_DIFF, lw=1.6, label="train MSE")
    axes[1].plot(history_diff["val_loss"], color=COLOR_DIFF, lw=1.6, ls="--", label="val MSE")
    axes[1].set_title("Conditional DDPM training curve")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel(r"$\epsilon$-prediction MSE")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, figures_dir, "06_training_loss")


# ---------------------------------------------------------------------------
# 7. Sampling speed bar chart
# ---------------------------------------------------------------------------
def fig_sampling_speed(rate_nf: float, rate_diff: float, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.3))
    bars = ax.bar(["Cond. RealNVP", "Cond. DDPM"], [rate_nf, rate_diff],
                   color=[COLOR_NF, COLOR_DIFF])
    ax.set_yscale("log")
    ax.set_ylabel("samples / second (log scale)")
    ax.set_title("Sampling speed")
    for b, v in zip(bars, [rate_nf, rate_diff]):
        ax.annotate(f"{v:,.0f}/s", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_fig(fig, figures_dir, "07_sampling_speed")


# ---------------------------------------------------------------------------
# 8. GPU memory usage bar chart
# ---------------------------------------------------------------------------
def fig_gpu_memory(mem_nf_bytes: float, mem_diff_bytes: float, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.3))
    vals_mb = [mem_nf_bytes / 1e6, mem_diff_bytes / 1e6]
    ax.bar(["Cond. RealNVP", "Cond. DDPM"], vals_mb, color=[COLOR_NF, COLOR_DIFF])
    ax.set_ylabel("peak GPU memory during training (MB)")
    ax.set_title("GPU memory usage")
    fig.tight_layout()
    save_fig(fig, figures_dir, "08_gpu_memory")


# ---------------------------------------------------------------------------
# 9. Rollout comparison
# ---------------------------------------------------------------------------
def fig_rollout(rollout_results: dict, figures_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    steps = np.arange(1, len(rollout_results["ground_truth"]["survival"]) + 1)
    for name, color, label in [("ground_truth", COLOR_GT, "Ground Truth"),
                                 ("nf", COLOR_NF, "Cond. RealNVP"),
                                 ("diffusion", COLOR_DIFF, "Cond. DDPM")]:
        axes[0].plot(steps, rollout_results[name]["survival"], "-o", ms=3,
                      color=color, label=label)
    axes[0].set_title("Surviving fraction vs. rollout step")
    axes[0].set_xlabel("big step"); axes[0].set_ylabel("fraction still in domain")
    axes[0].legend()

    all_term = np.concatenate([rollout_results[k]["terminal"] for k in rollout_results])
    lo, hi = np.quantile(all_term, [0.001, 0.999])
    bins = np.linspace(lo, hi, 70)
    for name, color, label in [("ground_truth", COLOR_GT, "Ground Truth"),
                                 ("nf", COLOR_NF, "Cond. RealNVP"),
                                 ("diffusion", COLOR_DIFF, "Cond. DDPM")]:
        term = rollout_results[name]["terminal"]
        if len(term) == 0:
            continue
        style = dict(alpha=0.35, color=color) if name == "ground_truth" \
            else dict(histtype="step", lw=1.8, color=color)
        axes[1].hist(term, bins=bins, density=True, label=label, **style)
    axes[1].set_title("Terminal surviving density")
    axes[1].set_xlabel(r"$X_T$")
    axes[1].legend()

    fig.suptitle("Side-by-side rollout comparison (composed transition kernel)")
    fig.tight_layout()
    save_fig(fig, figures_dir, "09_rollout_comparison")
