"""Generate title-free publication figures for Sections 1-2 of the draft.

This reads the fresh checkpoints in jump1d/artifacts_bd and writes only to
paper_figures/. It intentionally mirrors jump1d/inference.py's seeded B1/B2
evaluation without modifying the source artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JUMP1D = ROOT / "jump1d"
sys.path.insert(0, str(JUMP1D))

from common import (  # noqa: E402
    Config,
    get_device,
    set_seed,
    simulate_bounded_step,
    wasserstein1,
)
from inference import (  # noqa: E402
    ExitHead,
    FlowMap,
    bounded_rollout_ml,
    bounded_rollout_num,
)


OUT = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "savefig.transparent": False,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=10,
    )


def make_b1(fm: FlowMap, eh: ExitHead, cfg: Config, rng: np.random.Generator) -> None:
    grid = np.linspace(cfg.x_min, cfg.x_max, 200)
    pe_pred = eh.prob(grid)
    pe_true = np.empty_like(grid)
    for k, xv in enumerate(grid):
        _, alive = simulate_bounded_step(np.full(4000, xv), cfg, rng)
        pe_true[k] = np.mean(~alive)

    test_x = [
        cfg.x_min + 0.25 * (cfg.x_max - cfg.x_min),
        0.5 * (cfg.x_min + cfg.x_max),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)

    axes[0].plot(grid, pe_true, color="black", label="Numerical")
    axes[0].plot(grid, pe_pred, color="#c43c39", linestyle="--", label="Exit indicator")
    axes[0].set(xlabel=r"$x$", ylabel=r"$P(\mathrm{exit}\mid x)$", xlim=(0, 6), ylim=(0, 1))
    axes[0].legend(frameon=False, loc="upper center")
    panel_label(axes[0], "(a)")

    bins = np.linspace(cfg.x_min, cfg.x_max, 70)
    for j, x0 in enumerate(test_x, start=1):
        n = 60000
        endpoint, alive = simulate_bounded_step(np.full(n, x0), cfg, rng)
        truth = endpoint[alive]
        learned = fm.step(np.full(n, x0), rng)
        learned = learned[(learned >= cfg.x_min) & (learned <= cfg.x_max)]
        axes[j].hist(
            truth,
            bins=bins,
            density=True,
            color="0.75",
            edgecolor="none",
            label="Numerical survivors",
        )
        axes[j].hist(
            learned,
            bins=bins,
            density=True,
            histtype="step",
            color="#c43c39",
            label="Flow map",
        )
        axes[j].set(xlabel=r"$X_{t+\Delta t}$", ylabel="Conditional density", xlim=(0, 6))
        axes[j].text(
            0.97,
            0.95,
            rf"$X_t={x0:g}$",
            transform=axes[j].transAxes,
            ha="right",
            va="top",
        )
        if j == 1:
            axes[j].legend(frameon=False, loc="lower right")
        panel_label(axes[j], f"({chr(97 + j)})")

    save_figure(fig, "B1_onestep")


def make_b2(fm: FlowMap, eh: ExitHead, cfg: Config, rng: np.random.Generator) -> None:
    initial = rng.uniform(cfg.x_min, cfg.x_max, size=cfg.n_ref_mc)
    learned_x, learned_alive, learned_survival = bounded_rollout_ml(
        fm, eh, cfg, initial, rng
    )
    numerical_x, numerical_alive, numerical_survival = bounded_rollout_num(
        cfg, initial, rng
    )

    steps = np.arange(1, cfg.rollout_K + 1)
    terminal_time = cfg.rollout_K * cfg.dt
    fig, axes = plt.subplots(1, 2, figsize=(5.25, 2.4), constrained_layout=True)

    axes[0].plot(steps, numerical_survival, "o-", color="black", ms=3.5, label="Numerical")
    axes[0].plot(
        steps,
        learned_survival,
        "s--",
        color="#c43c39",
        ms=3.5,
        label="ML rollout",
    )
    axes[0].set(
        xlabel="Coarse step",
        ylabel="Surviving fraction",
        xlim=(1, cfg.rollout_K),
    )
    axes[0].legend(frameon=False)
    panel_label(axes[0], "(a)")

    bins = np.linspace(cfg.x_min, cfg.x_max, 70)
    axes[1].hist(
        numerical_x[numerical_alive],
        bins=bins,
        density=True,
        color="0.75",
        edgecolor="none",
        label="Numerical survivors",
    )
    axes[1].hist(
        learned_x[learned_alive],
        bins=bins,
        density=True,
        histtype="step",
        color="#c43c39",
        label="ML rollout",
    )
    axes[1].set(xlabel=rf"$X_T$ at $T={terminal_time:g}$", ylabel="Conditional density", xlim=(0, 6))
    axes[1].legend(frameon=False)
    panel_label(axes[1], "(b)")

    save_figure(fig, "B2_rollout")


def main() -> None:
    cfg = Config()
    cfg.data_dir = str(JUMP1D / "artifacts_bd")
    set_seed(cfg.seed + 2)
    rng = np.random.default_rng(cfg.seed + 2)
    device = get_device(cfg)
    flow_map = FlowMap(cfg, device)
    exit_head = ExitHead(cfg, device)

    make_b1(flow_map, exit_head, cfg, rng)
    make_b2(flow_map, exit_head, cfg, rng)
    print(f"Wrote publication figures to {OUT}")


if __name__ == "__main__":
    main()
