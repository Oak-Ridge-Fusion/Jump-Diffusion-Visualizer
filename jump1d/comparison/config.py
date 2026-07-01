"""
config.py
=========
Central configuration for the ``comparison`` benchmark.

Every script (``train_nf.py``, ``train_diffusion.py``, ``evaluate.py``,
``compare.py``) builds a :class:`Config` and optionally overrides individual
fields from the command line via :func:`parse_args`.  Nothing here touches the
existing ``jump1d`` project -- this is a self-contained sibling package that
reads the ground-truth ``data_pairs.npz`` produced by ``data_generation.py``.

The benchmark is deliberately dataset-agnostic: any ``.npz`` file that stores
two equally-shaped arrays -- one for the conditioning state ``X_t`` and one
for the transitioned state ``X_{t+Delta t}`` -- can be plugged in by pointing
``data_path`` (and, if the key names differ, ``x_key`` / ``y_key``) at it.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMPARISON_ROOT = os.path.dirname(os.path.abspath(__file__))
JUMP1D_ROOT = os.path.dirname(COMPARISON_ROOT)

DEFAULT_DATA_PATH = os.path.join(JUMP1D_ROOT, "artifacts_bd", "data_pairs.npz")
DEFAULT_CHECKPOINT_DIR = os.path.join(COMPARISON_ROOT, "checkpoints")
DEFAULT_FIGURES_DIR = os.path.join(COMPARISON_ROOT, "figures")
DEFAULT_RESULTS_DIR = os.path.join(COMPARISON_ROOT, "results")


@dataclass
class Config:
    """Single source of truth for every hyperparameter in the benchmark."""

    # ---- reproducibility ---------------------------------------------------
    seed: int = 0

    # ---- data ---------------------------------------------------------------
    data_path: str = DEFAULT_DATA_PATH
    x_key: str = "x_sample"          # conditioning state X_t
    y_key: str = "y_sample"          # target state X_{t+Delta t}
    val_fraction: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # ---- device / precision --------------------------------------------------
    device: str = "auto"             # "auto" | "cuda" | "cpu"
    amp: bool = True                 # torch.cuda.amp mixed precision (CUDA only)

    # ---- optimisation (shared) ------------------------------------------------
    batch_size: int = 1024
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # ---- Conditional RealNVP --------------------------------------------------
    nf_epochs: int = 200
    nf_n_coupling: int = 8
    nf_hidden_dim: int = 128
    nf_n_hidden_layers: int = 3
    nf_context_dim: int = 64
    nf_use_actnorm: bool = True

    # ---- Conditional DDPM -----------------------------------------------------
    diff_epochs: int = 200
    diff_n_timesteps: int = 1000
    diff_schedule: str = "cosine"    # "cosine" | "linear"
    diff_hidden_dim: int = 256
    diff_n_res_blocks: int = 4
    diff_time_embed_dim: int = 128
    diff_cond_embed_dim: int = 128
    diff_ema_decay: float = 0.999

    # ---- evaluation / sampling --------------------------------------------------
    eval_n_samples: int = 20000
    eval_n_anchors: int = 5
    eval_anchor_tolerance: float = 0.15   # half-width window (in x units) around each anchor
    rollout_steps: int = 8
    rollout_n_particles: int = 20000

    # ---- paths ---------------------------------------------------------------
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    figures_dir: str = DEFAULT_FIGURES_DIR
    results_dir: str = DEFAULT_RESULTS_DIR

    def ensure_dirs(self) -> None:
        for d in (self.checkpoint_dir, self.figures_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every :class:`Config` field as an optional CLI flag."""
    defaults = Config()
    for name, value in defaults.to_dict().items():
        flag = f"--{name}"
        if isinstance(value, bool):
            parser.add_argument(flag, dest=name, action="store_true", default=None)
            parser.add_argument(f"--no_{name}", dest=name, action="store_false", default=None)
        else:
            parser.add_argument(flag, dest=name, type=type(value), default=None)
    return parser


def build_config(args: argparse.Namespace) -> Config:
    """Merge parsed CLI args on top of the :class:`Config` defaults."""
    cfg = Config()
    for name in cfg.to_dict():
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            setattr(cfg, name, cli_value)
    cfg.ensure_dirs()
    return cfg


def parse_args(description: str = "") -> Config:
    parser = argparse.ArgumentParser(description=description)
    add_config_arguments(parser)
    args = parser.parse_args()
    return build_config(args)
