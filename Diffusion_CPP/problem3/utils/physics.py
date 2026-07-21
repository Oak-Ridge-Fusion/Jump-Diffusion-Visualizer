"""
utils/physics.py
=================
Read-only bridge into the ground-truth 2D runaway-electron physics that
already lives in Diffusion_CPP/code_2d/common_2.py.  We never modify that
file; we just put it on sys.path and re-export what the figure/ML scripts
need, so every script imports one thing ("from utils.physics import ...")
instead of repeating the path hack.

CODE_2D_DIR is exported too: it's where data_generation_2.py writes its
artifacts (artifacts_re2d/data_pairs.npz, data_exit.npz, ...), and the ML
training/inference scripts need it to build cfg.data_dir themselves (this
problem3/ package intentionally lives outside code_2d/, so Config's default
relative data_dir="artifacts_re2d" would otherwise resolve to the wrong
place).
"""

import os
import sys

CODE_2D_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "code_2d")
)
if CODE_2D_DIR not in sys.path:
    sys.path.insert(0, CODE_2D_DIR)

from common_2 import (  # noqa: E402  (import after sys.path fix, by design)
    Config,
    gamma_of_p,
    lnL_reduction,
    re2d_coeffs,
    moller_sigma_hat,
    knock_rate,
    sample_eps,
    apply_knock_jump,
    simulate_re2d_step,
    simulate_re2d_rollout,
    exit_features,
    set_seed,
    get_device,
    wasserstein1,
    sliced_w1,
    generate_labels,
    FN_Net,
    ExitNet,
)

__all__ = [
    "CODE_2D_DIR",
    "Config",
    "gamma_of_p",
    "lnL_reduction",
    "re2d_coeffs",
    "moller_sigma_hat",
    "knock_rate",
    "sample_eps",
    "apply_knock_jump",
    "simulate_re2d_step",
    "simulate_re2d_rollout",
    "exit_features",
    "set_seed",
    "get_device",
    "wasserstein1",
    "sliced_w1",
    "generate_labels",
    "FN_Net",
    "ExitNet",
]
