"""
dataset.py
==========
Loading and normalising the ground-truth transition-pair dataset
``(X_t, X_{t+Delta t})`` produced by ``jump1d/data_generation.py``.

Reuses whatever ``.npz`` file ``config.data_path`` points at -- by default the
bounded jump-diffusion pairs in ``artifacts_bd/data_pairs.npz`` -- without
regenerating anything.  Any future dataset with the same two-array layout can
be loaded by changing ``data_path`` (and ``x_key`` / ``y_key`` if needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


@dataclass
class Normalizer:
    """Per-dimension standardisation, invertible for sampling / evaluation."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_dict(d: dict) -> "Normalizer":
        return Normalizer(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]))

    @staticmethod
    def fit(x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        return Normalizer(mean=mean, std=std)


class TransitionPairDataset(Dataset):
    """A dataset of ``(x_t, x_{t+dt})`` pairs, both stored pre-normalised."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _load_raw(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(cfg.data_path)
    x = np.asarray(data[cfg.x_key], dtype=np.float32).reshape(len(data[cfg.x_key]), -1)
    y = np.asarray(data[cfg.y_key], dtype=np.float32).reshape(len(data[cfg.y_key]), -1)
    return x, y


def load_splits(cfg: Config):
    """Load, shuffle, split (train/val) and standardise the transition pairs.

    Returns
    -------
    (x_train, y_train, x_val, y_val) : raw (unnormalised) numpy arrays
    (x_norm, y_norm) : fitted :class:`Normalizer` objects (train-set statistics)
    """
    x, y = _load_raw(cfg)
    n = len(x)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]

    n_val = max(1, int(round(n * cfg.val_fraction)))
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]

    x_norm = Normalizer.fit(x_train)
    y_norm = Normalizer.fit(y_train)
    return (x_train, y_train, x_val, y_val), (x_norm, y_norm)


def make_loader(x: np.ndarray, y: np.ndarray, cfg: Config, shuffle: bool) -> DataLoader:
    ds = TransitionPairDataset(x, y)
    use_pw = cfg.persistent_workers and cfg.num_workers > 0
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=use_pw,
        drop_last=shuffle,
    )


def build_dataloaders(cfg: Config):
    """Convenience wrapper returning train/val loaders plus normalisers."""
    (x_train, y_train, x_val, y_val), (x_norm, y_norm) = load_splits(cfg)

    train_loader = make_loader(x_norm.transform(x_train), y_norm.transform(y_train),
                                cfg, shuffle=True)
    val_loader = make_loader(x_norm.transform(x_val), y_norm.transform(y_val),
                              cfg, shuffle=False)

    meta = {
        "n_train": len(x_train),
        "n_val": len(x_val),
        "x_dim": x_train.shape[1],
        "y_dim": y_train.shape[1],
        "x_domain": (float(x_train.min()), float(x_train.max())),
        "y_domain": (float(y_train.min()), float(y_train.max())),
    }
    raw = {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}
    return train_loader, val_loader, (x_norm, y_norm), meta, raw
