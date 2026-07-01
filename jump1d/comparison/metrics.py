"""
metrics.py
==========
Distributional-comparison metrics used to benchmark generated samples against
ground-truth samples: Wasserstein-1 distance, histogram-based KL divergence,
Hellinger distance, and (RBF-kernel) Maximum Mean Discrepancy.  Also a thin
wrapper for exact normalizing-flow negative log-likelihood.

All distance/divergence functions accept 1-D or (N, D) arrays. For D == 1
exact formulas are used; for D > 1 Wasserstein falls back to the standard
sliced approximation and KL/Hellinger to independent per-dimension histograms
averaged across dimensions (a reasonable approximation for benchmarking, not
an exact joint-density estimate).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.stats import wasserstein_distance as _wasserstein_1d


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(len(a), -1) if a.ndim == 1 else a


def wasserstein(a: np.ndarray, b: np.ndarray, n_projections: int = 50,
                 rng: Optional[np.random.Generator] = None) -> float:
    """Wasserstein-1 distance. Exact for 1-D data, sliced-Wasserstein for D>1."""
    a, b = _as_2d(a), _as_2d(b)
    if a.shape[1] == 1:
        return float(_wasserstein_1d(a.ravel(), b.ravel()))
    rng = rng or np.random.default_rng(0)
    d = a.shape[1]
    dists = []
    for _ in range(n_projections):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v) + 1e-12
        dists.append(_wasserstein_1d(a @ v, b @ v))
    return float(np.mean(dists))


def _hist_density(samples: np.ndarray, lo: float, hi: float, nbins: int):
    grid = np.linspace(lo, hi, nbins + 1)
    hist, _ = np.histogram(samples, bins=grid, density=True)
    width = grid[1] - grid[0]
    return hist, width


def _kl_1d(p_samples: np.ndarray, q_samples: np.ndarray, nbins: int, eps: float) -> float:
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    p, width = _hist_density(p_samples, lo, hi, nbins)
    q, _ = _hist_density(q_samples, lo, hi, nbins)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))) * width)


def kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray, nbins: int = 100,
                   eps: float = 1e-8) -> float:
    """Forward KL(p || q) between two empirical sample sets via histogram density
    estimation. ``p`` is the reference (ground truth), ``q`` the model."""
    p, q = _as_2d(p_samples), _as_2d(q_samples)
    return float(np.mean([_kl_1d(p[:, i], q[:, i], nbins, eps) for i in range(p.shape[1])]))


def _hellinger_1d(p_samples: np.ndarray, q_samples: np.ndarray, nbins: int) -> float:
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    p, width = _hist_density(p_samples, lo, hi, nbins)
    q, _ = _hist_density(q_samples, lo, hi, nbins)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) * width))


def hellinger_distance(p_samples: np.ndarray, q_samples: np.ndarray, nbins: int = 100) -> float:
    p, q = _as_2d(p_samples), _as_2d(q_samples)
    return float(np.mean([_hellinger_1d(p[:, i], q[:, i], nbins) for i in range(p.shape[1])]))


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None,
            max_samples: int = 2000, seed: int = 0) -> float:
    """Unbiased squared MMD with an RBF kernel (median-distance bandwidth heuristic)."""
    rng = np.random.default_rng(seed)
    x, y = _as_2d(x), _as_2d(y)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]

    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()

    if sigma is None:
        z = torch.cat([xt, yt], dim=0)
        pdists = torch.cdist(z, z)
        sigma = torch.median(pdists[pdists > 0]).item() + 1e-8

    def kernel(a, b):
        d2 = torch.cdist(a, b) ** 2
        return torch.exp(-d2 / (2 * sigma ** 2))

    n, m = len(xt), len(yt)
    kxx = kernel(xt, xt)
    kyy = kernel(yt, yt)
    kxy = kernel(xt, yt)

    kxx_sum = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    kyy_sum = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    kxy_sum = kxy.sum() / (n * m)
    return float(kxx_sum + kyy_sum - 2 * kxy_sum)


@torch.no_grad()
def flow_negative_log_likelihood(flow_model, y: torch.Tensor, x_cond: torch.Tensor) -> float:
    """Mean exact NLL (nats) of a Conditional RealNVP on given data."""
    return float((-flow_model.log_prob(y, x_cond)).mean().item())


def compute_all_distribution_metrics(gt_samples: np.ndarray, model_samples: np.ndarray) -> dict:
    """Bundle of the four sample-vs-sample metrics used throughout the benchmark."""
    return {
        "wasserstein": wasserstein(gt_samples, model_samples),
        "kl_divergence": kl_divergence(gt_samples, model_samples),
        "hellinger": hellinger_distance(gt_samples, model_samples),
        "mmd": mmd_rbf(gt_samples, model_samples),
    }
