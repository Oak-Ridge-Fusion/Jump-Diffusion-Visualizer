"""
models/embeddings.py
=====================
Embedding modules shared by the normalizing flow and the diffusion model:

  * ``SinusoidalTimeEmbedding`` -- classic transformer-style sinusoidal
    embedding of the diffusion timestep ``t`` (Ho et al. 2020 / Vaswani et al.
    2017), followed by a small MLP.
  * ``ConditionEmbedding`` -- embeds the conditioning state ``X_t`` (an
    arbitrary-dimensional vector, 1-D for the current dataset) into a fixed
    size context vector consumed by both the flow's coupling-layer
    conditioners and the diffusion denoiser.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Maps integer/float diffusion timesteps to a ``dim``-dimensional vector."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float or long tensor of timesteps -> (B, dim)."""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding followed by a 2-layer MLP (as in Ho et al. 2020)."""

    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class ConditionEmbedding(nn.Module):
    """MLP embedding of the conditioning state ``X_t`` into a context vector."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
