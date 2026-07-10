"""
models/realnvp.py
==================
Conditional RealNVP / Neural Spline Flow for ``p(X_{t+Delta t} | X_t)``.

Architecture: 8 conditional coupling layers, each preceded by an ActNorm and
followed by a fixed permutation, with a standard normal base distribution.
Trained by maximising the exact log-likelihood of the data under the
change-of-variables formula.

    log p(y | x) = log p(z) + log|det dz/dy|,      z = f(y; ctx(x))

``coupling_type`` selects the per-layer transform:

  * "affine" (original RealNVP) -- ``y = x * exp(s(ctx)) + t(ctx)``. At
    ``dim == 1`` this can only represent a conditional Gaussian ``p(y|x)``
    (see ``models/layers.py`` docstring for why).
  * "spline" (Neural Spline Flows, Durkan et al. 2019) -- a monotonic
    rational-quadratic spline instead of an affine map. Same exact-NLL
    training and sampling interface, but not capped at conditional-Gaussian
    shapes -- can represent skew, multimodality, and boundary compression.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.embeddings import ConditionEmbedding
from models.layers import (ActNorm, ConditionalAffineCoupling, ConditionalSplineCoupling,
                            Permutation, make_checkerboard_mask)

LOG_2PI = math.log(2.0 * math.pi)


class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        n_coupling: int = 8,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        context_dim: int = 64,
        use_actnorm: bool = True,
        coupling_type: str = "affine",     # "affine" | "spline"
        n_bins: int = 8,                   # spline only
        tail_bound: float = 5.0,           # spline only
    ):
        super().__init__()
        self.dim = dim
        self.use_actnorm = use_actnorm
        self.coupling_type = coupling_type
        self.context_embed = ConditionEmbedding(cond_dim, context_dim, hidden_dim=hidden_dim)

        self.actnorms = nn.ModuleList()
        self.couplings = nn.ModuleList()
        self.permutations = nn.ModuleList()
        for i in range(n_coupling):
            self.actnorms.append(ActNorm(dim) if use_actnorm else nn.Identity())
            mask = make_checkerboard_mask(dim, i)
            if coupling_type == "spline":
                coupling = ConditionalSplineCoupling(
                    dim, context_dim, hidden_dim, n_hidden_layers, mask,
                    num_bins=n_bins, tail_bound=tail_bound,
                )
            elif coupling_type == "affine":
                coupling = ConditionalAffineCoupling(dim, context_dim, hidden_dim, n_hidden_layers, mask)
            else:
                raise ValueError(f"unknown coupling_type {coupling_type!r}")
            self.couplings.append(coupling)
            self.permutations.append(Permutation(dim))

        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_log_std", torch.zeros(dim))

    def _to_latent(self, y: torch.Tensor, ctx: torch.Tensor):
        z = y
        logdet = torch.zeros(y.shape[0], device=y.device)
        for actnorm, coupling, perm in zip(self.actnorms, self.couplings, self.permutations):
            if self.use_actnorm:
                z, ld = actnorm(z)
                logdet = logdet + ld
            else:
                ld = torch.zeros(z.shape[0], device=z.device)
            z, ld = coupling(z, ctx)
            logdet = logdet + ld
            z, ld = perm(z)
            logdet = logdet + ld
        return z, logdet

    def _to_data(self, z: torch.Tensor, ctx: torch.Tensor):
        y = z
        for actnorm, coupling, perm in reversed(
            list(zip(self.actnorms, self.couplings, self.permutations))
        ):
            y = perm.inverse(y)
            y = coupling.inverse(y, ctx)
            if self.use_actnorm:
                y = actnorm.inverse(y)
        return y

    def log_prob(self, y: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        ctx = self.context_embed(x_cond)
        z, logdet = self._to_latent(y, ctx)
        log_pz = -0.5 * (z ** 2 + LOG_2PI).sum(dim=-1)
        return log_pz + logdet

    @torch.no_grad()
    def sample(self, x_cond: torch.Tensor) -> torch.Tensor:
        ctx = self.context_embed(x_cond)
        z = torch.randn(x_cond.shape[0], self.dim, device=x_cond.device)
        return self._to_data(z, ctx)

    def forward(self, y: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """Returns the per-sample negative log-likelihood (training loss)."""
        return -self.log_prob(y, x_cond)
