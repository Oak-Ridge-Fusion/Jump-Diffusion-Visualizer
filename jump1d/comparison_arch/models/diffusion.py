"""
models/diffusion.py
====================
Conditional DDPM for ``p(X_{t+Delta t} | X_t)`` (Ho et al. 2020, cosine
schedule from Nichol & Dhariwal 2021).

  * ``cosine_beta_schedule`` / ``linear_beta_schedule`` -- noise schedules.
  * ``ResBlock``              -- FiLM-style residual block: a linear layer
                                  conditioned on a (timestep + X_t) embedding.
  * ``ConditionalDenoiser``   -- a small UNet-style network (down-project,
                                  bottleneck, up-project with skip
                                  connections) built from ``ResBlock``s. It is
                                  an MLP rather than a conv-net because the
                                  state here is a scalar (no spatial grid),
                                  but keeps the encoder/bottleneck/decoder +
                                  skip-connection structure of a UNet.
  * ``GaussianDiffusion``     -- wraps a denoiser with the forward noising
                                  process, the training loss, and sampling.

Two independent knobs beyond the original (both configurable, both optional):

  * ``parameterization`` ("eps" | "v") -- what the denoiser predicts.
    "eps" (Ho et al. 2020) is the original noise-prediction loss.
    "v" (Salimans & Ho, "Progressive Distillation", 2022) predicts
    ``v = alpha_t * eps - sigma_t * y0`` instead; the same network output
    is converted back to an (x0, eps) pair analytically at sampling time
    (see ``_predict_x0_eps``). v-prediction is often better-conditioned near
    the two ends of the noise schedule, where "eps" targets become nearly
    degenerate (t~0: eps barely affects y_t; t~T: eps is ~all of y_t).
  * ``sampler`` ("ancestral" | "ddim") -- how the reverse process is walked.
    "ancestral" is the original stochastic DDPM sampler, one network call
    per one of ``n_timesteps`` steps. "ddim" (Song, Meng & Ermon, 2020) is a
    deterministic, non-Markovian sampler that reuses the *same* trained
    network but takes a strided subsequence of ``ddim_steps`` timesteps --
    typically 10-50x fewer network calls for comparable sample quality.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embeddings import ConditionEmbedding, TimestepEmbedder


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal, 2021), better-conditioned than linear."""
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999).float()


def linear_beta_schedule(n_timesteps: int, beta_start: float = 1e-4,
                          beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)


class ResBlock(nn.Module):
    """Pre-norm residual block, FiLM-conditioned on a shared embedding vector."""

    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.emb_proj = nn.Linear(emb_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.lin1(self.act(self.norm1(x)))
        h = h + self.emb_proj(emb)
        h = self.lin2(self.act(self.norm2(h)))
        return x + h


class ConditionalDenoiser(nn.Module):
    """epsilon_theta(y_t, t, x_0) -- predicts the injected noise."""

    def __init__(self, data_dim: int, cond_dim: int, hidden_dim: int = 256,
                 n_res_blocks: int = 4, time_embed_dim: int = 128,
                 cond_embed_dim: int = 128):
        super().__init__()
        emb_dim = hidden_dim
        self.time_embedder = TimestepEmbedder(time_embed_dim, emb_dim)
        self.cond_embedder = ConditionEmbedding(cond_dim, emb_dim, hidden_dim=cond_embed_dim)
        self.in_proj = nn.Linear(data_dim, hidden_dim)

        n_down = max(1, n_res_blocks // 2)
        widths = [hidden_dim * (2 ** i) for i in range(n_down + 1)]

        self.down_blocks = nn.ModuleList([ResBlock(widths[i], emb_dim) for i in range(n_down)])
        self.down_proj = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(n_down)]
        )
        self.bottleneck = ResBlock(widths[-1], emb_dim)
        self.up_proj = nn.ModuleList(
            [nn.Linear(widths[n_down - i], widths[n_down - i - 1]) for i in range(n_down)]
        )
        self.up_blocks = nn.ModuleList(
            [ResBlock(widths[n_down - i - 1], emb_dim) for i in range(n_down)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, data_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_embedder(t) + self.cond_embedder(x_cond)
        h = self.in_proj(y_t)
        skips = []
        for block, proj in zip(self.down_blocks, self.down_proj):
            h = block(h, emb)
            skips.append(h)
            h = proj(h)
        h = self.bottleneck(h, emb)
        for proj, block, skip in zip(self.up_proj, self.up_blocks, reversed(skips)):
            h = proj(h) + skip
            h = block(h, emb)
        return self.out_proj(F.silu(self.out_norm(h)))


class GaussianDiffusion(nn.Module):
    """Forward noising process + training loss + sampling (ancestral or DDIM)."""

    def __init__(self, denoiser: ConditionalDenoiser, n_timesteps: int = 1000,
                 schedule: str = "cosine", parameterization: str = "eps",
                 sampler: str = "ancestral", ddim_steps: int = 50, ddim_eta: float = 0.0):
        super().__init__()
        self.denoiser = denoiser
        self.n_timesteps = n_timesteps
        self.parameterization = parameterization
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        betas = cosine_beta_schedule(n_timesteps) if schedule == "cosine" \
            else linear_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = torch.randn_like(y0) if noise is None else noise
        sac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        somac = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sac * y0 + somac * noise

    def p_losses(self, y0: torch.Tensor, x_cond: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(y0)
        y_t = self.q_sample(y0, t, noise)
        raw_pred = self.denoiser(y_t, t.float(), x_cond)
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            sac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            somac = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            target = sac * noise - somac * y0    # v = alpha*eps - sigma*y0
        else:
            raise ValueError(f"unknown parameterization {self.parameterization!r}")
        return F.mse_loss(raw_pred, target)

    def _predict_x0_eps(self, y_t: torch.Tensor, t: torch.Tensor, raw_pred: torch.Tensor):
        """Convert the raw denoiser output to an (x0, eps) pair, regardless of
        ``parameterization`` -- both samplers below are written purely in
        terms of this pair so they don't need to know which one was trained."""
        sac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        somac = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        if self.parameterization == "eps":
            eps_pred = raw_pred
            x0_pred = (y_t - somac * eps_pred) / sac
        else:  # "v": x0 = alpha*y_t - sigma*v ,  eps = sigma*y_t + alpha*v
            x0_pred = sac * y_t - somac * raw_pred
            eps_pred = somac * y_t + sac * raw_pred
        return x0_pred, eps_pred

    @torch.no_grad()
    def p_sample_step(self, y_t: torch.Tensor, t_index: int, x_cond: torch.Tensor) -> torch.Tensor:
        """One ancestral (stochastic, Ho et al. 2020) reverse step."""
        b = y_t.shape[0]
        t = torch.full((b,), t_index, device=y_t.device, dtype=torch.long)
        raw_pred = self.denoiser(y_t, t.float(), x_cond)
        _, eps_pred = self._predict_x0_eps(y_t, t, raw_pred)
        beta_t = self.betas[t_index]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t_index]
        model_mean = sqrt_recip_alpha * (y_t - beta_t / sqrt_one_minus * eps_pred)
        if t_index == 0:
            return model_mean
        var = self.posterior_variance[t_index]
        noise = torch.randn_like(y_t)
        return model_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def ancestral_sample(self, x_cond: torch.Tensor, dim: int) -> torch.Tensor:
        y = torch.randn(x_cond.shape[0], dim, device=x_cond.device)
        for t_index in reversed(range(self.n_timesteps)):
            y = self.p_sample_step(y, t_index, x_cond)
        return y

    def _ddim_timesteps(self, n_steps: int) -> np.ndarray:
        """Evenly-spaced descending subsequence of {0, ..., n_timesteps-1},
        length <= n_steps (deduplicated after rounding)."""
        idx = np.linspace(0, self.n_timesteps - 1, n_steps).round().astype(int)
        return np.unique(idx)[::-1]

    @torch.no_grad()
    def ddim_sample(self, x_cond: torch.Tensor, dim: int,
                     n_steps: Optional[int] = None, eta: Optional[float] = None) -> torch.Tensor:
        """Deterministic (eta=0) or partially-stochastic (Song, Meng & Ermon,
        2020) reverse process over a strided subsequence of timesteps. Reuses
        the same trained denoiser as ``ancestral_sample`` -- no retraining
        needed to switch samplers."""
        n_steps = self.ddim_steps if n_steps is None else n_steps
        eta = self.ddim_eta if eta is None else eta
        timesteps = self._ddim_timesteps(n_steps)

        y = torch.randn(x_cond.shape[0], dim, device=x_cond.device)
        for i, t_index in enumerate(timesteps):
            b = y.shape[0]
            t = torch.full((b,), int(t_index), device=y.device, dtype=torch.long)
            raw_pred = self.denoiser(y, t.float(), x_cond)
            x0_pred, eps_pred = self._predict_x0_eps(y, t, raw_pred)

            acp_t = self.alphas_cumprod[int(t_index)]
            if i + 1 < len(timesteps):
                acp_prev = self.alphas_cumprod[int(timesteps[i + 1])]
            else:
                acp_prev = torch.ones((), device=y.device, dtype=acp_t.dtype)   # "t = -1"

            if eta > 0:
                sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp_t)) * torch.sqrt(1 - acp_t / acp_prev)
            else:
                sigma = torch.zeros((), device=y.device, dtype=acp_t.dtype)
            dir_coef = torch.sqrt(torch.clamp(1 - acp_prev - sigma ** 2, min=0.0))

            y = torch.sqrt(acp_prev) * x0_pred + dir_coef * eps_pred
            if eta > 0:
                y = y + sigma * torch.randn_like(y)
        return y

    @torch.no_grad()
    def sample(self, x_cond: torch.Tensor, dim: int) -> torch.Tensor:
        if self.sampler == "ddim":
            return self.ddim_sample(x_cond, dim)
        return self.ancestral_sample(x_cond, dim)
