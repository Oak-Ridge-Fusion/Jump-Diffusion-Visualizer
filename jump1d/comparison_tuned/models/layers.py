"""
models/layers.py
=================
Building blocks for the Conditional RealNVP normalizing flow:

  * ``ActNorm``            -- data-dependent per-channel affine layer
                               (Kingma & Dhariwal, Glow 2018).
  * ``Permutation``         -- fixed random channel permutation (zero log-det).
  * ``ConditionalAffineCoupling`` -- affine coupling layer whose scale/shift
                               network is conditioned on an external context
                               vector (the embedded ``X_t``) in addition to
                               the usual masked half of the flow's own input.

All layers implement ``forward`` (data -> latent, used for log-likelihood)
and ``inverse`` (latent -> data, used for sampling), each returning the
transformed tensor and the (forward) log-determinant of the Jacobian.

Design note on dimensionality
------------------------------
The current dataset is a scalar SDE (``dim == 1``), so a coupling layer has
no second half of its own dimensions to split against.  We handle this
generally: the binary mask marks which of the flow's dimensions are held
fixed ("identity", used only as extra conditioner input) vs. transformed.
For ``dim == 1`` the mask is all-transformed (there is nothing to hold
fixed), so every coupling layer is a full conditional affine transform driven
entirely by the external context -- a valid, well-defined degenerate case of
RealNVP.  For ``dim >= 2`` the mask alternates in a checkerboard pattern by
layer index, recovering standard RealNVP behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_checkerboard_mask(dim: int, layer_idx: int) -> torch.Tensor:
    """Boolean mask, True = held fixed ("identity") this layer, False = transformed."""
    if dim == 1:
        return torch.zeros(dim, dtype=torch.bool)
    mask = (torch.arange(dim) % 2) == 0
    if layer_idx % 2 == 1:
        mask = ~mask
    return mask


class ActNorm(nn.Module):
    """Per-channel affine layer, initialised from the statistics of the first
    batch it sees so that its output is zero-mean/unit-variance (Glow, 2018)."""

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _data_dependent_init(self, x: torch.Tensor) -> None:
        std = x.std(dim=0) + 1e-6
        mean = x.mean(dim=0)
        self.log_scale.data.copy_(-torch.log(std))
        self.bias.data.copy_(-mean)
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor):
        if self.training and not bool(self.initialized):
            self._data_dependent_init(x)
        y = (x + self.bias) * torch.exp(self.log_scale)
        logdet = self.log_scale.sum().expand(x.shape[0])
        return y, logdet

    def inverse(self, y: torch.Tensor):
        x = y * torch.exp(-self.log_scale) - self.bias
        return x


class Permutation(nn.Module):
    """Fixed random permutation of the flow's own dimensions (log-det = 0)."""

    def __init__(self, dim: int, generator: torch.Generator | None = None):
        super().__init__()
        if dim > 1:
            perm = torch.randperm(dim, generator=generator)
        else:
            perm = torch.tensor([0])
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x: torch.Tensor):
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y: torch.Tensor):
        return y[:, self.inv_perm]


class ConditionalAffineCoupling(nn.Module):
    """Affine coupling layer: y = x * exp(s(x_id, ctx)) + t(x_id, ctx).

    ``x_id`` is the masked ("identity") part of the flow's own input, ``ctx``
    is the externally supplied conditioning context (embedded ``X_t``).
    Entries marked identity by the mask are passed through unchanged.
    """

    def __init__(self, dim: int, context_dim: int, hidden_dim: int,
                 n_hidden_layers: int, mask: torch.Tensor, clamp: float = 2.0):
        super().__init__()
        self.register_buffer("mask", mask.float())  # 1 = identity, 0 = transformed
        self.clamp = clamp

        layers = [nn.Linear(dim + context_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, n_hidden_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 2 * dim)]
        self.net = nn.Sequential(*layers)
        # zero-init the last layer -> the layer starts as an identity map,
        # which stabilises early training of deep coupling stacks.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _scale_shift(self, x_or_y: torch.Tensor, context: torch.Tensor):
        held = x_or_y * self.mask
        inp = torch.cat([held, context], dim=-1)
        log_scale, shift = self.net(inp).chunk(2, dim=-1)
        log_scale = self.clamp * torch.tanh(log_scale)
        keep = 1.0 - self.mask
        return log_scale * keep, shift * keep

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        log_scale, shift = self._scale_shift(x, context)
        y = x * torch.exp(log_scale) + shift
        logdet = log_scale.sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        # The identity part of y equals the identity part of x, so the same
        # scale/shift network call is valid for inverting the transform.
        log_scale, shift = self._scale_shift(y, context)
        x = (y - shift) * torch.exp(-log_scale)
        return x
