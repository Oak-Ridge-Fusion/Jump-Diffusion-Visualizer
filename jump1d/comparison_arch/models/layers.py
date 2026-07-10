"""
models/layers.py
=================
Building blocks for the Conditional RealNVP / Neural Spline Flow:

  * ``ActNorm``            -- data-dependent per-channel affine layer
                               (Kingma & Dhariwal, Glow 2018).
  * ``Permutation``         -- fixed random channel permutation (zero log-det).
  * ``ConditionalAffineCoupling`` -- affine coupling layer whose scale/shift
                               network is conditioned on an external context
                               vector (the embedded ``X_t``) in addition to
                               the usual masked half of the flow's own input.
  * ``ConditionalSplineCoupling``  -- same idea, but the per-dimension map is
                               a monotonic rational-quadratic spline instead
                               of an affine transform (Neural Spline Flows,
                               Durkan et al. 2019). See the docstring below
                               for why this matters at ``dim == 1``.

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
fixed), so every coupling layer is a full conditional transform driven
entirely by the external context. For ``dim >= 2`` the mask alternates in a
checkerboard pattern by layer index, recovering standard RealNVP/NSF
behaviour.

Why the affine coupling has a ceiling at dim == 1
--------------------------------------------------
With an *affine* map, ``y = x * exp(s(ctx)) + t(ctx)``, the coefficients
depend only on ``ctx`` (never on ``x`` itself, since there is nothing left
to hold out). Composing affine-in-``x`` maps whose coefficients don't depend
on ``x`` is still affine in ``x`` -- so stacking any number of
``ConditionalAffineCoupling`` layers at ``dim == 1`` can only ever represent
a conditional Gaussian ``p(y|x)`` (mean/std are complex functions of ``x``,
but the shape is always Gaussian). ``ConditionalSplineCoupling`` replaces
the affine map with an arbitrary monotonic warp, so the same stack can
represent skewed / multimodal / boundary-compressed conditional shapes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Neural Spline Flow machinery (Durkan, Bekasov, Murray & Papamakarios, 2019,
# "Neural Spline Flows"). Monotonic piecewise rational-quadratic transform on
# [-tail_bound, tail_bound], identity (slope 1) outside -- so the map is
# defined and invertible on all of R, with a closed-form log-det.
# ---------------------------------------------------------------------------
_MIN_BIN_WIDTH = 1e-3
_MIN_BIN_HEIGHT = 1e-3
_MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def _rational_quadratic_spline(
    inputs: torch.Tensor, unnormalized_widths: torch.Tensor, unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor, inverse: bool,
    left: float, right: float, bottom: float, top: float,
):
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = _MIN_BIN_WIDTH + (1 - _MIN_BIN_WIDTH * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = _MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = _MIN_BIN_HEIGHT + (1 - _MIN_BIN_HEIGHT * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = _searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]
    bin_idx = bin_idx.clamp(0, num_bins - 1)

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]
    input_delta = (heights / widths).gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives[..., :-1].gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        discriminant = torch.clamp(b.pow(2) - 4 * a * c, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor, unnormalized_widths: torch.Tensor, unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor, inverse: bool = False, tail_bound: float = 5.0,
):
    """Spline on ``[-tail_bound, tail_bound]``, identity (slope 1) outside it."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Pad derivatives with the boundary slope (=1) so the spline joins the
    # identity tails continuously and differentiably.
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = float(np.log(np.exp(1 - _MIN_DERIVATIVE) - 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs = torch.where(inside, outputs, inputs)

    if inside.any():
        in_outputs, in_logabsdet = _rational_quadratic_spline(
            inputs=inputs[inside],
            unnormalized_widths=unnormalized_widths[inside, :],
            unnormalized_heights=unnormalized_heights[inside, :],
            unnormalized_derivatives=unnormalized_derivatives[inside, :],
            inverse=inverse,
            left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        )
        outputs = outputs.masked_scatter(inside, in_outputs)
        logabsdet = logabsdet.masked_scatter(inside, in_logabsdet)
    return outputs, logabsdet


class ConditionalSplineCoupling(nn.Module):
    """Monotonic rational-quadratic spline coupling layer, conditioned on an
    external context vector (embedded ``X_t``) exactly like
    ``ConditionalAffineCoupling`` -- same mask convention, same interface
    (``forward``/``inverse`` returning ``(transformed, logdet)`` /
    ``transformed``). See the module docstring for why this fixes the
    conditional-Gaussian ceiling that the affine version hits at ``dim==1``.
    """

    def __init__(self, dim: int, context_dim: int, hidden_dim: int, n_hidden_layers: int,
                 mask: torch.Tensor, num_bins: int = 8, tail_bound: float = 5.0):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.register_buffer("mask", mask.float())  # 1 = identity, 0 = transformed

        out_dim = dim * (3 * num_bins - 1)   # widths + heights + internal derivatives, per dim
        layers = [nn.Linear(dim + context_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, n_hidden_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        # zero-init -> uniform bins + unit derivatives -> starts near-identity,
        # same stabilisation trick as the affine coupling's zero-init.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _spline_params(self, x_or_y: torch.Tensor, context: torch.Tensor):
        held = x_or_y * self.mask
        inp = torch.cat([held, context], dim=-1)
        raw = self.net(inp).view(-1, self.dim, 3 * self.num_bins - 1)
        widths, heights, derivatives = raw.split(
            [self.num_bins, self.num_bins, self.num_bins - 1], dim=-1
        )
        return widths, heights, derivatives

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        widths, heights, derivatives = self._spline_params(x, context)
        y, logabsdet = unconstrained_rational_quadratic_spline(
            x, widths, heights, derivatives, inverse=False, tail_bound=self.tail_bound,
        )
        keep = 1.0 - self.mask
        y = self.mask * x + keep * y
        logdet = (logabsdet * keep).sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        # Identity part of y equals identity part of x -> same params are
        # valid for the inverse spline call, exactly like the affine case.
        widths, heights, derivatives = self._spline_params(y, context)
        x, _ = unconstrained_rational_quadratic_spline(
            y, widths, heights, derivatives, inverse=True, tail_bound=self.tail_bound,
        )
        x = self.mask * y + (1.0 - self.mask) * x
        return x
