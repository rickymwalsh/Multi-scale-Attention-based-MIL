"""Feature-space augmentations for offline MIL training.

Augmentations are applied stochastically at __getitem__ time so each
training epoch sees independently augmented bags.

All augmentations accept ``x`` of shape ``(N, *feat_shape)`` and return
the same shape.  Spatially-aware augmentations (rotation_scaling,
simulated_low_res, mirroring, gaussian_blur) require ``x.dim() == 4``,
i.e. ``(N, C, H, W)`` — the format in which FPN patch features are stored.
For flat ``(N, D)`` features these augmentations are silently skipped;
only gaussian_noise applies regardless of shape.
"""

from __future__ import annotations

import math
import random
import torch
import torch.nn.functional as F
from typing import Union


def _resolve(value):
    """Resolve a YAML parameter value that may be a sampled distribution.

    Scalar values (int, float) are returned as-is.  A dict with a ``type``
    key is treated as a distribution specification and sampled once:

        type: uniform    → random.uniform(low, high)
        type: normal     → abs(random.gauss(mean, std))   (always positive)
        type: loguniform → exp(uniform(log(low), log(high)))

    This allows YAML entries such as::

        sigma:
          type: uniform
          low: 0.0
          high: 1.0

    which ``yaml.safe_load`` loads as a plain dict — no unsafe eval needed.
    """
    if not isinstance(value, dict) or 'type' not in value:
        return value
    dist = value['type'].lower()
    if dist == 'uniform':
        return random.uniform(float(value['low']), float(value['high']))
    if dist == 'normal':
        return abs(random.gauss(float(value['mean']), float(value['std'])))
    if dist == 'loguniform':
        lo, hi = math.log(float(value['low'])), math.log(float(value['high']))
        return math.exp(random.uniform(lo, hi))
    raise ValueError(f"Unknown distribution type '{dist}' in augmentation config")


class OfflineFeatureAugmentor:
    """Apply a configurable pipeline of feature-space augmentations to a MIL bag.

    Parameters
    ----------
    config : dict
        Parsed contents of ``data_augmentation.yaml``.  Each key is an
        augmentation name mapping to a sub-dict with at least a ``p`` field
        (application probability in [0, 1]) plus augmentation-specific
        parameters.

    Supported augmentation keys
    ---------------------------
    gaussian_noise      : p, sigma, delta_p
                          Works on any input shape.
                          Optionally accepts a ``per_scale`` sub-dict to apply
                          different noise parameters to individual scale keys
                          (e.g. C4, C5) when the input is a dict.  Each entry
                          in ``per_scale`` may override ``sigma`` and/or
                          ``delta_p``; ``p`` is always taken from the top level
                          (the roll is shared across all scales).  Example::

                              gaussian_noise:
                                p: 0.5
                                sigma: 0.05
                                per_scale:
                                  C4:
                                    sigma: 0.03
                                  C5:
                                    sigma: 0.10
                                    delta_p: 0.8

    rotation_scaling    : p, angle (degrees; scalar or distribution),
                          scale_range ([lo, hi]).
                          Requires (N, C, H, W) input; skipped otherwise.
    simulated_low_res   : p, factor_range ([min_int, max_int]).
                          Requires (N, C, H, W) input; skipped otherwise.
    mirroring           : p.  Per-axis flip probability fixed at 0.5 internally.
                          Requires (N, C, H, W) input; skipped otherwise.
    gaussian_blur       : p, kernel_size (odd int), sigma.
                          Requires (N, C, H, W) input; skipped otherwise.
    """

    def __init__(self, config: dict):
        self.cfg = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: Union[torch.Tensor, dict, list],
    ) -> Union[torch.Tensor, dict, list]:
        """Apply the augmentation pipeline to a bag.

        Parameters
        ----------
        x : Tensor (N, *feat_shape) | dict {scale: Tensor} | list [Tensor, ...]
        """
        if isinstance(x, torch.Tensor):
            return self._augment_tensor(x)
        elif isinstance(x, dict):
            return {
                k: self._augment_tensor(v) for k, v in x.items()
            }
        elif isinstance(x, list):
            # Assumes that list inputs are ordered by scale x[0] = C4 and x[1] = C5
            scale_keys = ['C4', 'C5'] if len(x) == 2 else [None] * len(x)
            return [self._augment_tensor(t, scale_key=scale_key) for t, scale_key in zip(x, scale_keys)]
        else:
            raise TypeError(
                f"OfflineFeatureAugmentor: unsupported x type {type(x)}"
            )

    # ------------------------------------------------------------------
    # Per-tensor pipeline
    # ------------------------------------------------------------------

    def _augment_tensor(
        self,
        x: torch.Tensor,
        *,
        scale_key: str | None = None,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        # Cast to float32: always creates a NEW tensor (safe to mutate),
        # and avoids float16 underflow in noise / Gaussian ops.
        x = x.float()

        cfg = self.cfg

        if 'gaussian_noise' in cfg:
            noise_cfg = cfg['gaussian_noise']
            if torch.rand(1).item() < noise_cfg['p']:
                # Merge per-scale overrides when a scale key is given.
                per_scale = noise_cfg.get('per_scale', {})
                if scale_key is not None and scale_key in per_scale:
                    noise_cfg = {**noise_cfg, **per_scale[scale_key]}
                x = self._gaussian_noise(
                    x,
                    _resolve(noise_cfg['sigma']),
                    _resolve(noise_cfg.get('delta_p', 1.0)),
                )

        if 'rotation_scaling' in cfg and torch.rand(1).item() < cfg['rotation_scaling']['p']:
            lo, hi = cfg['rotation_scaling']['scale_range']
            x = self._rotation_scaling(
                x,
                _resolve(cfg['rotation_scaling']['angle']),
                _resolve(lo),
                _resolve(hi),
            )

        if 'simulated_low_res' in cfg and torch.rand(1).item() < cfg['simulated_low_res']['p']:
            x = self._simulated_low_res(x, cfg['simulated_low_res']['factor_range'])

        if 'mirroring' in cfg and torch.rand(1).item() < cfg['mirroring']['p']:
            x = self._mirroring(x)

        if 'gaussian_blur' in cfg and torch.rand(1).item() < cfg['gaussian_blur']['p']:
            x = self._gaussian_blur(
                x,
                _resolve(cfg['gaussian_blur']['kernel_size']),
                _resolve(cfg['gaussian_blur']['sigma']),
            )

        return x.to(orig_dtype)

    # ------------------------------------------------------------------
    # Individual augmentations
    # Each accepts (N, *feat_shape) and returns the same shape.
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_noise(x: torch.Tensor, sigma: float, delta_p: float = 1.0) -> torch.Tensor:
        """Add masked Gaussian noise: x + noise * delta.

        ``noise`` ~ N(0, sigma²), ``delta`` ~ Bernoulli(delta_p) element-wise.
        When delta_p=1.0 all elements receive noise; lower values sparsify the
        corruption so only a fraction of feature values are perturbed.
        Works on any input shape.
        """
        noise = sigma * torch.randn_like(x)
        if delta_p < 1.0:
            noise = noise * (torch.rand_like(x) < delta_p).to(x.dtype)
        return x + noise

    @staticmethod
    def _rotation_scaling(
        x: torch.Tensor,
        angle_deg: float,
        scale_lo: float,
        scale_hi: float,
    ) -> torch.Tensor:
        """Rotate and isotropically scale the spatial feature maps.

        Requires ``x.dim() == 4`` i.e. ``(N, C, H, W)``; returns x unchanged
        for flat inputs.  The same transformation is applied to all N feature
        maps (simulating a consistently rotated view across the bag).

        Uses F.affine_grid + F.grid_sample (bilinear, border padding) so H and
        W are preserved and out-of-bounds positions use edge values.

        Parameters
        ----------
        angle_deg : float
            Rotation angle in degrees.  Positive = counter-clockwise.
        scale_lo, scale_hi : float
            Isotropic scale factor sampled uniformly in [scale_lo, scale_hi].
            Scale > 1 zooms in; scale < 1 zooms out.
        """
        if x.dim() != 4:
            return x

        N = x.shape[0]
        scale = scale_lo + (scale_hi - scale_lo) * torch.rand(1).item()
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta) * scale
        sin_t = math.sin(theta) * scale

        # 2×3 affine matrix broadcast across all N feature maps
        mat = torch.tensor(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0]],
            dtype=x.dtype, device=x.device,
        ).unsqueeze(0).expand(N, -1, -1)  # (N, 2, 3)

        grid = F.affine_grid(mat, x.shape, align_corners=False)  # (N, H, W, 2)
        return F.grid_sample(
            x, grid, mode='bilinear', padding_mode='border', align_corners=False
        )

    @staticmethod
    def _simulated_low_res(x: torch.Tensor, factor_range: list) -> torch.Tensor:
        """Downsample the spatial feature maps then upsample back (bilinear).

        Requires ``x.dim() == 4``; returns x unchanged for flat inputs.
        Simulates extracting features from a lower-resolution image: the
        coarser representation forces spatial blending of neighbouring positions.

        Parameters
        ----------
        factor_range : [min_factor, max_factor]
            Integer downsampling factor sampled uniformly in the given range.
        """
        if x.dim() != 4:
            return x

        _, _, H, W = x.shape
        factor = random.randint(int(factor_range[0]), int(factor_range[1]))
        if factor <= 1:
            return x

        H_d = max(1, H // factor)
        W_d = max(1, W // factor)

        x_down = F.interpolate(x, size=(H_d, W_d), mode='bilinear', align_corners=False)
        return F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)

    @staticmethod
    def _mirroring(x: torch.Tensor) -> torch.Tensor:
        """Randomly flip the spatial feature maps along H and/or W.

        Requires ``x.dim() == 4``; returns x unchanged for flat inputs.
        The outer probability ``p`` controls whether mirroring fires at all.
        Conditionally on firing, H and W are each flipped independently at p=0.5.
        """
        if x.dim() != 4:
            return x

        axes = []
        if torch.rand(1).item() < 0.5:
            axes.append(2)   # flip H
        if torch.rand(1).item() < 0.5:
            axes.append(3)   # flip W

        if axes:
            x = x.flip(axes)
        return x

    @staticmethod
    def _gaussian_blur(
        x: torch.Tensor,
        kernel_size: int,
        sigma: float,
    ) -> torch.Tensor:
        """Apply a 2D Gaussian blur over the spatial feature maps.

        Requires ``x.dim() == 4``; returns x unchanged for flat inputs.
        Uses depthwise F.conv2d (groups=C) so each channel is blurred
        independently.  Kernel is clamped to min(H, W) and forced odd.
        """
        if x.dim() != 4:
            return x

        _, C, H, W = x.shape

        ks = min(int(kernel_size), min(H, W))
        if ks % 2 == 0:
            ks -= 1
        if ks < 1:
            return x

        half = ks // 2

        # Separable 2D Gaussian kernel (ks, ks)
        coords = torch.arange(ks, dtype=x.dtype, device=x.device) - half
        k1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
        k2d = k2d / k2d.sum()

        # Depthwise weight: (C, 1, ks, ks)
        weight = k2d.unsqueeze(0).unsqueeze(0).expand(C, 1, ks, ks)

        return F.conv2d(x, weight, bias=None, padding=half, groups=C)
