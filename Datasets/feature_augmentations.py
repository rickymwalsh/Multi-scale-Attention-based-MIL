"""Feature-space augmentations for offline MIL training.

Augmentations are applied stochastically at __getitem__ time so each
training epoch sees independently augmented bags.

Spatially-aware augmentations (simulated_low_res, mirroring, gaussian_blur)
accept a ``grid_shape=(H, W)`` argument to reconstruct the 2D patch grid
from the sorted (N=H*W, D) feature tensor.  Patches are sorted row-major
(y ascending, then x ascending), so ``x.view(H, W, D)`` gives the correct
spatial layout.
"""

from __future__ import annotations

import random
import torch
import torch.nn.functional as F
from typing import Union


def _resolve(value):
    """Resolve a YAML parameter value that may be a sampled distribution.

    Scalar values (int, float) are returned as-is.  A dict with a ``type``
    key is treated as a distribution specification and sampled once:

        type: uniform   → random.uniform(low, high)
        type: normal    → abs(random.gauss(mean, std))   (always positive)
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
        import math
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
    gaussian_noise      : p, sigma
    rotation_scaling    : p, scale_range ([lo, hi])
    simulated_low_res   : p, factor_range ([min_int, max_int])
    mirroring           : p  (per-axis flip probability is fixed at 0.5 internally)
    gaussian_blur       : p, kernel_size (odd int), sigma
    """

    def __init__(self, config: dict):
        self.cfg = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: Union[torch.Tensor, dict, list],
        grid_shape=None,
    ) -> Union[torch.Tensor, dict, list]:
        """Apply the augmentation pipeline to a bag.

        Parameters
        ----------
        x : Tensor (N, D) | dict {scale: Tensor} | list [Tensor, ...]
        grid_shape : (H, W) | dict {scale: (H, W)} | list [(H,W), ...] | None
            Spatial grid dimensions matching the structure of ``x``.
            Required for simulated_low_res, mirroring, and gaussian_blur.
        """
        if isinstance(x, torch.Tensor):
            return self._augment_tensor(x, grid_shape)
        elif isinstance(x, dict):
            return {
                k: self._augment_tensor(
                    v,
                    grid_shape[k] if isinstance(grid_shape, dict) else grid_shape,
                )
                for k, v in x.items()
            }
        elif isinstance(x, list):
            return [
                self._augment_tensor(
                    t,
                    grid_shape[i] if isinstance(grid_shape, (list, tuple)) and not (
                        len(grid_shape) == 2
                        and isinstance(grid_shape[0], int)
                    ) else grid_shape,
                )
                for i, t in enumerate(x)
            ]
        else:
            raise TypeError(
                f"OfflineFeatureAugmentor: unsupported x type {type(x)}"
            )

    # ------------------------------------------------------------------
    # Per-tensor pipeline
    # ------------------------------------------------------------------

    def _augment_tensor(
        self, x: torch.Tensor, grid_shape=None
    ) -> torch.Tensor:
        """Apply each enabled augmentation in sequence to a single (N, ...) tensor.

        Features may be stored as (N, D) or higher-dimensional (e.g. (N, D1, D2)
        for FPN levels that retain spatial structure).  All augmentation methods
        assume 2-D input, so we flatten to (N, D_flat) before the pipeline and
        restore the original trailing shape afterwards.  N is always preserved.
        """
        orig_dtype = x.dtype
        orig_shape = x.shape  # e.g. (N,) or (N, D) or (N, D1, D2)

        # Cast to float32: always creates a NEW tensor (safe to mutate),
        # and avoids float16 underflow in noise / Gaussian ops.
        x = x.float()

        # Flatten to 2-D so every augmentation method sees (N, D).
        if x.dim() != 2:
            x = x.flatten(start_dim=1)  # (N, D1*D2*...)

        cfg = self.cfg

        if 'gaussian_noise' in cfg and torch.rand(1).item() < cfg['gaussian_noise']['p']:
            x = self._gaussian_noise(
                x,
                _resolve(cfg['gaussian_noise']['sigma']),
                _resolve(cfg['gaussian_noise'].get('delta_p', 1.0)),
            )

        if 'rotation_scaling' in cfg and torch.rand(1).item() < cfg['rotation_scaling']['p']:
            lo, hi = cfg['rotation_scaling']['scale_range']
            x = self._rotation_scaling(x, _resolve(lo), _resolve(hi))

        if 'simulated_low_res' in cfg and torch.rand(1).item() < cfg['simulated_low_res']['p']:
            if grid_shape is not None:
                x = self._simulated_low_res(
                    x, grid_shape, cfg['simulated_low_res']['factor_range']
                )

        if 'mirroring' in cfg and torch.rand(1).item() < cfg['mirroring']['p']:
            if grid_shape is not None:
                x = self._mirroring(x, grid_shape)

        if 'gaussian_blur' in cfg and torch.rand(1).item() < cfg['gaussian_blur']['p']:
            if grid_shape is not None:
                x = self._gaussian_blur(
                    x,
                    grid_shape,
                    _resolve(cfg['gaussian_blur']['kernel_size']),
                    _resolve(cfg['gaussian_blur']['sigma']),
                )

        # Restore original trailing shape (N unchanged; D_flat splits back).
        if len(orig_shape) != 2:
            x = x.view(orig_shape[0], *orig_shape[1:])

        return x.to(orig_dtype)

    # ------------------------------------------------------------------
    # Individual augmentations (accept/return float32, shape (N, D))
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_noise(x: torch.Tensor, sigma: float, delta_p: float = 1.0) -> torch.Tensor:
        """Add masked Gaussian noise: x + noise * delta.

        ``noise`` ~ N(0, sigma²), ``delta`` ~ Bernoulli(delta_p) element-wise.
        When delta_p=1.0 (default) all elements receive noise; lower values
        sparsify the corruption so only a fraction of feature values are perturbed.
        ``sigma`` is resolved before this call so it is always a plain float.
        """
        noise = sigma * torch.randn_like(x)
        if delta_p < 1.0:
            delta = (torch.rand_like(x) < delta_p).to(x.dtype)
            noise = noise * delta
        return x + noise

    @staticmethod
    def _rotation_scaling(
        x: torch.Tensor, scale_lo: float, scale_hi: float
    ) -> torch.Tensor:
        """Random isotropic scale followed by a Householder reflection.

        The Householder transform H(x) = x − 2(x·v)vᵀ applies an orthogonal
        reflection to each row of x.  Cost: O(N·D) — no D×D matrix.
        """
        # Random isotropic scale
        scale = scale_lo + (scale_hi - scale_lo) * torch.rand(1).item()
        x = x * scale

        # Householder reflection with a random unit vector
        D = x.shape[1]
        v = torch.randn(D, device=x.device, dtype=x.dtype)
        v = v / v.norm()
        proj = x @ v                                          # (N,)
        x = x - 2.0 * proj.unsqueeze(1) * v.unsqueeze(0)    # (N, D)
        return x

    @staticmethod
    def _simulated_low_res(
        x: torch.Tensor,
        grid_shape: tuple,
        factor_range: list,
    ) -> torch.Tensor:
        """Downsample the 2D feature grid then upsample back (bilinear).

        Simulates the effect of extracting features from a lower-resolution
        image: the coarser grid forces spatial blending of adjacent patches.
        N is preserved so collate_fn / torch.stack is unaffected.

        Parameters
        ----------
        factor_range : [min_factor, max_factor]
            Integer downsampling factor sampled uniformly in the given range.
        """
        H, W = grid_shape
        N, D = x.shape

        factor = random.randint(int(factor_range[0]), int(factor_range[1]))
        if factor <= 1:
            return x

        H_d = max(1, H // factor)
        W_d = max(1, W // factor)

        # (N, D) → (1, D, H, W) for F.interpolate
        x_grid = x.view(H, W, D).permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
        x_down = F.interpolate(
            x_grid, size=(H_d, W_d), mode='bilinear', align_corners=False
        )
        x_up = F.interpolate(
            x_down, size=(H, W), mode='bilinear', align_corners=False
        )
        return x_up.squeeze(0).permute(1, 2, 0).reshape(N, D)

    @staticmethod
    def _mirroring(x: torch.Tensor, grid_shape: tuple) -> torch.Tensor:
        """Randomly flip the 2D patch grid along horizontal and/or vertical axes.

        The outer probability ``p`` controls whether mirroring fires at all.
        Conditionally on firing, each axis is independently flipped with
        probability 0.5.  If neither axis is selected, x is returned unchanged.
        """
        H, W = grid_shape
        N, D = x.shape

        x_grid = x.view(H, W, D)

        axes = []
        if torch.rand(1).item() < 0.5:
            axes.append(0)   # vertical flip
        if torch.rand(1).item() < 0.5:
            axes.append(1)   # horizontal flip

        if axes:
            x_grid = x_grid.flip(axes)

        return x_grid.reshape(N, D)

    @staticmethod
    def _gaussian_blur(
        x: torch.Tensor,
        grid_shape: tuple,
        kernel_size: int,
        sigma: float,
    ) -> torch.Tensor:
        """Apply a 2D Gaussian blur over the spatial patch grid.

        Uses depthwise F.conv2d (groups=D) so each feature channel is
        smoothed independently.  The kernel is clamped to min(H, W) and
        forced to be odd so padding = kernel_size // 2 gives exact output size.
        """
        H, W = grid_shape
        N, D = x.shape

        # Clamp to spatial dimensions and force odd
        ks = min(int(kernel_size), min(H, W))
        if ks % 2 == 0:
            ks -= 1
        if ks < 1:
            return x

        half = ks // 2

        # Separable 2D Gaussian kernel (ks, ks)
        coords = torch.arange(ks, dtype=x.dtype, device=x.device) - half
        k1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)   # outer product
        k2d = k2d / k2d.sum()

        # (N, D) → (1, D, H, W) for conv2d
        x_grid = x.view(H, W, D).permute(2, 0, 1).unsqueeze(0)   # (1, D, H, W)

        # Depthwise weight: (D, 1, ks, ks)
        weight = k2d.unsqueeze(0).unsqueeze(0).expand(D, 1, ks, ks)

        x_blur = F.conv2d(x_grid, weight, bias=None, padding=half, groups=D)
        return x_blur.squeeze(0).permute(1, 2, 0).reshape(N, D)
