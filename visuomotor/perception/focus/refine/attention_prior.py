"""An auxiliary keypose-attention loss, shared by the RGB and voxel focus-pool encoders.

Both :class:`~visuomotor.perception.focus.refine.planar.FocusRefine2d`
(RGB, 2D image-stage grid) and
:class:`~visuomotor.perception.focus.refine.volumetric.FocusRefine3d` (voxel, 3D
grid) attend over a :class:`~visuomotor.geometry.grid.FeatureGridGeometry`
feature grid, and both are supervised from the same input: a world-space
keypose position (``focus_target_pos``). Only how that world position maps
into the feature grid's normalized index space differs by rank -- world->voxel
grid for the 3D case, world->pixel->stage-grid (via camera projection) for the
2D case -- so this module holds the rank-agnostic half: turning a target
position (already expressed in the feature grid's coordinate space) into a
soft Gaussian heatmap, and the cross-entropy loss against a predicted
attention map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn

from visuomotor.geometry.grid import FeatureGridGeometry


def gaussian_feature_target(
    target_positions: torch.Tensor, geometry: FeatureGridGeometry, sigma_cells: float = 1.2
) -> torch.Tensor:
    """A soft probability mass over feature-grid cells, peaked at ``target_positions``.

    ``target_positions`` must already be expressed in ``geometry``'s
    coordinate space (voxel-grid-normalized for 3D, stage-grid pixel/index
    units for 2D) -- one position per batch element, ``[B, rank]``.
    """
    centers = geometry.centers.to(device=target_positions.device, dtype=target_positions.dtype)
    spacing = geometry.spacing.to(device=target_positions.device, dtype=target_positions.dtype)
    rank = centers.shape[-1]
    target = target_positions.reshape((-1,) + (1,) * rank + (rank,))
    diff = (target - centers) / spacing
    weight = torch.exp(-0.5 * diff.pow(2).sum(dim=-1) / (sigma_cells**2))
    flat = weight.flatten(1)
    return flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class FocusAttentionPrior(nn.Module):
    """An auxiliary loss supervising a focus-pool attention map toward a target cell."""

    _ALLOWED_KEYS = {"enabled", "weight", "sigma_cells", "bootstrap_steps"}

    def __init__(self, config: Optional[Mapping] = None) -> None:
        super().__init__()
        config = dict(config or {})
        unknown = set(config) - self._ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown FocusAttentionPrior config keys: {sorted(unknown)}")
        self.enabled = bool(config.get("enabled", False))
        self.weight = float(config.get("weight", 2e-4))
        self.sigma_cells = float(config.get("sigma_cells", 1.2))
        bootstrap_steps = config.get("bootstrap_steps")
        self.bootstrap_steps = None if bootstrap_steps is None else int(bootstrap_steps)
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        if self.sigma_cells <= 0:
            raise ValueError("sigma_cells must be positive")
        if self.bootstrap_steps is not None and self.bootstrap_steps <= 0:
            raise ValueError("bootstrap_steps must be positive when set")

    def is_bootstrapping(self, global_step: Optional[int]) -> bool:
        """Whether the loss should be active at ``global_step``.

        ``bootstrap_steps=None`` means no decay and keeps the loss active,
        matching this module's pre-decay default.
        """
        if self.bootstrap_steps is None:
            return True
        return global_step is not None and global_step < self.bootstrap_steps

    def active_weight(self, bootstrapping: bool) -> float:
        return self.weight if (self.enabled and bootstrapping) else 0.0

    def loss(
        self, attention: torch.Tensor, target: Mapping[str, torch.Tensor], geometry: FeatureGridGeometry
    ) -> torch.Tensor:
        geometry.validate_attention(attention)
        predicted = attention.mean(dim=1).flatten(1)
        predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        target_dist = gaussian_feature_target(target["pos"], geometry, self.sigma_cells)
        valid = target["valid"].to(dtype=predicted.dtype)
        cross_entropy = -(target_dist * predicted.clamp_min(1e-12).log()).sum(dim=-1)
        return (cross_entropy * valid).sum() / valid.sum().clamp_min(1.0)

    def runtime_config(self) -> str:
        if not self.enabled:
            return "FocusAttentionPrior: disabled"
        schedule = "always active" if self.bootstrap_steps is None else f"active for the first {self.bootstrap_steps} steps"
        return f"FocusAttentionPrior: weight={self.weight}, sigma_cells={self.sigma_cells}, {schedule}"


@dataclass(frozen=True)
class FocusPoolSpec:
    """Parsed ``focus_pool``/``pooling`` config: iteration count, head count, attention prior."""

    iters: int
    heads: int
    attention_prior: dict

    _ALLOWED_KEYS = {"iters", "heads", "attention_prior"}

    @classmethod
    def parse(cls, config: Optional[Mapping]) -> "FocusPoolSpec":
        config = dict(config or {})
        unknown = set(config) - cls._ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown focus_pool config keys: {sorted(unknown)}")
        iters = int(config.get("iters", 3))
        heads = int(config.get("heads", 4))
        if iters < 1:
            raise ValueError("iters must be positive")
        if heads < 1:
            raise ValueError("heads must be positive")
        return cls(iters=iters, heads=heads, attention_prior=dict(config.get("attention_prior", {})))
