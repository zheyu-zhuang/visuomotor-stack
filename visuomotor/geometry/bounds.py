"""Reachable-set / coordinate-extrema math for spatial augmentation."""

from __future__ import annotations

from typing import Tuple

import torch


class PlanarWorkspace:
    """A square workspace footprint in the world XY plane, for yaw augmentation."""

    def __init__(self, center_xy, size: float) -> None:
        self.center_xy = tuple(float(v) for v in center_xy)
        if len(self.center_xy) != 2:
            raise ValueError("center_xy must have two values")
        self.size = float(size)

    def center(self, device=None, dtype=None) -> torch.Tensor:
        return torch.tensor(self.center_xy, device=device, dtype=dtype)

    def relative_xy(self, positions: torch.Tensor) -> torch.Tensor:
        return positions[..., :2] - self.center(positions.device, positions.dtype)

    def restore_xy(self, relative: torch.Tensor) -> torch.Tensor:
        return relative + self.center(relative.device, relative.dtype)

    def contains_xy(self, positions: torch.Tensor) -> torch.Tensor:
        half = self.size / 2
        relative = self.relative_xy(positions)
        return torch.isfinite(relative).all(dim=-1) & (relative.abs() <= half).all(dim=-1)

    def position_bounds(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact XYZ normalization bounds for workspace-constrained yaw.

        Accepted yaw augmentation keeps X/Y inside this workspace, while yaw
        leaves each position's Z coordinate unchanged.
        """
        half = self.size / 2
        center = self.center(positions.device, positions.dtype)
        lo_xy = (center - half).expand(*positions.shape[:-1], 2)
        hi_xy = (center + half).expand(*positions.shape[:-1], 2)
        return (
            torch.cat((lo_xy, positions[..., 2:3]), dim=-1),
            torch.cat((hi_xy, positions[..., 2:3]), dim=-1),
        )


def yaw_envelope_bounds(p: torch.Tensor, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conservative axis-aligned bound for ``p`` under a full-circle yaw about ``center``.

    Intentionally conservative (covers the reachable range, not the sampled
    yaw distribution): for a point ``p=[x,y,z]`` and centre ``c=[cx,cy,cz]``,
    with ``r = ||p[:2] - c[:2]||``, the bound is ``x,y in [c-r, c+r]`` and
    ``z`` unchanged. Use the same rotation centre/coordinate convention as the
    yaw augmentation this bounds.
    """
    if p.shape[-1] != 3 or center.shape[-1] != 3:
        raise ValueError("p and center must both be [...,3]")
    d_xy = p[..., :2] - center[..., :2]
    r = torch.linalg.vector_norm(d_xy, dim=-1, keepdim=True)
    lo_xy = center[..., :2] - r
    hi_xy = center[..., :2] + r
    lo = torch.cat((lo_xy, p[..., 2:3]), dim=-1)
    hi = torch.cat((hi_xy, p[..., 2:3]), dim=-1)
    return lo, hi
