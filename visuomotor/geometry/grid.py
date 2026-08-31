"""Grid / voxel / feature-grid index geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch


def points_to_voxel_indices(
    points: torch.Tensor,
    bounds: Union[torch.Tensor, Sequence[Sequence[float]]],
    resolution: Union[int, Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ``[...,3]`` points to integer voxel indices and an in-bounds mask."""
    bounds = torch.as_tensor(bounds, dtype=points.dtype, device=points.device)
    if bounds.shape == (2, 3):
        lower, upper = bounds[0], bounds[1]
    elif bounds.shape == (3, 2):
        lower, upper = bounds[:, 0], bounds[:, 1]
    else:
        raise ValueError("bounds must have shape [2,3] or [3,2]")
    if isinstance(resolution, int):
        resolution = (resolution,) * 3
    resolution_tensor = points.new_tensor(tuple(resolution))
    if resolution_tensor.numel() != 3 or torch.any(resolution_tensor <= 0):
        raise ValueError("resolution must contain three positive values")
    unit = (points - lower) / (upper - lower)
    valid = ((unit >= 0) & (unit < 1)).all(dim=-1)
    indices = torch.floor(unit * resolution_tensor).long()
    maximum = resolution_tensor.long() - 1
    indices = torch.maximum(torch.zeros_like(indices), torch.minimum(indices, maximum))
    return indices, valid


def voxelize_points(
    points: torch.Tensor,
    features: torch.Tensor,
    bounds,
    resolution: Union[int, Sequence[int]],
) -> torch.Tensor:
    """Average point features into a dense ``[B,C,D,H,W]`` voxel grid."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,N,3]")
    if features.ndim != 3 or features.shape[:2] != points.shape[:2]:
        raise ValueError("features must have shape [B,N,C]")
    if isinstance(resolution, int):
        resolution = (resolution,) * 3
    depth, height, width = map(int, resolution)
    indices, valid = points_to_voxel_indices(points, bounds, resolution)
    batch_size, _, channels = features.shape
    flat_size = depth * height * width
    grid = features.new_zeros((batch_size, channels, flat_size))
    counts = features.new_zeros((batch_size, 1, flat_size))
    flat_indices = (
        indices[..., 0] * height * width + indices[..., 1] * width + indices[..., 2]
    )
    for batch_index in range(batch_size):
        selected = valid[batch_index]
        if not selected.any():
            continue
        target = flat_indices[batch_index, selected]
        source = features[batch_index, selected].transpose(0, 1)
        grid[batch_index].scatter_add_(1, target.unsqueeze(0).expand(channels, -1), source)
        counts[batch_index].scatter_add_(
            1, target.unsqueeze(0), features.new_ones((1, target.numel()))
        )
    grid = grid / counts.clamp_min(1)
    return grid.reshape(batch_size, channels, depth, height, width)


def voxel_cell_centers(
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    resolution: Sequence[int],
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return exact cell centers as ``[X,Y,Z,3]`` coordinates."""
    lower = torch.as_tensor(bounds_min, device=device, dtype=dtype)
    upper = torch.as_tensor(bounds_max, device=device, dtype=dtype)
    shape = tuple(int(size) for size in resolution)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("voxel bounds must contain three values")
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("voxel resolution must contain three positive values")
    if torch.any(lower >= upper):
        raise ValueError("voxel lower bounds must be below upper bounds")
    axes = [
        torch.linspace(
            lower[axis],
            upper[axis],
            shape[axis] + 1,
            device=device,
            dtype=dtype,
        )
        for axis in range(3)
    ]
    axes = [0.5 * (axis[:-1] + axis[1:]) for axis in axes]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def normalized_world_coordinate_channels(
    *,
    global_bounds_min: Sequence[float],
    global_bounds_max: Sequence[float],
    resolution: Sequence[int],
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Generate normalized absolute-world XYZ channels for a fixed world grid."""
    lower = torch.as_tensor(global_bounds_min, device=device, dtype=dtype)
    upper = torch.as_tensor(global_bounds_max, device=device, dtype=dtype)
    center = 0.5 * (lower + upper)
    half_width = 0.5 * float(upper[0] - lower[0])
    if half_width <= 0:
        raise ValueError("global workspace half-width must be positive")

    world = voxel_cell_centers(
        global_bounds_min,
        global_bounds_max,
        resolution,
        device=device,
        dtype=dtype,
    )
    return ((world - center) / half_width).movedim(-1, 0)


@dataclass(frozen=True)
class SourceVoxelGeometry:
    """World<->[-1,1] map for the raw voxel grid a scene is rendered into."""

    workspace_min: Tuple[float, float, float]
    workspace_size: float
    shape: Tuple[int, int, int]

    @classmethod
    def optional(cls, workspace_min, workspace_size, shape) -> Optional["SourceVoxelGeometry"]:
        """All-or-nothing constructor: ``None`` unless every argument is given."""
        if workspace_min is None or workspace_size is None or shape is None:
            return None
        return cls(
            tuple(float(v) for v in workspace_min),
            float(workspace_size),
            tuple(int(v) for v in shape),
        )

    @property
    def uniform_scale(self) -> float:
        if len(set(self.shape)) != 1:
            raise ValueError("uniform_scale requires a cubic grid")
        return self.workspace_size / self.shape[0]

    @property
    def pitch(self) -> Tuple[float, ...]:
        """Per-axis voxel edge length, matching the producer's ``ws/res + 1e-4``."""
        return tuple(self.workspace_size / size + 1e-4 for size in self.shape)

    @property
    def extent(self) -> Tuple[float, ...]:
        """World span the array actually covers -- ``pitch * shape``, not ``workspace_size``."""
        return tuple(step * size for step, size in zip(self.pitch, self.shape))

    @property
    def center(self) -> Tuple[float, ...]:
        """World position of the array centre, i.e. the fixed point of any grid rotation."""
        return tuple(
            minimum + span / 2 for minimum, span in zip(self.workspace_min, self.extent)
        )

    def _extent(self, device, dtype) -> torch.Tensor:
        return torch.tensor(self.extent, device=device, dtype=dtype)

    def world_to_grid(self, points: torch.Tensor) -> torch.Tensor:
        minimum = points.new_tensor(self.workspace_min)
        extent = self._extent(points.device, points.dtype)
        return (points - minimum) / extent * 2 - 1

    def grid_to_world(self, points: torch.Tensor) -> torch.Tensor:
        minimum = points.new_tensor(self.workspace_min)
        extent = self._extent(points.device, points.dtype)
        return (points + 1) / 2 * extent + minimum


@dataclass
class VoxelCropTransform:
    """Per-sample integer-window crop of a source voxel grid, as an affine map."""

    starts: torch.Tensor  # [B,3] float, integer-valued
    source_shape: Tuple[int, int, int]
    crop_shape: Tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.starts.ndim != 2 or self.starts.shape[-1] != 3:
            raise ValueError("starts must have shape [B,3]")
        max_start = self.starts.new_tensor(
            [s - c for s, c in zip(self.source_shape, self.crop_shape)]
        )
        if torch.any(self.starts < 0) or torch.any(self.starts > max_start + 1e-4):
            raise ValueError("starts must lie within [0, source_shape - crop_shape]")

    @property
    def scale(self) -> torch.Tensor:
        source = self.starts.new_tensor(self.source_shape)
        crop = self.starts.new_tensor(self.crop_shape)
        return source / crop

    @property
    def offset(self) -> torch.Tensor:
        crop = self.starts.new_tensor(self.crop_shape)
        return self.scale - 1 - 2 * self.starts / crop

    def source_to_crop(self, points: torch.Tensor) -> torch.Tensor:
        return points * self.scale + self.offset

    def contains_crop(self, points: torch.Tensor) -> torch.Tensor:
        return (points.abs() <= 1.0).all(dim=-1)

    def closest_crop_voxel(self, points: torch.Tensor) -> torch.Tensor:
        crop = self.starts.new_tensor(self.crop_shape)
        clamped = points.clamp(-1.0, 1.0)
        index = ((clamped + 1) / 2 * (crop - 1)).round()
        return index / (crop - 1) * 2 - 1

    def project_source_to_crop(self, points: torch.Tensor) -> torch.Tensor:
        """Crop-normalize ``points``; snap finite-but-outside-crop points to the crop edge."""
        crop_points = self.source_to_crop(points)
        finite = torch.isfinite(crop_points).all(dim=-1, keepdim=True)
        inside = self.contains_crop(crop_points).unsqueeze(-1)
        snapped = self.closest_crop_voxel(crop_points)
        return torch.where(finite & ~inside, snapped, crop_points)


@dataclass(frozen=True)
class FeatureGridGeometry:
    """Exact centres/spacing of a strided conv feature grid, in [-1,1] space.

    Rank-agnostic: ``centers`` is ``[*grid_shape, rank]`` and ``spacing`` is
    ``[rank]``, so the same class describes either a 3D voxel feature grid
    (``rank == 3``) or a 2D image-stage feature grid (``rank == 2``).
    """

    centers: torch.Tensor  # [*grid_shape, rank]
    spacing: torch.Tensor  # [rank]

    @classmethod
    def from_stride(cls, input_shape, output_shape, stride) -> "FeatureGridGeometry":
        rank = len(input_shape)
        if isinstance(stride, int):
            stride = (stride,) * rank
        axes, spacing = [], []
        for axis in range(rank):
            cell_width = 2.0 / input_shape[axis]
            axis_spacing = stride[axis] * cell_width
            first_center = -1.0 + axis_spacing / 2
            count = output_shape[axis]
            axes.append(first_center + torch.arange(count, dtype=torch.float32) * axis_spacing)
            spacing.append(axis_spacing)
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return cls(centers=grid, spacing=torch.tensor(spacing, dtype=torch.float32))

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(size) for size in self.centers.shape[:-1])

    def validate_attention(self, attention: torch.Tensor) -> None:
        rank = len(self.shape)
        if tuple(attention.shape[-rank:]) != self.shape:
            raise ValueError(
                f"attention map trailing dims {tuple(attention.shape[-rank:])} do not match "
                f"feature grid shape {self.shape}"
            )
