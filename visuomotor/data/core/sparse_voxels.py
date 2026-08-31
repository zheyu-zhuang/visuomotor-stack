"""Sparse transport for occupancy-plus-colour voxel grids.

Only occupied cell indices and colours cross the dataloader boundary. Dense
grids are reconstructed on device. Cache arrays use this layout:

``voxel/index``    int32  ``[total_cells]``      ascending within each frame
``voxel/colour``   uint8  ``[total_cells, 3]``
``voxel/offsets``  int64  ``[n_frames + 1]``     cumulative cell counts

Frames are concatenated on disk and padded to ``voxel_max_points`` when read.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

INDEX_SUFFIX = "_index"
COLOUR_SUFFIX = "_colour"
PADDING_INDEX = -1
DENSE_CHANNELS = 4

INDEX_ARRAY = "voxel/index.npy"
COLOUR_ARRAY = "voxel/colour.npy"
OFFSETS_ARRAY = "voxel/offsets.npy"


def array_names(voxel_key: str) -> Tuple[str, str, str]:
    """Persistent sparse-array names for one voxel observation key."""
    key = str(voxel_key)
    if key == "voxel":
        return INDEX_ARRAY, COLOUR_ARRAY, OFFSETS_ARRAY
    return f"{key}/index.npy", f"{key}/colour.npy", f"{key}/offsets.npy"


def sparse_keys(voxel_key: str) -> Tuple[str, str]:
    """The ``(index, colour)`` observation keys carrying one voxel field."""
    return f"{voxel_key}{INDEX_SUFFIX}", f"{voxel_key}{COLOUR_SUFFIX}"


def cell_count(resolution: Sequence[int]) -> int:
    count = 1
    for size in resolution:
        count *= int(size)
    return count


def validate_dense_grid(grid: np.ndarray) -> None:
    """Reject a grid the sparse form could not represent exactly."""
    if grid.dtype != np.uint8:
        raise ValueError(f"voxel grid must be uint8, got {grid.dtype}")
    if grid.ndim != 4 or grid.shape[0] != DENSE_CHANNELS:
        raise ValueError(
            f"voxel grid must be [occupancy,R,G,B],X,Y,Z-shaped, got {grid.shape}"
        )
    occupancy = grid[0]
    if not np.all((occupancy == 0) | (occupancy == 1)):
        raise ValueError("voxel occupancy must be binary")
    if grid[1:, occupancy == 0].any():
        raise ValueError(
            "voxel colour must be zero wherever occupancy is zero; the sparse "
            "form carries colour only for occupied cells and would lose it"
        )


def encode(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce one dense ``[4,X,Y,Z]`` uint8 grid to its occupied cells."""
    validate_dense_grid(grid)
    index = np.flatnonzero(grid[0].reshape(-1)).astype(np.int32)
    colour = np.ascontiguousarray(grid[1:].reshape(3, -1)[:, index].T)
    return index, colour


def decode(
    index: np.ndarray, colour: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad one frame's cells to ``max_points`` for fixed-shape collation."""
    count = int(index.shape[0])
    if count > max_points:
        raise ValueError(
            f"voxel frame has {count} occupied cells, above the cache's "
            f"recorded maximum of {max_points}"
        )
    padded_index = np.full(max_points, PADDING_INDEX, dtype=np.int32)
    padded_colour = np.zeros((max_points, 3), dtype=np.uint8)
    padded_index[:count] = index
    padded_colour[:count] = colour
    return padded_index, padded_colour


def pack(frames: Sequence[Tuple[np.ndarray, np.ndarray]]) -> dict:
    """Concatenate per-frame cells into the three cache arrays."""
    counts = [int(index.shape[0]) for index, _ in frames]
    empty_index = np.zeros(0, dtype=np.int32)
    empty_colour = np.zeros((0, 3), dtype=np.uint8)
    return {
        INDEX_ARRAY: (
            np.concatenate([index for index, _ in frames]) if frames else empty_index
        ),
        COLOUR_ARRAY: (
            np.concatenate([colour for _, colour in frames]) if frames else empty_colour
        ),
        OFFSETS_ARRAY: np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        ).astype(np.int64),
        "max_points": max(counts) if counts else 0,
    }


def materialize(
    index: torch.Tensor, colour: torch.Tensor, shape: Sequence[int]
) -> torch.Tensor:
    """Scatter padded cells into a dense ``[...,4,X,Y,Z]`` grid."""
    channels, size_x, size_y, size_z = (int(value) for value in shape)
    if channels != DENSE_CHANNELS:
        raise ValueError(f"sparse voxels reconstruct 4 channels, not {channels}")
    if colour.shape[:-1] != index.shape or colour.shape[-1] != 3:
        raise ValueError(
            f"colour {tuple(colour.shape)} does not match index {tuple(index.shape)}"
        )
    leading, points = index.shape[:-1], index.shape[-1]
    cells = size_x * size_y * size_z

    flat_index = index.reshape(-1, points).long()
    frames = flat_index.shape[0]
    occupied = flat_index >= 0
    # Route padding to a discarded scratch cell.
    slot = torch.where(occupied, flat_index, flat_index.new_full((1,), cells))
    source = torch.cat(
        (
            occupied.to(colour.dtype).unsqueeze(1),
            colour.reshape(frames, points, 3).permute(0, 2, 1),
        ),
        dim=1,
    )
    grid = torch.zeros(
        (frames, channels, cells + 1), dtype=colour.dtype, device=colour.device
    )
    grid.scatter_(2, slot.unsqueeze(1).expand(frames, channels, points), source)
    return grid[:, :, :cells].reshape(*leading, channels, size_x, size_y, size_z)


class VoxelMaterializer:
    """Materialize sparse voxel fields using contract-declared shapes."""

    def __init__(self, voxel_shapes: Mapping[str, Sequence[int]]) -> None:
        self.voxel_shapes = {
            str(key): tuple(int(size) for size in shape)
            for key, shape in dict(voxel_shapes).items()
        }

    @property
    def enabled(self) -> bool:
        return bool(self.voxel_shapes)

    def get_runtime_config(self) -> str:
        if not self.enabled:
            return "VoxelMaterializer: no voxel fields"
        fields = ", ".join(
            f"{key}{tuple(shape)}" for key, shape in self.voxel_shapes.items()
        )
        return f"VoxelMaterializer: sparse cells -> dense grid on device for {fields}"

    def __call__(self, batch: dict) -> dict:
        if not self.enabled:
            return batch
        obs = batch["obs"]
        for key, shape in self.voxel_shapes.items():
            index_key, colour_key = sparse_keys(key)
            if index_key not in obs:
                # Rollouts already provide dense simulator grids.
                continue
            obs[key] = materialize(obs.pop(index_key), obs.pop(colour_key), shape)
        return batch
