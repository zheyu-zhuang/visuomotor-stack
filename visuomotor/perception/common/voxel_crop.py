"""Train-time random / eval-time centered crop of a source voxel grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from visuomotor.geometry import grid as Grid


@dataclass
class VoxelCropResult:
    voxels: torch.Tensor
    transform: Grid.VoxelCropTransform


class VoxelCropper(nn.Module):
    """Crops ``[B,C,X,Y,Z]`` voxel grids to ``crop_shape``, tracking the crop as a transform."""

    def __init__(self, crop_shape: Optional[Tuple[int, int, int]] = None) -> None:
        super().__init__()
        self.crop_shape = tuple(int(size) for size in crop_shape) if crop_shape is not None else None

    def forward(self, voxels: torch.Tensor) -> VoxelCropResult:
        if voxels.ndim != 5:
            raise ValueError(f"voxels must have shape [B,C,X,Y,Z], got {tuple(voxels.shape)}")
        source_shape = tuple(int(size) for size in voxels.shape[2:])
        crop_shape = self.crop_shape or source_shape
        max_start = [source - crop for source, crop in zip(source_shape, crop_shape)]
        if any(margin < 0 for margin in max_start):
            raise ValueError(f"crop_shape {crop_shape} must not exceed source_shape {source_shape}")

        batch = voxels.shape[0]
        if crop_shape == source_shape:
            starts = torch.zeros(batch, 3, device=voxels.device, dtype=torch.float32)
            transform = Grid.VoxelCropTransform(
                starts=starts, source_shape=source_shape, crop_shape=crop_shape
            )
            return VoxelCropResult(voxels=voxels, transform=transform)

        if self.training:
            starts = torch.stack(
                [
                    torch.randint(0, margin + 1, (batch,), device=voxels.device)
                    for margin in max_start
                ],
                dim=-1,
            ).to(dtype=torch.float32)
        else:
            centered = torch.tensor([margin // 2 for margin in max_start], device=voxels.device)
            starts = centered.unsqueeze(0).expand(batch, -1).to(dtype=torch.float32)

        cropped = torch.stack(
            [
                voxels[
                    index,
                    :,
                    int(start[0]) : int(start[0]) + crop_shape[0],
                    int(start[1]) : int(start[1]) + crop_shape[1],
                    int(start[2]) : int(start[2]) + crop_shape[2],
                ]
                for index, start in enumerate(starts)
            ],
            dim=0,
        )
        transform = Grid.VoxelCropTransform(starts=starts, source_shape=source_shape, crop_shape=crop_shape)
        return VoxelCropResult(voxels=cropped, transform=transform)
