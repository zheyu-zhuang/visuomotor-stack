"""A plain (non-equivariant) 3D ResNet trunk for voxel-grid observations."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

_STAGE_WIDTH_DIVISORS = (8, 2, 1, 0.5)
_STAGE_BLOCK_COUNTS = (1, 2, 2, 2)


def group_norm_3d(num_channels: int, channels_per_group: int = 16) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, num_channels // channels_per_group), num_channels)


class Bottleneck3d(nn.Module):
    """A 1x1 -> 3x3(stride) -> 1x1 Conv3d bottleneck block with a GroupNorm3d shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        mid_channels = out_channels // 4
        self.reduce = nn.Conv3d(in_channels, mid_channels, kernel_size=1)
        self.reduce_norm = group_norm_3d(mid_channels)
        self.conv = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.conv_norm = group_norm_3d(mid_channels)
        self.expand = nn.Conv3d(mid_channels, out_channels, kernel_size=1)
        self.expand_norm = group_norm_3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                group_norm_3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.relu(self.reduce_norm(self.reduce(value)))
        residual = self.relu(self.conv_norm(self.conv(residual)))
        residual = self.expand_norm(self.expand(residual))
        return self.relu(residual + self.shortcut(value))


def build_voxel_resnet_stages(
    obs_channel: int, n_out: int, stem: int = 32, max_stage: int = 4
) -> Tuple[List[nn.Module], int]:
    """A Conv3d stem (stride 2) followed by up to four bottleneck stages."""
    if not 1 <= max_stage <= 4:
        raise ValueError("max_stage must be between 1 and 4")
    layers: List[nn.Module] = [
        nn.Sequential(
            nn.Conv3d(obs_channel, stem, kernel_size=3, stride=2, padding=1),
            group_norm_3d(stem),
            nn.ReLU(inplace=True),
        )
    ]
    in_channels = stem
    for stage_index in range(max_stage):
        out_channels = max(1, int(n_out / _STAGE_WIDTH_DIVISORS[stage_index]))
        blocks = []
        for block_index in range(_STAGE_BLOCK_COUNTS[stage_index]):
            stride = 2 if (block_index == 0 and stage_index > 0) else 1
            blocks.append(Bottleneck3d(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        layers.append(nn.Sequential(*blocks))
    return layers, in_channels


class VoxelResNetBackbone(nn.Module):
    """A full 4-stage 3D ResNet trunk, pooled to a fixed-size feature vector."""

    equivariant = False
    needs_proprio = False

    def __init__(self, obs_channel: int = 4, n_out: int = 128, stem: int = 32, head_grid: int = 2) -> None:
        super().__init__()
        layers, channels = build_voxel_resnet_stages(obs_channel, n_out, stem=stem, max_stage=4)
        self.trunk = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(head_grid)
        self.projection = nn.Linear(channels * head_grid**3, n_out)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.trunk(voxels))
        return self.projection(features.flatten(1))
