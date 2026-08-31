"""The simple voxel backbone."""

from __future__ import annotations

import torch
from torch import nn

from visuomotor.perception.backbone.resnet.voxel import group_norm_3d


def _conv3d_block(in_channels: int, out_channels: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False),
        group_norm_3d(out_channels),
        nn.ReLU(inplace=True),
    )


def _source_conv3d_block(
    in_channels: int, out_channels: int, padding: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding),
        nn.ReLU(inplace=True),
    )


class VoxelSimpleBackbone(nn.Module):
    """The source EquiDiff voxel CNN."""

    equivariant = False
    needs_proprio = False

    def __init__(self, obs_channel: int = 4, n_out: int = 256) -> None:
        super().__init__()
        widths = (n_out // 16, n_out // 8, n_out // 4, n_out // 2, n_out)
        self.stage1 = nn.Sequential(
            _source_conv3d_block(obs_channel, widths[0], padding=0), nn.MaxPool3d(2)
        )
        self.stage2 = nn.Sequential(
            _source_conv3d_block(widths[0], widths[1], padding=1),
            _source_conv3d_block(widths[1], widths[1], padding=1),
            nn.MaxPool3d(2),
        )
        self.stage3 = nn.Sequential(
            _source_conv3d_block(widths[1], widths[2], padding=0),
            _source_conv3d_block(widths[2], widths[2], padding=1),
            nn.MaxPool3d(2),
        )
        self.stage4 = nn.Sequential(
            _source_conv3d_block(widths[2], widths[3], padding=1),
            _source_conv3d_block(widths[3], widths[3], padding=1),
            nn.MaxPool3d(2),
        )
        self.stage5 = _source_conv3d_block(widths[3], widths[4], padding=0)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        value = self.stage1(voxels)
        value = self.stage2(value)
        value = self.stage3(value)
        value = self.stage4(value)
        return self.stage5(value)


class VoxelSimpleLocalBackbone(nn.Module):
    """A three-stage voxel-simple encoder for a 32-cell local volume."""

    equivariant = False
    needs_proprio = False

    def __init__(self, obs_channel: int = 4, n_out: int = 128) -> None:
        super().__init__()
        widths = (n_out // 16, n_out // 8, n_out // 4)
        self.stage1 = nn.Sequential(
            _conv3d_block(obs_channel, widths[0], padding=0), nn.MaxPool3d(2)
        )
        self.stage2 = nn.Sequential(
            _conv3d_block(widths[0], widths[1], padding=1),
            _conv3d_block(widths[1], widths[1], padding=1),
            nn.MaxPool3d(2),
        )
        self.stage3 = nn.Sequential(
            _conv3d_block(widths[1], widths[2], padding=0),
            _conv3d_block(widths[2], widths[2], padding=1),
            nn.MaxPool3d(2),
        )
        self.contract = nn.Sequential(
            nn.Conv3d(widths[2], n_out, kernel_size=2, bias=False),
            group_norm_3d(n_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        value = self.stage1(voxels)
        value = self.stage2(value)
        return self.contract(self.stage3(value))
