"""Small RGB backbones used by :mod:`visuomotor.perception.encoder.voxel`'s eye-in-hand branch."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from visuomotor.perception.backbone.resnet.build import build_resnet18_backbone


class ResNet18Backbone(nn.Module):
    """A ResNet-18 trunk (GroupNorm or BatchNorm) followed by a linear projection."""

    equivariant = False

    def __init__(
        self,
        out_size: int = 128,
        weights: Optional[str] = "IMAGENET1K_V1",
        norm: str = "groupnorm",
    ) -> None:
        super().__init__()
        backbone = build_resnet18_backbone(pretrained_imagenet=weights is not None, norm=norm)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(512, out_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(image))
