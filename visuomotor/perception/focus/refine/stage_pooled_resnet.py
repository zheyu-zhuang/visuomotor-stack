"""A ResNet-18 truncated at a residual stage, pooled by :class:`FocusRefine2d`.

Ported from ``seeker-dev``'s ``focus-pool`` branch
(``seeker/model/stage_pooled_resnet.py::StagePooledResNet``).
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import torch
from torch import nn

from visuomotor.perception.backbone.resnet.build import (
    build_resnet18_stages,
    probe_resnet18_stage_shapes,
    run_resnet18_until,
)
from visuomotor.perception.focus.refine.planar import (
    FocusRefine2d,
    FocusRefine2dOutput,
)


class StagePooledResNet2d(nn.Module):
    """Truncate a ResNet-18 at ``pooling_stage`` and pool the result with :class:`FocusRefine2d`."""

    def __init__(
        self,
        *,
        input_res: int,
        feat_dim: int,
        input_channels: int = 3,
        pretrained_imagenet: bool = True,
        norm: str = "groupnorm",
        pooling_stage: str = "l2",
        query_cond: Tuple[str, ...] = ("gripper",),
        iters: int = 3,
        heads: int = 4,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_res = int(input_res)
        self.feat_dim = int(feat_dim)
        self.pooling_stage = str(pooling_stage)
        self.backbone = build_resnet18_stages(pretrained_imagenet=pretrained_imagenet, norm=norm)
        shapes = probe_resnet18_stage_shapes(
            self.backbone,
            input_res=self.input_res,
            stage=self.pooling_stage,
            input_channels=input_channels,
        )
        self.stage_grid = (shapes.stage_grid_h, shapes.stage_grid_w)
        self.pool = FocusRefine2d(
            in_channels=shapes.stage_channels,
            grid_h=shapes.stage_grid_h,
            grid_w=shapes.stage_grid_w,
            query_cond=query_cond,
            iters=iters,
            heads=heads,
            head_dim=head_dim,
        )
        self.out_proj = nn.Linear(self.pool.ctx_dim, self.feat_dim)

    def _forward_backbone(self, image: torch.Tensor) -> torch.Tensor:
        return run_resnet18_until(self.backbone, image, self.pooling_stage)

    def forward(
        self,
        image: torch.Tensor,
        *,
        composer_in: Mapping[str, torch.Tensor],
        prop_noise: float = 0.0,
        return_pool_map: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        feat = self._forward_backbone(image)
        pool_out: FocusRefine2dOutput = self.pool(feat, composer_in, prop_noise)
        out = self.out_proj(pool_out.ctx)
        if not return_pool_map:
            return out
        return out, pool_out.pool_map, pool_out.keypoints
