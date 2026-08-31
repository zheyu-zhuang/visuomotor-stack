"""Planar-equivalent iterative attention pooling over a voxel feature grid."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import nn

from visuomotor.geometry import grid as Grid
from visuomotor.perception.backbone.resnet.voxel import build_voxel_resnet_stages
from visuomotor.perception.focus.refine.attention_prior import (
    FocusAttentionPrior,
    FocusPoolSpec,
)
from visuomotor.perception.focus.refine.position_encoding import (
    get_freq_position_embedding_3d,
)
from visuomotor.perception.focus.refine.query_composer import QueryComposer


class FocusRefine3d(nn.Module):
    """Iterative cross-attention pooling over a voxel feature grid."""

    def __init__(
        self,
        *,
        in_channels: int,
        grid_d: int,
        grid_h: int,
        grid_w: int,
        iters: int = 3,
        heads: int = 4,
        head_dim: int = 128,
        query_cond: Tuple[str, ...] = ("gripper",),
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if iters < 1:
            raise ValueError("iters must be positive")
        self.num_heads = int(heads)
        self.head_dim = int(head_dim)
        self.dim = self.num_heads * self.head_dim
        self.iters = int(iters)
        self.ctx_dim = self.head_dim

        self.query_builder = QueryComposer(
            dim=self.dim, num_heads=self.num_heads, query_cond=query_cond
        )
        self.pos_dim = self.head_dim
        self.register_buffer(
            "pos_enc",
            get_freq_position_embedding_3d(
                int(grid_d), int(grid_h), int(grid_w), self.pos_dim
            ),
            persistent=False,
        )
        self.grid = (int(grid_d), int(grid_h), int(grid_w))
        self.to_key = nn.Linear(in_channels + self.pos_dim, self.dim)
        self.to_value = nn.Linear(in_channels + self.pos_dim, self.dim)
        self.to_query = nn.Linear(self.dim, self.dim)
        self.query_norm = nn.LayerNorm(self.dim)
        self.key_norm = nn.LayerNorm(self.dim)
        self.context_norm = nn.LayerNorm(self.head_dim)
        self.context_film = nn.Sequential(
            nn.Linear(self.head_dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, 2 * self.dim),
            nn.Tanh(),
        )

    def forward(
        self,
        features: torch.Tensor,
        composer_in: Mapping[str, torch.Tensor],
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if features.ndim != 5:
            raise ValueError(f"features must be [B,C,D,H,W], got {tuple(features.shape)}")
        batch_size = features.shape[0]
        spatial_shape = tuple(features.shape[2:])
        if spatial_shape != self.grid:
            raise ValueError(
                f"features must use configured grid {self.grid}, got {spatial_shape}"
            )
        query = self.query_builder(composer_in, batch_size)[:, None]

        feature_tokens = features.flatten(2).transpose(1, 2)
        position_tokens = self.pos_enc.to(
            device=features.device, dtype=features.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat((feature_tokens, position_tokens), dim=-1)
        keys = self.key_norm(self.to_key(tokens))
        values = self.to_value(tokens)
        keys = keys.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        values = values.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        keys_t = keys.transpose(-2, -1)

        attention: Optional[torch.Tensor] = None
        context: Optional[torch.Tensor] = None
        for step in range(self.iters):
            projected = self.to_query(self.query_norm(query)).view(
                batch_size, 1, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)
            attention = torch.softmax(
                projected @ keys_t / math.sqrt(self.head_dim), dim=-1
            )
            context_heads = attention @ values
            context = context_heads.mean(dim=(1, 2))
            if step + 1 < self.iters:
                normalized = self.context_norm(context).unsqueeze(1)
                gamma, beta = self.context_film(normalized).chunk(2, -1)
                query = (1 + gamma) * query + beta

        if not return_attn:
            return context
        attention_map = attention.reshape(
            batch_size, self.num_heads, *spatial_shape
        )
        return context, attention_map


class FocusVoxelBackbone(nn.Module):
    """A stage-truncated 3D ResNet trunk pooled by :class:`FocusRefine3d`."""

    equivariant = False

    def __init__(
        self,
        obs_channel: int = 4,
        n_out: int = 128,
        in_size: int = 58,
        stem: int = 32,
        pool_stage: int = 3,
        focus_pool: Optional[Mapping] = None,
        query_cond: Tuple[str, ...] = ("gripper",),
    ) -> None:
        super().__init__()
        self.needs_proprio = True
        pool_spec = FocusPoolSpec.parse(focus_pool)
        layers, channels = build_voxel_resnet_stages(obs_channel, n_out, stem=stem, max_stage=pool_stage)
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, obs_channel, in_size, in_size, in_size))
        self.grid = tuple(int(size) for size in probe.shape[2:])
        self.total_stride = 2**pool_stage
        geometry = Grid.FeatureGridGeometry.from_stride((in_size,) * 3, self.grid, self.total_stride)
        self.register_buffer("_feature_centers", geometry.centers, persistent=False)
        self.register_buffer("_feature_spacing", geometry.spacing, persistent=False)

        self.focus = FocusRefine3d(
            in_channels=channels,
            grid_d=self.grid[0],
            grid_h=self.grid[1],
            grid_w=self.grid[2],
            iters=pool_spec.iters,
            heads=pool_spec.heads,
            head_dim=n_out,
            query_cond=query_cond,
        )
        self.attention_prior = FocusAttentionPrior(pool_spec.attention_prior)

    @property
    def feature_geometry(self) -> Grid.FeatureGridGeometry:
        return Grid.FeatureGridGeometry(centers=self._feature_centers, spacing=self._feature_spacing)

    def forward(
        self,
        x: torch.Tensor,
        gripper: torch.Tensor,
        eef_pos: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if tuple(gripper.shape) != (x.shape[0], 2):
            raise ValueError(
                f"gripper must have shape {(x.shape[0], 2)}, got {tuple(gripper.shape)}"
            )
        opening = (gripper[:, 0] - gripper[:, 1]).abs().unsqueeze(-1)
        composer_in = {"gripper_opening": opening}
        if "eef" in self.focus.query_builder.query_cond:
            if eef_pos is None or tuple(eef_pos.shape) != (x.shape[0], 3):
                shape = None if eef_pos is None else tuple(eef_pos.shape)
                raise ValueError(
                    f"eef_pos must have shape {(x.shape[0], 3)}, got {shape}"
                )
            composer_in["eef_pos"] = eef_pos
        features = self.conv(x)
        return self.focus(features, composer_in, return_attn=return_attn)
