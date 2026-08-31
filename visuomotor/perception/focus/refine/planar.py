"""Planar iterative cross-attention focus pooling.

Ported verbatim from ``seeker-dev``'s ``focus-pool`` branch
(``seeker/model/stage_pooled_resnet.py::FocusRefine``), except the query
conditioning is fixed to gripper-only (see :mod:`.query_composer`). Do not
confuse this with :class:`~visuomotor.perception.focus.refine.volumetric.FocusRefine3d`,
the voxel counterpart, which follows the same frequency-position,
gripper-query, and averaged-head context contract on a 3D feature grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
from torch import nn

from visuomotor.perception.focus.refine.position_encoding import (
    get_freq_position_embedding,
    get_normalized_grid_coordinates,
)
from visuomotor.perception.focus.refine.query_composer import QueryComposer


def _init_focus_refine_2d_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


@dataclass
class FocusRefine2dOutput:
    """Result of one :class:`FocusRefine2d` forward pass.

    ``ctx`` is ``[B, head_dim]`` -- the reference architecture averages the
    final context across heads, so ``ctx_dim == head_dim`` (not
    ``heads * head_dim``).
    """

    ctx: torch.Tensor
    pool_map: torch.Tensor
    keypoints: torch.Tensor


class FocusRefine2d(nn.Module):
    """Iterative cross-attention pooling with a gripper-conditioned learned query."""

    def __init__(
        self,
        *,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        query_cond: Tuple[str, ...] = ("gripper",),
        iters: int = 3,
        heads: int = 4,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if iters < 1:
            raise ValueError("iters must be positive")
        self.dim = head_dim * heads
        self.num_heads = int(heads)
        self.iters = int(iters)
        self.head_dim = int(head_dim)
        self.ctx_dim = self.head_dim

        self.query_builder = QueryComposer(dim=self.dim, num_heads=self.num_heads, query_cond=query_cond)
        self.pos_dim = head_dim
        self.register_buffer(
            "pos_enc", get_freq_position_embedding(grid_h, grid_w, self.pos_dim), persistent=False
        )
        self.register_buffer(
            "coord_tokens", get_normalized_grid_coordinates(grid_h, grid_w), persistent=False
        )

        self.query_norm = nn.LayerNorm(self.dim)
        self.key_norm = nn.LayerNorm(self.dim)
        self.ctx_norm = nn.LayerNorm(self.head_dim)

        self.to_k = nn.Linear(in_channels + self.pos_dim, self.dim)
        self.to_v = nn.Linear(in_channels + self.pos_dim, self.dim)
        self.to_query = nn.Linear(self.dim, self.dim)
        self.ctx_to_film = nn.Sequential(
            nn.Linear(self.head_dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, 2 * self.dim),
            nn.Tanh(),
        )
        self.apply(_init_focus_refine_2d_module)

    def forward(
        self, feat: torch.Tensor, composer_in: Mapping[str, torch.Tensor], prop_noise: float = 0.0
    ) -> FocusRefine2dOutput:
        batch_size, _, height, width = feat.shape
        query = self.query_builder(composer_in, batch_size, prop_noise)[:, None]
        num_tokens = height * width
        num_heads, head_dim = self.num_heads, self.head_dim

        feat_tokens = feat.flatten(2).transpose(1, 2)
        pos_tokens = self.pos_enc.to(device=feat.device, dtype=feat.dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        value_tokens = torch.cat([feat_tokens, pos_tokens], dim=-1)
        keys = self.key_norm(self.to_k(value_tokens))
        values = self.to_v(value_tokens)
        keys = keys.view(batch_size, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
        values = values.view(batch_size, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
        keys_t = keys.transpose(-2, -1)

        attention = None
        pool_ctx = None
        for step_idx in range(self.iters):
            projected_query = self.to_query(self.query_norm(query))
            projected_query = projected_query.view(
                batch_size, query.shape[1], num_heads, head_dim
            ).permute(0, 2, 1, 3)
            scores = torch.matmul(projected_query, keys_t) / math.sqrt(head_dim)
            attention = torch.softmax(scores, dim=-1)
            ctx_heads = torch.matmul(attention, values)
            pool_ctx = ctx_heads.mean(dim=(1, 2))

            if step_idx < self.iters - 1:
                norm_ctx = self.ctx_norm(pool_ctx).unsqueeze(1)
                gamma, beta = self.ctx_to_film(norm_ctx).chunk(2, dim=-1)
                query = (1.0 + gamma) * query + beta

        pool_map = attention.reshape(batch_size, num_heads * query.shape[1], height, width)
        coord_tokens = self.coord_tokens.to(device=feat.device, dtype=feat.dtype)
        keypoints = torch.matmul(pool_map.detach().flatten(2), coord_tokens)
        return FocusRefine2dOutput(ctx=pool_ctx, pool_map=pool_map, keypoints=keypoints.detach())
