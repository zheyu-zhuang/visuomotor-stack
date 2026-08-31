"""DP3 observation encoder over point clouds and selected proprioception."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch
from torch import nn

from visuomotor.perception.backbone.pointnet import PointNetBackbone
from visuomotor.perception.common.types import EncoderOutput


class PointCloudObservationEncoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str = "dp3",
        point_cloud_key: str = "point_cloud",
        source_shape: Tuple[int, int] = (1024, 6),
        proprio_fields: Sequence[str] = (),
        proprio_dims: Sequence[int] = (),
        feature_dim: int = 64,
        state_mlp_dims: Sequence[int] = (64, 64),
        use_color: bool = True,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
    ) -> None:
        super().__init__()
        if len(tuple(proprio_fields)) != len(tuple(proprio_dims)):
            raise ValueError("proprio_fields and proprio_dims must have equal length")
        if len(source_shape) != 2 or source_shape[1] not in (3, 6):
            raise ValueError("point clouds must have shape [points, 3 or 6]")
        state_mlp_dims = tuple(int(width) for width in state_mlp_dims)
        if not state_mlp_dims:
            raise ValueError("DP3 state MLP must have at least one output layer")
        self.encoder_name = str(encoder_name)
        self.point_cloud_key = str(point_cloud_key)
        self.source_shape = tuple(int(size) for size in source_shape)
        self.proprio_fields = tuple(proprio_fields)
        self.use_color = bool(use_color)
        point_dim = self.source_shape[1] if self.use_color else 3
        self.pointnet = PointNetBackbone(
            input_dim=point_dim,
            output_dim=int(feature_dim),
            use_layernorm=use_layernorm,
            final_norm=final_norm,
        )
        state_layers = []
        width = sum(int(dim) for dim in proprio_dims)
        for index, hidden in enumerate(state_mlp_dims):
            state_layers.append(nn.Linear(width, hidden))
            if index != len(state_mlp_dims) - 1:
                state_layers.append(nn.ReLU())
            width = hidden
        self.state_mlp = nn.Sequential(*state_layers)
        self.output_dim = int(feature_dim) + state_mlp_dims[-1]

    @staticmethod
    def _flatten_time(value: torch.Tensor, event_rank: int):
        if value.ndim == event_rank + 1:
            return value, None
        if value.ndim == event_rank + 2:
            batch, steps = value.shape[:2]
            return value.reshape(batch * steps, *value.shape[2:]), (batch, steps)
        raise ValueError(f"unexpected input shape {tuple(value.shape)}")

    def forward(self, observations: Mapping[str, torch.Tensor]) -> EncoderOutput:
        points, temporal_shape = self._flatten_time(
            observations[self.point_cloud_key], 2
        )
        if tuple(points.shape[1:]) != self.source_shape:
            raise ValueError(
                f"expected point cloud shape {self.source_shape}, got {tuple(points.shape[1:])}"
            )
        point_feature = self.pointnet(points if self.use_color else points[..., :3])
        state = []
        for key in self.proprio_fields:
            value, shape = self._flatten_time(observations[key], 1)
            if shape != temporal_shape:
                raise ValueError("proprio and point cloud must share batch/time dimensions")
            state.append(value)
        state_feature = self.state_mlp(torch.cat(state, dim=-1))
        features = torch.cat((point_feature, state_feature), dim=-1)
        streams = {"point_cloud": point_feature, "proprio": state_feature}
        if temporal_shape is not None:
            features = features.reshape(*temporal_shape, self.output_dim)
            streams = {
                key: value.reshape(*temporal_shape, value.shape[-1])
                for key, value in streams.items()
            }
        prepared = points
        if temporal_shape is not None:
            prepared = points.reshape(*temporal_shape, *points.shape[1:])
        return EncoderOutput(
            features=features,
            streams=streams,
            prepared_inputs={self.point_cloud_key: prepared},
        )
