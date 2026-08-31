"""Composed ResNet-18 branches for canonical RGB inputs."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from visuomotor.config.schema import RandomCropSpec
from visuomotor.perception.backbone.resnet.rgb import ResNet18Backbone
from visuomotor.perception.common.augmentation import ResizeCropRandomizer
from visuomotor.perception.common.inputs import validate_model_rgb
from visuomotor.perception.common.types import EncoderOutput


class MultiViewRgbEncoder(nn.Module):
    """One ResNet-18 per selected view, fused with exactly the selected proprio fields."""

    def __init__(
        self,
        *,
        encoder_name: str,
        rgb_keys: Sequence[str],
        proprio_fields: Sequence[str],
        proprio_dims: Sequence[int],
        feature_dim: int,
        random_crop: RandomCropSpec,
        pretrained_imagenet: bool = True,
        norm: str = "groupnorm",
    ) -> None:
        super().__init__()
        self.encoder_name = str(encoder_name)
        self.rgb_keys = tuple(rgb_keys)
        self.proprio_fields = tuple(proprio_fields)
        self.proprio_dims = tuple(int(dim) for dim in proprio_dims)
        weights = "IMAGENET1K_V1" if pretrained_imagenet else None
        self.backbones = nn.ModuleDict(
            {key: ResNet18Backbone(int(feature_dim), weights=weights, norm=norm) for key in self.rgb_keys}
        )
        self.augmentation = ResizeCropRandomizer(random_crop)
        self.output_dim = len(self.rgb_keys) * int(feature_dim) + sum(self.proprio_dims)

    @staticmethod
    def _flatten_time(value: torch.Tensor, event_rank: int):
        if value.ndim == event_rank + 1:
            return value, None
        if value.ndim == event_rank + 2:
            batch, steps = value.shape[:2]
            return value.reshape(batch * steps, *value.shape[2:]), (batch, steps)
        raise ValueError(f"unexpected input shape {tuple(value.shape)}")

    def forward(self, observations: Mapping[str, torch.Tensor]) -> EncoderOutput:
        parts = []
        temporal_shape = None
        streams = {}
        prepared = {}
        for key in self.rgb_keys:
            value, shape = self._flatten_time(observations[key], 3)
            validate_model_rgb(value)
            if temporal_shape is None:
                temporal_shape = shape
            elif temporal_shape != shape:
                raise ValueError("all RGB views must share batch/time dimensions")
            value = self.augmentation(value)
            prepared[key] = value
            feature = self.backbones[key](value)
            parts.append(feature)
            streams[key] = feature
        for key in self.proprio_fields:
            value, shape = self._flatten_time(observations[key], 1)
            if temporal_shape != shape:
                raise ValueError("proprio and RGB must share batch/time dimensions")
            parts.append(value)
        features = torch.cat(parts, dim=-1)
        if temporal_shape is not None:
            features = features.reshape(*temporal_shape, self.output_dim)
            streams = {
                key: value.reshape(*temporal_shape, value.shape[-1]) for key, value in streams.items()
            }
            prepared = {
                key: value.reshape(*temporal_shape, *value.shape[1:])
                for key, value in prepared.items()
            }
        return EncoderOutput(
            features=features,
            streams=streams,
            prepared_inputs=prepared,
            metadata={"encoder": self.encoder_name},
        )
