"""Composed voxel, optional RGB, and selected-proprio observation encoder."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from visuomotor.config.schema import RandomCropSpec
from visuomotor.geometry import grid as Grid
from visuomotor.perception.backbone.resnet.rgb import ResNet18Backbone
from visuomotor.perception.backbone.resnet.voxel import VoxelResNetBackbone
from visuomotor.perception.backbone.voxel_simple import VoxelSimpleBackbone
from visuomotor.perception.common.augmentation import ResizeCropRandomizer
from visuomotor.perception.common.inputs import (
    validate_model_rgb,
    validate_model_voxel,
)
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.perception.common.voxel_crop import VoxelCropper
from visuomotor.perception.focus.refine.volumetric import FocusVoxelBackbone


def _build_voxel_backbone(architecture, channels, feature_dim, input_size, focus_pool):
    if architecture == "voxel_simple":
        return VoxelSimpleBackbone(obs_channel=channels, n_out=feature_dim)
    if architecture == "voxel_resnet3d":
        return VoxelResNetBackbone(obs_channel=channels, n_out=feature_dim)
    if architecture == "voxel_focus_pool3d":
        return FocusVoxelBackbone(
            obs_channel=channels,
            n_out=feature_dim,
            in_size=input_size,
            pool_stage=int(focus_pool.pop("pool_stage")),
            focus_pool=focus_pool,
        )
    raise ValueError(f"unknown voxel encoder architecture {architecture!r}")


class VoxelObservationEncoder(nn.Module):
    """Voxel branch plus zero or more ResNet-18 RGB branches and selected proprio."""

    def __init__(
        self,
        *,
        encoder_name: str = "voxel_resnet3d",
        voxel_key: str = "voxel",
        source_shape: Tuple[int, int, int, int] = (4, 64, 64, 64),
        crop_size: Optional[int] = None,
        voxel_architecture: str = "voxel_resnet3d",
        rgb_keys: Sequence[str] = (),
        rgb_architecture: str = "resnet18",
        proprio_fields: Sequence[str] = (),
        proprio_dims: Sequence[int] = (),
        feature_dim: int = 256,
        rgb_feature_dim: int = 256,
        rgb_pretrained_imagenet: bool = True,
        rgb_norm: str = "groupnorm",
        rgb_random_crop: Optional[RandomCropSpec] = None,
        coord_conv: bool = False,
        num_heads: int = 4,
        num_iterations: int = 3,
        pool_stage: int = 3,
        attention_prior=None,
        source_workspace_min=None,
        source_workspace_size=None,
    ) -> None:
        super().__init__()
        self.encoder_name = str(encoder_name)
        if len(tuple(proprio_fields)) != len(tuple(proprio_dims)):
            raise ValueError("proprio_fields and proprio_dims must have equal length")
        source_channels, *source_grid = source_shape
        if len(source_grid) != 3 or len(set(source_grid)) != 1:
            raise ValueError("source voxel shape must be cubic")
        self.voxel_key = str(voxel_key)
        self.rgb_keys = tuple(rgb_keys)
        if self.rgb_keys and rgb_architecture != "resnet18":
            raise ValueError(f"unknown voxel RGB branch architecture {rgb_architecture!r}")
        self.rgb_architecture = str(rgb_architecture)
        self.proprio_fields = tuple(proprio_fields)
        self.proprio_dims = tuple(int(dim) for dim in proprio_dims)
        self.n_hidden = int(feature_dim)
        self.voxel_feature_dim = self.n_hidden
        self.voxel_architecture = str(voxel_architecture)
        self.source_grid = tuple(int(size) for size in source_grid)
        encoder_input_size = int(crop_size) if crop_size is not None else int(source_grid[0])
        self.encoder_input_size = encoder_input_size
        self.cropper = VoxelCropper(
            crop_shape=None if crop_size is None else (encoder_input_size,) * 3
        )
        self.coord_conv = bool(coord_conv)
        if self.coord_conv:
            # At the source resolution, so the cropper slices coordinates and
            # content together: a cell's coordinate is its place in the source
            # grid, not in a window the training crop puts down at random.
            axes = [torch.linspace(-1, 1, int(size)) for size in source_grid]
            self.register_buffer(
                "coord_grid",
                torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0),
                persistent=False,
            )
        focus_cfg = {
            "iters": int(num_iterations),
            "heads": int(num_heads),
            "pool_stage": int(pool_stage),
        }
        if attention_prior is not None:
            focus_cfg["attention_prior"] = {
                "enabled": attention_prior.enabled,
                "weight": attention_prior.weight,
                "sigma_cells": attention_prior.sigma_cells,
                "bootstrap_steps": attention_prior.bootstrap_steps,
            }
        input_channels = int(source_channels) + (3 if self.coord_conv else 0)
        self.voxel_backbone = _build_voxel_backbone(
            voxel_architecture, input_channels, self.n_hidden, encoder_input_size, focus_cfg
        )
        self.rgb_backbones = nn.ModuleDict(
            {
                key: ResNet18Backbone(
                    int(rgb_feature_dim),
                    weights="IMAGENET1K_V1" if rgb_pretrained_imagenet else None,
                    norm=rgb_norm,
                )
                for key in self.rgb_keys
            }
        )
        self.rgb_augmentation = (
            None if rgb_random_crop is None else ResizeCropRandomizer(rgb_random_crop)
        )
        visual_dim = self.n_hidden + len(self.rgb_keys) * int(rgb_feature_dim)
        self.output_dim = visual_dim + sum(self.proprio_dims)
        self.source_geometry = Grid.SourceVoxelGeometry.optional(
            source_workspace_min, source_workspace_size, tuple(source_grid)
        )

    def get_runtime_config(self) -> dict:
        source = "×".join(str(size) for size in self.source_grid)
        if self.encoder_input_size == self.source_grid[0]:
            voxel_crop = f"Disabled ({source}, no margin)"
        else:
            voxel_crop = f"Enabled ({source}→{self.encoder_input_size}³ cells)"
        if not self.rgb_keys:
            rgb_crop = "n/a"
        elif self.rgb_augmentation is None:
            rgb_crop = "Disabled"
        else:
            spec = self.rgb_augmentation.spec
            rgb_crop = f"Enabled ({spec.input_res}→{spec.output_res} px)"
        prior = self.attention_prior
        return {
            "Voxel Encoder": {
                "Architecture": self.voxel_architecture,
                "RGB Keys": ", ".join(self.rgb_keys) if self.rgb_keys else "n/a",
                "Proprio": ", ".join(self.proprio_fields) if self.proprio_fields else "n/a",
                "Voxel Crop": voxel_crop,
                "RGB Crop": rgb_crop,
                # Coordinate channels ride along through the crop, so the count
                # the backbone sees is the honest thing to report.
                "Coord Conv": (
                    f"Enabled ({self.voxel_backbone_in_channels} input channels)"
                    if self.coord_conv
                    else "Disabled"
                ),
                "Attention Prior": (
                    "disabled" if prior is None else prior.runtime_config()
                ),
            }
        }

    @property
    def voxel_backbone_in_channels(self) -> int:
        for module in self.voxel_backbone.modules():
            if isinstance(module, nn.Conv3d):
                return int(module.in_channels)
        raise ValueError("voxel backbone has no Conv3d stage")

    @property
    def attention_prior(self):
        if isinstance(self.voxel_backbone, FocusVoxelBackbone):
            return self.voxel_backbone.attention_prior
        return None

    @property
    def supports_attention(self) -> bool:
        return isinstance(self.voxel_backbone, FocusVoxelBackbone)

    @staticmethod
    def _flatten_time(value: torch.Tensor, event_rank: int):
        if value.ndim == event_rank + 1:
            return value, None
        if value.ndim == event_rank + 2:
            batch, steps = value.shape[:2]
            return value.reshape(batch * steps, *value.shape[2:]), (batch, steps)
        raise ValueError(f"unexpected input shape {tuple(value.shape)}")

    def _prepare_voxel(self, value):
        validate_model_voxel(value)
        if self.coord_conv:
            grid = self.coord_grid.unsqueeze(0).expand(value.shape[0], -1, -1, -1, -1)
            value = torch.cat((value, grid.to(dtype=value.dtype)), dim=1)
        crop = self.cropper(value)
        return crop.voxels, crop.transform

    def _attention_target(self, world, transform, temporal_shape):
        if world is None:
            return None
        if self.source_geometry is None:
            raise ValueError("attention target requires source voxel geometry")
        position, valid = world["pos"], world["valid"]
        if temporal_shape is not None:
            position, valid = position.reshape(-1, 3), valid.reshape(-1)
        source = self.source_geometry.world_to_grid(position)
        return {"pos": transform.project_source_to_crop(source), "valid": valid}

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        focus_target: Optional[Mapping[str, torch.Tensor]] = None,
        attention_target_world: Optional[Mapping[str, torch.Tensor]] = None,
        collect_attention: bool = False,
        global_step: Optional[int] = None,
    ) -> EncoderOutput:
        voxels, temporal_shape = self._flatten_time(observations[self.voxel_key], 4)
        voxels, crop_transform = self._prepare_voxel(voxels)
        target = self._attention_target(
            attention_target_world if attention_target_world is not None else focus_target,
            crop_transform,
            temporal_shape,
        )
        want_attention = target is not None or (collect_attention and self.supports_attention)
        if getattr(self.voxel_backbone, "needs_proprio", False):
            gripper, gripper_shape = self._flatten_time(observations["gripper_qpos"], 1)
            if gripper_shape != temporal_shape:
                raise ValueError("gripper and voxel must share batch/time dimensions")
            if want_attention:
                voxel_feature, attention = self.voxel_backbone(voxels, gripper, return_attn=True)
            else:
                voxel_feature, attention = self.voxel_backbone(voxels, gripper), None
        else:
            if want_attention:
                raise ValueError(f"encoder {type(self.voxel_backbone).__name__} has no attention")
            voxel_feature, attention = self.voxel_backbone(voxels), None
        voxel_feature = voxel_feature.flatten(1)
        visual = [voxel_feature]
        streams = {"voxel": voxel_feature}
        for key, backbone in self.rgb_backbones.items():
            image, shape = self._flatten_time(observations[key], 3)
            if shape != temporal_shape:
                raise ValueError("RGB and voxel must share batch/time dimensions")
            validate_model_rgb(image)
            if self.rgb_augmentation is not None:
                image = self.rgb_augmentation(image)
            rgb_feature = backbone(image)
            visual.append(rgb_feature)
            streams[key] = rgb_feature
        parts = list(visual)
        for key in self.proprio_fields:
            value, shape = self._flatten_time(observations[key], 1)
            if shape != temporal_shape:
                raise ValueError("proprio and voxel must share batch/time dimensions")
            parts.append(value)
        features = torch.cat(parts, dim=-1)
        losses = {}
        if target is not None:
            prior = self.attention_prior
            if prior is None or not prior.enabled:
                raise ValueError("attention target given but attention prior is disabled")
            attention_prior_loss = prior.loss(
                attention, target, self.voxel_backbone.feature_geometry
            )
            weight = prior.active_weight(prior.is_bootstrapping(global_step))
            losses["attention_prior"] = weight * attention_prior_loss
        if temporal_shape is not None:
            features = features.reshape(*temporal_shape, self.output_dim)
            streams = {
                key: value.reshape(*temporal_shape, value.shape[-1]) for key, value in streams.items()
            }
            if attention is not None:
                attention = attention.reshape(*temporal_shape, *attention.shape[1:])
        return EncoderOutput(
            features=features,
            streams=streams,
            attention=attention,
            geometry=self.voxel_backbone.feature_geometry if self.supports_attention else None,
            prepared_inputs={self.voxel_key: observations[self.voxel_key]},
            attention_geometry=self.voxel_backbone.feature_geometry if self.supports_attention else None,
            voxel_crop_geometry=getattr(self, "source_geometry", None),
            voxel_crop_transform=crop_transform,
            auxiliary_losses=losses,
            metadata={"encoder": self.encoder_name},
        )
