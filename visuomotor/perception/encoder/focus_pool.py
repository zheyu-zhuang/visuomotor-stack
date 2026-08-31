"""Policy-facing planar and voxel encoders built from reference focus-pool architectures.

The planar (``spatial_rank == 2``) path ports ``seeker-dev``'s ``focus-pool``
branch (:class:`~visuomotor.perception.focus.refine.stage_pooled_resnet.StagePooledResNet2d`
+ :class:`~visuomotor.perception.focus.refine.planar.FocusRefine2d`);
the volumetric (``spatial_rank == 3``) path uses EquiDiff's ``local-policy``
port (:class:`~visuomotor.perception.focus.refine.volumetric.FocusVoxelBackbone`
+ :class:`~visuomotor.perception.focus.refine.volumetric.FocusRefine3d`). The two
reference architectures have genuinely different query contracts -- 2D is
conditioned on gripper opening with a fixed positional encoding; 3D normally
uses the full 2-dim gripper state and can also use its learned query alone. The
3D path has no positional encoding of its own, so this class dispatches to one
or the other rather than sharing a single pooling module across ranks.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import torch
from torch import nn

from visuomotor.config.schema import AttentionPriorSpec, RandomCropSpec
from visuomotor.geometry.grid import FeatureGridGeometry, SourceVoxelGeometry
from visuomotor.geometry.projection import world_xyz_to_pixel_row_col
from visuomotor.perception.common.augmentation import ResizeCropRandomizer
from visuomotor.perception.common.inputs import (
    validate_model_rgb,
    validate_model_voxel,
)
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.perception.focus.refine.attention_prior import FocusAttentionPrior
from visuomotor.perception.focus.refine.stage_pooled_resnet import StagePooledResNet2d
from visuomotor.perception.focus.refine.volumetric import FocusVoxelBackbone

_PROP_NOISE = 0.01  # seeker-dev's default train-time query proprio noise std


class FocusRefineEncoder(nn.Module):
    """Encode RGB views or a voxel grid with the matching reference focus-pool architecture."""

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        spatial_rank: int = 2,
        input_channels: int = 3,
        input_res: int = 256,
        rgb_keys: Sequence[str] = ("rgb_external",),
        voxel_key: str = "voxel",
        gripper_key: str = "gripper_opening",
        proprio_fields: Sequence[str] = (),
        proprio_dims: Sequence[int] = (),
        random_crop: Optional[RandomCropSpec] = None,
        num_heads: int = 4,
        num_iterations: int = 3,
        pool_stage: int = 2,
        pretrained_imagenet: bool = True,
        norm: str = "groupnorm",
        attention_prior: Optional[AttentionPriorSpec] = None,
        attention_prior_view: Optional[str] = None,
        attention_prior_raw_res: Optional[int] = None,
        workspace_min: Optional[Sequence[float]] = None,
        ws_size: Optional[float] = None,
    ) -> None:
        super().__init__()
        if spatial_rank not in (2, 3):
            raise ValueError("spatial_rank must be 2 or 3")
        if random_crop is not None and spatial_rank != 2:
            raise ValueError("random_crop is only meaningful for the planar (spatial_rank=2) encoder")
        if spatial_rank == 3 and attention_prior is not None and attention_prior.enabled:
            if workspace_min is None or ws_size is None:
                raise ValueError("attention_prior requires workspace_min/ws_size for the voxel encoder")
        if spatial_rank == 2 and attention_prior is not None and attention_prior.enabled:
            if attention_prior_view is None or attention_prior_raw_res is None:
                raise ValueError(
                    "attention_prior requires attention_prior_view/attention_prior_raw_res"
                )
            if attention_prior_view not in rgb_keys:
                raise ValueError(f"attention_prior_view {attention_prior_view!r} must be one of {rgb_keys}")
        self.feature_dim = int(feature_dim)
        self.spatial_rank = int(spatial_rank)
        self.rgb_keys = tuple(rgb_keys)
        self.voxel_key = str(voxel_key)
        self.gripper_key = str(gripper_key)
        self.proprio_fields = tuple(proprio_fields)
        if len(self.proprio_fields) != len(tuple(proprio_dims)):
            raise ValueError("proprio_fields and proprio_dims must have equal length")
        self.pool_stage = int(pool_stage)
        self.random_crop = random_crop
        self.rgb_augmentation = (
            None
            if random_crop is None
            else ResizeCropRandomizer(random_crop, channels=int(input_channels))
        )

        if self.spatial_rank == 2:
            branch_res = random_crop.output_res if random_crop is not None else int(input_res)
            self.rgb_backbones = nn.ModuleDict(
                {
                    key: StagePooledResNet2d(
                        input_res=branch_res,
                        feat_dim=self.feature_dim,
                        input_channels=int(input_channels),
                        pretrained_imagenet=pretrained_imagenet,
                        norm=norm,
                        pooling_stage=f"l{int(pool_stage)}",
                        iters=num_iterations,
                        heads=num_heads,
                    )
                    for key in self.rgb_keys
                }
            )
            self.voxel_backbone = None
            self.attention_prior_view = attention_prior_view
            self.attention_prior_raw_res = attention_prior_raw_res
            self.attention_prior = None if attention_prior is None else FocusAttentionPrior(
                {
                    "enabled": attention_prior.enabled,
                    "weight": attention_prior.weight,
                    "sigma_cells": attention_prior.sigma_cells,
                    "bootstrap_steps": attention_prior.bootstrap_steps,
                }
            )
            geometry_view = attention_prior_view or self.rgb_keys[0]
            grid_h, grid_w = self.rgb_backbones[geometry_view].stage_grid
            stride = branch_res // grid_h
            geometry = FeatureGridGeometry.from_stride(
                (branch_res, branch_res), (grid_h, grid_w), stride
            )
            self.register_buffer(
                "_attention_prior_centers", geometry.centers, persistent=False
            )
            self.register_buffer(
                "_attention_prior_spacing", geometry.spacing, persistent=False
            )
        else:
            self.rgb_backbones = None
            self.source_geometry = SourceVoxelGeometry.optional(
                workspace_min, ws_size, (int(input_res),) * 3
            )
            self.voxel_backbone = FocusVoxelBackbone(
                obs_channel=int(input_channels),
                n_out=self.feature_dim,
                in_size=int(input_res),
                pool_stage=int(pool_stage),
                focus_pool={
                    "iters": int(num_iterations),
                    "heads": int(num_heads),
                    **(
                        {}
                        if attention_prior is None
                        else {
                            "attention_prior": {
                                "enabled": attention_prior.enabled,
                                "weight": attention_prior.weight,
                                "sigma_cells": attention_prior.sigma_cells,
                                "bootstrap_steps": attention_prior.bootstrap_steps,
                            }
                        }
                    ),
                },
            )

        branch_count = len(self.rgb_keys) if spatial_rank == 2 else 1
        self.output_dim = branch_count * self.feature_dim + sum(int(dim) for dim in proprio_dims)

    @property
    def attention_prior_geometry(self) -> FeatureGridGeometry:
        return FeatureGridGeometry(centers=self._attention_prior_centers, spacing=self._attention_prior_spacing)

    @property
    def observation_contract(self) -> dict:
        return {
            "spatial_rank": self.spatial_rank,
            "rgb": list(self.rgb_keys) if self.spatial_rank == 2 else [],
            "voxel": self.voxel_key if self.spatial_rank == 3 else None,
            "query": self.gripper_key,
            "proprio": list(self.proprio_fields),
            "uses_task_embedding": False,
            "uses_robot_id": False,
        }

    def initialize_internal(
        self,
        dataset_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize runtime-owned state (none is required for this encoder)."""

    def get_runtime_config(self) -> dict:
        if self.random_crop is None:
            crop = "Disabled"
        elif self.random_crop.enabled:
            crop = f"Enabled ({self.random_crop.input_res}→{self.random_crop.output_res} px)"
        else:
            crop = f"Disabled ({self.random_crop.output_res}px, no margin)"
        attention_prior = self.attention_prior if self.spatial_rank == 2 else self.voxel_backbone.attention_prior
        attention_prior_status = "disabled" if attention_prior is None else attention_prior.runtime_config()
        return {
            "FocusRefine Encoder": {
                "Spatial Rank": f"{self.spatial_rank}D",
                "RGB Keys": ", ".join(self.rgb_keys) if self.rgb_keys else "n/a",
                "Pool Stage": self.pool_stage,
                "Random Crop": crop,
                "Attention Prior": attention_prior_status,
            }
        }

    @staticmethod
    def _flatten_time(value: torch.Tensor, spatial_rank: int):
        expected_without_time = spatial_rank + 2
        if value.ndim == expected_without_time:
            return value, None
        if value.ndim == expected_without_time + 1:
            batch, steps = value.shape[:2]
            return value.reshape(batch * steps, *value.shape[2:]), (batch, steps)
        raise ValueError(f"unexpected spatial input shape: {tuple(value.shape)}")

    def _prepare_rgb_branch(self, value: torch.Tensor) -> torch.Tensor:
        """Contract check, then this modality's own crop augmentation."""
        validate_model_rgb(value)
        if self.rgb_augmentation is None:
            return value
        return self.rgb_augmentation(value)

    def _gripper(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.gripper_key not in observations:
            raise KeyError(f"missing focus-pool query field: {self.gripper_key}")
        return observations[self.gripper_key]

    def _gripper_opening(self, observations: Mapping[str, torch.Tensor], temporal_shape) -> torch.Tensor:
        """A scalar gripper-opening query, for the 2D reference architecture's ``QueryComposer``."""
        value = self._gripper(observations)
        if value.shape[-1:] == (2,):
            value = (value[..., 0] - value[..., 1]).abs().unsqueeze(-1)
        elif value.ndim == 1 or value.shape[-1:] != (1,):
            value = value.unsqueeze(-1) if value.ndim == 1 else value
        if temporal_shape is not None:
            value = value.reshape(-1, 1)
        return value

    def _gripper_state(self, observations: Mapping[str, torch.Tensor], temporal_shape) -> torch.Tensor:
        """The raw 2-dim gripper state, for the 3D reference architecture's FiLM query."""
        value = self._gripper(observations)
        if value.shape[-1:] != (2,):
            raise ValueError(f"voxel focus-pool requires a 2-dim gripper state, got {tuple(value.shape)}")
        if temporal_shape is not None:
            value = value.reshape(-1, 2)
        return value

    @staticmethod
    def _flatten_batch_time(value: torch.Tensor, temporal_shape) -> torch.Tensor:
        if temporal_shape is None:
            return value
        return value.reshape(-1, *value.shape[2:])

    def _attention_prior_loss(
        self,
        attention_by_key: Mapping[str, torch.Tensor],
        oracle_info: Optional[Mapping[str, torch.Tensor]],
        focus_target: Optional[Mapping[str, torch.Tensor]],
        temporal_shape,
        global_step: Optional[int],
    ) -> Optional[torch.Tensor]:
        if self.attention_prior is None or not self.attention_prior.enabled or focus_target is None:
            return None
        if oracle_info is None:
            raise ValueError("attention_prior is enabled but no oracle_info was given")
        view = self.attention_prior_view.removeprefix("rgb_")
        camera_key = f"camera_matrix_{view}"
        if camera_key not in oracle_info:
            raise KeyError(f"oracle_info is missing {camera_key!r}")
        camera_matrix = self._flatten_batch_time(oracle_info[camera_key], temporal_shape)
        pos = self._flatten_batch_time(focus_target["pos"], temporal_shape)
        valid = self._flatten_batch_time(focus_target["valid"], temporal_shape)

        row_col = world_xyz_to_pixel_row_col(
            pos.to(dtype=camera_matrix.dtype), camera_matrix, self.attention_prior_raw_res
        )
        # This coarse prior deliberately ignores the sampled crop offset.
        target_pos = 2 * (row_col / self.attention_prior_raw_res) - 1
        valid = valid & torch.isfinite(target_pos).all(dim=-1)
        target_pos = torch.nan_to_num(target_pos, nan=0.0)

        attention = attention_by_key[self.attention_prior_view]
        loss = self.attention_prior.loss(
            attention, {"pos": target_pos, "valid": valid}, self.attention_prior_geometry
        )
        bootstrapping = self.attention_prior.is_bootstrapping(global_step)
        return self.attention_prior.active_weight(bootstrapping) * loss

    def _attention_prior_loss_voxel(
        self,
        attention: torch.Tensor,
        focus_target: Optional[Mapping[str, torch.Tensor]],
        temporal_shape,
        global_step: Optional[int],
    ) -> Optional[torch.Tensor]:
        prior = self.voxel_backbone.attention_prior
        if not prior.enabled or focus_target is None:
            return None
        if self.source_geometry is None:
            raise ValueError("attention_prior is enabled but no source voxel geometry was configured")
        pos = self._flatten_batch_time(focus_target["pos"], temporal_shape)
        valid = self._flatten_batch_time(focus_target["valid"], temporal_shape)
        target_pos = self.source_geometry.world_to_grid(pos.to(dtype=attention.dtype))
        loss = prior.loss(attention, {"pos": target_pos, "valid": valid}, self.voxel_backbone.feature_geometry)
        bootstrapping = prior.is_bootstrapping(global_step)
        return prior.active_weight(bootstrapping) * loss

    def _forward_rgb(
        self,
        observations: Union[Mapping[str, torch.Tensor], torch.Tensor],
        gripper_opening: Optional[torch.Tensor],
        oracle_info: Optional[Mapping[str, torch.Tensor]] = None,
        focus_target: Optional[Mapping[str, torch.Tensor]] = None,
        global_step: Optional[int] = None,
    ) -> EncoderOutput:
        if torch.is_tensor(observations):
            if gripper_opening is None:
                raise ValueError("gripper_opening is required for tensor inputs")
            values = {self.rgb_keys[0]: observations}
        else:
            missing = [key for key in self.rgb_keys if key not in observations]
            if missing:
                raise KeyError(f"missing observation fields: {missing}")
            values = {key: observations[key] for key in self.rgb_keys}

        contexts = []
        attention_by_key = {}
        prepared_by_key = {}
        temporal_shape = None
        for key, value in values.items():
            value, this_temporal_shape = self._flatten_time(value, spatial_rank=2)
            if temporal_shape is None:
                temporal_shape = this_temporal_shape
            elif temporal_shape != this_temporal_shape:
                raise ValueError("all FocusRefine branches must share batch/time dimensions")
            value = self._prepare_rgb_branch(value)
            prepared_by_key[key] = value

            if gripper_opening is None:
                query = self._gripper_opening(observations, temporal_shape)
            else:
                query = gripper_opening.reshape(-1, 1) if temporal_shape is not None else gripper_opening
            composer_in = {"gripper_opening": query.to(device=value.device, dtype=value.dtype)}

            out, pool_map, _keypoints = self.rgb_backbones[key](
                value, composer_in=composer_in, prop_noise=_PROP_NOISE, return_pool_map=True
            )
            contexts.append(out)
            attention_by_key[key] = pool_map

        weighted_prior = self._attention_prior_loss(
            attention_by_key, oracle_info, focus_target, temporal_shape, global_step
        )

        if not torch.is_tensor(observations):
            for key in self.proprio_fields:
                if key not in observations:
                    raise KeyError(f"missing selected proprio field: {key}")
                contexts.append(self._flatten_batch_time(observations[key], temporal_shape))
        visual_features = torch.cat(contexts[: len(self.rgb_keys)], dim=-1)
        features = torch.cat(contexts, dim=-1)
        attentions = list(attention_by_key.values())
        if temporal_shape is not None:
            features = features.reshape(*temporal_shape, self.output_dim)
            visual_features = visual_features.reshape(*temporal_shape, visual_features.shape[-1])
            attentions = [value.reshape(*temporal_shape, *value.shape[1:]) for value in attentions]
            prepared_by_key = {
                key: value.reshape(*temporal_shape, *value.shape[1:])
                for key, value in prepared_by_key.items()
            }
        attention = attentions[0] if len(attentions) == 1 else torch.stack(attentions, dim=2)
        return EncoderOutput(
            features=features,
            streams={"rgb": visual_features},
            attention=attention,
            prepared_inputs=prepared_by_key,
            attention_geometry=self.attention_prior_geometry,
            auxiliary_losses=(
                {}
                if weighted_prior is None
                else {"attention_prior": weighted_prior}
            ),
            metadata={"stem": "focus_refine", **self.observation_contract},
        )

    def _forward_voxel(
        self,
        observations: Union[Mapping[str, torch.Tensor], torch.Tensor],
        gripper_opening: Optional[torch.Tensor],
        focus_target: Optional[Mapping[str, torch.Tensor]] = None,
        global_step: Optional[int] = None,
    ) -> EncoderOutput:
        if torch.is_tensor(observations):
            if gripper_opening is None:
                raise ValueError("gripper_opening is required for tensor inputs")
            value = observations
        else:
            if self.voxel_key not in observations:
                raise KeyError(f"missing observation fields: [{self.voxel_key}]")
            value = observations[self.voxel_key]

        value, temporal_shape = self._flatten_time(value, spatial_rank=3)
        validate_model_voxel(value)

        if gripper_opening is None:
            gripper = self._gripper_state(observations, temporal_shape)
        else:
            gripper = gripper_opening.reshape(-1, 2) if temporal_shape is not None else gripper_opening
        gripper = gripper.to(device=value.device, dtype=value.dtype)

        out, attention = self.voxel_backbone(value, gripper, return_attn=True)
        weighted_prior = self._attention_prior_loss_voxel(
            attention, focus_target, temporal_shape, global_step
        )
        parts = [out]
        for key in self.proprio_fields:
            if key not in observations:
                raise KeyError(f"missing selected proprio field: {key}")
            parts.append(self._flatten_batch_time(observations[key], temporal_shape))
        context = torch.cat(parts, dim=-1)
        if temporal_shape is not None:
            context = context.reshape(*temporal_shape, self.output_dim)
            attention = attention.reshape(*temporal_shape, *attention.shape[1:])
        return EncoderOutput(
            features=context,
            streams={"voxel": out},
            attention=attention,
            prepared_inputs={self.voxel_key: value.reshape(*temporal_shape, *value.shape[1:]) if temporal_shape is not None else value},
            attention_geometry=self.voxel_backbone.feature_geometry,
            voxel_crop_geometry=self.source_geometry,
            auxiliary_losses=(
                {}
                if weighted_prior is None
                else {"attention_prior": weighted_prior}
            ),
            metadata={"stem": "focus_refine", **self.observation_contract},
        )

    def forward(
        self,
        observations: Union[Mapping[str, torch.Tensor], torch.Tensor],
        gripper_opening: Optional[torch.Tensor] = None,
        oracle_info: Optional[Mapping[str, torch.Tensor]] = None,
        focus_target: Optional[Mapping[str, torch.Tensor]] = None,
        global_step: Optional[int] = None,
    ) -> EncoderOutput:
        if self.spatial_rank == 2:
            return self._forward_rgb(
                observations,
                gripper_opening,
                oracle_info=oracle_info,
                focus_target=focus_target,
                global_step=global_step,
            )
        return self._forward_voxel(
            observations, gripper_opening, focus_target=focus_target, global_step=global_step
        )
