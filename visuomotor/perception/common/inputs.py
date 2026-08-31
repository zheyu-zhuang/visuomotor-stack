"""Shared observation input preprocessing and validation utilities.

This module defines the processor used by model encoders to transform
already-normalized observation dictionaries into flattened `EncoderInputs`
tensors consumed by Seeker-based encoder variants. The policy boundary
normalizes the observation (``normalize_obs``); the only normalization left
here is the Seeker composer's, because a released Seeker owns the fitted
normalizer it was trained with and so defines its own model space.
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from visuomotor.data.core.images import resize_image
from visuomotor.data.core.normalization import normalize_obs, obs_robot_id

# Fields the Seeker query composer reads, normalized into Seeker's model space.
_COMPOSER_FIELDS = ("eef_pos", "eef_rot6d", "gripper_qpos")


def validate_model_rgb(image: torch.Tensor) -> None:
    """Validate ImageNet-normalized RGB at the encoder boundary."""
    if image.dtype != torch.float32 or image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("expected model RGB: float32, channel-first, three channels")


def validate_model_voxel(voxel: torch.Tensor) -> None:
    """Validate normalized voxel input at the encoder boundary."""
    if voxel.dtype != torch.float32 or voxel.ndim < 4 or voxel.shape[-4] != 4:
        raise ValueError("expected model voxel: float32, channel-first, [occupancy,R,G,B]")


@dataclass
class EncoderInputs:
    external: torch.Tensor  # [B*T, 3, vit_in, vit_in]
    wrist: Optional[torch.Tensor]  # [B*T, 3, vit_in, vit_in] or None
    proprio: torch.Tensor  # [B*T, 11]
    robot_id: torch.Tensor  # [B*T, num_robots]
    composer_in: dict  # dict of tensors [B*T, ...]
    task_embedding: torch.Tensor  # [B*T, D]
    T: int


class ObsInputProcessor(nn.Module):
    """Common observation input path for policy-facing encoders.

    Responsibilities:
    - flatten temporal observations from [B, T, ...] to [B*T, ...];
    - normalize the Seeker composer's fields into Seeker's own model space;
    - build validated `composer_in` tensors used by Seeker components.
    """

    def __init__(
        self,
        *,
        num_robots: int,
        input_res: Optional[int] = None,
        enable_wrist_view: Optional[bool] = None,
    ):
        super().__init__()

        self.num_robots = int(num_robots)
        if self.num_robots <= 0:
            raise ValueError("num_robots must be positive")
        self.input_res = None if input_res is None else int(input_res)
        self.enable_wrist_view = None if enable_wrist_view is None else bool(enable_wrist_view)

    def set_normalizer(self, normalizer: nn.Module):
        raise NotImplementedError

    @staticmethod
    def _flatten_time(obs: Mapping[str, torch.Tensor]) -> dict:
        return {
            key: value.reshape(-1, *value.shape[2:])
            if torch.is_tensor(value) and value.dim() >= 2
            else value
            for key, value in obs.items()
        }

    @staticmethod
    def _expand_task_context(
        task_context: Mapping[str, torch.Tensor], batch_size: int, steps: int
    ) -> dict:
        expanded = {}
        for key, value in task_context.items():
            if not torch.is_tensor(value) or value.shape[:1] != (batch_size,):
                raise ValueError(
                    f"task context {key!r} must have leading batch size {batch_size}"
                )
            expanded[key] = value[:, None].expand(
                batch_size, steps, *value.shape[1:]
            ).reshape(batch_size * steps, *value.shape[1:])
        return expanded

    def obs_to_input(
        self,
        model_obs: Mapping[str, torch.Tensor],
        canonical_obs: Mapping[str, torch.Tensor],
        task_context: Mapping[str, torch.Tensor],
        composer_normalizer,
        resize: bool = False,
    ) -> EncoderInputs:
        """Flatten a normalized observation into encoder inputs.

        ``model_obs`` is already in the policy's model space (``normalize_obs`` ran at
        the policy boundary); ``canonical_obs`` is the physical observation it
        came from, which the Seeker composer re-normalizes
        into ``composer_normalizer``'s own space.
        """
        if canonical_obs is None:
            raise ValueError(
                "obs_to_input needs the canonical observation for the Seeker composer; "
                "the policy boundary passes it as the `canonical_obs` side channel"
            )
        if task_context is None or "robot_id" not in task_context:
            raise ValueError(
                "robot_id must be provided in task_context for multi-robot normalization"
            )

        img_dim = model_obs["rgb_external"].dim()
        assert img_dim == 5, f"Expect input to contain temporal dim, got dim={img_dim}"

        T = model_obs["rgb_external"].shape[1]

        obs_flat = self._flatten_time(model_obs)
        canonical_flat = self._flatten_time(canonical_obs)
        context_flat = self._expand_task_context(
            task_context, model_obs["rgb_external"].shape[0], T
        )

        composer_norm = normalize_obs(
            {key: canonical_flat[key] for key in _COMPOSER_FIELDS},
            composer_normalizer,
            observation_kinds={},
            robot_id=obs_robot_id(context_flat),
        )

        external_image = obs_flat["rgb_external"]
        validate_model_rgb(external_image)
        if resize:
            external_image = resize_image(external_image, self.input_res)

        if self.enable_wrist_view:
            wrist_image = obs_flat["rgb_wrist"]
            validate_model_rgb(wrist_image)
            if resize:
                wrist_image = resize_image(wrist_image, self.input_res)
        else:
            wrist_image = None

        # convert robot_id to one-hot
        robot_id = context_flat["robot_id"].view(-1).long()
        robot_id_one_hot = F.one_hot(robot_id, num_classes=self.num_robots).float()
        composer_robot_id = robot_id
        composer_robot_id_one_hot = F.one_hot(
            composer_robot_id, num_classes=self.num_robots
        ).float()
        if composer_norm["gripper_qpos"].shape[-1] != 2:
            raise ValueError(
                "gripper must have 2 finger joints (Panda parallel-jaw "
                f"gripper), got shape {tuple(composer_norm['gripper_qpos'].shape)}"
            )
        gripper_open = (
            torch.abs(
                composer_norm["gripper_qpos"][:, 0] - composer_norm["gripper_qpos"][:, 1]
            )
            - 1.0
        )

        composer_in = {
            "eef_pos": composer_norm["eef_pos"],
            "eef_rot": composer_norm["eef_rot6d"],
            "gripper": composer_norm["gripper_qpos"],
            "gripper_opening": gripper_open.unsqueeze(-1),
            "robot_id": composer_robot_id_one_hot,
            "task_embedding": context_flat["task_embedding"],
            "raw_eef_pos": canonical_flat["eef_pos"],
            "raw_gripper_opening": (
                torch.abs(
                    canonical_flat["gripper_qpos"][:, 0] - canonical_flat["gripper_qpos"][:, 1]
                )
                - 1.0
            ).unsqueeze(-1),
        }
        if "task_language_tokens" in context_flat:
            composer_in["task_language_tokens"] = context_flat[
                "task_language_tokens"
            ].float()
        self._validate_composer_in(composer_in)

        proprio = torch.cat(
            [obs_flat["eef_pos"], obs_flat["eef_rot6d"], obs_flat["gripper_qpos"]],
            dim=-1,
        )

        return EncoderInputs(
            external=external_image,
            wrist=wrist_image,
            proprio=proprio,
            robot_id=robot_id_one_hot,
            composer_in=composer_in,
            task_embedding=context_flat["task_embedding"],
            T=T,
        )

    def _validate_composer_in(self, composer_in: dict):
        if not isinstance(composer_in, dict):
            raise TypeError(f"composer_in must be a dict, got {type(composer_in)}")

        expected = {
            "eef_pos": (3,),
            "eef_rot": (6,),
            "gripper": (2,),
            "gripper_opening": (1,),
            "robot_id": (self.num_robots,),
        }

        missing = [
            k for k in [*expected.keys(), "task_embedding"] if k not in composer_in
        ]
        if missing:
            raise ValueError(f"composer_in missing keys: {missing}")

        expected["task_embedding"] = (composer_in["task_embedding"].shape[-1],)

        for key, tail_shape in expected.items():
            x = composer_in[key]
            if x.ndim != 2 or tuple(x.shape[1:]) != tail_shape:
                raise ValueError(
                    f"composer_in['{key}'] must be [B, {tail_shape}], got {tuple(x.shape)}"
                )
