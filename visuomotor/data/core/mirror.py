"""Mirror-frame transforms and runtime obs/action pair augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch

from visuomotor.geometry import reflection as Reflection
from visuomotor.geometry import representation as Representation
from visuomotor.geometry import rigid as Rigid

DEFAULT_MIRROR_FRAME_POS = [-0.1, 0.0, 1.0]
DEFAULT_MIRROR_FRAME_ROT = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
]


@dataclass(frozen=True)
class MirrorAugmentationConfig:
    enable: bool = False
    prob: float = 0.5
    frame_pos: tuple[float, float, float] = tuple(DEFAULT_MIRROR_FRAME_POS)
    frame_rot: tuple[tuple[float, float, float], ...] = tuple(
        tuple(row) for row in DEFAULT_MIRROR_FRAME_ROT
    )

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "MirrorAugmentationConfig":
        if isinstance(cfg, cls):
            return cfg
        if cfg is None:
            return cls()
        enable = bool(_cfg_get(cfg, "enable", False))
        prob = float(_cfg_get(cfg, "prob", 0.5))
        frame_pose = _cfg_get(cfg, "frame_pose", {}) or {}
        frame_pos = _cfg_get(frame_pose, "pos", DEFAULT_MIRROR_FRAME_POS)
        frame_rot = _cfg_get(frame_pose, "rot", DEFAULT_MIRROR_FRAME_ROT)
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"mirror_augmentation.prob must be in [0, 1], got {prob}")
        return cls(
            enable=enable,
            prob=prob,
            frame_pos=tuple(float(x) for x in frame_pos),
            frame_rot=tuple(tuple(float(x) for x in row) for row in frame_rot),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enable": self.enable,
            "prob": self.prob,
            "frame_pose": {
                "pos": list(self.frame_pos),
                "rot": [list(row) for row in self.frame_rot],
            },
        }


def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _is_torch(x) -> bool:
    return torch.is_tensor(x)


def _as_torch(x) -> torch.Tensor:
    return x if _is_torch(x) else torch.from_numpy(np.asarray(x))


def _like(result: torch.Tensor, reference):
    return result if _is_torch(reference) else result.numpy()


def _frame_pos(config: MirrorAugmentationConfig, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(config.frame_pos, dtype=like.dtype, device=like.device)


def _frame_rot(config: MirrorAugmentationConfig, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(config.frame_rot, dtype=like.dtype, device=like.device)


def _rot_to_matrix(rot: torch.Tensor):
    rot_dim = int(rot.shape[-1])
    if rot_dim == 9:
        return rot.reshape(*rot.shape[:-1], 3, 3), 9
    if rot_dim == 6:
        return Representation.rot6d_to_mat(rot), 6
    raise ValueError(f"Expected rotation dim 6 or 9, got {rot_dim}")


def _matrix_to_rot(rot_mat: torch.Tensor, rot_dim: int):
    if rot_dim == 9:
        return rot_mat.reshape(*rot_mat.shape[:-2], 9)
    if rot_dim == 6:
        return Representation.mat_to_rot6d(rot_mat)
    raise ValueError(f"Expected rotation dim 6 or 9, got {rot_dim}")


def pose_world_to_mirror_frame(pos, rot, config: MirrorAugmentationConfig):
    """Convert world-frame positions/rotations into the configured mirror frame."""
    pos_t, rot_t = _as_torch(pos), _as_torch(rot)
    frame_pos, frame_rot = _frame_pos(config, pos_t), _frame_rot(config, pos_t)
    rot_mat, rot_dim = _rot_to_matrix(rot_t)
    R_frame_inv, t_frame_inv = Rigid.inv(frame_rot, frame_pos)
    centered_pos = Rigid.transform(R_frame_inv, t_frame_inv, pos_t)
    centered_rot = Rigid.transform_rotation(R_frame_inv, rot_mat)
    return _like(centered_pos, pos), _like(_matrix_to_rot(centered_rot, rot_dim), rot)


def pose_mirror_frame_to_world(pos, rot, config: MirrorAugmentationConfig):
    """Convert mirror-frame positions/rotations back to world frame."""
    pos_t, rot_t = _as_torch(pos), _as_torch(rot)
    frame_pos, frame_rot = _frame_pos(config, pos_t), _frame_rot(config, pos_t)
    rot_mat, rot_dim = _rot_to_matrix(rot_t)
    world_pos = Rigid.transform(frame_rot, frame_pos, pos_t)
    world_rot = Rigid.transform_rotation(frame_rot, rot_mat)
    return _like(world_pos, pos), _like(_matrix_to_rot(world_rot, rot_dim), rot)


def reflect_pose_in_mirror_frame(pos, rot):
    """Reflect centered pose coordinates across the mirror-frame yz plane."""
    pos_t, rot_t = _as_torch(pos), _as_torch(rot)
    reflection = Reflection.reflection_matrix(axis=0, dtype=pos_t.dtype, device=pos_t.device)
    rot_mat, rot_dim = _rot_to_matrix(rot_t)
    reflected_pos = Reflection.reflect_point(reflection, pos_t)
    reflected_rot = Reflection.reflect_rotation(reflection, rot_mat)
    return _like(reflected_pos, pos), _like(_matrix_to_rot(reflected_rot, rot_dim), rot)


def action_world_to_mirror_frame(
    action,
    config: MirrorAugmentationConfig,
    *,
    action_rep: str,
    action_dim: int = 10,
):
    """Center absolute actions; local delta actions are already frame-relative."""
    if action_rep == "delta":
        return action
    if action_rep != "absolute":
        raise ValueError(f"Unsupported action_rep for mirror centering: {action_rep!r}")
    return _map_action_pose(
        action,
        action_dim,
        lambda p, r: pose_world_to_mirror_frame(p, r, config),
    )


def action_mirror_frame_to_world(
    action,
    config: MirrorAugmentationConfig,
    *,
    action_rep: str,
    action_dim: int = 10,
):
    """Uncenter absolute actions; local delta actions pass through unchanged."""
    if action_rep == "delta":
        return action
    if action_rep != "absolute":
        raise ValueError(f"Unsupported action_rep for mirror uncentering: {action_rep!r}")
    return _map_action_pose(
        action,
        action_dim,
        lambda p, r: pose_mirror_frame_to_world(p, r, config),
    )


def reflect_action_in_mirror_frame(action, *, action_dim: int = 10):
    """Reflect action pose chunks in mirror-frame coordinates."""
    return _map_action_pose(action, action_dim, reflect_pose_in_mirror_frame)


def _map_action_pose(action, action_dim: int, pose_fn):
    if int(action.shape[-1]) != int(action_dim):
        raise ValueError(
            f"Expected action last dim {action_dim}, got {int(action.shape[-1])}"
        )
    if action_dim < 10:
        raise ValueError(f"Mirror augmentation requires action_dim >= 10, got {action_dim}")
    pos = action[..., :3]
    rot = action[..., 3:9]
    gripper = action[..., 9:]
    out_pos, out_rot = pose_fn(pos, rot)
    if _is_torch(action):
        return torch.cat([out_pos, out_rot, gripper], dim=-1)
    return np.concatenate([out_pos, out_rot, gripper], axis=-1)


def center_lowdim_observations(obs: dict, pos_key: str, rot_key: str, config):
    obs[pos_key], obs[rot_key] = pose_world_to_mirror_frame(
        obs[pos_key], obs[rot_key], config
    )


class MirrorObsActionAugmentor:
    """Runtime mirror-frame conversion plus optional obs/action pair reflection."""

    def __init__(
        self,
        *,
        shape_meta: dict,
        action_rep: str,
        config: Optional[Mapping[str, Any]],
        source_keys: Mapping[str, str],
    ):
        self.config = MirrorAugmentationConfig.from_config(config)
        self.action_rep = str(action_rep)
        self.action_dim = int(shape_meta["action"]["shape"][0])
        obs_meta = shape_meta["obs"]
        self.image_keys = [
            key for key, value in obs_meta.items() if value.get("type", "low_dim") == "rgb"
        ]
        self.pos_key = source_keys["eef_pos"]
        self.rot_key = source_keys["eef_rot"]
        if self.config.enable and not self.image_keys:
            raise ValueError("mirror_augmentation requires at least one RGB observation key")

    @property
    def enabled(self) -> bool:
        return bool(self.config.enable)

    def center_batch(self, batch: dict) -> dict:
        if not self.enabled:
            return batch
        obs = batch["obs"]
        center_lowdim_observations(obs, self.pos_key, self.rot_key, self.config)
        batch["action"] = action_world_to_mirror_frame(
            batch["action"],
            self.config,
            action_rep=self.action_rep,
            action_dim=self.action_dim,
        )
        return batch

    def augment_batch(self, batch: dict) -> dict:
        if not self.enabled or self.config.prob <= 0.0:
            return batch
        action = batch["action"]
        batch_size = int(action.shape[0])
        mask = torch.rand(batch_size, device=action.device) < self.config.prob
        if not bool(mask.any()):
            return batch

        obs = batch["obs"]
        for key in self.image_keys:
            if key in obs:
                obs[key][mask] = obs[key][mask].flip(-1)
        obs[self.pos_key][mask], obs[self.rot_key][mask] = reflect_pose_in_mirror_frame(
            obs[self.pos_key][mask], obs[self.rot_key][mask]
        )
        batch["action"][mask] = reflect_action_in_mirror_frame(
            batch["action"][mask], action_dim=self.action_dim
        )
        return batch

    def __call__(self, batch: dict) -> dict:
        self.center_batch(batch)
        self.augment_batch(batch)
        return batch
