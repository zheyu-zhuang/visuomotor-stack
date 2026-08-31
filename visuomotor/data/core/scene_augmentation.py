"""Coupled world-space yaw augmentation: rotates obs + actions + focus targets together.

The direct analog of :mod:`visuomotor.data.core.mirror`'s
``MirrorAugmentationConfig``/``MirrorObsActionAugmentor`` pattern (one
transform applied jointly to everything in a batch before normalization),
ported from equidiff's ``transform/scene.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from visuomotor.geometry import bounds as Bounds
from visuomotor.geometry import representation as Representation
from visuomotor.geometry import rigid as Rigid


@dataclass(frozen=True)
class SceneYawAugmentationConfig:
    """Config for :class:`SceneYawAugmenter`. Voxel-only; no rotation.py analog."""

    enable: bool = False
    min_deg: float = -180.0
    max_deg: float = 180.0
    max_attempts: int = 64
    identity_probability: float = 1 / 64
    workspace_center_xy: tuple[float, float] = (0.0, 0.0)
    workspace_size: float = 0.0

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "SceneYawAugmentationConfig":
        if isinstance(cfg, cls):
            return cfg
        if cfg is None:
            return cls()
        enable = bool(_cfg_get(cfg, "enable", False))
        min_deg = float(_cfg_get(cfg, "min_deg", -180.0))
        max_deg = float(_cfg_get(cfg, "max_deg", 180.0))
        max_attempts = int(_cfg_get(cfg, "max_attempts", 64))
        identity_probability = float(_cfg_get(cfg, "identity_probability", 1 / 64))
        if min_deg > max_deg:
            raise ValueError(
                f"scene_yaw_augmentation.min_deg ({min_deg}) must be <= max_deg ({max_deg})"
            )
        if max_attempts < 1:
            raise ValueError(f"scene_yaw_augmentation.max_attempts must be >= 1, got {max_attempts}")
        if not 0.0 <= identity_probability <= 1.0:
            raise ValueError(
                f"scene_yaw_augmentation.identity_probability must be in [0, 1], got {identity_probability}"
            )
        workspace = _cfg_get(cfg, "workspace", None)
        if enable and not workspace:
            raise ValueError(
                "scene_yaw_augmentation.workspace (center_xy, size) is required when enable=True"
            )
        workspace = workspace or {}
        center_xy = _cfg_get(workspace, "center_xy", None)
        size = _cfg_get(workspace, "size", None)
        if enable and (center_xy is None or size is None):
            raise ValueError(
                "scene_yaw_augmentation.workspace must set both center_xy and size when enable=True"
            )
        center_xy = center_xy if center_xy is not None else (0.0, 0.0)
        size = float(size) if size is not None else 0.0
        if enable and size <= 0.0:
            raise ValueError(f"scene_yaw_augmentation.workspace.size must be > 0, got {size}")
        if len(center_xy) != 2:
            raise ValueError(
                f"scene_yaw_augmentation.workspace.center_xy must have length 2, got {center_xy!r}"
            )
        return cls(
            enable=enable,
            min_deg=min_deg,
            max_deg=max_deg,
            max_attempts=max_attempts,
            identity_probability=identity_probability,
            workspace_center_xy=(float(center_xy[0]), float(center_xy[1])),
            workspace_size=size,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enable": self.enable,
            "min_deg": self.min_deg,
            "max_deg": self.max_deg,
            "max_attempts": self.max_attempts,
            "identity_probability": self.identity_probability,
            "workspace": {
                "center_xy": list(self.workspace_center_xy),
                "size": self.workspace_size,
            },
        }


def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class FocusTargets:
    pos: torch.Tensor
    valid: torch.Tensor


@dataclass
class WorldSceneBatch:
    obs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    focus: Optional[FocusTargets] = None
    reference_frame: Optional[Dict[str, torch.Tensor]] = None
    progress: Optional[torch.Tensor] = None


@dataclass
class SceneTransformMetrics:
    """Lazily reduced identity and rejection-fallback metrics."""

    identity: torch.Tensor
    fallback: torch.Tensor

    @property
    def identity_fraction(self) -> float:
        return float(self.identity.float().mean())

    @property
    def fallback_fraction(self) -> float:
        return float(self.fallback.float().mean())


@dataclass
class SceneTransformResult:
    scene: WorldSceneBatch
    metrics: SceneTransformMetrics


def yaw_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrices about the world Z axis."""
    cos, sin = angles.cos(), angles.sin()
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    row0 = torch.stack((cos, -sin, zero), dim=-1)
    row1 = torch.stack((sin, cos, zero), dim=-1)
    row2 = torch.stack((zero, zero, one), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def randomize_pose(
    rotation: torch.Tensor,
    position: torch.Tensor,
    translation_m: float,
    rotation_deg: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple:
    """Bounded random translation + isotropic body-rotation jitter, for ground-truth noise."""
    batch = rotation.shape[0]
    translation_noise = (
        torch.rand(
            batch,
            3,
            device=rotation.device,
            dtype=rotation.dtype,
            generator=generator,
        )
        * 2
        - 1
    )
    position = position + translation_noise * translation_m
    axis = F.normalize(
        torch.randn(
            batch,
            3,
            device=rotation.device,
            dtype=rotation.dtype,
            generator=generator,
        ),
        dim=-1,
    )
    angle = (
        torch.rand(
            batch,
            device=rotation.device,
            dtype=rotation.dtype,
            generator=generator,
        )
        * 2
        - 1
    ) * math.radians(rotation_deg)
    noise_rotation = Representation.rotvec_to_mat(axis * angle.unsqueeze(-1))
    return Rigid.transform_rotation(rotation, noise_rotation), position


def _yaw_frame(
    rotation: torch.Tensor, workspace: Bounds.PlanarWorkspace, batch_dims: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(R, t)`` for a yaw about ``workspace``'s vertical axis, broadcast to ``batch_dims``."""
    middle = (1,) * (batch_dims - 1)
    rotation_b = rotation.reshape(rotation.shape[0], *middle, 3, 3)
    center = torch.cat(
        (workspace.center(rotation.device, rotation.dtype), rotation.new_zeros(1))
    )
    translation = center - torch.einsum("...ij,j->...i", rotation_b, center)
    return rotation_b, translation


def _rotate_points(points: torch.Tensor, rotation: torch.Tensor, workspace: Bounds.PlanarWorkspace) -> torch.Tensor:
    """Rotate world XYZ points about ``workspace``'s vertical axis through its center."""
    return Rigid.transform(*_yaw_frame(rotation, workspace, points.ndim - 1), points)


def _rotate_pose(pose: torch.Tensor, rotation: torch.Tensor, workspace: Bounds.PlanarWorkspace) -> torch.Tensor:
    """Rotate ``[position(3), rotation6d(6), extras...]`` poses or action chunks."""
    return Rigid.transform_pose(*_yaw_frame(rotation, workspace, pose.ndim - 1), pose)


def _rotate_orientation(orientation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a canonical rot6d or source flattened-matrix orientation."""
    width = int(orientation.shape[-1])
    if width == 6:
        matrix = Representation.rot6d_to_mat(orientation)
    elif width == 9:
        matrix = orientation.reshape(*orientation.shape[:-1], 3, 3)
    else:
        raise ValueError(f"orientation must have width 6 or 9, got {width}")
    middle = (1,) * (matrix.ndim - 3)
    rotation_b = rotation.reshape(rotation.shape[0], *middle, 3, 3)
    new_matrix = Rigid.transform_rotation(rotation_b, matrix)
    if width == 6:
        return Representation.mat_to_rot6d(new_matrix)
    return new_matrix.reshape(*orientation.shape[:-1], 9)


def _rotate_voxels(voxels: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a ``[B,T,C,X,Y,Z]`` voxel grid via an X/Y nearest-neighbour gather."""
    if voxels.ndim != 6:
        raise ValueError(f"voxels must have shape [B,T,C,X,Y,Z], got {tuple(voxels.shape)}")
    batch, steps, channels, size_x, size_y, size_z = voxels.shape
    if size_x != size_y:
        raise ValueError("voxel yaw rotation requires square X/Y axes")
    rot = rotation.repeat_interleave(steps, dim=0)

    axis_xy = torch.linspace(-1, 1, size_x, device=voxels.device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(axis_xy, axis_xy, indexing="ij")
    coords = torch.stack((grid_x, grid_y), dim=-1)
    inverse_rotation = Rigid.inv_rotation(rot)[:, :2, :2].to(torch.float32)
    source = torch.einsum("bij,xyj->bxyi", inverse_rotation, coords)

    index = ((source + 1.0) * (0.5 * (size_x - 1))).round().to(torch.int64)
    inside = ((index >= 0) & (index < size_x)).all(dim=-1)
    index = index.clamp(0, size_x - 1)
    plane = (index[..., 0] * size_y + index[..., 1]).reshape(
        batch * steps, 1, size_x * size_y, 1
    )

    planar = voxels.reshape(batch * steps, channels, size_x * size_y, size_z)
    rotated = planar.gather(
        2, plane.expand(batch * steps, channels, size_x * size_y, size_z)
    )
    rotated.mul_(inside.reshape(batch * steps, 1, size_x * size_y, 1))
    return rotated.reshape(batch, steps, channels, size_x, size_y, size_z)


class SceneYawAugmenter(nn.Module):
    """Rejection-samples one world-yaw angle per batch row that keeps everything in-workspace."""

    def __init__(
        self,
        workspace: Bounds.PlanarWorkspace,
        *,
        enabled: bool = True,
        min_deg: float = -180.0,
        max_deg: float = 180.0,
        max_attempts: int = 64,
        identity_probability: float = 1 / 64,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.enabled = bool(enabled)
        self.min_deg = float(min_deg)
        self.max_deg = float(max_deg)
        self.max_attempts = int(max_attempts)
        self.identity_probability = float(identity_probability)

    @classmethod
    def from_config(cls, config: "SceneYawAugmentationConfig") -> "SceneYawAugmenter":
        workspace = Bounds.PlanarWorkspace(config.workspace_center_xy, config.workspace_size)
        return cls(
            workspace,
            enabled=config.enable,
            min_deg=config.min_deg,
            max_deg=config.max_deg,
            max_attempts=config.max_attempts,
            identity_probability=config.identity_probability,
        )

    def get_runtime_config(self) -> str:
        if not self.enabled:
            return "SceneYawAugmenter: disabled"
        return (
            f"SceneYawAugmenter: yaw in [{self.min_deg}, {self.max_deg}] deg, "
            f"max_attempts={self.max_attempts}, identity_probability={self.identity_probability}"
        )

    def _points_inside(self, points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        rotated = _rotate_points(points, rotation, self.workspace)
        inside = self.workspace.contains_xy(rotated)
        return inside.reshape(inside.shape[0], -1).all(dim=-1)

    def _candidates_valid(
        self, scene: WorldSceneBatch, rotation: torch.Tensor, attempts: int
    ) -> torch.Tensor:
        """Return the ``[B, attempts]`` in-workspace mask."""
        batch = scene.actions.shape[0]

        def inside(points: torch.Tensor) -> torch.Tensor:
            fanned = (
                points.reshape(batch, 1, -1, 3)
                .expand(batch, attempts, -1, 3)
                .reshape(batch * attempts, -1, 3)
            )
            return self._points_inside(fanned, rotation).reshape(batch, attempts)

        valid = inside(scene.actions[..., :3])
        if "eef_pos" in scene.obs:
            valid = valid & inside(scene.obs["eef_pos"])
        if scene.focus is not None:
            focus_matters = scene.focus.valid.reshape(batch, -1).any(dim=-1, keepdim=True)
            valid = valid & (inside(scene.focus.pos) | ~focus_matters)
        if scene.reference_frame is not None:
            for pose in scene.reference_frame.values():
                valid = valid & inside(pose[..., :3])
        return valid

    def _sample_valid_yaw(
        self, scene: WorldSceneBatch, angles: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched rejection sampling returning ``(angle, use_identity, resolved)``."""
        batch = scene.actions.shape[0]
        device, dtype = scene.actions.device, scene.actions.dtype

        use_identity = torch.rand(batch, device=device) < self.identity_probability
        if angles is not None:
            candidates = angles.reshape(batch, 1)
        else:
            candidates = (
                torch.rand(batch, self.max_attempts, device=device, dtype=dtype)
                * (self.max_deg - self.min_deg)
                + self.min_deg
            ) * (math.pi / 180.0)
        attempts = int(candidates.shape[1])

        valid = self._candidates_valid(
            scene, yaw_matrices(candidates.reshape(-1)), attempts
        )
        accepted = valid.any(dim=-1)
        first = valid.to(dtype).argmax(dim=-1, keepdim=True)
        best_angle = candidates.gather(1, first).squeeze(1)
        best_angle = torch.where(
            accepted & ~use_identity, best_angle, torch.zeros_like(best_angle)
        )
        return best_angle, use_identity, accepted | use_identity

    def bounds(self, scene: WorldSceneBatch) -> Dict[str, Any]:
        """Exact workspace-constrained bounds for yaw-affected positions.

        Candidate yaw rotations are accepted only when every relevant X/Y
        position remains inside this same square workspace. Z is data-derived
        because a world-Z yaw leaves it unchanged.
        """
        out: Dict[str, Any] = {
            "action_pos": self.workspace.position_bounds(scene.actions[..., :3])
        }
        if "eef_pos" in scene.obs:
            out["eef_pos"] = self.workspace.position_bounds(scene.obs["eef_pos"])
        if scene.focus is not None:
            out["focus_pos"] = self.workspace.position_bounds(scene.focus.pos)
        if scene.reference_frame is not None:
            out["reference_frame"] = {
                key: self.workspace.position_bounds(pose[..., :3])
                for key, pose in scene.reference_frame.items()
            }
        return out

    def forward(
        self, scene: WorldSceneBatch, angles: Optional[torch.Tensor] = None
    ) -> SceneTransformResult:
        batch = scene.actions.shape[0]
        device, dtype = scene.actions.device, scene.actions.dtype

        if not self.enabled:
            rotation = yaw_matrices(torch.zeros(batch, device=device, dtype=dtype))
            return SceneTransformResult(
                scene=self._apply_rotation_to_scene(scene, rotation),
                metrics=SceneTransformMetrics(
                    identity=torch.ones(batch, device=device, dtype=torch.bool),
                    fallback=torch.zeros(batch, device=device, dtype=torch.bool),
                ),
            )

        best_angle, use_identity, resolved = self._sample_valid_yaw(scene, angles)
        rotation = yaw_matrices(best_angle)
        return SceneTransformResult(
            scene=self._apply_rotation_to_scene(scene, rotation),
            metrics=SceneTransformMetrics(
                identity=use_identity | ~resolved,
                fallback=~resolved,
            ),
        )

    def _apply_rotation_to_scene(self, scene: WorldSceneBatch, rotation: torch.Tensor) -> WorldSceneBatch:
        obs = dict(scene.obs)
        if "eef_pos" in obs:
            obs["eef_pos"] = _rotate_points(obs["eef_pos"], rotation, self.workspace)
        if "eef_rot" in obs:
            obs["eef_rot"] = _rotate_orientation(obs["eef_rot"], rotation)
        if "voxel" in obs:
            obs["voxel"] = _rotate_voxels(obs["voxel"], rotation)

        focus = None
        if scene.focus is not None:
            focus = FocusTargets(
                pos=_rotate_points(scene.focus.pos, rotation, self.workspace), valid=scene.focus.valid
            )
        reference_frame = None
        if scene.reference_frame is not None:
            reference_frame = {
                key: _rotate_pose(pose, rotation, self.workspace)
                for key, pose in scene.reference_frame.items()
            }
        return WorldSceneBatch(
            obs=obs,
            actions=_rotate_pose(scene.actions, rotation, self.workspace),
            focus=focus,
            reference_frame=reference_frame,
            progress=scene.progress,
        )


class SceneYawObsActionAugmentor:
    """Runtime world-yaw augmentation of a training batch (voxel policies only)."""

    def __init__(
        self,
        *,
        shape_meta: dict,
        action_rep: str,
        config: Optional[Mapping[str, Any]],
        source_keys: Mapping[str, str],
        fixed_camera_rgb_keys: Sequence[str],
    ):
        self.config = SceneYawAugmentationConfig.from_config(config)
        self.augmenter = SceneYawAugmenter.from_config(self.config)
        self.action_rep = str(action_rep)
        self.action_dim = int(shape_meta["action"]["shape"][0])
        obs_meta = shape_meta["obs"]
        voxel_keys = [key for key, value in obs_meta.items() if value.get("type", "low_dim") == "voxel"]
        self.pos_key = source_keys["eef_pos"]
        self.rot_key = source_keys["eef_rot"]
        if self.config.enable:
            if len(voxel_keys) != 1:
                raise ValueError(
                    f"scene_yaw_augmentation requires exactly one voxel observation, got {voxel_keys}"
                )
            size_x, size_y = obs_meta[voxel_keys[0]]["shape"][1:3]
            if size_x != size_y:
                raise ValueError(
                    "scene_yaw_augmentation requires square voxel X/Y axes, got "
                    f"{tuple(obs_meta[voxel_keys[0]]['shape'])}"
                )
            if fixed_camera_rgb_keys:
                raise ValueError(
                    "scene_yaw_augmentation cannot rotate a fixed-camera RGB observation "
                    f"(rotates the world without re-rendering): {list(fixed_camera_rgb_keys)}"
                )
            world_frame_keys = [
                key
                for key, value in obs_meta.items()
                if value.get("type", "low_dim") in ("point_cloud", "depth")
            ]
            if world_frame_keys:
                raise ValueError(
                    "scene_yaw_augmentation only rotates the voxel grid; these "
                    f"world-frame observations would be left unrotated: {world_frame_keys}"
                )
            if self.action_rep != "absolute":
                raise ValueError("scene_yaw_augmentation requires action_rep='absolute'")
        self.voxel_key = voxel_keys[0] if voxel_keys else None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enable)

    def get_runtime_config(self) -> str:
        return self.augmenter.get_runtime_config()

    def augment_batch(self, batch: dict) -> dict:
        if not self.enabled:
            return batch
        obs = batch["obs"]
        world_obs = {
            "eef_pos": obs[self.pos_key],
            "eef_rot": obs[self.rot_key],
            "voxel": obs[self.voxel_key],
        }
        targets = batch.get("targets", {})
        focus = None
        if "focus_target_pos" in targets and "focus_target_valid" in targets:
            focus = FocusTargets(
                pos=targets["focus_target_pos"],
                valid=targets["focus_target_valid"],
            )
        reference_frame_keys = [
            key for key in targets if key.startswith("reference_frame_")
        ]
        reference_frame = (
            {key: targets[key] for key in reference_frame_keys}
            if reference_frame_keys
            else None
        )

        scene = WorldSceneBatch(
            obs=world_obs,
            actions=batch["action"][..., : self.action_dim],
            focus=focus,
            reference_frame=reference_frame,
        )
        rotated = self.augmenter(scene).scene

        obs[self.pos_key] = rotated.obs["eef_pos"]
        obs[self.rot_key] = rotated.obs["eef_rot"]
        obs[self.voxel_key] = rotated.obs["voxel"]
        batch["action"][..., : self.action_dim] = rotated.actions
        if focus is not None:
            targets["focus_target_pos"] = rotated.focus.pos
        if reference_frame is not None:
            for key in reference_frame_keys:
                targets[key] = rotated.reference_frame[key]
        return batch

    def __call__(self, batch: dict) -> dict:
        return self.augment_batch(batch)
