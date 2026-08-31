"""Fit model normalizers from canonical MimicGen training values."""

import numpy as np
import torch

from visuomotor.data.core import keypose_targets as KeyposeTargets
from visuomotor.data.core import mirror as CoreMirror
from visuomotor.data.core import normalization as CoreNormalization
from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.core import scene_augmentation as CoreSceneAugmentation


def build_normalizer(dataset, kind: str = "multi_robot_linear"):
    """Fit the public dataset normalizer without making the dataset own fitting."""
    _validate_kind(kind)
    normalizer = CoreNormalization.Normalizer()
    robot_ids = _fit_normalizer(
        dataset, normalizer, single_robot=(kind == "linear")
    )
    if dataset.target_adapter is not None:
        if kind == "linear" and len(robot_ids) == 1:
            fit_target_fields(dataset, normalizer, mask=None, robot_id=None)
        elif kind != "linear":
            step_robot_ids = _step_robot_ids(dataset)
            for robot_id in robot_ids:
                fit_target_fields(
                    dataset,
                    normalizer,
                    mask=step_robot_ids == robot_id,
                    robot_id=robot_id,
                )
    normalizer.finalize()
    _validate_single_robot(kind, robot_ids)
    return normalizer


def fit_target_fields(dataset, normalizer, *, mask, robot_id) -> None:
    """Fit target-derived fields for one robot's active training rows."""
    target_adapter = dataset.target_adapter
    if target_adapter is None:
        raise ValueError("target normalizer fields require a target adapter")
    active_indices = (
        dataset.active_step_indices
        if mask is None
        else dataset.active_step_indices[mask]
    )
    position = dataset.lowdim[dataset.pos_key]
    rotation_flat = dataset.lowdim[dataset.rot_key]
    first_idx = target_adapter.first_indices[active_indices]
    last_idx = target_adapter.last_indices[active_indices]
    first_poses = KeyposeTargets.reference_frame_poses(position, rotation_flat, first_idx)
    last_poses = KeyposeTargets.reference_frame_poses(position, rotation_flat, last_idx)
    if dataset.scene_yaw_augmentation.enable:
        scene_yaw = CoreSceneAugmentation.SceneYawAugmenter.from_config(
            dataset.scene_yaw_augmentation
        )
        scene = CoreSceneAugmentation.WorldSceneBatch(
            obs={},
            actions=torch.from_numpy(dataset.action[active_indices, :3]),
            reference_frame={
                "first": torch.from_numpy(first_poses),
                "last": torch.from_numpy(last_poses),
            },
        )
        bounds = scene_yaw.bounds(scene)["reference_frame"]
        lo = torch.cat((bounds["first"][0], bounds["last"][0]), dim=0)
        hi = torch.cat((bounds["first"][1], bounds["last"][1]), dim=0)
        normalizer.update_bounds(KeyposeTargets.FOCUS_POSE_POS_KEY, lo, hi, robot_id)
    else:
        focus_pose_samples = KeyposeTargets.focus_pose_position_samples(
            first_poses[:, :3], last_poses[:, :3]
        )
        normalizer.update_samples(
            KeyposeTargets.FOCUS_POSE_POS_KEY, focus_pose_samples, robot_id
        )


def _fit_normalizer(dataset, normalizer, *, single_robot: bool) -> list[int]:
    active_action = dataset.action[dataset.active_step_indices]
    active_lowdim = {
        key: dataset.lowdim[key][dataset.active_step_indices].astype(
            np.float32, copy=False
        )
        for key in dataset.lowdim_keys
    }
    active_derived = {
        key: dataset.derived_lowdim[key][dataset.active_step_indices]
        for key in dataset.derived_keys
    }
    action_dim = int(dataset.action_dim)
    if action_dim < 9:
        raise ValueError(
            "action_dim must include pos(3)+rot6d(6) to build the action "
            f"normalizer, got {action_dim}"
        )
    if (
        dataset.mirror_augmentation.enable
        and dataset.scene_yaw_augmentation.enable
    ):
        raise ValueError(
            "mirror_augmentation (RGB policies) and scene_yaw_augmentation "
            "(voxel policies) are mutually exclusive and must not both be enabled"
        )

    mirrored_lowdim = None
    mirrored_action = None
    if dataset.mirror_augmentation.enable:
        active_lowdim = dict(active_lowdim)
        active_lowdim[dataset.pos_key], active_lowdim[dataset.rot_key] = (
            CoreMirror.pose_world_to_mirror_frame(
                active_lowdim[dataset.pos_key],
                active_lowdim[dataset.rot_key],
                dataset.mirror_augmentation,
            )
        )
        active_action = CoreMirror.action_world_to_mirror_frame(
            active_action,
            dataset.mirror_augmentation,
            action_rep=dataset.action_rep,
            action_dim=action_dim,
        )
        mirrored_pos, mirrored_rot = CoreMirror.reflect_pose_in_mirror_frame(
            active_lowdim[dataset.pos_key], active_lowdim[dataset.rot_key]
        )
        mirrored_lowdim = dict(active_lowdim)
        mirrored_lowdim[dataset.pos_key] = mirrored_pos
        mirrored_lowdim[dataset.rot_key] = mirrored_rot
        mirrored_action = CoreMirror.reflect_action_in_mirror_frame(
            active_action, action_dim=action_dim
        )

    scene_yaw = None
    if dataset.scene_yaw_augmentation.enable:
        scene_yaw = CoreSceneAugmentation.SceneYawAugmenter.from_config(
            dataset.scene_yaw_augmentation
        )

    step_robot_ids = _step_robot_ids(dataset)
    robot_ids = [int(robot_id) for robot_id in np.unique(step_robot_ids)]
    if single_robot and len(robot_ids) != 1:
        return robot_ids

    for robot_id in robot_ids:
        fit_robot_id = None if single_robot else robot_id
        mask = step_robot_ids == robot_id
        _fit_point_cloud_fields(
            dataset,
            normalizer,
            global_indices=dataset.active_step_indices[mask],
            robot_id=fit_robot_id,
        )
        robot_action = active_action[mask]
        width = int(robot_action.shape[-1])
        if width % action_dim != 0:
            raise ValueError(
                "action width must be divisible by per-step action_dim: "
                f"width={width} action_dim={action_dim}"
            )
        steps = width // action_dim
        robot_action = robot_action.reshape(-1, steps, action_dim)
        action_pos = robot_action[..., :3]
        normalizer.update_samples(
            "action_gripper", robot_action[..., 9:action_dim], fit_robot_id
        )

        if scene_yaw is not None:
            scene = CoreSceneAugmentation.WorldSceneBatch(
                obs={}, actions=torch.from_numpy(action_pos)
            )
            lo, hi = scene_yaw.bounds(scene)["action_pos"]
            normalizer.update_bounds("action_pos", lo, hi, fit_robot_id)
        else:
            normalizer.update_samples("action_pos", action_pos, fit_robot_id)
            if mirrored_action is not None:
                mirrored_pos = mirrored_action[mask].reshape(
                    -1, steps, action_dim
                )[..., :3]
                normalizer.update_samples("action_pos", mirrored_pos, fit_robot_id)

        for key in dataset.lowdim_keys:
            values = active_lowdim[key][mask]
            if CoreObservations.is_matched_key("rot", key):
                continue
            if CoreObservations.is_matched_key("qpos", key):
                field = "gripper_qpos" if key == dataset.gripper_key else key
                normalizer.update_samples(field, values, fit_robot_id)
            elif CoreObservations.is_matched_key("pos", key):
                field = "eef_pos" if key == dataset.pos_key else key
                if scene_yaw is not None:
                    scene = CoreSceneAugmentation.WorldSceneBatch(
                        obs={"eef_pos": torch.from_numpy(values)},
                        actions=torch.from_numpy(action_pos),
                    )
                    lo, hi = scene_yaw.bounds(scene)["eef_pos"]
                    normalizer.update_bounds(field, lo, hi, fit_robot_id)
                else:
                    normalizer.update_samples(field, values, fit_robot_id)
                    if mirrored_lowdim is not None:
                        normalizer.update_samples(
                            field, mirrored_lowdim[key][mask], fit_robot_id
                        )
            else:
                raise RuntimeError(f"unsupported lowdim key: {key}")

        for key in dataset.derived_keys:
            # Body-frame deltas are invariant to the scene yaw, so their fitted
            # range is the observed one; widening it over the rotation orbit the
            # way `eef_pos` needs would only cost resolution.
            normalizer.update_samples(
                key, active_derived[key][mask], fit_robot_id
            )

    return robot_ids


def _fit_point_cloud_fields(dataset, normalizer, *, global_indices, robot_id) -> None:
    """Fit point channels in bounded chunks without retaining the full cloud cache."""
    if not dataset.point_cloud_keys:
        return
    for key in dataset.point_cloud_keys:
        for start in range(0, len(global_indices), 256):
            indices = global_indices[start : start + 256]
            values = dataset.observation_adapter.read(indices)[key]
            flat = torch.from_numpy(values).reshape(-1, values.shape[-1])
            normalizer.update_bounds(
                key, flat.min(dim=0).values, flat.max(dim=0).values, robot_id
            )


def _step_robot_ids(dataset) -> np.ndarray:
    return np.repeat(
        np.asarray(dataset.robot_id_episode, dtype=np.int64).reshape(-1),
        np.asarray(dataset.episode_lengths_active, dtype=np.int64),
    )


def _validate_kind(kind: str) -> None:
    if kind not in ("linear", "multi_robot_linear"):
        raise ValueError(f"unknown normalizer kind {kind!r}")


def _validate_single_robot(kind: str, robot_ids: list[int]) -> None:
    if kind == "linear" and len(robot_ids) != 1:
        raise ValueError(
            "a global normalizer needs a single-robot dataset, but this "
            f"one holds robots {sorted(robot_ids)}"
        )
