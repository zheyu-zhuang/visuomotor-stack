"""Canonical observation conversion and validation."""

import re
from typing import Mapping

import numpy as np
import torch

from visuomotor.geometry import representation as Representation

CANONICAL_PROPRIO_FIELDS = {
    "eef_pos": "eef_pos",
    "eef_rot": "eef_rot6d",
    "gripper_qpos": "gripper_qpos",
}
CANONICAL_PROPRIO_SHAPES = {
    "eef_pos": (3,),
    "eef_rot6d": (6,),
    "gripper_qpos": (2,),
    "eef_delta_pos": (3,),
    "eef_delta_rotvec": (3,),
    "gripper_qpos_delta": (2,),
}
# Proprioception derived from a pair of consecutive frames rather than read from
# a source field, and the canonical fields each one consumes.
DERIVED_PROPRIO_FIELDS = {
    "eef_delta_pos": ("eef_pos", "eef_rot6d"),
    "eef_delta_rotvec": ("eef_rot6d",),
    "gripper_qpos_delta": ("gripper_qpos",),
}


def derived_proprio_sources(fields) -> tuple:
    """Canonical fields the selected derived proprioception reads from."""
    required = {
        source
        for field in fields
        if field in DERIVED_PROPRIO_FIELDS
        for source in DERIVED_PROPRIO_FIELDS[field]
    }
    return tuple(key for key in CANONICAL_PROPRIO_SHAPES if key in required)


def proprio_deltas(
    previous: Mapping[str, torch.Tensor], current: Mapping[str, torch.Tensor]
) -> dict:
    """Body-frame proprioceptive change between two consecutive frames.

    Expressing the pose change in the previous frame's own body frame makes it
    invariant to the world frame, so scene-yaw augmentation leaves it alone.
    The rotation delta is a rotation vector rather than rot6d: over one control
    step it stays far from the antipodal wrap that motivates rot6d for absolute
    orientation, and three components share one scale where six do not.
    """
    available = {
        field
        for field, sources in DERIVED_PROPRIO_FIELDS.items()
        if all(source in previous and source in current for source in sources)
    }
    deltas = {}
    if {"eef_delta_pos", "eef_delta_rotvec"} & available:
        previous_rotation = Representation.rot6d_to_mat(previous["eef_rot6d"])
    if "eef_delta_pos" in available:
        translation = (current["eef_pos"] - previous["eef_pos"]).unsqueeze(-1)
        deltas["eef_delta_pos"] = (
            previous_rotation.transpose(-1, -2) @ translation
        ).squeeze(-1)
    if "eef_delta_rotvec" in available:
        relative = previous_rotation.transpose(-1, -2) @ Representation.rot6d_to_mat(
            current["eef_rot6d"]
        )
        deltas["eef_delta_rotvec"] = Representation.mat_to_rotvec(relative)
    if "gripper_qpos_delta" in available:
        deltas["gripper_qpos_delta"] = current["gripper_qpos"] - previous["gripper_qpos"]
    return {key: value.to(torch.float32) for key, value in deltas.items()}


def is_matched_key(pattern, key, reject_pattern=None):
    matched = re.search(rf"(^|_)({re.escape(pattern)})(_|$)", key)
    rejected = reject_pattern and re.search(
        rf"(^|_)({re.escape(reject_pattern)})(_|$)", key
    )
    return bool(matched and not rejected)


def canonicalize_rgb_from_uint8(image: torch.Tensor) -> torch.Tensor:
    """Canonicalize channel-first uint8 RGB without widening it."""
    if image.dtype != torch.uint8 or image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("RGB source must be channel-first uint8 with three channels")
    return image


def canonicalize_rgb_from_float01(image: torch.Tensor) -> torch.Tensor:
    """Canonicalize source float RGB to channel-first uint8."""
    if not image.dtype.is_floating_point or image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("RGB source must be channel-first floating point with three channels")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError("RGB source float values must lie in [0, 1]")
    return image.mul(255.0).round().to(torch.uint8)


def validate_canonical_rgb(image: torch.Tensor) -> None:
    """Validate canonical channel-first uint8 RGB."""
    if image.dtype != torch.uint8 or image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("expected canonical RGB: uint8, channel-first, three channels")


def validate_canonical_voxel(voxel: torch.Tensor) -> None:
    """Validate canonical uint8 occupancy and RGB voxel channels."""
    if voxel.dtype != torch.uint8 or voxel.ndim < 4 or voxel.shape[-4] != 4:
        raise ValueError("expected canonical voxel: uint8, channel-first, [occupancy,R,G,B]")


def canonicalize_voxel_from_uint8(
    voxel: torch.Tensor, *, validate_values: bool = True
) -> torch.Tensor:
    """Canonicalize uint8 occupancy and RGB voxel channels without widening."""
    if voxel.dtype != torch.uint8 or voxel.ndim < 4 or voxel.shape[-4] != 4:
        raise ValueError("voxel source must be channel-first [occupancy,R,G,B] uint8")
    if validate_values:
        occupancy = voxel.select(-4, 0)
        if not bool(((occupancy == 0) | (occupancy == 1)).all()):
            raise ValueError("voxel occupancy must be binary")
    return voxel


def canonicalize_proprio(
    observations: Mapping[str, torch.Tensor], source_keys: Mapping[str, str]
) -> dict:
    """Map adapter-resolved source proprio fields to canonical semantic names."""
    observations = dict(observations)

    pos_key = source_keys["eef_pos"]
    if pos_key is not None:
        observations["eef_pos"] = observations.pop(pos_key).to(torch.float32)

    rot_key = source_keys["eef_rot"]
    if rot_key is not None:
        rot = observations.pop(rot_key).to(torch.float32)
        observations["eef_rot6d"] = Representation.mat_to_rot6d(
            rot.reshape(*rot.shape[:-1], 3, 3)
        )

    gripper_key = source_keys["gripper_qpos"]
    if gripper_key is not None:
        observations["gripper_qpos"] = observations.pop(gripper_key).to(torch.float32)

    return observations


def canonicalize_proprio_numpy(
    observations: Mapping[str, np.ndarray], source_keys: Mapping[str, str]
) -> dict:
    """:func:`canonicalize_proprio` for NumPy batches.

    Datasets read NumPy and rollouts read torch, but a second conversion would
    be a second definition of what canonical proprioception is, free to drift
    from the first. This lifts the source fields into torch, applies the one
    implementation, and lowers the result back.
    """
    observations = dict(observations)
    source_fields = {
        key: observations.pop(key)
        for key in dict.fromkeys(source_keys.values())
        if key is not None and key in observations
    }
    canonical = canonicalize_proprio(
        {
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in source_fields.items()
        },
        source_keys,
    )
    observations.update({key: value.numpy() for key, value in canonical.items()})
    return observations


def _validate_canonical_proprio(observations, float32) -> None:
    """Check canonical proprio dtype and trailing shape, in torch or NumPy."""
    for field, shape in CANONICAL_PROPRIO_SHAPES.items():
        if field not in observations:
            continue
        value = observations[field]
        if value.dtype != float32 or tuple(value.shape[-1:]) != shape:
            raise ValueError(
                f"expected canonical {field}: float32 with trailing shape {shape}"
            )


def canonicalize_cameras(
    observations: Mapping[str, object], camera_keys: Mapping[str, str]
) -> dict:
    """Rename adapter-resolved source camera keys to canonical view names."""
    observations = dict(observations)
    for view, source_key in camera_keys.items():
        if source_key is None or source_key == view:
            continue
        observations[view] = observations.pop(source_key)
    return observations


def validate_obs(
    observations: Mapping[str, torch.Tensor],
    shape_meta_obs: Mapping[str, Mapping],
    *,
    source_keys: Mapping[str, str],
    camera_keys: Mapping[str, str],
) -> None:
    """Validate an already canonicalized observation batch."""
    canonical_name = {
        source_key: view for view, source_key in camera_keys.items() if source_key is not None
    }
    for key, field_spec in shape_meta_obs.items():
        name = canonical_name.get(key, key)
        if name not in observations:
            continue
        kind = field_spec.get("type")
        if kind == "rgb":
            validate_canonical_rgb(observations[name])
        elif kind == "voxel":
            validate_canonical_voxel(observations[name])

    _validate_canonical_names(
        observations, source_keys=source_keys, camera_keys=camera_keys
    )
    _validate_canonical_proprio(observations, torch.float32)


def _validate_canonical_names(observations, *, source_keys, camera_keys) -> None:
    for view, source_key in camera_keys.items():
        if source_key is None or source_key == view:
            continue
        if source_key in observations:
            raise ValueError(f"source camera key survived canonicalization: {source_key}")
        if view not in observations:
            raise ValueError(f"missing canonical camera field: {view}")

    for source_field, canonical_field in CANONICAL_PROPRIO_FIELDS.items():
        source_key = source_keys.get(source_field)
        if source_key is None:
            continue
        if source_key != canonical_field and source_key in observations:
            raise ValueError(f"source proprio key survived canonicalization: {source_key}")
        if canonical_field not in observations:
            raise ValueError(f"missing canonical proprio field: {canonical_field}")


def canonicalize_visuals(
    observations: Mapping[str, torch.Tensor],
    shape_meta_obs: Mapping[str, Mapping],
    *,
    canonicalize_rgb,
    validate_values: bool = True,
) -> dict:
    """Convert source visual encodings to canonical uint8 tensors."""
    observations = dict(observations)
    for key, field_spec in shape_meta_obs.items():
        if key not in observations:
            continue
        kind = field_spec.get("type")
        if kind == "rgb":
            observations[key] = canonicalize_rgb(observations[key])
        elif kind == "voxel":
            observations[key] = canonicalize_voxel_from_uint8(
                observations[key], validate_values=validate_values
            )
    return observations


def canonicalize_obs(
    observations: Mapping[str, torch.Tensor],
    shape_meta_obs: Mapping[str, Mapping],
    *,
    canonicalize_rgb,
    source_proprio_keys,
    source_camera_keys,
    validate_values: bool = True,
) -> dict:
    """Convert a source-native tensor batch to the canonical contract."""
    observations = canonicalize_visuals(
        observations,
        shape_meta_obs,
        canonicalize_rgb=canonicalize_rgb,
        validate_values=validate_values,
    )
    camera_keys = source_camera_keys(observations.keys())
    observations = canonicalize_cameras(observations, camera_keys)
    source_keys = source_proprio_keys(observations.keys())
    observations = canonicalize_proprio(observations, source_keys)
    validate_obs(
        observations, shape_meta_obs, source_keys=source_keys, camera_keys=camera_keys
    )
    return observations


def canonicalize_numpy_obs(
    observations: Mapping[str, np.ndarray],
    *,
    rgb_source_keys,
    source_proprio_keys,
    source_camera_keys,
) -> dict:
    """Convert source-native NumPy arrays to canonical physical observations."""
    canonical = dict(observations)
    for key in rgb_source_keys:
        if key not in canonical:
            continue
        image = canonical[key]
        if image.dtype != np.uint8 or image.ndim < 3 or image.shape[-3] != 3:
            raise ValueError(
                "source RGB must be channel-first uint8 with three channels"
            )

    camera_keys = source_camera_keys(canonical.keys())
    canonical = canonicalize_cameras(canonical, camera_keys)
    source_keys = source_proprio_keys(canonical.keys())

    canonical = canonicalize_proprio_numpy(canonical, source_keys)

    _validate_canonical_names(
        canonical, source_keys=source_keys, camera_keys=camera_keys
    )
    for source_key in rgb_source_keys:
        canonical_key = next(
            (view for view, matched in camera_keys.items() if matched == source_key),
            source_key,
        )
        if canonical_key not in canonical:
            continue
        image = canonical[canonical_key]
        if image.dtype != np.uint8 or image.shape[-3] != 3:
            raise ValueError("expected canonical RGB: uint8 channel-first RGB")
    _validate_canonical_proprio(canonical, np.float32)
    return canonical
