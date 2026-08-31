"""RVT2Heatmap heuristic labels, keypoints, and projection helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from visuomotor.geometry.projection import world_xyz_to_pixel_row_col


def gripper_open_from_signal(gripper: np.ndarray) -> np.ndarray:
    """Convert continuous or binary gripper observations to open/closed booleans."""
    g = np.asarray(gripper)
    if g.ndim == 0:
        raise ValueError("gripper must contain one value per timestep")
    if g.ndim > 2:
        g = g.reshape(g.shape[0], -1)

    if g.ndim == 2 and g.shape[1] == 2:
        signal = np.abs(g[:, 0] - g[:, 1])
    elif g.ndim == 2:
        signal = g[:, 0]
    else:
        signal = g

    signal = np.asarray(signal).reshape(-1)
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        raise ValueError("gripper signal has no finite values")

    uniq = np.unique(finite)
    if uniq.size <= 2:
        return signal > float(np.mean(uniq))

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if np.isclose(lo, hi):
        return np.ones(signal.shape[0], dtype=bool)
    return signal > (lo + hi) * 0.5


def discover_rvt2_heatmap_keypoints(
    joint_velocities: np.ndarray,
    gripper_open: Union[Sequence[bool], np.ndarray],
    *,
    atol: float = 0.1,
    stopped_buffer_len: int = 4,
    include_gripper_changes: bool = True,
    include_final: bool = True,
    mute_initial_gripper_open: bool = False,
    remove_adjacent: bool = True,
) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    """
    Discover keypoints with the fixed heuristic used by the RVT2Heatmap baseline.

    A timestep is selected when the gripper state changes, the robot has stopped
    while the gripper state is stable around that frame, or it is the final frame.
    """
    velocities = np.asarray(joint_velocities, dtype=np.float32)
    if velocities.ndim == 1:
        velocities = velocities[:, None]
    if velocities.ndim != 2:
        raise ValueError(f"joint_velocities must be [T,D], got {velocities.shape}")

    gripper = np.asarray(gripper_open)
    if gripper.dtype != np.bool_:
        gripper = gripper_open_from_signal(gripper)
    else:
        gripper = gripper.reshape(-1)

    T = int(velocities.shape[0])
    if gripper.shape[0] != T:
        raise ValueError(
            f"gripper_open length mismatch: got {gripper.shape[0]}, expected {T}"
        )
    if T == 0:
        return np.empty((0,), dtype=int), {}

    keypoints: List[int] = []
    reasons: Dict[int, List[str]] = {}
    prev_gripper_open = bool(gripper[0])
    stopped_buffer = 0

    for i in range(T):
        gripper_stable = (
            i < T - 2
            and bool(gripper[i]) == bool(gripper[i + 1])
            and bool(gripper[i]) == bool(gripper[max(0, i - 1)])
            and bool(gripper[max(0, i - 2)]) == bool(gripper[max(0, i - 1)])
        )
        stopped = (
            stopped_buffer <= 0
            and np.allclose(velocities[i], 0.0, atol=float(atol))
            and gripper_stable
            and i != T - 2
        )

        if stopped:
            stopped_buffer = int(stopped_buffer_len)
        else:
            stopped_buffer -= 1

        last = include_final and i == T - 1
        changed = bool(gripper[i]) != prev_gripper_open

        if i != 0 and ((include_gripper_changes and changed) or stopped or last):
            keypoints.append(i)
            frame_reasons = []
            if include_gripper_changes and changed:
                frame_reasons.append("gripper")
            if stopped:
                frame_reasons.append("stopped")
            if last:
                frame_reasons.append("final")
            reasons[i] = frame_reasons

        prev_gripper_open = bool(gripper[i])

    if remove_adjacent and len(keypoints) >= 2 and keypoints[-1] == keypoints[-2] + 1:
        removed = keypoints.pop(-2)
        reasons.pop(removed, None)

    if mute_initial_gripper_open:
        for keypoint in list(keypoints):
            frame_reasons = reasons.get(int(keypoint), [])
            if frame_reasons == ["gripper"] and bool(gripper[int(keypoint)]):
                keypoints.remove(keypoint)
                reasons.pop(int(keypoint), None)
                break

    return np.asarray(keypoints, dtype=int), reasons


JOINT_VELOCITY_KEY = "robot0_joint_vel"


def _project_world_xyz_to_row_col(
    xyz: np.ndarray,
    world_to_pixel: np.ndarray,
    image_size: int,
) -> Optional[np.ndarray]:
    """Snap one projected world point to an integer pixel, or ``None`` if unprojectable."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(3)
    matrix = np.asarray(world_to_pixel, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    if not np.isfinite(xyz).all():
        return None

    row_col = world_xyz_to_pixel_row_col(
        torch.from_numpy(xyz), torch.from_numpy(matrix), image_size
    ).numpy()
    if not np.isfinite(row_col).all():
        return None
    return np.clip(np.round(row_col), 0, image_size - 1).astype(np.float32)
