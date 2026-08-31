"""Keypose (keyframe) segmentation from commanded gripper / velocity signals."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks


def segment_settled_gripper_commands(
    gripper_command: np.ndarray,
    gripper_qpos: np.ndarray,
    episode_ends: Sequence[int],
    *,
    command_threshold: float = 1e-3,
    motion_threshold: float = 5e-4,
    valley_threshold: float = 2e-4,
    valley_window: int = 4,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Place every gripper command transition at its measured motion valley.

    Command transitions determine how many mandatory keyposes exist. Measured
    finger motion only moves each keypose within the interval governed by that
    command, so opening and closing events cannot be dropped or merged.

    Returns ``(keyframes, command_keyframes)``. Both contain episode-local
    indices; ``keyframes`` additionally contains the episode boundaries.
    """
    command = np.asarray(gripper_command, dtype=np.float32).reshape(-1)
    qpos = np.asarray(gripper_qpos, dtype=np.float32)
    ends = np.asarray(episode_ends, dtype=np.int64)
    valley_window = int(valley_window)
    if qpos.ndim != 2 or qpos.shape[1] != 2:
        raise ValueError(f"gripper_qpos must have shape [T,2], got {qpos.shape}")
    if command.shape[0] != qpos.shape[0]:
        raise ValueError("gripper command and qpos lengths must match")
    if ends.size and int(ends[-1]) != command.shape[0]:
        raise ValueError("episode_ends must terminate at the gripper signal length")
    if valley_window < 1:
        raise ValueError("valley_window must be positive")
    if min(command_threshold, motion_threshold, valley_threshold) <= 0:
        raise ValueError("gripper thresholds must be positive")
    if not np.isfinite(command).all() or not np.isfinite(qpos).all():
        raise ValueError("gripper command and qpos must be finite")

    keyframes_per_episode = []
    command_keyframes_per_episode = []
    start = 0
    for end_value in ends.tolist():
        end = int(end_value)
        length = end - start
        if length <= 0:
            raise ValueError(f"episode has non-positive length: {length}")

        local_command = command[start:end]
        local_qpos = qpos[start:end]
        changes = (
            np.flatnonzero(np.abs(np.diff(local_command)) > float(command_threshold))
            + 1
        )
        interval_ends = np.concatenate(
            [changes[1:], np.asarray([length], dtype=np.int64)]
        )

        opening = np.abs(local_qpos[:, 0] - local_qpos[:, 1])
        opening_change = np.zeros(length, dtype=np.float32)
        opening_change[1:] = np.abs(np.diff(opening))
        command_keyframes = []
        for change, interval_end in zip(changes.tolist(), interval_ends.tolist()):
            motion_candidates = np.flatnonzero(
                opening_change[change + 1 : interval_end] > float(motion_threshold)
            )
            settled = None
            if motion_candidates.size:
                motion_start = change + 1 + int(motion_candidates[0])
                latest_start = interval_end - valley_window
                for valley_start in range(motion_start + 1, latest_start + 1):
                    valley = opening_change[valley_start : valley_start + valley_window]
                    if np.all(valley <= float(valley_threshold)):
                        settled = valley_start
                        break
            if settled is None:
                settled = interval_end - 1
            command_keyframes.append(int(settled))

        command_keyframes = np.asarray(command_keyframes, dtype=np.int64)
        if command_keyframes.size:
            if np.any(np.diff(command_keyframes) <= 0):
                raise RuntimeError("command keyposes must be strictly increasing")
            if np.any(command_keyframes < changes) or np.any(
                command_keyframes >= interval_ends
            ):
                raise RuntimeError("command keypose escaped its command interval")

        keyframes = np.unique(
            np.concatenate(
                [
                    np.asarray([0, length - 1], dtype=np.int64),
                    command_keyframes,
                ]
            )
        )
        keyframes_per_episode.append(keyframes)
        command_keyframes_per_episode.append(command_keyframes)
        start = end

    return keyframes_per_episode, command_keyframes_per_episode


def segment_velocity_scipy(
    gripper_command: np.ndarray,
    episode_ends: Sequence[int],
    *,
    use_gripper: bool = True,
    use_vel: bool = False,
    velocity: Optional[np.ndarray] = None,
    gripper_threshold: float = 1e-3,
    vel_threshold: float = 1e-3,
) -> List[np.ndarray]:
    """Per-episode keyframe indices (local to each episode).

    Every episode always has ``{0, length-1}`` as keyframes. When
    ``use_gripper``, an index is also a keyframe wherever the *commanded*
    (not measured) gripper action changes by more than ``gripper_threshold``
    since the previous step. When ``use_vel``, an index is also a keyframe
    wherever ``velocity`` has a local minimum at or below ``vel_threshold``
    (a pause in motion).
    """
    gripper_command = np.asarray(gripper_command).reshape(-1)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    keyframes_per_episode = []
    start = 0
    for end in episode_ends.tolist():
        length = int(end) - start
        if length <= 0:
            raise ValueError(f"episode has non-positive length: {length}")
        local_keyframes = {0, length - 1}
        if use_gripper and length > 1:
            command = gripper_command[start:end]
            changes = np.flatnonzero(np.abs(np.diff(command)) > gripper_threshold) + 1
            local_keyframes.update(changes.tolist())
        if use_vel and velocity is not None and length > 2:
            speed = np.abs(np.asarray(velocity[start:end]).reshape(length, -1)).sum(
                axis=-1
            )
            minima, _ = find_peaks(-speed, height=-vel_threshold)
            local_keyframes.update(minima.tolist())
        keyframes_per_episode.append(
            np.asarray(sorted(local_keyframes), dtype=np.int64)
        )
        start = end
    return keyframes_per_episode


def build_keypose_segments(
    gripper_command: np.ndarray,
    episode_ends: Sequence[int],
    segment_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every global timestep: the keypose interval it targets and its progress through it.

    Returns ``(first_index, last_index, progress, valid)``, each length
    ``episode_ends[-1]`` (or 0 if there are no episodes); ``valid`` is
    ``False`` only for degenerate (single-frame) episodes.
    """
    keyframes_per_episode = segment_velocity_scipy(
        gripper_command, episode_ends, **(segment_kwargs or {})
    )
    return build_keypose_segments_from_keyframes(
        keyframes_per_episode,
        episode_ends,
    )


def build_keypose_segments_from_keyframes(
    keyframes_per_episode: Sequence[np.ndarray],
    episode_ends: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-frame segment targets from preselected episode keyframes."""
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    if len(keyframes_per_episode) != len(episode_ends):
        raise ValueError("keyframe and episode counts must match")
    total = int(episode_ends[-1]) if len(episode_ends) else 0
    first_index = np.zeros(total, dtype=np.int64)
    last_index = np.zeros(total, dtype=np.int64)
    progress = np.zeros(total, dtype=np.float32)
    valid = np.zeros(total, dtype=bool)

    start = 0
    for keyframes, end in zip(keyframes_per_episode, episode_ends):
        end = int(end)
        length = end - start
        if length <= 1:
            first_index[start:end] = start
            last_index[start:end] = start
            valid[start:end] = False
            start = end
            continue
        keyframes = np.asarray(keyframes, dtype=np.int64)
        if keyframes.ndim != 1 or keyframes.size < 2:
            raise ValueError("non-degenerate episodes need at least two keyframes")
        if keyframes[0] != 0 or keyframes[-1] != length - 1:
            raise ValueError("episode keyframes must include both boundaries")
        if np.any(np.diff(keyframes) <= 0):
            raise ValueError("episode keyframes must be strictly increasing")
        for local_t in range(length):
            segment_index = (
                int(np.searchsorted(keyframes, local_t, side="right")) - 1
            )
            segment_index = max(0, min(segment_index, len(keyframes) - 2))
            local_first = int(keyframes[segment_index])
            local_last = int(keyframes[segment_index + 1])
            span = max(1, local_last - local_first)
            global_t = start + local_t
            first_index[global_t] = start + local_first
            last_index[global_t] = start + local_last
            progress[global_t] = np.clip((local_t - local_first) / span, 0.0, 1.0)
            valid[global_t] = True
        start = end
    return first_index, last_index, progress, valid
