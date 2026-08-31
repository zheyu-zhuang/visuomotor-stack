"""Action/cache helpers for MimicGen datasets."""

import os
from typing import Dict, Sequence

import numpy as np
import torch

from visuomotor.data.core import actions as CoreActions
from visuomotor.data.core.observations import is_matched_key
from visuomotor.data.mimicgen.cache import load_numpy_array
from visuomotor.geometry import representation as Representation
from visuomotor.geometry import rigid as Rigid


def _float64_tensor(array: np.ndarray) -> torch.Tensor:
    """Copy a NumPy array (possibly a read-only mmap view) into a float64 tensor."""
    return torch.from_numpy(np.array(array, dtype=np.float64))


def action_to_posmat(action: np.ndarray) -> np.ndarray:
    """Convert 7D controller targets to 13D pos+rotmat+command storage."""
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"action_to_posmat expects (T, 7), got {action.shape}")
    pos = action[:, :3]
    rotvec = action[:, 3:6]
    gripper = action[:, 6:]
    rot_mat = Representation.rotvec_to_mat(_float64_tensor(rotvec))
    rot_9d = rot_mat.reshape(rot_mat.shape[0], 9).numpy()
    return np.concatenate([pos, rot_9d, gripper], axis=1)


def action_posmat_to_pos6d(action: np.ndarray) -> np.ndarray:
    """Convert 13D pos+rotmat+command storage to the 10D model layout."""
    if action.ndim != 2 or action.shape[1] != 13:
        raise ValueError(f"action_posmat_to_pos6d expects (T, 13), got {action.shape}")
    pos = action[:, :3]
    rot = action[:, 3:12]
    gripper = action[:, 12:]
    rot_6d = Representation.mat_to_rot6d(_float64_tensor(rot).reshape(-1, 3, 3)).numpy()
    return np.concatenate([pos, rot_6d, gripper], axis=1)


def absolute_posmat_to_delta_chunks(
    eef_pos: np.ndarray,
    eef_rot: np.ndarray,
    action_posmat: np.ndarray,
    horizon: int,
    *,
    episode_lengths: Sequence[int],
) -> np.ndarray:
    """Build first-frame-relative chunked delta actions from absolute pos+rotmat actions.

    ``episode_lengths`` partitions the concatenated trajectory. Each chunk stays
    inside its own episode and pads with that episode's last action, matching the
    per-episode index clamp the absolute path applies when it samples a window.
    """
    horizon = int(horizon)
    num_steps, action_dim = action_posmat.shape
    if action_dim != 13:
        raise ValueError(f"absolute_action must have 13 dims, got {action_dim}")
    if eef_pos.shape[0] != num_steps or eef_rot.shape[0] != num_steps:
        raise ValueError("eef_pos/eef_rot length mismatch with action")
    lengths = [int(length) for length in episode_lengths]
    if any(length < 1 for length in lengths):
        raise ValueError(f"episode lengths must be positive, got {lengths}")
    if sum(lengths) != num_steps:
        raise ValueError(
            "episode lengths must sum to the action length: "
            f"{sum(lengths)} != {num_steps}"
        )

    out = []
    for start, length in zip(np.cumsum([0, *lengths]).tolist(), lengths):
        end = start + length
        for i in range(start, end):
            chunk = action_posmat[i:min(i + horizon, end)]
            if chunk.shape[0] < horizon:
                pad = horizon - chunk.shape[0]
                chunk = np.concatenate([chunk, np.tile(chunk[-1:], (pad, 1))], axis=0)

            # The eef frame at step i is frame B expressed in world frame A; deltas
            # are the chunk's world poses re-expressed in that body frame.
            R_AB = _float64_tensor(eef_rot[i]).reshape(3, 3)
            t_AB = _float64_tensor(eef_pos[i]).reshape(3)
            R_BA, t_BA = Rigid.inv(R_AB, t_AB)

            pt = _float64_tensor(chunk[:, :3])
            rt = _float64_tensor(chunk[:, 3:12]).reshape(-1, 3, 3)
            gr = chunk[:, 12:]

            dp_body = Rigid.transform(R_BA, t_BA, pt)
            r_rel = Rigid.transform_rotation(R_BA, rt)
            rot_delta = Representation.mat_to_rot6d(r_rel)

            delta = np.concatenate([dp_body.numpy(), rot_delta.numpy(), gr], axis=1)
            out.append(delta.reshape(1, -1))

    return np.concatenate(out, axis=0).astype(np.float32)


def load_action_array(
    *,
    cache_dir: str,
    meta: Dict,
    horizon: int,
    action_dim: int,
    action_rep: str,
) -> np.ndarray:
    """Load action array for the selected action representation."""
    action_rep = CoreActions.validate_action_rep(action_rep)
    if action_rep == "delta":
        return load_delta_action(
            cache_dir=cache_dir,
            meta=meta,
            horizon=horizon,
            action_dim=action_dim,
        ).astype(np.float32, copy=False)

    action = load_numpy_array(
        cache_dir, os.path.join("action", f"{action_rep}_action.npy")
    )
    action = action_posmat_to_pos6d(action)
    if action.shape[1] != action_dim:
        raise ValueError(
            "absolute model action dim does not match the configured contract: "
            f"{action.shape[1]} != {action_dim}"
        )
    return action.astype(np.float32, copy=False)


class MimicGenActionAdapter:
    """Load and sample one configured target representation from a trajectory."""

    def __init__(
        self,
        *,
        cache_dir: str,
        meta: Dict,
        horizon: int,
        action_dim: int,
        action_rep: str,
    ) -> None:
        self.cache_dir = cache_dir
        self.meta = meta
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.action_rep = CoreActions.validate_action_rep(action_rep)

    def load(self) -> np.ndarray:
        return load_action_array(
            cache_dir=self.cache_dir,
            meta=self.meta,
            horizon=self.horizon,
            action_dim=self.action_dim,
            action_rep=self.action_rep,
        )

    def sample_window(
        self,
        action: np.ndarray,
        *,
        observation_index: int,
        action_indices,
    ) -> np.ndarray:
        """Select and shape the target window for the configured representation."""
        if self.action_rep == "delta":
            window = action[int(observation_index)]
        else:
            window = action[np.asarray(action_indices, dtype=np.int64)]
        return window.reshape(self.horizon, -1)


def load_delta_action(
    *,
    cache_dir: str,
    meta: Dict,
    horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Load precomputed chunked delta actions under the exact per-step contract."""
    rel_path = os.path.join("action", f"delta_action_h{int(horizon)}.npy")
    try:
        action = load_numpy_array(cache_dir, rel_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing precomputed delta-action cache "
            f"{rel_path!r}. Rebuild rerender/merged cache with "
            f"--delta-horizons {int(horizon)} or train with action_rep=absolute."
        ) from exc
    if action.ndim != 2:
        raise ValueError(f"delta-action cache must be 2D, got shape {action.shape}")
    if action.shape[1] % horizon != 0:
        raise ValueError(
            "delta-action cache width must be divisible by the requested "
            f"horizon, got width={action.shape[1]} horizon={horizon}"
        )

    per_step_dim = action.shape[1] // horizon
    if per_step_dim != action_dim:
        raise ValueError(
            "delta action per-step dim does not match the configured action dim: "
            f"{per_step_dim} != {action_dim}"
        )

    return action.reshape(-1, horizon * action_dim)
