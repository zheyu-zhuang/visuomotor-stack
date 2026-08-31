"""Pure numpy helpers building local-policy dataset targets from keypose segments.

Kept dataset-machinery-free (plain arrays in, plain arrays out) so the
geometry is unit-testable without a rendered LMDB cache;
the MimicGen target and normalization adapters wire them into sample and fitted
model fields.
"""

from __future__ import annotations

import numpy as np
import torch

from visuomotor.geometry import representation as Representation

FOCUS_POSE_POS_KEY = "focus_pose_pos"


def reference_frame_poses(
    position: np.ndarray, rotation_matrix_flat: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """``[pos(3), rotation_6d(6)]`` poses at ``index`` into the per-step arrays."""
    pos = np.asarray(position)[index].astype(np.float32, copy=False)
    rot = np.asarray(rotation_matrix_flat)[index].reshape(-1, 3, 3).astype(np.float32, copy=False)
    rot6d = Representation.mat_to_rot6d(torch.from_numpy(rot)).numpy()
    return np.concatenate((pos, rot6d), axis=-1)


def focus_pose_position_samples(
    first_position: np.ndarray, last_position: np.ndarray
) -> np.ndarray:
    """Pooled absolute reference-frame positions, for fitting :data:`FOCUS_POSE_POS_KEY`."""
    return np.concatenate((first_position, last_position), axis=0).astype(np.float32, copy=False)
