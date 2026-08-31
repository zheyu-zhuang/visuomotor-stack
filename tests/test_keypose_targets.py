import numpy as np
import torch

from visuomotor.data.core.keypose_targets import (
    focus_pose_position_samples,
    reference_frame_poses,
)
from visuomotor.geometry.representation import mat_to_rot6d as matrix_to_rotation_6d


def test_reference_frame_poses_extracts_position_and_rotation6d():
    position = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rotation_flat = np.tile(np.eye(3, dtype=np.float32).reshape(-1), (2, 1))
    poses = reference_frame_poses(position, rotation_flat, index=np.array([1, 0]))
    np.testing.assert_allclose(poses[:, :3], position[[1, 0]])
    expected_rot6d = matrix_to_rotation_6d(torch.eye(3)).numpy()
    np.testing.assert_allclose(poses[:, 3:], np.tile(expected_rot6d, (2, 1)), atol=1e-6)


def test_focus_pose_position_samples_concatenates_first_and_last():
    first = np.ones((2, 3), dtype=np.float32)
    last = np.zeros((2, 3), dtype=np.float32)
    pooled = focus_pose_position_samples(first, last)
    assert pooled.shape == (4, 3)
    np.testing.assert_array_equal(pooled[:2], first)
    np.testing.assert_array_equal(pooled[2:], last)
