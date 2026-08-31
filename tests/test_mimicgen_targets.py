import numpy as np
import pytest
import torch

from visuomotor.data.core import normalization as CoreNormalization
from visuomotor.data.core.keypose_targets import (
    FOCUS_POSE_POS_KEY,
)
from visuomotor.data.core.scene_augmentation import SceneYawAugmentationConfig
from visuomotor.data.mimicgen import dataset as MimicgenDataset
from visuomotor.data.mimicgen import normalization as MimicgenNormalization
from visuomotor.data.mimicgen import targets as MimicgenTargets
from visuomotor.geometry.representation import mat_to_rot6d as matrix_to_rotation_6d


def _bare_dataset(**overrides):
    """A dataset with only the attributes the target adapter methods touch."""
    dataset = MimicgenDataset.MimicGenDataset.__new__(MimicgenDataset.MimicGenDataset)
    n = 6
    identity_rot_flat = np.tile(np.eye(3, dtype=np.float32).reshape(-1), (n, 1))
    dataset.pos_key = "robot0_eef_pos"
    dataset.rot_key = "robot0_eef_rot"
    dataset.lowdim = {
        "robot0_eef_pos": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        "robot0_eef_rot": identity_rot_flat,
    }
    dataset.target_adapter = MimicgenTargets.MimicGenKeyposeTargetAdapter(
        first_indices=np.array([0, 0, 0, 3, 3, 3]),
        last_indices=np.array([3, 3, 3, 5, 5, 5]),
        valid=np.ones(n, dtype=bool),
    )
    dataset.action_rep = "absolute"
    dataset.scene_yaw_augmentation = SceneYawAugmentationConfig()
    for key, value in overrides.items():
        setattr(dataset, key, value)
    return dataset


def test_attention_target_fields_shapes_and_values():
    dataset = _bare_dataset()
    global_obs_indices = np.array([0, 4])
    fields = dataset.target_adapter.fields(dataset, global_obs_indices)

    assert fields["focus_target_pos"].shape == (2, 3)
    assert fields["focus_target_valid"].shape == (2,)

    np.testing.assert_allclose(
        fields["focus_target_pos"][1], dataset.lowdim["robot0_eef_pos"][5]
    )

def test_fit_attention_normalizer_produces_a_finite_range_normalizer():
    action_dim = 10
    robot_action = np.zeros((6, action_dim), dtype=np.float32)
    dataset = _bare_dataset(active_step_indices=np.arange(6))
    robot_action[:, :3] = dataset.lowdim["robot0_eef_pos"] + 0.1
    robot_action[:, 3:9] = matrix_to_rotation_6d(torch.eye(3)).numpy()
    dataset.action = robot_action

    normalizer = CoreNormalization.Normalizer()
    MimicgenNormalization.fit_target_fields(
        dataset, normalizer, mask=None, robot_id=None
    )
    normalizer.finalize()

    assert normalizer.has(FOCUS_POSE_POS_KEY)
    sample = torch.zeros(1, 3)
    assert torch.isfinite(normalizer.normalize(FOCUS_POSE_POS_KEY, sample)).all()


def test_construction_requires_absolute_action_rep():
    with pytest.raises(ValueError, match="absolute"):
        MimicgenDataset.MimicGenDataset.__init__(
            MimicgenDataset.MimicGenDataset.__new__(MimicgenDataset.MimicGenDataset),
            shape_meta={},
            dataset_path="unused",
            image_size=84,
            action_rep="delta",
            keypose_targets=True,
        )


def test_target_adapter_uses_commanded_events_shifted_by_measured_valley():
    n = 10
    dataset = _bare_dataset()
    dataset.action = np.zeros((n, 10), dtype=np.float32)
    dataset.action[:, -1] = np.array([-1, -1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    opening = np.array(
        [0.08, 0.08, 0.08, 0.07, 0.05, 0.04, 0.0399, 0.03985, 0.0398, 0.0398],
        dtype=np.float32,
    )
    dataset.gripper_key = "robot0_gripper_qpos"
    dataset.lowdim[dataset.gripper_key] = np.stack([opening / 2, -opening / 2], axis=-1)
    dataset.cum_lengths_all = np.array([0, n], dtype=np.int64)

    adapter = MimicgenTargets.MimicGenKeyposeTargetAdapter.from_dataset(
        dataset,
        gripper_motion_threshold=5e-4,
        gripper_valley_threshold=2e-4,
        gripper_valley_window=4,
    )

    assert adapter.command_keyframes_per_episode[0].tolist() == [6]
    assert adapter.last_indices[6] == 9
