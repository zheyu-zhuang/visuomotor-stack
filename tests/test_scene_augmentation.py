import math

import pytest
import torch
from torch.nn import functional as F

from visuomotor.data.core.scene_augmentation import (
    FocusTargets,
    SceneYawAugmentationConfig,
    SceneYawAugmenter,
    SceneYawObsActionAugmentor,
    WorldSceneBatch,
    _rotate_points,
    _rotate_voxels,
    randomize_pose,
    yaw_matrices,
)
from visuomotor.data.mimicgen.observations import (
    fixed_camera_rgb_keys,
    source_proprio_keys,
)
from visuomotor.geometry.bounds import PlanarWorkspace
from visuomotor.geometry.grid import SourceVoxelGeometry
from visuomotor.geometry.representation import mat_to_rot6d as matrix_to_rotation_6d

_VOXEL_SHAPE_META = {
    "action": {"shape": [10]},
    "obs": {
        "voxel": {"type": "voxel", "shape": [4, 16, 16, 16]},
        "robot0_eef_pos": {"type": "low_dim", "shape": [3]},
        "robot0_eef_rot": {"type": "low_dim", "shape": [9]},
        "robot0_eye_in_hand_image": {"type": "rgb", "shape": [3, 84, 84]},
    },
}


_VOXEL_SOURCE_KEYS = source_proprio_keys(_VOXEL_SHAPE_META["obs"].keys())
_VOXEL_FIXED_CAMERA_RGB_KEYS = fixed_camera_rgb_keys(_VOXEL_SHAPE_META["obs"])


def _enabled_config(**overrides):
    cfg = {
        "enable": True,
        "min_deg": 90.0,
        "max_deg": 90.0,
        "max_attempts": 1,
        "identity_probability": 0.0,
        "workspace": {"center_xy": [0.0, 0.0], "size": 2.0},
    }
    cfg.update(overrides)
    return cfg


def _voxel_batch(batch=2, steps=1, size=16):
    voxels = torch.zeros(batch, steps, 4, size, size, size, dtype=torch.uint8)
    voxels[:, :, 0, size // 2 + 4, size // 2, size // 2] = 1
    return {
        "obs": {
            "voxel": voxels,
            "robot0_eef_pos": torch.tensor([0.1, 0.0, 0.5]).expand(batch, steps, 3).clone(),
            "robot0_eef_rot": torch.eye(3).reshape(1, 1, 9).expand(batch, steps, 9).clone(),
            "robot0_eye_in_hand_image": torch.zeros(batch, steps, 3, 84, 84, dtype=torch.uint8),
        },
        "action": torch.cat(
            [
                torch.tensor([0.1, 0.0, 0.5, 1, 0, 0, 0, 1, 0]).expand(batch, 9).clone(),
                torch.zeros(batch, 1),
            ],
            dim=-1,
        ),
    }


def _scene(batch=2, steps=3):
    actions = torch.zeros(batch, steps, 10)
    actions[..., :3] = torch.tensor([0.1, 0.1, 0.5])
    actions[..., 3:9] = matrix_to_rotation_6d(torch.eye(3)).reshape(1, 1, 6).expand(batch, steps, 6)
    obs = {
        "eef_pos": actions[..., :3].clone(),
        "eef_rot": torch.eye(3).reshape(1, 1, 9).expand(batch, steps, 9).clone(),
    }
    return WorldSceneBatch(
        obs=obs,
        actions=actions,
        focus=FocusTargets(pos=actions[..., :3].clone(), valid=torch.ones(batch, steps, dtype=torch.bool)),
        reference_frame={"last": actions[..., :9].clone().unsqueeze(2)},
        progress=torch.zeros(batch, steps, 1),
    )


def test_yaw_matrices_rotate_a_point_as_expected():
    rotation = yaw_matrices(torch.tensor([math.pi / 2]))
    point = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = torch.einsum("bij,bj->bi", rotation, point)
    torch.testing.assert_close(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=1e-6)


def test_disabled_augmenter_is_identity():
    workspace = PlanarWorkspace(center_xy=(0.0, 0.0), size=1.2)
    augmenter = SceneYawAugmenter(workspace, enabled=False)
    scene = _scene()
    result = augmenter(scene)
    torch.testing.assert_close(result.scene.actions, scene.actions)
    assert result.metrics.identity_fraction == 1.0
    assert result.metrics.fallback_fraction == 0.0


def test_augmenter_rotates_actions_by_the_given_angle():
    workspace = PlanarWorkspace(center_xy=(0.0, 0.0), size=1.2)
    augmenter = SceneYawAugmenter(workspace, enabled=True, identity_probability=0.0)
    scene = _scene()
    angles = torch.tensor([math.pi / 2, math.pi])
    result = augmenter(scene, angles=angles)
    torch.testing.assert_close(
        result.scene.actions[0, :, :3], torch.tensor([-0.1, 0.1, 0.5]).expand(3, 3), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        result.scene.actions[1, :, :3], torch.tensor([-0.1, -0.1, 0.5]).expand(3, 3), atol=1e-5, rtol=1e-5
    )


def test_augmenter_bounds_contain_every_sampled_rotation_of_the_action_position():
    workspace = PlanarWorkspace(center_xy=(0.0, 0.0), size=1.2)
    augmenter = SceneYawAugmenter(workspace, enabled=True, identity_probability=0.0)
    scene = _scene()
    bounds = augmenter.bounds(scene)
    lo, hi = bounds["action_pos"]
    for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 1.7):
        rotated = augmenter(scene, angles=torch.full((scene.actions.shape[0],), angle)).scene
        position = rotated.actions[..., :3]
        assert bool((position >= lo - 1e-5).all())
        assert bool((position <= hi + 1e-5).all())


def test_augmenter_bounds_use_exact_workspace_xy_limits():
    workspace = PlanarWorkspace(center_xy=(0.2, -0.1), size=1.2)
    augmenter = SceneYawAugmenter(workspace, enabled=True, identity_probability=0.0)
    scene = _scene()

    lo, hi = augmenter.bounds(scene)["action_pos"]

    torch.testing.assert_close(lo[..., :2], torch.tensor([-0.4, -0.7]).expand_as(lo[..., :2]))
    torch.testing.assert_close(hi[..., :2], torch.tensor([0.8, 0.5]).expand_as(hi[..., :2]))
    torch.testing.assert_close(lo[..., 2], scene.actions[..., 2])
    torch.testing.assert_close(hi[..., 2], scene.actions[..., 2])


def test_augmenter_falls_back_to_identity_when_no_candidate_fits():
    # A workspace far too small for the scripted 0.1m offset action to stay inside after
    # any rotation forces every candidate to be rejected.
    workspace = PlanarWorkspace(center_xy=(0.0, 0.0), size=0.01)
    augmenter = SceneYawAugmenter(
        workspace, enabled=True, identity_probability=0.0, max_attempts=4
    )
    scene = _scene()
    result = augmenter(scene)
    torch.testing.assert_close(result.scene.actions, scene.actions)
    assert result.metrics.fallback_fraction == 1.0


def test_rotate_voxels_zero_degrees_is_identity():
    voxels = torch.rand(1, 2, 4, 6, 6, 5)
    rotation = yaw_matrices(torch.zeros(1))
    torch.testing.assert_close(_rotate_voxels(voxels, rotation), voxels)


def test_rotate_voxels_moves_a_marker_as_expected_for_a_90_degree_yaw():
    voxels = torch.zeros(1, 1, 4, 16, 16, 16)
    voxels[0, 0, 0, 12, 8, 8] = 1.0
    rotated = _rotate_voxels(voxels, yaw_matrices(torch.tensor([math.pi / 2])))
    nonzero = rotated[0, 0, 0].nonzero()
    assert nonzero.shape[0] == 1
    assert nonzero[0].tolist() == [7, 12, 8]


def test_rotate_voxels_matches_equidiff_affine_grid_convention():
    torch.manual_seed(3)
    voxels = torch.randint(0, 5, (4, 1, 4, 16, 16, 16)).float()
    rotation = yaw_matrices(torch.tensor([-2.3, -0.7, 0.4, 1.9]))

    batch, steps, channels, size_x, size_y, size_z = voxels.shape
    source = voxels.permute(0, 1, 2, 5, 4, 3).flip((3, 4))
    source = source.reshape(batch, steps * channels * size_z, size_y, size_x)
    grid = F.affine_grid(rotation[:, :2], source.size(), align_corners=True)
    expected = F.grid_sample(
        source, grid, align_corners=True, mode="nearest"
    )
    expected = expected.reshape(batch, steps, channels, size_z, size_y, size_x)
    expected = expected.flip((3, 4)).permute(0, 1, 2, 5, 4, 3)

    torch.testing.assert_close(_rotate_voxels(voxels, rotation), expected)


def test_randomize_pose_stays_within_bounds():
    rotation = torch.eye(3).unsqueeze(0).expand(5, 3, 3).clone()
    position = torch.zeros(5, 3)
    new_rotation, new_position = randomize_pose(
        rotation, position, translation_m=0.05, rotation_deg=10.0
    )
    assert new_position.abs().max() <= 0.05 + 1e-6
    # Determinant 1 and orthogonal: still a valid rotation matrix.
    torch.testing.assert_close(
        new_rotation @ new_rotation.transpose(-1, -2), torch.eye(3).expand(5, 3, 3), atol=1e-5, rtol=1e-5
    )


def test_rotate_voxels_preserves_uint8_dtype_and_binary_occupancy():
    voxels = torch.zeros(1, 1, 4, 16, 16, 16, dtype=torch.uint8)
    voxels[0, 0, 0, 12, 8, 8] = 1
    rotated = _rotate_voxels(voxels, yaw_matrices(torch.tensor([math.pi / 2])))
    assert rotated.dtype == torch.uint8
    assert set(rotated[0, 0, 0].unique().tolist()) <= {0, 1}
    nonzero = rotated[0, 0, 0].nonzero()
    assert nonzero.shape[0] == 1
    assert nonzero[0].tolist() == [7, 12, 8]


def test_scene_yaw_config_from_config_round_trips():
    config = SceneYawAugmentationConfig.from_config(_enabled_config())
    assert config.enable is True
    assert config.workspace_center_xy == (0.0, 0.0)
    assert config.workspace_size == 2.0
    restored = SceneYawAugmentationConfig.from_config(config.as_dict())
    assert restored == config


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_deg": 10.0, "max_deg": -10.0},
        {"max_attempts": 0},
        {"identity_probability": 1.5},
        {"workspace": {"center_xy": [0.0, 0.0], "size": 0.0}},
        {"workspace": None},
    ],
)
def test_scene_yaw_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        SceneYawAugmentationConfig.from_config(_enabled_config(**overrides))


def test_scene_yaw_obs_action_augmentor_is_identity_when_disabled():
    aug = SceneYawObsActionAugmentor(
        shape_meta=_VOXEL_SHAPE_META,
        action_rep="absolute",
        config=None,
        source_keys=_VOXEL_SOURCE_KEYS,
        fixed_camera_rgb_keys=_VOXEL_FIXED_CAMERA_RGB_KEYS,
    )
    assert aug.enabled is False
    batch = _voxel_batch()
    before = {key: value.clone() for key, value in batch["obs"].items()}
    aug(batch)
    for key, value in before.items():
        torch.testing.assert_close(batch["obs"][key], value)


def test_scene_yaw_obs_action_augmentor_rotates_obs_action_and_voxel_consistently():
    aug = SceneYawObsActionAugmentor(
        shape_meta=_VOXEL_SHAPE_META,
        action_rep="absolute",
        config=_enabled_config(),
        source_keys=_VOXEL_SOURCE_KEYS,
        fixed_camera_rgb_keys=_VOXEL_FIXED_CAMERA_RGB_KEYS,
    )
    batch = _voxel_batch()
    wrist = batch["obs"]["robot0_eye_in_hand_image"].clone()
    aug(batch)
    # 90 degree yaw about the origin: (x, y) -> (-y, x).
    torch.testing.assert_close(
        batch["obs"]["robot0_eef_pos"][0, 0], torch.tensor([0.0, 0.1, 0.5]), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        batch["action"][0, :3], torch.tensor([0.0, 0.1, 0.5]), atol=1e-5, rtol=1e-5
    )
    nonzero = batch["obs"]["voxel"][0, 0, 0].nonzero()
    assert nonzero.shape[0] == 1
    assert batch["obs"]["voxel"].dtype == torch.uint8
    torch.testing.assert_close(batch["obs"]["robot0_eye_in_hand_image"], wrist)


def test_scene_yaw_obs_action_augmentor_rejects_fixed_camera_rgb():
    shape_meta = dict(_VOXEL_SHAPE_META)
    shape_meta["obs"] = dict(shape_meta["obs"])
    shape_meta["obs"]["agentview_image"] = {"type": "rgb", "shape": [3, 84, 84]}
    with pytest.raises(ValueError, match="fixed-camera"):
        SceneYawObsActionAugmentor(
            shape_meta=shape_meta,
            action_rep="absolute",
            config=_enabled_config(),
            source_keys=source_proprio_keys(shape_meta["obs"].keys()),
            fixed_camera_rgb_keys=fixed_camera_rgb_keys(shape_meta["obs"]),
        )


def test_scene_yaw_obs_action_augmentor_permits_wrist_rgb():
    SceneYawObsActionAugmentor(
        shape_meta=_VOXEL_SHAPE_META,
        action_rep="absolute",
        config=_enabled_config(),
        source_keys=_VOXEL_SOURCE_KEYS,
        fixed_camera_rgb_keys=_VOXEL_FIXED_CAMERA_RGB_KEYS,
    )


def test_scene_yaw_obs_action_augmentor_rejects_delta_actions():
    with pytest.raises(ValueError, match="absolute"):
        SceneYawObsActionAugmentor(
            shape_meta=_VOXEL_SHAPE_META,
            action_rep="delta",
            config=_enabled_config(),
            source_keys=_VOXEL_SOURCE_KEYS,
            fixed_camera_rgb_keys=_VOXEL_FIXED_CAMERA_RGB_KEYS,
        )


def test_scene_yaw_obs_action_augmentor_requires_a_voxel_field():
    shape_meta = {
        "action": {"shape": [10]},
        "obs": {
            "robot0_eef_pos": {"type": "low_dim", "shape": [3]},
            "robot0_eef_rot": {"type": "low_dim", "shape": [9]},
            "robot0_eye_in_hand_image": {"type": "rgb", "shape": [3, 84, 84]},
        },
    }
    with pytest.raises(ValueError, match="voxel"):
        SceneYawObsActionAugmentor(
            shape_meta=shape_meta,
            action_rep="absolute",
            config=_enabled_config(),
            source_keys=source_proprio_keys(shape_meta["obs"].keys()),
            fixed_camera_rgb_keys=fixed_camera_rgb_keys(shape_meta["obs"]),
        )


def test_scene_yaw_obs_action_augmentor_handles_optional_focus_target():
    aug = SceneYawObsActionAugmentor(
        shape_meta=_VOXEL_SHAPE_META,
        action_rep="absolute",
        config=_enabled_config(),
        source_keys=_VOXEL_SOURCE_KEYS,
        fixed_camera_rgb_keys=_VOXEL_FIXED_CAMERA_RGB_KEYS,
    )
    batch = _voxel_batch()
    batch["targets"] = {
        "focus_target_pos": batch["action"][:, :3].unsqueeze(1).clone(),
        "focus_target_valid": torch.ones(2, 1, dtype=torch.bool),
    }
    aug(batch)
    torch.testing.assert_close(
        batch["targets"]["focus_target_pos"][0, 0], torch.tensor([0.0, 0.1, 0.5]), atol=1e-5, rtol=1e-5
    )


def test_yaw_centre_is_the_voxel_array_centre():
    """A rotated world point stays on the voxel it started on.

    The augmenter turns the grid about the array centre, so the workspace it
    turns points about has to be that same centre -- resolved from the grid's
    own footprint, not from the nominal ``ws_size`` box.
    """
    geometry = SourceVoxelGeometry((0.1, -0.2, 0.0), 0.6, (64, 64, 64))
    workspace = PlanarWorkspace(center_xy=geometry.center[:2], size=geometry.extent[0])
    index = (44, 26, 30)

    voxels = torch.zeros(1, 1, 1, 64, 64, 64)
    voxels[(0, 0, 0) + index] = 1.0
    rotation = yaw_matrices(torch.tensor([math.pi]))

    point = torch.tensor(
        [[minimum + (i + 0.5) * step for minimum, i, step in zip(geometry.workspace_min, index, geometry.pitch)]]
    )
    rotated_point = _rotate_points(point, rotation, workspace)
    rotated_index = _rotate_voxels(voxels, rotation).nonzero()[0, 3:]

    landed = torch.tensor(
        [
            [minimum + (i + 0.5) * step
             for minimum, i, step in zip(geometry.workspace_min, rotated_index.tolist(), geometry.pitch)]
        ]
    )
    assert float((rotated_point - landed).abs().max()) < geometry.pitch[0] / 2
    centre = torch.tensor([list(geometry.center)])
    torch.testing.assert_close(
        _rotate_points(centre, yaw_matrices(torch.tensor([math.pi / 3])), workspace), centre
    )


@pytest.mark.parametrize("kind", ("point_cloud", "depth"))
def test_scene_yaw_obs_action_augmentor_rejects_unrotatable_world_frame_obs(kind):
    shape_meta = {
        "action": {"shape": [10]},
        "obs": dict(_VOXEL_SHAPE_META["obs"], extra={"type": kind, "shape": [3, 1024]}),
    }
    with pytest.raises(ValueError, match="would be left unrotated"):
        SceneYawObsActionAugmentor(
            shape_meta=shape_meta,
            action_rep="absolute",
            config=_enabled_config(),
            source_keys=_VOXEL_SOURCE_KEYS,
            fixed_camera_rgb_keys=[],
        )
