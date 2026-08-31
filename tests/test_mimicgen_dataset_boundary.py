import numpy as np
import pytest
import torch

from visuomotor.data.core import images as CoreImages
from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.core import sparse_voxels as SparseVoxels
from visuomotor.data.mimicgen import action as MimicgenAction
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.geometry import representation as Representation


def _adapter(*, rgb_keys=(), voxel_keys=(), lowdim_keys=()):
    adapter = MimicgenObservations.MimicGenObservationAdapter.__new__(
        MimicgenObservations.MimicGenObservationAdapter
    )
    adapter.rgb_keys = list(rgb_keys)
    adapter.voxel_keys = list(voxel_keys)
    adapter.point_cloud_keys = []
    adapter.lowdim_keys = list(lowdim_keys)
    adapter.source_keys = MimicgenObservations.source_proprio_keys(lowdim_keys)
    return adapter


def test_adapter_owns_mimicgen_source_sensor_contract():
    assert MimicgenObservations.source_camera_key("external") == "agentview_image"
    assert MimicgenObservations.source_camera_key("wrist") == (
        "robot0_eye_in_hand_image"
    )
    assert (
        MimicgenObservations.source_camera_name_for_key(
            MimicgenObservations.source_camera_key("wrist")
        )
        == "robot0_eye_in_hand"
    )
    assert MimicgenObservations.source_proprio_field("eef_pos") == (
        "robot0_eef_pos",
        (3,),
    )
    assert MimicgenObservations.source_proprio_field("eef_rot6d") == (
        "robot0_eef_rot",
        (9,),
    )

    source_meta = MimicgenObservations.default_source_observation_meta(128)
    assert source_meta["agentview_image"] == {
        "shape": [3, 128, 128],
        "type": "rgb",
    }
    assert source_meta["robot0_gripper_qpos"] == {"shape": [2]}


def test_oracle_metadata_is_canonicalized_recursively():
    matrix = np.eye(4, dtype=np.float32)
    canonical = MimicgenObservations.canonicalize_oracle_info(
        {
            "camera_matrix_agentview": matrix,
            "focus": {
                "target_patch_mask_robot0_eye_in_hand": np.ones((2, 2)),
            },
        }
    )

    assert canonical["camera_matrix_external"] is matrix
    assert "camera_matrix_agentview" not in canonical
    assert "target_patch_mask_wrist" in canonical["focus"]


def test_observation_adapter_emits_canonical_physical_observations():
    adapter = _adapter(
        rgb_keys=("agentview_image",),
        lowdim_keys=(
            "robot0_eef_pos",
            "robot0_eef_rot",
            "robot0_gripper_qpos",
        ),
    )
    rotation = np.array(
        [[[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    source = {
        "agentview_image": np.full((1, 1, 3, 2, 2), 255, dtype=np.uint8),
        "robot0_eef_pos": np.zeros((1, 1, 3), dtype=np.float32),
        "robot0_eef_rot": rotation,
        "robot0_gripper_qpos": np.zeros((1, 1, 2), dtype=np.float32),
    }

    canonical = adapter.canonicalize_obs(source)

    assert set(canonical) == {
        "rgb_external",
        "eef_pos",
        "eef_rot6d",
        "gripper_qpos",
    }
    # Cached uint8 is also the canonical visual encoding.
    assert canonical["rgb_external"].dtype == np.uint8
    np.testing.assert_array_equal(canonical["rgb_external"], 255)
    canonical_modalities = CoreObservations.canonicalize_visuals(
        {"rgb_external": torch.from_numpy(canonical["rgb_external"])},
        {"rgb_external": {"type": "rgb", "shape": [3, 2, 2]}},
        canonicalize_rgb=CoreObservations.canonicalize_rgb_from_uint8,
    )
    assert canonical_modalities["rgb_external"].dtype == torch.uint8
    torch.testing.assert_close(
        canonical_modalities["rgb_external"],
        torch.full_like(canonical_modalities["rgb_external"], 255),
    )
    expected_rot6d = Representation.mat_to_rot6d(
        torch.from_numpy(rotation).reshape(1, 1, 3, 3)
    ).numpy()
    np.testing.assert_allclose(canonical["eef_rot6d"], expected_rot6d)


def test_sparse_voxel_storage_rejects_a_grid_it_could_not_represent():
    """Reject dense grids that cannot round-trip through sparse storage."""
    grid = np.zeros((4, 2, 2, 2), dtype=np.uint8)
    grid[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        SparseVoxels.encode(grid)

    grid = np.zeros((4, 2, 2, 2), dtype=np.uint8)
    grid[1, 1, 1, 1] = 200  # colour on a cell occupancy says is empty
    with pytest.raises(ValueError, match="colour must be zero"):
        SparseVoxels.encode(grid)


def test_sparse_voxel_round_trip_reconstructs_the_dense_grid_exactly():
    rng = np.random.default_rng(0)
    occupancy = rng.integers(0, 2, (1, 8, 8, 8)).astype(np.uint8)
    colour = (rng.integers(0, 256, (3, 8, 8, 8)) * occupancy).astype(np.uint8)
    grid = np.concatenate((occupancy, colour), axis=0)

    index, cells = SparseVoxels.encode(grid)
    padded_index, padded_colour = SparseVoxels.decode(index, cells, max_points=600)
    dense = SparseVoxels.materialize(
        torch.from_numpy(padded_index).unsqueeze(0),
        torch.from_numpy(padded_colour).unsqueeze(0),
        (4, 8, 8, 8),
    )
    assert torch.equal(dense[0], torch.from_numpy(grid))


def test_sparse_voxel_decode_rejects_a_frame_wider_than_the_recorded_maximum():
    grid = np.zeros((4, 2, 2, 2), dtype=np.uint8)
    grid[0] = 1
    index, colour = SparseVoxels.encode(grid)
    with pytest.raises(ValueError, match="above the cache's recorded maximum"):
        SparseVoxels.decode(index, colour, max_points=4)


def test_observation_adapter_decodes_each_rgb_view_at_its_load_resolution(monkeypatch):
    adapter = _adapter(rgb_keys=("agentview_image", "robot0_eye_in_hand_image"))
    adapter.image_size = None
    adapter.rgb_load_resolutions = {"rgb_external": 224, "rgb_wrist": 84}
    monkeypatch.setattr(adapter, "_read_value", lambda *_: b"jpeg")
    calls = []

    def decode(_value, *, image_size, to_float, fmt):
        calls.append((image_size, to_float, fmt))
        return np.zeros((3, image_size, image_size), dtype=np.uint8)

    monkeypatch.setattr(
        "visuomotor.data.mimicgen.observations.CoreImages.decode_jpg_bytes", decode
    )

    external = adapter._decode_image(None, "agentview_image", 0)
    wrist = adapter._decode_image(None, "robot0_eye_in_hand_image", 0)

    assert external.shape == (3, 224, 224)
    assert wrist.shape == (3, 84, 84)
    assert external.dtype == wrist.dtype == np.uint8
    assert calls == [(224, False, "CHW"), (84, False, "CHW")]


def test_resized_jpeg_decode_preserves_cached_rgb_channel_convention():
    source = np.zeros((24, 24, 3), dtype=np.uint8)
    source[..., 0] = 240
    source[..., 1] = 120
    source[..., 2] = 20
    encoded = CoreImages.encode_rgb_to_jpg_bytes(source, quality=100)

    decoded = CoreImages.decode_jpg_bytes(
        encoded, image_size=8, to_float=False, fmt="HWC"
    )

    assert decoded.shape == (8, 8, 3)
    assert decoded.dtype == np.uint8
    assert decoded[..., 0].mean() > decoded[..., 1].mean() > decoded[..., 2].mean()


def test_native_resolution_jpeg_decode_preserves_native_decoder_path():
    source = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    encoded = CoreImages.encode_rgb_to_jpg_bytes(source, quality=90)

    native = CoreImages.decode_jpg_bytes(
        encoded, image_size=None, to_float=False, fmt="HWC"
    )
    requested_native = CoreImages.decode_jpg_bytes(
        encoded, image_size=24, to_float=False, fmt="HWC"
    )

    np.testing.assert_array_equal(requested_native, native)


def test_observation_adapter_matches_rollout_canonicalization():
    adapter = _adapter(rgb_keys=("agentview_image",))
    dataset_source = {
        "agentview_image": np.array(
            [[[[[255, 0], [0, 255]], [[0, 255], [0, 255]], [[0, 0], [255, 255]]]]],
            dtype=np.uint8,
        )
    }
    rollout_source = {
        "agentview_image": torch.from_numpy(dataset_source["agentview_image"]).float()
        / 255.0
    }
    shape_meta = {"agentview_image": {"type": "rgb", "shape": [3, 2, 2]}}

    dataset_canonical = adapter.canonicalize_obs(dataset_source)
    # Dataset and rollout converge on the same canonical uint8 encoding.
    dataset_canonical = CoreObservations.canonicalize_visuals(
        {key: torch.from_numpy(value) for key, value in dataset_canonical.items()},
        {"rgb_external": {"type": "rgb", "shape": [3, 2, 2]}},
        canonicalize_rgb=CoreObservations.canonicalize_rgb_from_uint8,
    )
    rollout_canonical = CoreObservations.canonicalize_obs(
        rollout_source,
        shape_meta,
        canonicalize_rgb=CoreObservations.canonicalize_rgb_from_float01,
        source_proprio_keys=MimicgenObservations.source_proprio_keys,
        source_camera_keys=MimicgenObservations.source_camera_keys,
    )

    torch.testing.assert_close(
        dataset_canonical["rgb_external"],
        rollout_canonical["rgb_external"],
        rtol=0,
        atol=0,
    )


def test_action_adapter_owns_representation_specific_window_selection():
    absolute = MimicgenAction.MimicGenActionAdapter(
        cache_dir="unused",
        meta={},
        horizon=2,
        action_dim=10,
        action_rep="absolute",
    )
    action = np.arange(40, dtype=np.float32).reshape(4, 10)
    np.testing.assert_array_equal(
        absolute.sample_window(
            action, observation_index=0, action_indices=np.array([1, 3])
        ),
        action[[1, 3]],
    )

    delta = MimicgenAction.MimicGenActionAdapter(
        cache_dir="unused",
        meta={},
        horizon=2,
        action_dim=10,
        action_rep="delta",
    )
    chunked = np.arange(60, dtype=np.float32).reshape(3, 20)
    np.testing.assert_array_equal(
        delta.sample_window(
            chunked, observation_index=2, action_indices=np.array([0, 1])
        ),
        chunked[2].reshape(2, 10),
    )


def test_action_cache_round_trip_preserves_controller_command_and_pose():
    action = np.array(
        [
            [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
            [-0.4, 0.5, 0.6, 0.2, -0.1, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    posmat = MimicgenAction.action_to_posmat(action)
    model_action = MimicgenAction.action_posmat_to_pos6d(posmat)

    assert posmat.shape == (2, 13)
    assert model_action.shape == (2, 10)
    np.testing.assert_allclose(model_action[:, :3], action[:, :3])
    np.testing.assert_array_equal(model_action[:, -1], action[:, -1])


def test_action_cache_rejects_noncanonical_widths():
    rejected = (
        (MimicgenAction.action_to_posmat, (6, 8), r"\(T, 7\)"),
        (MimicgenAction.action_posmat_to_pos6d, (12, 14), r"\(T, 13\)"),
    )
    for convert, widths, match in rejected:
        for width in widths:
            with pytest.raises(ValueError, match=match):
                convert(np.zeros((2, width), dtype=np.float32))


def _posmat_trajectory(rng, length):
    position = rng.normal(size=(length, 3))
    rotation = Representation.rotvec_to_mat(
        torch.from_numpy(rng.normal(size=(length, 3)) * 0.4)
    ).numpy().reshape(length, 9)
    gripper = rng.choice([-1.0, 1.0], size=(length, 1))
    return np.concatenate([position, rotation, gripper], axis=1)


def test_delta_chunks_stay_inside_their_own_episode():
    """Windows must clamp at the episode end, matching the absolute path's index clamp."""
    rng = np.random.default_rng(0)
    lengths = [12, 15]
    horizon = 8
    episodes = [_posmat_trajectory(rng, length) for length in lengths]
    episodes[1][:, :3] += 100.0  # a reset pose nowhere near where episode 0 ends
    action = np.concatenate(episodes, axis=0)
    eef_pos, eef_rot = action[:, :3] - 0.01, action[:, 3:12]

    chunks = MimicgenAction.absolute_posmat_to_delta_chunks(
        eef_pos=eef_pos,
        eef_rot=eef_rot,
        action_posmat=action,
        horizon=horizon,
        episode_lengths=lengths,
    ).reshape(-1, horizon, 10)

    for index, episode in enumerate(episodes):
        start = sum(lengths[:index])
        expected = MimicgenAction.absolute_posmat_to_delta_chunks(
            eef_pos=eef_pos[start : start + lengths[index]],
            eef_rot=eef_rot[start : start + lengths[index]],
            action_posmat=episode,
            horizon=horizon,
            episode_lengths=[lengths[index]],
        ).reshape(-1, horizon, 10)
        np.testing.assert_allclose(
            chunks[start : start + lengths[index]], expected, atol=1e-6
        )


def test_delta_chunks_reject_episode_lengths_that_miss_the_trajectory():
    rng = np.random.default_rng(1)
    action = _posmat_trajectory(rng, 6)
    with pytest.raises(ValueError, match="must sum to the action length"):
        MimicgenAction.absolute_posmat_to_delta_chunks(
            eef_pos=action[:, :3],
            eef_rot=action[:, 3:12],
            action_posmat=action,
            horizon=4,
            episode_lengths=[3, 2],
        )
