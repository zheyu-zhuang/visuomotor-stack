"""Dataset <-> rollout canonical-contract parity.

Both sources must converge, via the shared canonicalization primitives in
``data/core/observations.py``, to the same canonical representation
before model normalization runs -- exactly once -- downstream of that.
"""

import numpy as np
import pytest
import torch

from visuomotor.data.core import images as CoreImages
from visuomotor.data.core import normalization as CoreNormalization
from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.core.observations import (
    canonicalize_obs,
    canonicalize_proprio,
    canonicalize_rgb_from_float01,
    canonicalize_rgb_from_uint8,
    canonicalize_voxel_from_uint8,
    validate_obs,
)
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.environment.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
from visuomotor.environment.runner import SeekerRobomimicImageRunner
from visuomotor.geometry import representation as Representation


def _rollout_get_observation(shape_meta_obs, raw_obs, rgb_load_resolutions=None):
    """Call the real RobomimicImageWrapper.get_observation() without a live env."""
    wrapper = object.__new__(RobomimicImageWrapper)
    wrapper.shape_meta = {"obs": shape_meta_obs}
    wrapper.observation_space = dict.fromkeys(shape_meta_obs)
    wrapper.render_obs_key = next(iter(shape_meta_obs))
    wrapper._validated_rgb_keys = set()
    wrapper._validated_spatial_keys = set()
    wrapper._observation_needed = True
    wrapper._last_visual_obs = {}
    wrapper.skipped_observations = 0
    wrapper.produced_observations = 0
    wrapper.rgb_load_resolutions = dict(rgb_load_resolutions or {})
    wrapper.rgb_jpeg_quality = CoreImages.JPEG_QUALITY_DEFAULT
    return RobomimicImageWrapper.get_observation(wrapper, raw_obs=raw_obs)


def _dataset_decode_cached_rgb(source_hwc, *, canonical_key, load_resolution):
    """Read one frame back through the real dataset cache write/read pair."""
    cached_bytes = CoreImages.encode_rgb_to_jpg_bytes(
        source_hwc, quality=CoreImages.JPEG_QUALITY_DEFAULT
    )
    adapter = object.__new__(MimicgenObservations.MimicGenObservationAdapter)
    adapter.image_size = None
    adapter.rgb_load_resolutions = {canonical_key: int(load_resolution)}
    adapter._read_value = lambda transaction, key, index: cached_bytes
    return adapter._decode_image(
        None,
        MimicgenObservations.source_camera_key_for_canonical(canonical_key),
        0,
    )


def _rendered_frame(resolution: int) -> np.ndarray:
    """A deterministic HWC uint8 frame with edges JPEG and resampling both bite on."""
    rng = np.random.default_rng(0)
    grid = np.arange(resolution, dtype=np.float32)
    ramp = np.stack(
        [
            np.broadcast_to(grid[None, :], (resolution, resolution)),
            np.broadcast_to(grid[:, None], (resolution, resolution)),
            (grid[None, :] + grid[:, None]) / 2.0,
        ],
        axis=-1,
    )
    frame = (ramp / max(resolution - 1, 1) * 255.0).astype(np.uint8)
    frame[resolution // 3 : 2 * resolution // 3, resolution // 4 : resolution // 2] = 0
    noise = rng.integers(0, 32, size=frame.shape, dtype=np.uint8)
    return np.ascontiguousarray(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def _sentinel_rgb_chw(dtype, scale) -> np.ndarray:
    """A 2x2 red/green/blue/white block image, channel-first."""
    hwc = np.zeros((2, 2, 3), dtype=dtype)
    hwc[0, 0] = np.array([1, 0, 0]) * scale  # red
    hwc[0, 1] = np.array([0, 1, 0]) * scale  # green
    hwc[1, 0] = np.array([0, 0, 1]) * scale  # blue
    hwc[1, 1] = np.array([1, 1, 1]) * scale  # white
    return np.moveaxis(hwc, -1, 0).astype(dtype, copy=False)


# --------------------------------------------------------------- A. RGB pre-normalization


def test_rgb_pre_normalization_parity():
    dataset_raw = _sentinel_rgb_chw(np.uint8, 255)
    rollout_raw = _sentinel_rgb_chw(np.float32, 1.0)

    canonical_dataset = canonicalize_rgb_from_uint8(torch.from_numpy(dataset_raw))
    canonical_rollout = canonicalize_rgb_from_float01(torch.from_numpy(rollout_raw))

    for canonical in (canonical_dataset, canonical_rollout):
        assert canonical.dtype == torch.uint8
        assert canonical.shape[-3] == 3
        assert int(canonical.min()) >= 0 and int(canonical.max()) <= 255

    torch.testing.assert_close(canonical_dataset, canonical_rollout, rtol=0, atol=0)
    torch.testing.assert_close(
        canonical_dataset[:, 0, 0], torch.tensor([255, 0, 0], dtype=torch.uint8)
    )


def test_rgb_adapters_reject_the_other_source_encoding():
    rejected = (
        (canonicalize_rgb_from_uint8, torch.from_numpy(_sentinel_rgb_chw(np.float32, 1.0))),
        (canonicalize_rgb_from_float01, torch.from_numpy(_sentinel_rgb_chw(np.uint8, 255))),
        (
            canonicalize_rgb_from_float01,
            torch.from_numpy(_sentinel_rgb_chw(np.float32, 1.0)) * 255.0,
        ),
    )
    for adapter, bad in rejected:
        with pytest.raises(ValueError):
            adapter(bad)


@pytest.mark.parametrize("load_resolution", [84, 128, 256])
def test_rollout_rgb_is_byte_identical_to_the_cached_training_encoding(load_resolution):
    """One rendered frame must reach the policy identically from either source."""
    render_resolution = 256
    source_hwc = _rendered_frame(render_resolution)
    native = np.moveaxis(source_hwc, -1, 0).astype(np.float32) / 255.0

    shape_meta_obs = {
        "agentview_image": {
            "type": "rgb",
            "shape": (3, load_resolution, load_resolution),
        }
    }
    rollout = _rollout_get_observation(
        shape_meta_obs,
        {"agentview_image": native.copy()},
        rgb_load_resolutions={"agentview_image": load_resolution},
    )["agentview_image"]
    dataset = _dataset_decode_cached_rgb(
        source_hwc, canonical_key="rgb_external", load_resolution=load_resolution
    )

    assert rollout.dtype == np.uint8 == dataset.dtype
    assert rollout.shape == (3, load_resolution, load_resolution)
    np.testing.assert_array_equal(rollout, dataset)


def test_rollout_rgb_without_a_load_resolution_keeps_the_render_resolution():
    source_hwc = _rendered_frame(64)
    native = np.moveaxis(source_hwc, -1, 0).astype(np.float32) / 255.0
    shape_meta_obs = {"agentview_image": {"type": "rgb", "shape": (3, 64, 64)}}
    obs = _rollout_get_observation(shape_meta_obs, {"agentview_image": native})
    assert obs["agentview_image"].shape == (3, 64, 64)
    np.testing.assert_array_equal(
        obs["agentview_image"],
        CoreImages.canonical_rgb_from_source(source_hwc, load_resolution=None),
    )


def test_uncompressed_rollout_rgb_would_not_match_the_cached_encoding():
    """The parity above is earned by the shared codec, not by the frame being easy."""
    load_resolution = 84
    source_hwc = _rendered_frame(256)
    dataset = _dataset_decode_cached_rgb(
        source_hwc, canonical_key="rgb_external", load_resolution=load_resolution
    )
    naive = (
        torch.nn.functional.interpolate(
            torch.from_numpy(np.moveaxis(source_hwc, -1, 0)).float()[None],
            size=(load_resolution, load_resolution),
            mode="area",
        )[0]
        .round()
        .to(torch.uint8)
        .numpy()
    )
    assert not np.array_equal(naive, dataset)


# --------------------------------------------------------------- B. RGB post-normalization


def test_rgb_post_normalization_parity():
    dataset_raw = _sentinel_rgb_chw(np.uint8, 255)
    rollout_raw = _sentinel_rgb_chw(np.float32, 1.0)

    canonical_dataset = canonicalize_rgb_from_uint8(torch.from_numpy(dataset_raw))
    canonical_rollout = canonicalize_rgb_from_float01(torch.from_numpy(rollout_raw))

    normalizer = CoreNormalization.Normalizer()
    normalized_dataset = normalizer.normalize("rgb", canonical_dataset)
    normalized_rollout = normalizer.normalize("rgb", canonical_rollout)
    torch.testing.assert_close(normalized_dataset, normalized_rollout, rtol=0, atol=0)

    # Moving normalization into canonicalization would make this fail: the
    # canonical (pre-normalization) tensor must not already be ImageNet-scaled.
    assert canonical_dataset.dtype != normalized_dataset.dtype


# --------------------------------------------------------------- C. Voxel pre-normalization


def _sentinel_voxel(dtype) -> np.ndarray:
    """A 2x2x2 grid: one occupied red voxel, one occupied blue voxel, rest empty."""
    voxel = np.zeros((4, 2, 2, 2), dtype=dtype)
    voxel[0, 0, 0, 0] = 1
    voxel[1, 0, 0, 0] = 255  # red
    voxel[0, 1, 1, 1] = 1
    voxel[3, 1, 1, 1] = 255  # blue
    return voxel


def test_voxel_pre_normalization_parity():
    dataset_raw = _sentinel_voxel(np.uint8)
    rollout_raw = dataset_raw.copy()  # both sources synthesize the same raw uint8 grid

    canonical_dataset = canonicalize_voxel_from_uint8(torch.from_numpy(dataset_raw))
    canonical_rollout = canonicalize_voxel_from_uint8(torch.from_numpy(rollout_raw))
    torch.testing.assert_close(canonical_dataset, canonical_rollout, rtol=0, atol=0)

    occupancy = canonical_dataset[0]
    assert set(occupancy.unique().tolist()).issubset({0.0, 1.0})
    rgb = canonical_dataset[1:]
    assert int(rgb.min()) >= 0 and int(rgb.max()) <= 255
    # Empty source cells stay structurally empty (unoccupied) after canonicalization.
    empty_cell = canonical_dataset[:, 1, 0, 0]
    assert float(empty_cell[0]) == 0.0


def test_voxel_adapter_rejects_noncanonical_grids():
    nonbinary = torch.from_numpy(_sentinel_voxel(np.uint8)).clone()
    nonbinary[0, 0, 0, 0] = 2
    rejected = (
        (torch.from_numpy(_sentinel_voxel(np.uint8)).float(), None),
        (torch.from_numpy(_sentinel_voxel(np.uint8))[:3], None),
        (nonbinary, "binary"),
    )
    for bad, match in rejected:
        with pytest.raises(ValueError, match=match):
            canonicalize_voxel_from_uint8(bad)


def test_rollout_wrapper_rejects_degenerate_spatial_observations():
    nonfinite = np.ones((8, 6), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    rejected = (
        ("voxel", np.zeros((4, 2, 2, 2), dtype=np.uint8), "zero occupied cells"),
        ("point_cloud", np.zeros((8, 6), dtype=np.float32), "all-zero XYZ"),
        ("point_cloud", nonfinite, "non-finite"),
    )
    for key, observation, match in rejected:
        shape_meta_obs = {key: {"type": key, "shape": observation.shape}}
        with pytest.raises(RuntimeError, match=match):
            _rollout_get_observation(shape_meta_obs, {key: observation})


# --------------------------------------------------------------- D. Voxel post-normalization


def test_voxel_post_normalization_parity():
    dataset_raw = _sentinel_voxel(np.uint8)
    rollout_raw = dataset_raw.copy()

    canonical_dataset = canonicalize_voxel_from_uint8(torch.from_numpy(dataset_raw))
    canonical_rollout = canonicalize_voxel_from_uint8(torch.from_numpy(rollout_raw))

    normalizer = CoreNormalization.Normalizer()
    normalized_dataset = normalizer.normalize("voxel", canonical_dataset)
    normalized_rollout = normalizer.normalize("voxel", canonical_rollout)
    torch.testing.assert_close(normalized_dataset, normalized_rollout, rtol=0, atol=0)

    empty_cell = normalized_dataset[:, 1, 0, 0]
    torch.testing.assert_close(empty_cell, torch.zeros_like(empty_cell), rtol=0, atol=0)
    assert normalized_dataset[0].amin().item() == 0.0
    assert normalized_dataset[0].amax().item() == 1.0
    assert normalized_dataset[1:].amin().item() >= 0.0
    assert normalized_dataset[1:].amax().item() <= 1.0


# --------------------------------------------------------------- F. Proprio parity


def _proprio_source(pos, rot_mat_flat, gripper):
    return {
        "robot0_eef_pos": torch.tensor(pos, dtype=torch.float32),
        "robot0_eef_rot": torch.tensor(rot_mat_flat, dtype=torch.float32),
        "robot0_gripper_qpos": torch.tensor(gripper, dtype=torch.float32),
    }


def _tumbling_rotations(count: int) -> np.ndarray:
    """Flattened rotation matrices, float64 as the simulator emits them."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    axis = np.stack(
        [np.cos(angles * 0.7), np.sin(angles * 1.3), np.cos(angles * 0.3)], axis=-1
    )
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    rotvec = (axis * angles[:, None]).astype(np.float64)
    matrix = Representation.rotvec_to_mat(torch.from_numpy(rotvec)).numpy()
    return matrix.reshape(count, 9)


def test_proprio_canonicalization_parity_across_dataset_and_rollout_naming():
    pos = [[0.1, 0.2, 0.3]]
    # A non-identity rotation: 90 degrees about the z axis.
    rot_mat_flat = [[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    gripper = [[0.02, 0.03]]
    native = _proprio_source(pos, rot_mat_flat, gripper)

    canonical = canonicalize_proprio(
        native, MimicgenObservations.source_proprio_keys(native.keys())
    )

    expected_rot6d = Representation.mat_to_rot6d(
        torch.tensor(rot_mat_flat).reshape(-1, 3, 3)
    )
    torch.testing.assert_close(canonical["eef_pos"], torch.tensor(pos))
    torch.testing.assert_close(canonical["eef_rot6d"], expected_rot6d)
    torch.testing.assert_close(canonical["gripper_qpos"], torch.tensor(gripper))
    assert "robot0_eef_pos" not in canonical
    assert "robot0_eef_rot" not in canonical
    assert "robot0_gripper_qpos" not in canonical


def test_numpy_and_torch_proprio_converters_agree_bit_for_bit():
    """Datasets convert in NumPy and rollouts in torch; the values must not diverge.

    Source values are float64, as the simulator emits them, so the narrowing to
    float32 is exercised rather than assumed away.
    """
    count = 512
    rng = np.random.default_rng(0)
    source = {
        "robot0_eef_pos": rng.standard_normal((count, 3)).astype(np.float64),
        "robot0_eef_rot": _tumbling_rotations(count),
        "robot0_gripper_qpos": rng.standard_normal((count, 2)).astype(np.float64),
    }
    source_keys = MimicgenObservations.source_proprio_keys(source.keys())

    from_numpy = CoreObservations.canonicalize_proprio_numpy(source, source_keys)
    from_torch = canonicalize_proprio(
        {key: torch.from_numpy(value) for key, value in source.items()}, source_keys
    )

    assert set(from_numpy) == set(from_torch) == {
        "eef_pos",
        "eef_rot6d",
        "gripper_qpos",
    }
    for key, expected in from_torch.items():
        actual = torch.from_numpy(from_numpy[key])
        assert actual.dtype == expected.dtype == torch.float32
        assert torch.equal(actual, expected), f"{key} diverged"


def test_torch_proprio_conversion_is_device_independent():
    """Rollout canonicalizes on the policy device; the dataset does it on the host."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    source = {
        "robot0_eef_pos": torch.randn(256, 3, dtype=torch.float64),
        "robot0_eef_rot": torch.from_numpy(_tumbling_rotations(256)),
        "robot0_gripper_qpos": torch.randn(256, 2, dtype=torch.float64),
    }
    source_keys = MimicgenObservations.source_proprio_keys(source.keys())

    host = canonicalize_proprio(source, source_keys)
    device = canonicalize_proprio(
        {key: value.cuda() for key, value in source.items()}, source_keys
    )

    for key, expected in host.items():
        assert torch.equal(device[key].cpu(), expected), f"{key} diverged on device"


# --------------------------------------------------------------- F2. Whole-batch contract parity

_CONTRACT_SHAPE_META_OBS = {
    "agentview_image": {"type": "rgb", "shape": (3, 2, 2)},
    "robot0_eye_in_hand_image": {"type": "rgb", "shape": (3, 2, 2)},
    "voxel": {"type": "voxel", "shape": (4, 2, 2, 2)},
    "robot0_eef_pos": {"type": "low_dim", "shape": (3,)},
    "robot0_eef_rot": {"type": "low_dim", "shape": (9,)},
    "robot0_gripper_qpos": {"type": "low_dim", "shape": (2,)},
}


def _source_batch(image):
    batch = _proprio_source(
        [[0.1, 0.2, 0.3]],
        [[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        [[0.02, 0.03]],
    )
    batch["agentview_image"] = torch.from_numpy(image)
    batch["robot0_eye_in_hand_image"] = torch.from_numpy(image.copy())
    batch["voxel"] = torch.from_numpy(_sentinel_voxel(np.uint8))
    return batch


def test_train_and_rollout_canonicalize_obs_agree_on_the_whole_batch():
    """Only the source RGB adapter differs between the two paths."""
    dataset_canonical = canonicalize_obs(
        _source_batch(_sentinel_rgb_chw(np.uint8, 255)),
        _CONTRACT_SHAPE_META_OBS,
        canonicalize_rgb=canonicalize_rgb_from_uint8,
        source_proprio_keys=MimicgenObservations.source_proprio_keys,
        source_camera_keys=MimicgenObservations.source_camera_keys,
    )
    rollout_canonical = canonicalize_obs(
        _source_batch(_sentinel_rgb_chw(np.float32, 1.0)),
        _CONTRACT_SHAPE_META_OBS,
        canonicalize_rgb=canonicalize_rgb_from_float01,
        source_proprio_keys=MimicgenObservations.source_proprio_keys,
        source_camera_keys=MimicgenObservations.source_camera_keys,
    )

    assert set(dataset_canonical) == set(rollout_canonical)
    for key, value in dataset_canonical.items():
        torch.testing.assert_close(value, rollout_canonical[key], rtol=0, atol=0)


def test_camera_keys_are_canonical_after_canonicalization():
    canonical = canonicalize_obs(
        _source_batch(_sentinel_rgb_chw(np.uint8, 255)),
        _CONTRACT_SHAPE_META_OBS,
        canonicalize_rgb=canonicalize_rgb_from_uint8,
        source_proprio_keys=MimicgenObservations.source_proprio_keys,
        source_camera_keys=MimicgenObservations.source_camera_keys,
    )
    assert "rgb_external" in canonical and "rgb_wrist" in canonical
    assert "agentview_image" not in canonical
    assert "robot0_eye_in_hand_image" not in canonical


def test_validate_obs_rejects_a_batch_that_is_not_purely_canonical():
    def _canonical():
        return canonicalize_obs(
            _source_batch(_sentinel_rgb_chw(np.uint8, 255)),
            _CONTRACT_SHAPE_META_OBS,
            canonicalize_rgb=canonicalize_rgb_from_uint8,
            source_proprio_keys=MimicgenObservations.source_proprio_keys,
            source_camera_keys=MimicgenObservations.source_camera_keys,
        )

    def _surviving_camera_key(canonical):
        canonical["agentview_image"] = canonical["rgb_external"]

    def _surviving_proprio_key(canonical):
        canonical["robot0_eef_pos"] = canonical["eef_pos"]

    def _missing_proprio_field(canonical):
        del canonical["eef_rot6d"]

    rejected = (
        (_surviving_camera_key, "source camera key survived"),
        (_surviving_proprio_key, "survived canonicalization"),
        (_missing_proprio_field, "missing canonical proprio field"),
    )
    for mutate, match in rejected:
        canonical = _canonical()
        mutate(canonical)
        with pytest.raises(ValueError, match=match):
            validate_obs(
                canonical,
                _CONTRACT_SHAPE_META_OBS,
                source_keys=MimicgenObservations.source_proprio_keys(
                    _CONTRACT_SHAPE_META_OBS.keys()
                ),
                camera_keys=MimicgenObservations.source_camera_keys(
                    _CONTRACT_SHAPE_META_OBS.keys()
                ),
            )


def test_canonicalize_obs_rejects_a_non_canonical_rgb_adapter():
    with pytest.raises(ValueError, match="canonical RGB"):
        canonicalize_obs(
            _source_batch(_sentinel_rgb_chw(np.uint8, 255)),
            _CONTRACT_SHAPE_META_OBS,
            canonicalize_rgb=lambda image: image.float(),
            source_proprio_keys=MimicgenObservations.source_proprio_keys,
            source_camera_keys=MimicgenObservations.source_camera_keys,
        )


# --------------------------------------------------------------- G. Action parity


def test_action_normalizer_round_trip_preserves_pos_rot6d_gripper_semantics():
    normalizer = CoreNormalization.Normalizer()
    positions = torch.tensor([[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]])
    gripper = torch.tensor([[-1.0], [1.0]])
    normalizer.update_samples("action_pos", positions)
    normalizer.update_samples("action_gripper", gripper)
    normalizer.finalize()

    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    action = torch.cat(
        [torch.tensor([[0.1, -0.1, 0.05]]), rotation, torch.tensor([[0.3]])], dim=-1
    )

    normalized = normalizer.normalize_action(action)
    # Rotation passes through the action normalizer unchanged.
    torch.testing.assert_close(normalized[..., 3:9], rotation)
    round_tripped = normalizer.denormalize_action(normalized)
    torch.testing.assert_close(round_tripped, action)


def test_rollout_action_conversion_boundary_preserves_absolute_action():
    """``undo_transform_action`` converts a model-space rot6d action back to the
    environment's axis-angle controller action without altering pos/gripper."""
    runner = object.__new__(SeekerRobomimicImageRunner)
    runner.action_rep = "absolute"
    runner.rotation_transformer = Representation.RotationTransformer(
        "axis_angle", "rotation_6d"
    )

    axis_angle = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    rot6d = runner.rotation_transformer.forward(axis_angle[None])[0]
    pos = np.array([0.3, -0.1, 0.5], dtype=np.float32)
    gripper = np.array([0.7], dtype=np.float32)
    action = np.concatenate([pos, rot6d, gripper])[None, None]  # [B=1, T=1, D]

    env_action = runner.undo_transform_action(action)

    np.testing.assert_allclose(env_action[0, 0, :3], pos, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(env_action[0, 0, -1], gripper, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(env_action[0, 0, 3:-1], axis_angle, rtol=1e-4, atol=1e-4)


def test_rollout_action_conversion_rejects_an_extra_gripper_state_channel():
    runner = object.__new__(SeekerRobomimicImageRunner)
    runner.action_rep = "absolute"
    runner.rotation_transformer = Representation.RotationTransformer(
        "axis_angle", "rotation_6d"
    )

    with pytest.raises(ValueError, match=r"10D xyz\+rot6d\+gripper-command"):
        runner.undo_transform_action(np.zeros((1, 1, 11), dtype=np.float32))
