"""The canonical -> model-space boundary: ``normalize_obs`` and its callers.

Canonicalization stops at the physical representation; this boundary is the one
place that representation becomes a model input, so the tests here pin down
what it converts, what it leaves alone, that it refuses to degrade to raw
values, and that encoders no longer convert anything themselves.
"""

import numpy as np
import pytest
import torch

from visuomotor.action_generation.base import ActionGenerator
from visuomotor.data.core import sparse_voxels as SparseVoxels
from visuomotor.data.core.normalization import (
    Normalizer,
    denormalize_obs,
    normalize_obs,
)
from visuomotor.data.core.observations import (
    canonicalize_obs,
    canonicalize_rgb_from_float01,
    canonicalize_rgb_from_uint8,
    canonicalize_visuals,
    canonicalize_voxel_from_uint8,
)
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.perception.common.inputs import ObsInputProcessor
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.perception.focus.seeker.model import Seeker
from visuomotor.policy.generative import GenerativePolicy


def _fitted_normalizer(*, robot_id=None) -> Normalizer:
    normalizer = Normalizer()
    normalizer.update_samples(
        "eef_pos", torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), robot_id
    )
    normalizer.update_samples("gripper_qpos", torch.tensor([[0.0, 0.0], [1.0, 1.0]]), robot_id)
    normalizer.update_samples("action_pos", torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), robot_id)
    normalizer.update_samples("action_gripper", torch.tensor([[-1.0], [1.0]]), robot_id)
    normalizer.finalize()
    return normalizer


def _canonical_obs() -> dict:
    return {
        "rgb_external": canonicalize_rgb_from_uint8(
            torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
        ),
        "voxel": canonicalize_voxel_from_uint8(
            torch.cat(
                (
                    torch.randint(0, 2, (2, 1, 2, 2, 2), dtype=torch.uint8),
                    torch.randint(0, 256, (2, 3, 2, 2, 2), dtype=torch.uint8),
                ),
                dim=1,
            )
        ),
        "eef_pos": torch.tensor([[0.0, 0.5, -0.5], [1.0, -1.0, 0.0]]),
        "eef_rot6d": torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).repeat(2, 1),
        "gripper_qpos": torch.tensor([[0.0, 0.5], [1.0, 0.25]]),
    }


_OBSERVATION_KINDS = {
    "rgb_external": "rgb",
    "rgb_wrist": "rgb",
    "voxel": "voxel",
}


# ------------------------------------------------------- what the boundary converts


def test_normalize_obs_converts_model_inputs_and_passes_physical_fields_through():
    canonical = _canonical_obs()
    normalizer = _fitted_normalizer()

    normalized = normalize_obs(
        canonical, normalizer, observation_kinds=_OBSERVATION_KINDS
    )

    torch.testing.assert_close(
        normalized["rgb_external"], normalizer.normalize("rgb", canonical["rgb_external"])
    )
    torch.testing.assert_close(
        normalized["voxel"], normalizer.normalize("voxel", canonical["voxel"])
    )
    torch.testing.assert_close(
        normalized["eef_pos"], normalizer.normalize("eef_pos", canonical["eef_pos"])
    )
    torch.testing.assert_close(
        normalized["gripper_qpos"],
        normalizer.normalize("gripper_qpos", canonical["gripper_qpos"]),
    )
    # Rotations are already in [-1, 1] and are not model-scaled.
    torch.testing.assert_close(
        normalized["eef_rot6d"], canonical["eef_rot6d"], rtol=0, atol=0
    )


def test_normalize_obs_does_not_mutate_the_canonical_observation():
    canonical = _canonical_obs()
    before = {key: value.clone() for key, value in canonical.items()}
    normalize_obs(
        canonical, _fitted_normalizer(), observation_kinds=_OBSERVATION_KINDS
    )
    for key, value in before.items():
        torch.testing.assert_close(canonical[key], value, rtol=0, atol=0)


def test_normalize_obs_routes_a_contract_declared_custom_voxel_key():
    canonical = {"scene_grid": _canonical_obs()["voxel"]}
    normalized = normalize_obs(
        canonical,
        _fitted_normalizer(),
        observation_kinds={"scene_grid": "voxel"},
    )
    assert normalized["scene_grid"].dtype == torch.float32
    assert normalized["scene_grid"] is not canonical["scene_grid"]


def test_denormalize_obs_round_trips_low_dim_and_rgb():
    canonical = _canonical_obs()
    canonical.pop("voxel")
    normalizer = _fitted_normalizer()
    normalized = normalize_obs(
        canonical, normalizer, observation_kinds=_OBSERVATION_KINDS
    )
    restored = denormalize_obs(
        normalized, normalizer, observation_kinds=_OBSERVATION_KINDS
    )
    for field in ("eef_pos", "gripper_qpos", "rgb_external"):
        torch.testing.assert_close(restored[field], canonical[field])


def test_denormalize_obs_refuses_the_lossy_voxel_direction():
    normalizer = _fitted_normalizer()
    with pytest.raises(NotImplementedError):
        denormalize_obs(
            normalize_obs(
                _canonical_obs(), normalizer, observation_kinds=_OBSERVATION_KINDS
            ),
            normalizer,
            observation_kinds=_OBSERVATION_KINDS,
        )


# ------------------------------------------------------- failing loudly


def test_normalize_obs_raises_for_an_unfitted_canonical_field():
    """The previous regression: an unfitted lookup silently returned raw values."""
    normalizer = Normalizer()
    normalizer.update_samples("action_pos", torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    normalizer.finalize()
    with pytest.raises(KeyError, match="eef_pos"):
        normalize_obs(
            _canonical_obs(), normalizer, observation_kinds=_OBSERVATION_KINDS
        )


def test_normalize_obs_raises_when_a_per_robot_field_has_no_robot_id():
    normalizer = _fitted_normalizer(robot_id=0)
    canonical = _canonical_obs()
    with pytest.raises(KeyError, match="per robot"):
        normalize_obs(canonical, normalizer, observation_kinds=_OBSERVATION_KINDS)


def test_normalize_obs_requires_a_normalizer():
    with pytest.raises(ValueError, match="requires a normalizer"):
        normalize_obs(_canonical_obs(), None, observation_kinds=_OBSERVATION_KINDS)


def test_normalize_obs_routes_per_robot_state_over_a_temporal_window():
    normalizer = Normalizer()
    normalizer.update_samples("eef_pos", torch.tensor([[0.0], [2.0]]), robot_id=0)
    normalizer.update_samples("eef_pos", torch.tensor([[10.0], [14.0]]), robot_id=1)
    normalizer.finalize()

    observations = {
        "eef_pos": torch.tensor([[[1.0], [1.0]], [[12.0], [12.0]]]),  # [B=2, T=2, 1]
    }
    normalized = normalize_obs(
        observations,
        normalizer,
        observation_kinds={},
        robot_id=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(normalized["eef_pos"], torch.zeros(2, 2, 1))


def test_normalizer_normalize_field_raises_where_normalize_would_pass_through():
    normalizer = Normalizer()
    normalizer.update_samples("eef_pos", torch.tensor([[0.0], [2.0]]), robot_id=0)
    normalizer.finalize()
    value = torch.tensor([[1.0]])
    # The silent-degradation path this refactor removes.
    torch.testing.assert_close(normalizer.normalize("eef_pos", value), value)
    with pytest.raises(KeyError):
        normalizer.normalize_field("eef_pos", value)


# ------------------------------------------------------- train/rollout share the boundary

_SHAPE_META_OBS = {
    "agentview_image": {"type": "rgb", "shape": (3, 2, 2)},
    "robot0_eef_pos": {"type": "low_dim", "shape": (3,)},
    "robot0_eef_rot": {"type": "low_dim", "shape": (9,)},
    "robot0_gripper_qpos": {"type": "low_dim", "shape": (2,)},
}


def _source_batch(image):
    return {
        "agentview_image": torch.from_numpy(image),
        "robot0_eef_pos": torch.tensor([[0.1, 0.2, 0.3]]),
        "robot0_eef_rot": torch.tensor([[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "robot0_gripper_qpos": torch.tensor([[0.02, 0.03]]),
    }


def _rgb_chw(dtype, scale):
    hwc = np.zeros((2, 2, 3), dtype=dtype)
    hwc[0, 0] = np.array([1, 0, 0]) * scale
    hwc[1, 1] = np.array([1, 1, 1]) * scale
    return np.moveaxis(hwc, -1, 0).astype(dtype, copy=False)


def _through_the_boundary(image, canonicalize_rgb, normalizer):
    canonical = canonicalize_obs(
        _source_batch(image),
        _SHAPE_META_OBS,
        canonicalize_rgb=canonicalize_rgb,
        source_proprio_keys=MimicgenObservations.source_proprio_keys,
        source_camera_keys=MimicgenObservations.source_camera_keys,
    )
    return normalize_obs(
        canonical, normalizer, observation_kinds={"rgb_external": "rgb"}
    )


def test_training_and_rollout_reach_model_space_through_the_same_boundary():
    normalizer = _fitted_normalizer()
    training = _through_the_boundary(
        _rgb_chw(np.uint8, 255), canonicalize_rgb_from_uint8, normalizer
    )
    rollout = _through_the_boundary(
        _rgb_chw(np.float32, 1.0), canonicalize_rgb_from_float01, normalizer
    )

    assert set(training) == set(rollout)
    for key, value in training.items():
        torch.testing.assert_close(value, rollout[key], rtol=0, atol=0)
    # And the boundary really did rescale, rather than forwarding canonical values.
    assert float(training["rgb_external"].min()) < 0.0


def test_rollout_canonicalization_converts_float64_proprio_to_float32():
    source = _source_batch(_rgb_chw(np.float32, 1.0))
    for key in (
        "robot0_eef_pos",
        "robot0_eef_rot",
        "robot0_gripper_qpos",
    ):
        source[key] = source[key].to(torch.float64)
    canonical = canonicalize_obs(
        source,
        _SHAPE_META_OBS,
        canonicalize_rgb=canonicalize_rgb_from_float01,
        source_proprio_keys=MimicgenObservations.source_proprio_keys,
        source_camera_keys=MimicgenObservations.source_camera_keys,
    )
    assert all(
        canonical[key].dtype == torch.float32
        for key in ("eef_pos", "eef_rot6d", "gripper_qpos")
    )


# ------------------------------------------------------- the policy owns the transition


class _RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 3
        self.seen = None
        self.seen_canonical = None

    def forward(self, observations, canonical_obs=None):
        self.seen = observations
        self.seen_canonical = canonical_obs
        return EncoderOutput(observations["eef_pos"])


class _PassThroughGenerator(ActionGenerator):
    def loss(self, actions, condition, *, generator=None):
        return actions.mean()

    def sample(self, condition, *, generator=None):
        return condition


def test_generative_policy_normalizes_immediately_before_the_encoder():
    normalizer = _fitted_normalizer()
    encoder = _RecordingEncoder()
    policy = GenerativePolicy(
        encoder=encoder,
        generator=_PassThroughGenerator(),
        observation_kinds=_OBSERVATION_KINDS,
        observation_feature_dim=encoder.output_dim,
        action_normalizer=normalizer,
    )
    canonical = _canonical_obs()

    policy.encode(canonical)

    torch.testing.assert_close(
        encoder.seen["eef_pos"], normalizer.normalize("eef_pos", canonical["eef_pos"])
    )
    torch.testing.assert_close(
        encoder.seen["rgb_external"],
        normalizer.normalize("rgb", canonical["rgb_external"]),
    )
    # The canonical observation is still available to encoders that declare it,
    # for a separately-owned model space (a released Seeker's normalizer).
    torch.testing.assert_close(
        encoder.seen_canonical["eef_pos"], canonical["eef_pos"], rtol=0, atol=0
    )


def test_generative_policy_refuses_visual_observations_without_a_normalizer():
    encoder = _RecordingEncoder()
    policy = GenerativePolicy(
        encoder=encoder,
        generator=_PassThroughGenerator(),
        observation_kinds=_OBSERVATION_KINDS,
        observation_feature_dim=encoder.output_dim,
    )
    canonical = _canonical_obs()
    with pytest.raises(ValueError, match="require a policy normalizer"):
        policy.encode(canonical)


# ------------------------------------------------------- encoders no longer convert


def test_focus_refine_encoder_does_not_normalize_its_input():
    from visuomotor.perception.encoder.focus_pool import FocusRefineEncoder

    encoder = FocusRefineEncoder(
        spatial_rank=2,
        feature_dim=8,
        input_res=8,
        rgb_keys=("rgb_external",),
        gripper_key="gripper_qpos",
        num_heads=2,
        num_iterations=2,
        pretrained_imagenet=False,
    ).eval()
    assert not hasattr(encoder, "normalizer")
    assert not hasattr(encoder, "set_normalizer")

    canonical = {
        "rgb_external": canonicalize_rgb_from_uint8(
            torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
        ),
        "gripper_qpos": torch.rand(2, 2, 2),
    }
    normalizer = _fitted_normalizer()
    normalized = normalize_obs(
        canonical, normalizer, observation_kinds={"rgb_external": "rgb"}
    )

    with pytest.raises(ValueError, match="model RGB"):
        encoder(canonical)
    from_normalized = encoder(normalized).features
    assert from_normalized.dtype == torch.float32


def test_voxel_observation_encoder_does_not_normalize_its_input():
    from visuomotor.perception.encoder.voxel import VoxelObservationEncoder

    encoder = VoxelObservationEncoder(
        source_shape=(4, 64, 64, 64),
        crop_size=58,
        voxel_architecture="voxel_focus_pool3d",
        rgb_keys=("rgb_wrist",),
        proprio_fields=("eef_pos", "eef_rot6d", "gripper_qpos"),
        proprio_dims=(3, 6, 2),
        feature_dim=32,
        num_iterations=2,
        num_heads=4,
    ).eval()
    canonical = {
        "voxel": canonicalize_voxel_from_uint8(
            torch.cat(
                (
                    torch.randint(0, 2, (2, 1, 64, 64, 64), dtype=torch.uint8),
                    torch.randint(0, 256, (2, 3, 64, 64, 64), dtype=torch.uint8),
                ),
                dim=1,
            )
        ),
        "rgb_wrist": canonicalize_rgb_from_uint8(
            torch.randint(0, 256, (2, 3, 24, 24), dtype=torch.uint8)
        ),
        "eef_pos": torch.randn(2, 3) * 0.1,
        "eef_rot6d": torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).repeat(2, 1),
        "gripper_qpos": torch.rand(2, 2),
    }
    normalized = normalize_obs(
        canonical,
        _fitted_normalizer(),
        observation_kinds={"voxel": "voxel", "rgb_wrist": "rgb"},
    )

    with torch.no_grad():
        with pytest.raises(ValueError, match="model voxel"):
            encoder(canonical)
        from_normalized = encoder(normalized).features
    assert from_normalized.dtype == torch.float32


# ------------------------------------------------------- Seeker composer inputs


def test_obs_to_input_reads_policy_space_and_normalizes_the_composer_separately():
    processor = ObsInputProcessor(
        num_robots=2, input_res=4, enable_wrist_view=False
    )
    policy_normalizer = _fitted_normalizer()
    # A Seeker-owned normalizer fitted over a different range defines its own
    # model space, so composer values must not match the policy's.
    seeker_normalizer = Normalizer()
    seeker_normalizer.update_samples("eef_pos", torch.tensor([[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]]))
    seeker_normalizer.update_samples("gripper_qpos", torch.tensor([[0.0, 0.0], [4.0, 4.0]]))
    seeker_normalizer.finalize()

    canonical = {
        "rgb_external": canonicalize_rgb_from_uint8(
            torch.randint(0, 256, (2, 1, 3, 4, 4), dtype=torch.uint8)
        ),
        "eef_pos": torch.tensor([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]),
        "eef_rot6d": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]).repeat(2, 1, 1),
        "gripper_qpos": torch.tensor([[[0.2, 0.3]], [[0.4, 0.5]]]),
    }
    task_context = {
        "robot_id": torch.zeros(2, dtype=torch.long),
        "task_embedding": torch.randn(2, 8),
    }
    normalized = normalize_obs(
        canonical,
        policy_normalizer,
        observation_kinds={"rgb_external": "rgb"},
    )

    enc_in = processor.obs_to_input(
        normalized, canonical, task_context, seeker_normalizer
    )

    flat_pos = canonical["eef_pos"].reshape(-1, 3)
    # Proprio arrives already in the policy's model space, untouched here.
    torch.testing.assert_close(
        enc_in.proprio[:, :3], policy_normalizer.normalize("eef_pos", flat_pos)
    )
    # The composer sees Seeker's space, plus the canonical values it needs raw.
    torch.testing.assert_close(
        enc_in.composer_in["eef_pos"], seeker_normalizer.normalize("eef_pos", flat_pos)
    )
    torch.testing.assert_close(enc_in.composer_in["raw_eef_pos"], flat_pos, rtol=0, atol=0)
    assert not torch.allclose(enc_in.composer_in["eef_pos"], enc_in.proprio[:, :3])
    # RGB is normalized once, at the boundary.
    torch.testing.assert_close(
        enc_in.external,
        seeker_normalizer.normalize(
            "rgb", canonical["rgb_external"].reshape(-1, 3, 4, 4)
        ),
    )


def test_obs_to_input_requires_the_canonical_observation():
    processor = ObsInputProcessor(
        num_robots=2, input_res=4, enable_wrist_view=False
    )
    with pytest.raises(ValueError, match="canonical observation"):
        processor.obs_to_input(
            {"rgb_external": torch.zeros(1, 1, 3, 4, 4)},
            None,
            {},
            Normalizer(),
        )


# ------------------------------------------------------- Seeker checkpoint ownership


def _seeker_with_normalizer(normalizer: Normalizer) -> Seeker:
    seeker = object.__new__(Seeker)
    torch.nn.Module.__init__(seeker)
    seeker.normalizer = normalizer
    return seeker


def test_seeker_checkpoint_normalizer_accepts_canonical_fields():
    seeker = _seeker_with_normalizer(_fitted_normalizer(robot_id=0))
    seeker._validate_checkpoint_normalizer("seeker.pth")


def test_seeker_checkpoint_normalizer_rejects_pre_canonicalization_field_names():
    normalizer = Normalizer()
    normalizer.update_samples("robot0_eef_pos", torch.tensor([[0.0], [1.0]]), robot_id=0)
    normalizer.update_samples("robot0_gripper_qpos", torch.tensor([[0.0], [1.0]]), robot_id=0)
    normalizer.finalize()
    seeker = _seeker_with_normalizer(normalizer)
    with pytest.raises(ValueError, match="pre-canonicalization"):
        seeker._validate_checkpoint_normalizer("seeker.pth")


def test_seeker_checkpoint_normalizer_rejects_missing_canonical_fields():
    normalizer = Normalizer()
    normalizer.update_samples("eef_pos", torch.tensor([[0.0], [1.0]]), robot_id=0)
    normalizer.finalize()
    seeker = _seeker_with_normalizer(normalizer)
    with pytest.raises(ValueError, match="gripper_qpos"):
        seeker._validate_checkpoint_normalizer("seeker.pth")


# ------------------------------------------------------- end to end, through a real config


def _run_spec(input_name: str, encoder: str):
    from test_config_resolution import _resolve

    return _resolve(input=input_name, encoder=encoder)


def _source_window(obs_shape_meta, steps: int, batch: int = 1) -> dict:
    """A source-native training batch, exactly as the dataset caches it."""
    window = {}
    for key, field in obs_shape_meta.items():
        shape = tuple(field["shape"])
        kind = field.get("type", "low_dim")
        if kind == "rgb":
            window[key] = torch.randint(
                0, 256, (batch, steps, *shape), dtype=torch.uint8
            )
        elif kind == "voxel":
            occupancy = torch.randint(0, 2, (batch, steps, 1, *shape[1:]), dtype=torch.uint8)
            colour = torch.randint(0, 256, (batch, steps, 3, *shape[1:]), dtype=torch.uint8)
            # Sparse storage requires zero colour outside occupancy.
            window[key] = torch.cat((occupancy, colour * occupancy), dim=2)
        elif "rot" in key:
            window[key] = torch.eye(3).reshape(1, 1, 9).expand(batch, steps, 9).clone()
        else:
            window[key] = torch.rand(batch, steps, *shape) * 0.1
    return window


def _canonicalize_dataset_window(source: dict, obs_shape_meta: dict) -> dict:
    adapter = MimicgenObservations.MimicGenObservationAdapter.__new__(
        MimicgenObservations.MimicGenObservationAdapter
    )
    adapter.rgb_keys = [
        key for key, field in obs_shape_meta.items() if field.get("type") == "rgb"
    ]
    adapter.voxel_keys = [
        key for key, field in obs_shape_meta.items() if field.get("type") == "voxel"
    ]
    adapter.point_cloud_keys = [
        key
        for key, field in obs_shape_meta.items()
        if field.get("type") == "point_cloud"
    ]
    adapter.lowdim_keys = [
        key
        for key, field in obs_shape_meta.items()
        if field.get("type", "low_dim") == "low_dim"
    ]
    adapter.source_keys = MimicgenObservations.source_proprio_keys(
        adapter.lowdim_keys
    )
    source_arrays = {key: value.numpy() for key, value in source.items()}
    # Exercise the dataset path through sparse voxel storage.
    voxel_shapes = _sparsify_voxels(adapter, source_arrays)
    canonical = adapter.canonicalize_obs(source_arrays)
    tensors = {key: torch.from_numpy(value) for key, value in canonical.items()}
    SparseVoxels.VoxelMaterializer(voxel_shapes)({"obs": tensors})
    # Both paths expose the same canonical-keyed uint8 visual representation.
    canonical_meta = {
        (
            MimicgenObservations.canonical_camera_key(key)
            if field.get("type") == "rgb"
            else key
        ): field
        for key, field in obs_shape_meta.items()
    }
    return canonicalize_visuals(
        tensors,
        canonical_meta,
        canonicalize_rgb=canonicalize_rgb_from_uint8,
    )


def _sparsify_voxels(adapter, source_arrays: dict) -> dict:
    """Replace each dense voxel entry with the padded cells the cache hands over."""
    voxel_shapes = {}
    for key in adapter.voxel_keys:
        dense = source_arrays.pop(key)
        batch, steps = dense.shape[:2]
        voxel_shapes[key] = dense.shape[2:]
        frames = [
            SparseVoxels.encode(dense[b, t])
            for b in range(batch)
            for t in range(steps)
        ]
        max_points = max(int(index.shape[0]) for index, _ in frames)
        padded = [SparseVoxels.decode(index, colour, max_points) for index, colour in frames]
        index_key, colour_key = SparseVoxels.sparse_keys(key)
        source_arrays[index_key] = np.stack([entry[0] for entry in padded]).reshape(
            batch, steps, max_points
        )
        source_arrays[colour_key] = np.stack([entry[1] for entry in padded]).reshape(
            batch, steps, max_points, 3
        )
        adapter.voxel_max_points = max_points
        adapter.voxel_resolution = dense.shape[3:]
    return voxel_shapes


@pytest.mark.parametrize(
    "input_name,encoder",
    [
        ("rgb_external", "rgb_focus_pool2d"),
        ("voxel", "voxel_focus_pool3d"),
    ],
)
def test_training_and_rollout_share_one_canonical_and_normalized_path(input_name, encoder):
    """A resolved run's real encoder, reached from a source-native batch.

    The dataset adapter canonicalizes cached uint8 while rollout canonicalizes
    robomimic's native float01. Both then cross the same model boundary.
    """
    from visuomotor.config.build import build_policy

    run = _run_spec(input_name, encoder)
    trajectory = run.dataset.trajectory
    obs_shape_meta = run.dataset.source_observation.shape_meta(trajectory.action_dim)["obs"]
    policy = build_policy(run.model).eval()
    policy.set_normalizer(_fitted_normalizer())

    source = _source_window(obs_shape_meta, steps=trajectory.observation_horizon)
    training_canonical = _canonicalize_dataset_window(source, obs_shape_meta)
    rollout_source = {
        key: value.float().div(255.0)
        if obs_shape_meta.get(key, {}).get("type") == "rgb"
        else value
        for key, value in source.items()
    }
    rollout_canonical = canonicalize_obs(
        rollout_source,
        obs_shape_meta,
        canonicalize_rgb=canonicalize_rgb_from_float01,
        source_proprio_keys=MimicgenObservations.source_proprio_keys,
        source_camera_keys=MimicgenObservations.source_camera_keys,
    )

    with torch.no_grad():
        training_features = policy.encode(training_canonical).features
        rollout_features = policy.encode(rollout_canonical).features
    torch.testing.assert_close(training_features, rollout_features, rtol=0, atol=0)

    model_obs = policy.normalize_obs(training_canonical)
    for key, kind in policy.observation_kinds.items():
        if kind in ("rgb", "voxel"):
            assert model_obs[key].dtype == torch.float32
