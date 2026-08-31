"""Typed configuration boundary for one complete Visuomotor Stack experiment.

Hydra selects declarative input, encoder, policy, and regime presets. The
resolver turns those selections into immutable runtime specs; runtime domains
never read Hydra or YAML.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.core import spatial as Spatial
from visuomotor.data.core.mirror import MirrorAugmentationConfig
from visuomotor.data.core.scene_augmentation import SceneYawAugmentationConfig

OBS_KINDS = ("rgb", "depth", "low_dim", "point_cloud", "voxel", "frame")
_SHAPE_META_KINDS = {"rgb", "depth", "point_cloud", "voxel"}
PROPRIO_SHAPES = dict(CoreObservations.CANONICAL_PROPRIO_SHAPES)
# The absolute pose/state fields, excluding proprioception derived from a pair
# of frames. Consumers that mean "the robot's current configuration" -- Seeker
# queries, the pretraining input -- take this set, not every selectable field.
ABSOLUTE_PROPRIO_SHAPES = {
    key: shape
    for key, shape in PROPRIO_SHAPES.items()
    if key not in CoreObservations.DERIVED_PROPRIO_FIELDS
}
SOURCE_RGB_SHAPE = (3, 256, 256)

_SPEC_TYPES: Dict[str, type] = {
    "MirrorAugmentationConfig": MirrorAugmentationConfig,
    "PointCloudProducerSpec": Spatial.PointCloudProducerSpec,
    "SceneYawAugmentationConfig": SceneYawAugmentationConfig,
    "VoxelProducerSpec": Spatial.VoxelProducerSpec,
}


def _spec(cls):
    _SPEC_TYPES[cls.__name__] = cls
    return cls


def to_dict(value):
    """Serialize a spec tree to plain containers, tagging dataclasses."""
    if is_dataclass(value) and not isinstance(value, type):
        payload = {"__spec__": type(value).__name__}
        for entry in fields(value):
            payload[entry.name] = to_dict(getattr(value, entry.name))
        return payload
    if isinstance(value, Mapping):
        return {str(key): to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value


def from_dict(value):
    """Rebuild a spec tree produced by :func:`to_dict`."""
    if isinstance(value, Mapping):
        name = value.get("__spec__")
        if name is None:
            return {str(key): from_dict(item) for key, item in value.items()}
        try:
            cls = _SPEC_TYPES[str(name)]
        except KeyError as error:
            raise ValueError(f"unknown spec type {name!r}") from error
        return cls(
            **{
                entry.name: from_dict(value[entry.name])
                for entry in fields(cls)
                if entry.name in value
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(from_dict(item) for item in value)
    return value


# ---------------------------------------------------------------- observations


@_spec
@dataclass(frozen=True)
class ObsFieldSpec:
    key: str
    kind: str
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in OBS_KINDS:
            raise ValueError(f"unknown observation kind {self.kind!r}")
        if not self.shape or any(int(size) < 1 for size in self.shape):
            raise ValueError(f"observation {self.key!r} needs a positive shape")


@_spec
@dataclass(frozen=True)
class ObservationContract:
    """Exact canonical tensors visible at the model boundary."""

    fields: Tuple[ObsFieldSpec, ...]

    def __post_init__(self) -> None:
        keys = tuple(entry.key for entry in self.fields)
        if len(keys) != len(set(keys)):
            raise ValueError("observation keys must be unique")

    def keys(self, *kinds: str) -> Tuple[str, ...]:
        wanted = set(kinds) or set(OBS_KINDS)
        return tuple(entry.key for entry in self.fields if entry.kind in wanted)

    def field(self, key: str) -> ObsFieldSpec:
        for entry in self.fields:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def describe(self) -> str:
        return "\n".join(
            ["model observations:"]
            + [
                f"  {entry.key:<20s} {entry.kind:<8s} {entry.shape}"
                for entry in self.fields
            ]
        )

    def model_meta(self, action_dim: int) -> dict:
        """Derived tensor metadata for canonical-space pipeline utilities."""
        obs = {}
        for entry in self.fields:
            obs[entry.key] = {"shape": list(entry.shape)}
            if entry.kind in _SHAPE_META_KINDS:
                obs[entry.key]["type"] = entry.kind
        return {"obs": obs, "action": {"shape": [int(action_dim)]}}


@_spec
@dataclass(frozen=True)
class TrajectoryContract:
    """Shared action and temporal dimensions for model, data, and rollout."""

    action_dim: int
    prediction_horizon: int
    observation_horizon: int
    execution_horizon: int
    action_rep: str

    def __post_init__(self) -> None:
        dimensions = (
            self.action_dim,
            self.prediction_horizon,
            self.observation_horizon,
            self.execution_horizon,
        )
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("trajectory dimensions must be positive")
        if self.execution_horizon > self.prediction_horizon:
            raise ValueError("execution horizon cannot exceed prediction horizon")


@_spec
@dataclass(frozen=True)
class DataRequirements:
    """Data-side capabilities requested by a resolved model component."""

    keypose_targets: bool = False
    oracle_info: bool = False

    def merge(self, other: "DataRequirements") -> "DataRequirements":
        return DataRequirements(
            keypose_targets=self.keypose_targets or other.keypose_targets,
            oracle_info=self.oracle_info or other.oracle_info,
        )


@_spec
@dataclass(frozen=True)
class SourceObsFieldSpec:
    """One source-native dataset/environment field."""

    source_key: str
    kind: str
    shape: Tuple[int, ...]


@_spec
@dataclass(frozen=True)
class SourceObservationSpec:
    """Dataset/environment fetch bindings, separate from model observations."""

    fields: Tuple[SourceObsFieldSpec, ...]
    support_fields: Tuple[SourceObsFieldSpec, ...] = ()
    producers: Tuple[Spatial.SpatialProducerSpec, ...] = ()

    def shape_meta(self, action_dim: int) -> dict:
        obs = {}
        for entry in self.fields + self.support_fields:
            obs[entry.source_key] = {"shape": list(entry.shape)}
            if entry.kind in _SHAPE_META_KINDS:
                obs[entry.source_key]["type"] = entry.kind
        return {"obs": obs, "action": {"shape": [int(action_dim)]}}


@_spec
@dataclass(frozen=True)
class VoxelInputSpec:
    key: str = "voxel"
    frame: str = "world"
    resolution: Tuple[int, int, int] = (64, 64, 64)
    channels: Tuple[str, ...] = ("occupancy", "R", "G", "B")
    ws_size: float = 0.6

    def __post_init__(self) -> None:
        if self.frame not in ("world", "eef_centered"):
            raise ValueError(f"unknown voxel input frame {self.frame!r}")
        if len(self.resolution) != 3 or any(int(size) < 2 for size in self.resolution):
            raise ValueError("voxel input resolution must contain three values >= 2")
        if self.channels != ("occupancy", "R", "G", "B"):
            raise ValueError("voxel inputs require occupancy/R/G/B channels")
        if self.ws_size <= 0:
            raise ValueError("voxel input ws_size must be positive")


@_spec
@dataclass(frozen=True)
class PointCloudInputSpec:
    key: str = "point_cloud"
    num_points: int = 1024
    channels: Tuple[str, ...] = ("x", "y", "z", "R", "G", "B")
    ws_size: float = 0.6
    table_margin: float = 0.005

    def __post_init__(self) -> None:
        if self.num_points < 1:
            raise ValueError("point-cloud num_points must be positive")
        if self.table_margin <= 0:
            raise ValueError("point-cloud table_margin must be positive")
        if self.channels not in (("x", "y", "z"), ("x", "y", "z", "R", "G", "B")):
            raise ValueError("point-cloud channels must be XYZ or XYZRGB")


@_spec
@dataclass(frozen=True)
class InputSpec:
    """Declarative model input: modalities and selected proprioception only."""

    name: str
    rgb_views: Tuple[str, ...] = ()
    proprio: Tuple[str, ...] = ()
    voxel: Optional[VoxelInputSpec] = None
    point_cloud: Optional[PointCloudInputSpec] = None

    def __post_init__(self) -> None:
        if len(self.rgb_views) != len(set(self.rgb_views)):
            raise ValueError("RGB views must be unique")
        unknown = sorted(set(self.proprio).difference(PROPRIO_SHAPES))
        if unknown:
            raise ValueError(f"unknown canonical proprio fields: {unknown}")
        if self.voxel is not None and self.voxel.frame != "world":
            raise ValueError("input.voxel must use the world frame")
        if self.voxel is not None and self.point_cloud is not None:
            raise ValueError("input cannot select both voxel and point cloud")
        if not self.rgb_views and self.voxel is None and self.point_cloud is None:
            raise ValueError("input must select RGB, voxel, or point cloud")

    @property
    def voxels(self) -> Tuple[VoxelInputSpec, ...]:
        return tuple(value for value in (self.voxel,) if value is not None)


# ------------------------------------------------------------------- encoders


@_spec
@dataclass(frozen=True)
class OverlaySpec:
    prob: float = 0.0
    alpha_min: float = 0.6
    alpha_max: float = 0.6
    noise_std: float = 0.0
    warmup_steps: int = 0
    background_path: Optional[str] = None


@_spec
@dataclass(frozen=True)
class RandomCropSpec:
    input_res: int
    output_res: int

    def __post_init__(self) -> None:
        if self.input_res < 1 or not 1 <= self.output_res <= self.input_res:
            raise ValueError("random crop needs 0 < output_res <= input_res")

    @property
    def enabled(self) -> bool:
        return self.output_res < self.input_res


@_spec
@dataclass(frozen=True)
class FocusSourceSpec:
    name: str = "none"
    weights: Optional[str] = None
    strict_weights: bool = True
    checkpoint: Optional[str] = None
    checkpoint_views: Tuple[str, ...] = ()


@_spec
@dataclass(frozen=True)
class RgbEncoderSpec:
    name: str
    architecture: str
    rgb_keys: Tuple[str, ...]
    proprio_fields: Tuple[str, ...]
    feature_dim: int = 64
    random_crop: RandomCropSpec = field(default_factory=lambda: RandomCropSpec(84, 76))
    pretrained_imagenet: bool = True
    norm: str = "groupnorm"
    data_requirements: DataRequirements = field(default_factory=DataRequirements)


@_spec
@dataclass(frozen=True)
class FocusConditionedEncoderSpec:
    name: str
    feature_architecture: str
    source: FocusSourceSpec
    view_modes: Tuple[Tuple[str, str], ...]
    view_augmentations: Tuple[Tuple[str, str], ...]
    view_keys: Tuple[Tuple[str, str], ...]
    proprio_fields: Tuple[str, ...]
    num_robots: int
    feature_dim: int = 64
    resnet_pretrained_imagenet: bool = True
    vit_in: int = 224
    random_crop: RandomCropSpec = field(default_factory=lambda: RandomCropSpec(84, 76))
    guided_overlay: Optional[OverlaySpec] = None
    random_overlay: Optional[OverlaySpec] = None
    norm: str = "groupnorm"
    data_requirements: DataRequirements = field(default_factory=DataRequirements)

    def __post_init__(self) -> None:
        if not self.view_modes:
            raise ValueError("the active regime disables every selected view")
        if tuple(view for view, _ in self.view_modes) != tuple(
            view for view, _ in self.view_augmentations
        ):
            raise ValueError(
                "focus view transforms and augmentations must cover the same views"
            )


@_spec
@dataclass(frozen=True)
class AttentionPriorSpec:
    enabled: bool = False
    weight: float = 2e-4
    sigma_cells: float = 1.2
    bootstrap_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("attention-prior weight cannot be negative")
        if self.sigma_cells <= 0:
            raise ValueError("attention-prior sigma_cells must be positive")
        if self.bootstrap_steps is not None and self.bootstrap_steps <= 0:
            raise ValueError(
                "attention-prior bootstrap_steps must be positive when set"
            )


@_spec
@dataclass(frozen=True)
class FocusRefineEncoderSpec:
    name: str
    architecture: str
    rgb_keys: Tuple[str, ...]
    proprio_fields: Tuple[str, ...]
    gripper_key: str
    input_res: int
    input_channels: int = 3
    feature_dim: int = 128
    random_crop: Optional[RandomCropSpec] = None
    num_heads: int = 4
    num_iterations: int = 3
    pool_stage: int = 2
    pretrained_imagenet: bool = True
    norm: str = "groupnorm"
    attention_prior: Optional[AttentionPriorSpec] = None
    attention_prior_view: Optional[str] = None
    attention_prior_raw_res: Optional[int] = None
    data_requirements: DataRequirements = field(default_factory=DataRequirements)

    def __post_init__(self) -> None:
        if self.gripper_key not in self.proprio_fields:
            raise ValueError("focus-pool requires selected gripper proprioception")


@_spec
@dataclass(frozen=True)
class VoxelEncoderSpec:
    name: str
    voxel_architecture: str
    voxel_key: str
    voxel_shape: Tuple[int, int, int, int]
    rgb_keys: Tuple[str, ...]
    proprio_fields: Tuple[str, ...]
    feature_dim: int = 256
    rgb_architecture: str = "resnet18"
    rgb_feature_dim: int = 256
    rgb_pretrained_imagenet: bool = True
    rgb_norm: str = "groupnorm"
    rgb_random_crop: Optional[RandomCropSpec] = None
    crop_size: Optional[int] = None
    coord_conv: bool = False
    num_heads: int = 4
    num_iterations: int = 3
    pool_stage: int = 3
    attention_prior: Optional[AttentionPriorSpec] = None
    workspace_min: Optional[Tuple[float, float, float]] = None
    ws_size: Optional[float] = None
    data_requirements: DataRequirements = field(default_factory=DataRequirements)

    def __post_init__(self) -> None:
        if (
            self.voxel_architecture == "voxel_focus_pool3d"
            and "gripper_qpos" not in self.proprio_fields
        ):
            raise ValueError(
                "volumetric focus-pool requires selected gripper proprioception"
            )


@_spec
@dataclass(frozen=True)
class PointCloudEncoderSpec:
    name: str
    point_cloud_key: str
    point_cloud_shape: Tuple[int, int]
    proprio_fields: Tuple[str, ...]
    feature_dim: int = 64
    state_mlp_dims: Tuple[int, ...] = (64, 64)
    use_color: bool = True
    use_layernorm: bool = True
    final_norm: str = "layernorm"
    data_requirements: DataRequirements = field(default_factory=DataRequirements)

    def __post_init__(self) -> None:
        if not self.state_mlp_dims:
            raise ValueError("DP3 state_mlp_dims cannot be empty")
        object.__setattr__(
            self, "state_mlp_dims", tuple(int(size) for size in self.state_mlp_dims)
        )


# --------------------------------------------------------------- policy specs


def _validate_unet_contract(
    *,
    label: str,
    unet_channels,
    kernel_size: int,
    n_groups: int,
    horizon: Optional[int] = None,
) -> Tuple[int, ...]:
    widths = tuple(int(width) for width in unet_channels)
    kernel_size = int(kernel_size)
    n_groups = int(n_groups)
    if not widths or any(width < 1 for width in widths):
        raise ValueError(
            f"{label} unet_channels must contain positive widths, got {widths}"
        )
    if n_groups < 1:
        raise ValueError(f"{label} n_groups must be positive, got {n_groups}")
    if any(width % n_groups for width in widths):
        raise ValueError(
            f"{label} unet_channels must be divisible by n_groups={n_groups}, "
            f"got {widths}"
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"{label} kernel_size must be a positive odd integer, got {kernel_size}"
        )
    if horizon is not None:
        factor = 2 ** (len(widths) - 1)
        if int(horizon) % factor:
            raise ValueError(
                f"{label} horizon {horizon} must be divisible by UNet downsample "
                f"factor {factor} for unet_channels={widths}"
            )
    return widths


def _validate_time_embedding(label: str, dimension: int) -> None:
    dimension = int(dimension)
    if dimension < 4 or dimension % 2:
        raise ValueError(
            f"{label} must be an even integer of at least 4, got {dimension}"
        )


@_spec
@dataclass(frozen=True)
class GeneratorSpec:
    kind: str
    scheduler: str = "ddpm"
    unet_channels: Tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    num_inference_steps: int = 100
    diffusion_step_embed_dim: int = 128
    num_train_timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    variance_type: str = "fixed_small"
    clip_sample: bool = True
    prediction_type: str = "epsilon"
    integration_steps: int = 100
    time_embedding_dim: int = 128
    time_embedding_scale: float = 100.0

    def __post_init__(self) -> None:
        if self.kind not in ("diffusion", "flow"):
            raise ValueError(f"unknown generator kind {self.kind!r}")
        widths = _validate_unet_contract(
            label=f"{self.kind} generator",
            unet_channels=self.unet_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
        )
        object.__setattr__(self, "unet_channels", widths)
        if self.scheduler not in ("ddpm", "ddim"):
            raise ValueError(f"unknown diffusion scheduler {self.scheduler!r}")
        if self.kind == "flow":
            if self.integration_steps < 1:
                raise ValueError("flow integration_steps must be positive")
            _validate_time_embedding("flow time_embedding_dim", self.time_embedding_dim)
            if not math.isfinite(self.time_embedding_scale) or self.time_embedding_scale <= 0:
                raise ValueError("flow time_embedding_scale must be positive and finite")
        else:
            if self.num_inference_steps < 1 or self.num_train_timesteps < 1:
                raise ValueError("diffusion step counts must be positive")
            _validate_time_embedding(
                "diffusion diffusion_step_embed_dim", self.diffusion_step_embed_dim
            )


@_spec
@dataclass(frozen=True)
class GlobalPolicySpec:
    name: str
    generator: GeneratorSpec
    observation_feature_dim: int = 256
    data_requirements: DataRequirements = field(default_factory=DataRequirements)

    def __post_init__(self) -> None:
        if self.observation_feature_dim < 1:
            raise ValueError("global observation_feature_dim must be positive")


@_spec
@dataclass(frozen=True)
class ModelSpec:
    """Complete model construction spec, independent of source bindings."""

    input: InputSpec
    observation: ObservationContract
    encoder: Any
    policy: Any
    normalizer: str
    trajectory: TrajectoryContract

    @property
    def data_requirements(self) -> DataRequirements:
        return self.encoder.data_requirements.merge(self.policy.data_requirements)

    def describe(self) -> str:
        return "\n".join(
            [
                f"input: {self.input.name}",
                self.observation.describe(),
                f"encoder: {self.encoder.name}",
                f"conditioning: {self.policy.observation_feature_dim}D",
                f"policy: {self.policy.name}",
                f"normalizer: {self.normalizer}",
            ]
        )


# ------------------------------------------------------------ data and rollout


@_spec
@dataclass(frozen=True)
class KeyposeTargetSpec:
    gripper_motion_threshold: float = 5e-4
    gripper_valley_threshold: float = 2e-4
    gripper_valley_window: int = 4

    def __post_init__(self) -> None:
        if self.gripper_motion_threshold <= 0:
            raise ValueError("gripper_motion_threshold must be positive")
        if self.gripper_valley_threshold <= 0:
            raise ValueError("gripper_valley_threshold must be positive")
        if self.gripper_valley_window < 1:
            raise ValueError("gripper_valley_window must be positive")


@_spec
@dataclass(frozen=True)
class DatasetSpec:
    path: str
    observation: ObservationContract
    source_observation: SourceObservationSpec
    trajectory: TrajectoryContract
    rgb_load_resolutions: Tuple[Tuple[str, int], ...]
    n_demo: Optional[int] = None
    demo_count_mode: str = "total"
    val_ratio: float = 0.02
    cache_dir: Optional[str] = None
    lmdb_readahead: bool = False
    include_oracle_info: bool = False
    include_camera_matrices: bool = False
    mirror_augmentation: Optional[MirrorAugmentationConfig] = None
    scene_yaw_augmentation: Optional[SceneYawAugmentationConfig] = None
    keypose_targets: Optional[KeyposeTargetSpec] = None

    def __post_init__(self) -> None:
        resolutions = dict(self.rgb_load_resolutions)
        if len(resolutions) != len(self.rgb_load_resolutions):
            raise ValueError("RGB load-resolution keys must be unique")
        rgb_fields = self.observation.keys("rgb")
        if tuple(resolutions) != rgb_fields:
            raise ValueError(
                "RGB load resolutions must exactly cover canonical RGB fields"
            )
        for key, resolution in resolutions.items():
            if int(resolution) < 1:
                raise ValueError("RGB load resolutions must be positive")
            if self.observation.field(key).shape != (
                3,
                int(resolution),
                int(resolution),
            ):
                raise ValueError(
                    f"canonical RGB shape for {key!r} must match its load resolution"
                )


@_spec
@dataclass(frozen=True)
class MediaToggleSpec:
    images: bool = True
    videos: bool = True


@_spec
@dataclass(frozen=True)
class VisualizationSpec:
    enabled: bool = True
    num_samples: int = 6
    augmentation_preview: bool = True
    save: MediaToggleSpec = field(default_factory=MediaToggleSpec)
    upload: MediaToggleSpec = field(
        default_factory=lambda: MediaToggleSpec(images=False, videos=False)
    )

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("visualization.num_samples must be positive")
        if self.upload.images and not self.save.images:
            raise ValueError(
                "visualization cannot upload images when saving is disabled"
            )
        if self.upload.videos and not self.save.videos:
            raise ValueError(
                "visualization cannot upload videos when saving is disabled"
            )


@_spec
@dataclass(frozen=True)
class RunnerSpec:
    dataset_path: str
    observation: ObservationContract
    source_observation: SourceObservationSpec
    trajectory: TrajectoryContract
    rgb_load_resolutions: Tuple[Tuple[str, int], ...] = ()
    env_name: Optional[str] = None
    cache_dir: Optional[str] = None
    action_rep: str = "absolute"
    n_test: int = 50
    n_test_vis: int = 3
    test_start_seed: int = 100000
    max_steps: int = 400
    n_envs: int = 25
    render_obs_key: Optional[str] = None
    fps: int = 10
    crf: int = 28
    past_action: bool = False
    tqdm_interval_sec: float = 1.0
    shuffle_table_texture: bool = False
    strict_task_success: bool = True
    enable_oracle_subtask_info: bool = False
    oracle_projection_camera: Optional[str] = None
    enable_oracle_focus_info: bool = False
    oracle_focus_camera: Optional[str] = None
    oracle_focus_patch_size: int = 16
    oracle_focus_min_patch_area_fraction: float = 0.05
    oracle_focus_min_mask_pixels: int = 16
    mirror_augmentation: Optional[MirrorAugmentationConfig] = None
    output_dir: Optional[str] = None
    visualization: VisualizationSpec = field(default_factory=VisualizationSpec)

    def __post_init__(self) -> None:
        resolutions = dict(self.rgb_load_resolutions)
        if len(resolutions) != len(self.rgb_load_resolutions):
            raise ValueError("RGB load-resolution keys must be unique")
        if tuple(resolutions) != self.observation.keys("rgb"):
            raise ValueError(
                "rollout RGB load resolutions must exactly cover canonical RGB fields"
            )
        for key, resolution in resolutions.items():
            if self.observation.field(key).shape != (3, int(resolution), int(resolution)):
                raise ValueError(
                    f"canonical RGB shape for {key!r} must match its load resolution"
                )


@_spec
@dataclass(frozen=True)
class TrainingSpec:
    device: str = "cuda:0"
    seed: int = 0
    debug: bool = False
    resume: bool = True
    num_epochs: int = 500
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    use_ema: bool = True
    rollout_every: int = 20
    checkpoint_every: int = 10
    val_every: int = 10
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None
    tqdm_interval_sec: float = 1.0
    log_freq: int = 1000


@_spec
@dataclass(frozen=True)
class DataLoaderSpec:
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool = False


@_spec
@dataclass(frozen=True)
class EmaSpec:
    update_after_step: int = 0
    inv_gamma: float = 1.0
    power: float = 2 / 3
    min_value: float = 0.0
    max_value: float = 0.9999


@_spec
@dataclass(frozen=True)
class LoggingSpec:
    project: str
    resume: bool
    name: str
    group: str
    job_type: str
    tags: Tuple[str, ...]
    mode: Optional[str] = None
    run_id: Optional[str] = None


@_spec
@dataclass(frozen=True)
class TopKCheckpointSpec:
    monitor_key: str
    mode: str
    k: int
    format_str: str


@_spec
@dataclass(frozen=True)
class CheckpointSpec:
    topk: TopKCheckpointSpec
    save_last: bool
    save_topk_full: bool = False


@_spec
@dataclass(frozen=True)
class PolicyWorkspaceSpec:
    train_loader: DataLoaderSpec
    val_loader: DataLoaderSpec
    ema: EmaSpec
    logging: LoggingSpec
    checkpoint: CheckpointSpec
    visualization: VisualizationSpec = field(default_factory=VisualizationSpec)


@_spec
@dataclass(frozen=True)
class SamplerSpec:
    kind: str
    samples_per_epoch: int


@_spec
@dataclass(frozen=True)
class SeekerPretrainingModelSpec:
    visual_mode: str
    image_size: int
    obs_dropout: float
    stage_stride: int
    weights: str
    background_path: str
    generator: GeneratorSpec
    num_refinement_iters: int
    disable_proprio: bool
    disable_head_gating: bool


@_spec
@dataclass(frozen=True)
class SeekerPretrainingDatasetSpec:
    path: str
    source_observation: SourceObservationSpec
    trajectory: TrajectoryContract
    n_demo: int
    demo_count_mode: str
    image_size: int
    cache_dir: Optional[str] = None


@_spec
@dataclass(frozen=True)
class SeekerPretrainingCheckpointSpec:
    save_last: bool
    save_snapshot: bool
    seeker_light_path: Optional[str] = None


@_spec
@dataclass(frozen=True)
class SeekerPretrainingWorkspaceSpec:
    train_loader: DataLoaderSpec
    sampler: Optional[SamplerSpec]
    ema: EmaSpec
    logging: LoggingSpec
    checkpoint: SeekerPretrainingCheckpointSpec
    visualization: VisualizationSpec = field(default_factory=VisualizationSpec)


@_spec
@dataclass(frozen=True)
class SeekerPretrainingSpec:
    model: SeekerPretrainingModelSpec
    dataset: SeekerPretrainingDatasetSpec
    training: TrainingSpec
    workspace: SeekerPretrainingWorkspaceSpec


@_spec
@dataclass(frozen=True)
class Rvt2PretrainingModelSpec:
    heatmap: Mapping[str, Any]
    query_composer: Mapping[str, Any]
    background_overlay: Mapping[str, Any]
    stage_stride: int


@_spec
@dataclass(frozen=True)
class Rvt2PretrainingDatasetSpec:
    task_name: str
    path: str
    cache_dir: Optional[str]
    n_demo: Optional[int]
    demo_count_mode: str
    skip_first_episodes: int
    val_ratio: float


@_spec
@dataclass(frozen=True)
class Rvt2PretrainingTrainingSpec:
    epochs: int
    lr: float
    weight_decay: float
    device: str
    resume: Optional[str]
    load_weights: Optional[str]
    resume_optimizer: bool
    checkpoint_every: int
    max_train_steps: Optional[int]
    max_val_steps: Optional[int]
    seed: int


@_spec
@dataclass(frozen=True)
class Rvt2PretrainingWorkspaceSpec:
    train_loader: DataLoaderSpec
    val_loader: DataLoaderSpec
    sampler: Optional[SamplerSpec]
    logging: Optional[LoggingSpec]
    print_epoch_metrics: bool
    print_run_summary: bool
    show_label_progress: bool
    visualization: VisualizationSpec = field(default_factory=VisualizationSpec)
    visualization_alpha: float = 0.45
    visualization_sampling: str = "even"


@_spec
@dataclass(frozen=True)
class Rvt2PretrainingSpec:
    model: Rvt2PretrainingModelSpec
    dataset: Rvt2PretrainingDatasetSpec
    training: Rvt2PretrainingTrainingSpec
    workspace: Rvt2PretrainingWorkspaceSpec


@_spec
@dataclass(frozen=True)
class TaskSpec:
    name: str
    max_steps: int
    spatial_cameras: Tuple[str, ...]
    table_offset: Tuple[float, float, float]


@_spec
@dataclass(frozen=True)
class RegimeSpec:
    name: str
    dataset_suffix: str = ""
    shuffle_table_texture: bool = False


@_spec
@dataclass(frozen=True)
class RunSpec:
    task: TaskSpec
    regime: RegimeSpec
    model: ModelSpec
    dataset: DatasetSpec
    runner: RunnerSpec
    training: TrainingSpec
    workspace: PolicyWorkspaceSpec
    exp_name: str

    def __str__(self) -> str:
        return "\n".join(
            [
                f"task: {self.task.name} (max_steps={self.task.max_steps})",
                f"regime: {self.regime.name}",
                self.model.describe(),
                f"dataset: {self.dataset.path}",
                f"exp_name: {self.exp_name}",
            ]
        )


def _validate_rgb_encoder_contract(input_spec, encoder, contract) -> None:
    if (
        input_spec.voxel is not None
        or input_spec.point_cloud is not None
        or tuple(encoder.rgb_keys) != contract.keys("rgb")
    ):
        raise ValueError("RGB ResNet encoder requires only selected RGB views")


def _validate_focus_conditioned_contract(input_spec, encoder, _contract) -> None:
    if input_spec.voxel is not None:
        raise ValueError(f"encoder {encoder.name!r} does not support voxel input")
    if "external" not in input_spec.rgb_views:
        raise ValueError(f"encoder {encoder.name!r} requires the external RGB view")


def _validate_focus_refine_contract(input_spec, encoder, contract) -> None:
    if input_spec.voxel is not None or tuple(encoder.rgb_keys) != contract.keys("rgb"):
        raise ValueError("planar focus-pool requires selected RGB views and no voxel")


def _validate_voxel_encoder_contract(input_spec, encoder, contract) -> None:
    if input_spec.voxel is None:
        raise ValueError("single-scale voxel encoder requires one global voxel input")
    if tuple(encoder.rgb_keys) != contract.keys("rgb"):
        raise ValueError("voxel encoder RGB branches must match selected RGB views")


def _validate_point_cloud_encoder_contract(input_spec, encoder, contract) -> None:
    if input_spec.point_cloud is None:
        raise ValueError("point-cloud encoder requires point-cloud input")
    if contract.keys("point_cloud") != (encoder.point_cloud_key,):
        raise ValueError("point-cloud encoder key must match the observation contract")
    if input_spec.rgb_views or input_spec.voxel is not None:
        raise ValueError(
            "DP3 currently consumes point cloud without RGB or voxel branches"
        )


def _validate_encoder_contract(model: ModelSpec) -> None:
    input_spec = model.input
    encoder = model.encoder
    contract = model.observation
    if tuple(encoder.proprio_fields) != tuple(input_spec.proprio):
        raise ValueError(
            "encoder proprio fields must exactly match selected input proprioception"
        )
    if isinstance(encoder, RgbEncoderSpec):
        return _validate_rgb_encoder_contract(input_spec, encoder, contract)
    if isinstance(encoder, FocusConditionedEncoderSpec):
        return _validate_focus_conditioned_contract(input_spec, encoder, contract)
    if isinstance(encoder, FocusRefineEncoderSpec):
        return _validate_focus_refine_contract(input_spec, encoder, contract)
    if isinstance(encoder, VoxelEncoderSpec):
        return _validate_voxel_encoder_contract(input_spec, encoder, contract)
    if isinstance(encoder, PointCloudEncoderSpec):
        return _validate_point_cloud_encoder_contract(input_spec, encoder, contract)
    raise TypeError(f"unknown encoder spec {type(encoder).__name__}")


def _validate_policy_contract(model: ModelSpec) -> None:
    if isinstance(model.policy, GlobalPolicySpec):
        _validate_unet_contract(
            label=f"global {model.policy.generator.kind} generator",
            unet_channels=model.policy.generator.unet_channels,
            kernel_size=model.policy.generator.kernel_size,
            n_groups=model.policy.generator.n_groups,
            horizon=model.trajectory.prediction_horizon,
        )
    else:
        raise TypeError(f"unknown policy spec {type(model.policy).__name__}")


def _validate_augmentation_contract(run: RunSpec) -> None:
    input_spec = run.model.input
    mirror = run.dataset.mirror_augmentation
    scene_yaw = run.dataset.scene_yaw_augmentation
    if (
        mirror is not None
        and mirror.enable
        and scene_yaw is not None
        and scene_yaw.enable
    ):
        raise ValueError("mirror and scene-yaw augmentation cannot be combined")
    if scene_yaw is not None and scene_yaw.enable and input_spec.voxel is None:
        raise ValueError("scene-yaw augmentation is voxel-only")
    if mirror is not None and mirror.enable:
        if not input_spec.rgb_views:
            raise ValueError("mirror augmentation requires RGB input")
        if not {"eef_pos", "eef_rot6d"}.issubset(input_spec.proprio):
            raise ValueError("mirror augmentation requires eef_pos and eef_rot6d")
        derived = sorted(
            set(input_spec.proprio) & set(CoreObservations.DERIVED_PROPRIO_FIELDS)
        )
        if derived:
            # A reflection is not a rotation: it flips the body-frame delta's y
            # translation and negates the rotation vector's pseudo-vector axes.
            # The mirror augmenter reflects poses and actions only, so a cached
            # delta would survive it unreflected and disagree with rollout.
            raise ValueError(
                f"mirror augmentation cannot reflect derived proprioception {derived}"
            )
    if scene_yaw is not None and scene_yaw.enable:
        if "external" in input_spec.rgb_views:
            raise ValueError("scene-yaw augmentation cannot rotate fixed external RGB")
        if not {"eef_pos", "eef_rot6d"}.issubset(input_spec.proprio):
            raise ValueError("scene-yaw augmentation requires eef_pos and eef_rot6d")


def _validate_shared_runtime_contracts(run: RunSpec) -> None:
    model = run.model
    contract = model.observation
    if run.dataset.observation != contract:
        raise ValueError("dataset canonical contract differs from model input")
    runner_fields = tuple(
        (field.key, field.kind) for field in run.runner.observation.fields
    )
    model_fields = tuple((field.key, field.kind) for field in contract.fields)
    if runner_fields != model_fields:
        raise ValueError("runner and model canonical fields differ")
    if run.dataset.source_observation != run.runner.source_observation:
        raise ValueError("dataset/runner source bindings differ")
    if (
        run.dataset.trajectory != model.trajectory
        or run.runner.trajectory != model.trajectory
    ):
        raise ValueError("model, dataset, and runner trajectory contracts differ")
    source_kinds = tuple(field.kind for field in run.dataset.source_observation.fields)
    canonical_kinds = tuple(field.kind for field in contract.fields)
    if source_kinds != canonical_kinds:
        raise ValueError(
            "source field kinds must match the canonical observation contract order"
        )


def validate(run: RunSpec) -> RunSpec:
    """Validate requirements/capabilities instead of enumerating preset pairs."""
    _validate_encoder_contract(run.model)
    _validate_policy_contract(run.model)
    _validate_augmentation_contract(run)
    _validate_shared_runtime_contracts(run)
    return run
