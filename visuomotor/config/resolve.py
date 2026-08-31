"""Resolve one composed Hydra experiment into typed runtime specs."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping, Optional

from visuomotor.config import schema as Schema
from visuomotor.config import tasks as Tasks
from visuomotor.config.tasks import dataset_robot_ids, get_task_spec
from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.core import spatial as Spatial
from visuomotor.data.core.mirror import MirrorAugmentationConfig
from visuomotor.data.core.scene_augmentation import SceneYawAugmentationConfig
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.data.mimicgen import tasks as MimicgenTasks
from visuomotor.geometry.grid import SourceVoxelGeometry


def _plain(value):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        return value
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _tuple_sequences(value):
    if isinstance(value, Mapping):
        return {str(key): _tuple_sequences(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_tuple_sequences(item) for item in value)
    return value


def _rgb_crop(cfg) -> Schema.RandomCropSpec:
    cfg = dict(cfg or {})
    if cfg.get("train") != "random" or cfg.get("evaluation") != "center":
        raise ValueError(
            "rgb_crop must use random training crops and center evaluation crops"
        )
    return Schema.RandomCropSpec(
        input_res=int(cfg["resize"]), output_res=int(cfg["output"])
    )


def _attention(cfg) -> Optional[Schema.AttentionPriorSpec]:
    if not cfg:
        return None
    cfg = dict(cfg)
    bootstrap_steps = cfg.get("bootstrap_steps")
    return Schema.AttentionPriorSpec(
        enabled=bool(cfg.get("enabled", False)),
        weight=float(cfg.get("weight", 2e-4)),
        sigma_cells=float(cfg.get("sigma_cells", 1.2)),
        bootstrap_steps=None if bootstrap_steps is None else int(bootstrap_steps),
    )


def _overlay(cfg, backgrounds_dir: str) -> Schema.OverlaySpec:
    if not cfg:
        raise ValueError(
            "an active overlay strategy needs a matching augmentation profile"
        )
    cfg = dict(cfg)
    alpha = tuple(float(value) for value in cfg.get("alpha", (0.6, 0.6)))
    if len(alpha) != 2:
        raise ValueError("overlay alpha must contain [minimum, maximum]")
    return Schema.OverlaySpec(
        prob=float(cfg.get("probability", 0.0)),
        alpha_min=alpha[0],
        alpha_max=alpha[1],
        noise_std=float(cfg.get("mask_noise_std", 0.0)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        background_path=backgrounds_dir,
    )


def resolve_regime(cfg) -> Schema.RegimeSpec:
    cfg = dict(_plain(cfg))
    return Schema.RegimeSpec(
        name=str(cfg["name"]),
        dataset_suffix=str(cfg.get("dataset_suffix", "")),
        shuffle_table_texture=bool(cfg.get("shuffle_table_texture", False)),
    )


def resolve_input(cfg) -> Schema.InputSpec:
    """Resolve modalities without choosing their implementation."""
    cfg = dict(_plain(cfg))
    def voxel_input(name: str, default_key: str, default_frame: str):
        voxel_cfg = cfg.get(name)
        if not voxel_cfg:
            return None
        voxel_cfg = dict(voxel_cfg)
        return Schema.VoxelInputSpec(
            key=str(voxel_cfg.get("key", default_key)),
            frame=str(voxel_cfg.get("frame", default_frame)),
            resolution=tuple(
                int(size) for size in voxel_cfg.get("resolution", (64, 64, 64))
            ),
            channels=tuple(
                str(name)
                for name in voxel_cfg.get("channels", ("occupancy", "R", "G", "B"))
            ),
            ws_size=float(voxel_cfg.get("ws_size", 0.6)),
        )

    voxel = voxel_input("voxel", "voxel", "world")
    point_cloud_cfg = cfg.get("point_cloud")
    point_cloud = None
    if point_cloud_cfg:
        point_cloud_cfg = dict(point_cloud_cfg)
        point_cloud = Schema.PointCloudInputSpec(
            key=str(point_cloud_cfg.get("key", "point_cloud")),
            num_points=int(point_cloud_cfg.get("num_points", 1024)),
            channels=tuple(
                str(name)
                for name in point_cloud_cfg.get(
                    "channels", ("x", "y", "z", "R", "G", "B")
                )
            ),
            ws_size=float(point_cloud_cfg.get("ws_size", 0.6)),
            table_margin=float(point_cloud_cfg.get("table_margin", 0.005)),
        )
    return Schema.InputSpec(
        name=str(cfg["name"]),
        rgb_views=tuple(str(view) for view in cfg.get("rgb_views", ())),
        proprio=tuple(str(key) for key in cfg.get("proprio", ())),
        voxel=voxel,
        point_cloud=point_cloud,
    )


def resolve_source_observation(
    input_spec: Schema.InputSpec, *, task
) -> Schema.SourceObservationSpec:
    """Bind canonical input names to MimicGen dataset/environment fields."""
    fields = [
        Schema.SourceObsFieldSpec(
            source_key=MimicgenObservations.source_camera_key(view),
            kind="rgb",
            shape=Schema.SOURCE_RGB_SHAPE,
        )
        for view in input_spec.rgb_views
    ]
    producers = []
    for voxel in input_spec.voxels:
        if voxel.frame == "world":
            bounds_min, bounds_max = Tasks.voxel_bounds(task, voxel.ws_size)
        else:
            half = voxel.ws_size / 2.0
            bounds_min = (-half, -half, -half)
            bounds_max = (half, half, half)
        producer = Spatial.VoxelProducerSpec(
            cameras=task.spatial_cameras,
            output_key=voxel.key,
            frame=voxel.frame,
            resolution=voxel.resolution,
            channels=voxel.channels,
            ws_size=voxel.ws_size,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        producers.append(producer)
        fields.append(
            Schema.SourceObsFieldSpec(
                source_key=voxel.key,
                kind="voxel",
                shape=producer.observation_shape,
            )
        )
    if input_spec.point_cloud is not None:
        point_cloud = input_spec.point_cloud
        bounds_min, bounds_max = Tasks.point_cloud_bounds(
            task,
            point_cloud.ws_size,
            point_cloud.table_margin,
        )
        producer = Spatial.PointCloudProducerSpec(
            cameras=task.spatial_cameras,
            output_key=point_cloud.key,
            num_points=point_cloud.num_points,
            channels=point_cloud.channels,
            ws_size=point_cloud.ws_size,
            table_margin=point_cloud.table_margin,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        producers.append(producer)
        fields.append(
            Schema.SourceObsFieldSpec(
                source_key=point_cloud.key,
                kind="point_cloud",
                shape=producer.observation_shape,
            )
        )
    derived = tuple(
        key
        for key in input_spec.proprio
        if key in CoreObservations.DERIVED_PROPRIO_FIELDS
    )
    for canonical_key in input_spec.proprio:
        if canonical_key in derived:
            # Derived proprioception has no source field; the adapter builds it
            # from the canonical fields bound below.
            fields.append(
                Schema.SourceObsFieldSpec(
                    canonical_key, "low_dim", Schema.PROPRIO_SHAPES[canonical_key]
                )
            )
            continue
        source_key, source_shape = MimicgenObservations.source_proprio_field(
            canonical_key
        )
        fields.append(Schema.SourceObsFieldSpec(source_key, "low_dim", source_shape))
    support = []
    for canonical_key in CoreObservations.derived_proprio_sources(derived):
        if canonical_key in input_spec.proprio:
            continue
        source_key, source_shape = MimicgenObservations.source_proprio_field(
            canonical_key
        )
        support.append(Schema.SourceObsFieldSpec(source_key, "low_dim", source_shape))
    return Schema.SourceObservationSpec(
        fields=tuple(fields),
        support_fields=tuple(support),
        producers=tuple(producers),
    )


def _validate_encoder_compatibility(cfg, input_spec: Schema.InputSpec) -> None:
    compatibility = dict(cfg.get("compatibility") or {})
    representation = str(compatibility.get("representation", ""))
    selected_representation = (
        "voxel"
        if input_spec.voxel is not None
        else "point_cloud"
        if input_spec.point_cloud is not None
        else "rgb"
    )
    if representation != selected_representation:
        raise ValueError(
            f"encoder {cfg['name']!r} requires {representation!r} representation, "
            f"but input {input_spec.name!r} selects {selected_representation!r}"
        )

    required = tuple(str(view) for view in compatibility.get("requires_rgb_views", ()))
    accepted = tuple(str(view) for view in compatibility.get("accepts_rgb_views", ()))
    missing = tuple(view for view in required if view not in input_spec.rgb_views)
    unsupported = tuple(view for view in input_spec.rgb_views if view not in accepted)
    if missing:
        raise ValueError(f"encoder {cfg['name']!r} requires RGB views {missing}")
    if unsupported:
        raise ValueError(
            f"encoder {cfg['name']!r} does not accept RGB views {unsupported}"
        )


def _validate_fusion(cfg) -> None:
    """Encoders emit the concatenated streams; the policy owns the projection."""
    fusion = dict(cfg.get("fusion") or {})
    if fusion.get("architecture") != "concat":
        raise ValueError("encoder fusion.architecture must be 'concat'")
    if fusion.get("include_selected_proprio") is not True:
        raise ValueError(
            "encoder fusion must include exactly the selected proprioception"
        )
    if "output_dim" in fusion:
        raise ValueError(
            "encoder fusion.output_dim is owned by the policy as "
            "observation_feature_dim"
        )


def _view_strategies(cfg, input_spec: Schema.InputSpec, regime: Schema.RegimeSpec):
    configured = dict(cfg.get("view_strategies") or {})
    modes = []
    augmentations = []
    for view in input_spec.rgb_views:
        by_regime = dict(configured.get(view) or {})
        if regime.name not in by_regime:
            raise ValueError(
                f"encoder {cfg['name']!r} has no {regime.name!r} strategy for RGB view {view!r}"
            )
        strategy = dict(by_regime[regime.name] or {})
        mode = str(strategy.get("view_transform", ""))
        augmentation = str(strategy.get("augmentation", ""))
        if not mode or not augmentation:
            raise ValueError(
                f"encoder {cfg['name']!r} must declare both view_transform and augmentation "
                f"for {view!r}/{regime.name!r}"
            )
        if mode == "disabled":
            raise ValueError(
                f"encoder {cfg['name']!r} does not support input {input_spec.name!r} "
                f"under regime {regime.name!r}"
            )
        modes.append((view, mode))
        augmentations.append((view, augmentation))
    return tuple(modes), tuple(augmentations)


def _resolve_voxel_encoder(
    cfg,
    *,
    name,
    input_spec,
    source_observation,
    rgb_keys,
    proprio,
    crop_size,
    norm,
    rgb_crop,
):
    voxel_cfg = dict(cfg.get("voxel_branch") or {})
    voxel_architecture = str(voxel_cfg.get("architecture", ""))
    if voxel_architecture not in {
        "voxel_simple",
        "voxel_resnet3d",
        "voxel_focus_pool3d",
    }:
        raise ValueError(f"unknown voxel branch architecture: {voxel_architecture!r}")
    rgb_cfg = dict(cfg.get("rgb_branch") or {})
    if input_spec.rgb_views and rgb_cfg.get("architecture") != "resnet18":
        raise ValueError(
            "voxel RGB branches require an explicit resnet18 architecture"
        )
    if input_spec.rgb_views and rgb_cfg.get("augmentation") != "none":
        raise ValueError("voxel RGB branches currently require augmentation: none")
    query_field = voxel_cfg.get("query_field")
    if voxel_architecture == "voxel_focus_pool3d" and query_field not in proprio:
        raise ValueError(
            f"{voxel_architecture} requires {query_field!r} in input.proprio"
        )
    producer = next(
        item
        for item in source_observation.producers
        if isinstance(item, Spatial.VoxelProducerSpec)
    )
    attention_prior = _attention(voxel_cfg.get("attention_prior"))
    return Schema.VoxelEncoderSpec(
        name=name,
        voxel_architecture=voxel_architecture,
        voxel_key=input_spec.voxel.key,
        voxel_shape=(len(input_spec.voxel.channels),) + input_spec.voxel.resolution,
        rgb_keys=rgb_keys,
        proprio_fields=proprio,
        feature_dim=int(voxel_cfg["feature_dim"]),
        rgb_architecture=str(rgb_cfg.get("architecture", "resnet18")),
        rgb_feature_dim=int(rgb_cfg.get("feature_dim", 128)),
        rgb_pretrained_imagenet=bool(rgb_cfg.get("pretrained_imagenet", True)),
        rgb_norm=norm,
        rgb_random_crop=rgb_crop,
        crop_size=crop_size,
        coord_conv=bool(voxel_cfg.get("coord_conv", False)),
        num_heads=int(voxel_cfg.get("heads", 4)),
        num_iterations=int(voxel_cfg.get("iterations", 3)),
        pool_stage=int(voxel_cfg.get("pool_stage", 3)),
        attention_prior=attention_prior,
        workspace_min=producer.bounds_min,
        ws_size=producer.ws_size,
        data_requirements=Schema.DataRequirements(
            keypose_targets=bool(attention_prior and attention_prior.enabled)
        ),
    )


def _resolve_point_cloud_encoder(cfg, *, name, input_spec, proprio):
    point_cfg = dict(cfg.get("point_cloud_branch") or {})
    if point_cfg.get("architecture") != "pointnet":
        raise ValueError("point-cloud encoders require point_cloud_branch pointnet")
    fusion = dict(cfg.get("fusion") or {})
    state_dims = tuple(int(size) for size in fusion.get("state_mlp_dims", (64, 64)))
    feature_dim = int(point_cfg.get("feature_dim", 64))
    return Schema.PointCloudEncoderSpec(
        name=name,
        point_cloud_key=input_spec.point_cloud.key,
        point_cloud_shape=(
            input_spec.point_cloud.num_points,
            len(input_spec.point_cloud.channels),
        ),
        proprio_fields=proprio,
        feature_dim=feature_dim,
        state_mlp_dims=state_dims,
        use_color=bool(point_cfg.get("use_color", True)),
        use_layernorm=bool(point_cfg.get("use_layernorm", True)),
        final_norm=str(point_cfg.get("final_norm", "layernorm")),
    )


def _resolve_planar_encoder(
    cfg,
    *,
    name,
    input_spec,
    rgb_keys,
    proprio,
    norm,
    rgb_crop,
):
    feature_cfg = dict(cfg.get("feature_encoder") or {})
    architecture = str(feature_cfg.get("architecture", ""))
    if (
        architecture == "resnet18"
        and feature_cfg.get("weights_shared_between_views") is not False
    ):
        raise ValueError(
            "RGB feature encoders currently require unshared per-view weights"
        )
    if architecture == "resnet18":
        return Schema.RgbEncoderSpec(
            name=name,
            architecture=architecture,
            rgb_keys=rgb_keys,
            proprio_fields=proprio,
            feature_dim=int(feature_cfg["feature_dim"]),
            random_crop=rgb_crop,
            pretrained_imagenet=bool(feature_cfg.get("pretrained_imagenet", True)),
            norm=norm,
        )
    if architecture != "focus_pool2d":
        raise ValueError(f"unknown feature encoder architecture: {architecture!r}")
    query_field = str(feature_cfg.get("query_field", ""))
    if query_field not in proprio:
        raise ValueError(f"{architecture} requires {query_field!r} in input.proprio")
    attention_cfg = dict(cfg.get("attention_prior") or {})
    prior_view = str(attention_cfg.get("view", input_spec.rgb_views[0]))
    if attention_cfg.get("enabled") and prior_view not in input_spec.rgb_views:
        raise ValueError(
            f"attention-prior view {prior_view!r} is not selected by input "
            f"{input_spec.name!r}"
        )
    attention_prior = _attention(attention_cfg)
    return Schema.FocusRefineEncoderSpec(
        name=name,
        architecture=architecture,
        rgb_keys=rgb_keys,
        proprio_fields=proprio,
        gripper_key=query_field,
        input_res=Schema.SOURCE_RGB_SHAPE[-1],
        feature_dim=int(feature_cfg["feature_dim"]),
        random_crop=rgb_crop,
        num_heads=int(feature_cfg.get("heads", 4)),
        num_iterations=int(feature_cfg.get("iterations", 3)),
        pool_stage=int(feature_cfg.get("pool_stage", 2)),
        pretrained_imagenet=bool(feature_cfg.get("pretrained_imagenet", True)),
        norm=norm,
        attention_prior=attention_prior,
        attention_prior_view=f"rgb_{prior_view}",
        attention_prior_raw_res=Schema.SOURCE_RGB_SHAPE[-1],
        data_requirements=Schema.DataRequirements(
            keypose_targets=bool(attention_prior and attention_prior.enabled),
            oracle_info=bool(attention_prior and attention_prior.enabled),
        ),
    )


def _resolve_focus_conditioned_encoder(
    cfg,
    *,
    name,
    proprio,
    modes,
    view_augmentations,
    norm,
    rgb_crop,
    overlay_defaults,
    backgrounds_dir,
):
    feature_cfg = dict(cfg.get("feature_encoder") or {})
    architecture = str(feature_cfg.get("architecture", ""))
    if architecture != "resnet18":
        raise ValueError(
            "focus-conditioned encoders currently require feature_encoder resnet18"
        )
    focus_cfg = dict(cfg["focus_provider"])
    query_fields = tuple(str(field) for field in focus_cfg.get("query_fields", ()))
    if query_fields != tuple(Schema.ABSOLUTE_PROPRIO_SHAPES):
        raise ValueError(
            f"encoder {name!r} must declare all Seeker query fields in canonical order"
        )
    if tuple(proprio) != query_fields:
        raise ValueError(f"encoder {name!r} currently requires all declared query fields")
    source = Schema.FocusSourceSpec(
        name=str(focus_cfg["architecture"]),
        weights=focus_cfg.get("weights"),
        strict_weights=bool(focus_cfg.get("strict_weights", True)),
        checkpoint=focus_cfg.get("checkpoint"),
        checkpoint_views=tuple(
            str(view) for view in focus_cfg.get("checkpoint_views", ())
        ),
    )
    active_augmentations = {
        augmentation for _, augmentation in view_augmentations
    }
    unknown = active_augmentations.difference(
        {"none", "mask_guided_overlay", "random_background_overlay"}
    )
    if unknown:
        raise ValueError(f"unknown view augmentation strategies: {sorted(unknown)}")
    return Schema.FocusConditionedEncoderSpec(
        name=name,
        feature_architecture=architecture,
        source=source,
        view_modes=modes,
        view_augmentations=view_augmentations,
        view_keys=tuple((view, f"rgb_{view}") for view, _ in modes),
        proprio_fields=proprio,
        num_robots=MimicgenTasks.NUM_ROBOTS,
        feature_dim=int(feature_cfg["feature_dim"]),
        resnet_pretrained_imagenet=bool(
            feature_cfg.get("pretrained_imagenet", True)
        ),
        vit_in=int(cfg["focus_input_resolution"]),
        random_crop=rgb_crop,
        guided_overlay=(
            _overlay(overlay_defaults.get("mask_guided_overlay"), backgrounds_dir)
            if "mask_guided_overlay" in active_augmentations
            else None
        ),
        random_overlay=(
            _overlay(
                overlay_defaults.get("random_background_overlay"), backgrounds_dir
            )
            if "random_background_overlay" in active_augmentations
            else None
        ),
        norm=norm,
        data_requirements=Schema.DataRequirements(oracle_info=source.name == "oracle"),
    )


def _resolve_selected_rgb_crop(input_spec, rgb_crop):
    if not input_spec.rgb_views:
        return None
    rgb_crop = dict(rgb_crop)
    if not rgb_crop.pop("enable"):
        rgb_crop["resize"] = rgb_crop["output"]
    return _rgb_crop(rgb_crop)


def _resolve_selected_view_strategies(cfg, name, input_spec, regime):
    has_focus_provider = "focus_provider" in cfg
    view_strategy_cfg = cfg.get("view_strategies")
    if has_focus_provider:
        if view_strategy_cfg == "disabled":
            raise ValueError(
                f"focus-conditioned encoder {name!r} requires view strategies"
            )
        modes, augmentations = _view_strategies(cfg, input_spec, regime)
    else:
        if view_strategy_cfg != "disabled":
            raise ValueError(f"encoder {name!r} must set view_strategies: disabled")
        modes, augmentations = (), ()
    return has_focus_provider, modes, augmentations


def _resolve_selected_voxel_crop(input_spec, voxel_crop):
    if not bool(voxel_crop["enable"]):
        return None
    if input_spec.voxel is None:
        raise ValueError("voxel_crop augmentation requires voxel input")
    if voxel_crop["train"] != "random" or voxel_crop["evaluation"] != "center":
        raise ValueError(
            "voxel_crop must use random training crops and center evaluation crops"
        )
    crop_size = int(voxel_crop["output"])
    if crop_size <= 0:
        raise ValueError("voxel crop output must be positive")
    if crop_size > min(input_spec.voxel.resolution):
        raise ValueError(
            f"voxel crop output {crop_size} exceeds input resolution "
            f"{input_spec.voxel.resolution}"
        )
    return crop_size


def resolve_encoder(
    cfg,
    *,
    input_spec: Schema.InputSpec,
    source_observation: Schema.SourceObservationSpec,
    regime: Schema.RegimeSpec,
    backgrounds_dir: str,
    norm: str,
    rgb_crop,
    overlay_defaults,
    voxel_crop=None,
):
    """Resolve one explicit encoder preset against the selected input contract."""
    cfg = dict(_plain(cfg))
    name = str(cfg["name"])
    _validate_encoder_compatibility(cfg, input_spec)
    rgb_keys = tuple(f"rgb_{view}" for view in input_spec.rgb_views)
    rgb_crop = _resolve_selected_rgb_crop(input_spec, rgb_crop)
    proprio = input_spec.proprio
    _validate_fusion(cfg)
    has_focus_provider, modes, view_augmentations = (
        _resolve_selected_view_strategies(cfg, name, input_spec, regime)
    )
    crop_size = _resolve_selected_voxel_crop(input_spec, voxel_crop)

    if input_spec.voxel is not None:
        return _resolve_voxel_encoder(
            cfg,
            name=name,
            input_spec=input_spec,
            source_observation=source_observation,
            rgb_keys=rgb_keys,
            proprio=proprio,
            crop_size=crop_size,
            norm=norm,
            rgb_crop=rgb_crop,
        )

    if input_spec.point_cloud is not None:
        return _resolve_point_cloud_encoder(
            cfg, name=name, input_spec=input_spec, proprio=proprio
        )

    if not has_focus_provider:
        return _resolve_planar_encoder(
            cfg,
            name=name,
            input_spec=input_spec,
            rgb_keys=rgb_keys,
            proprio=proprio,
            norm=norm,
            rgb_crop=rgb_crop,
        )
    return _resolve_focus_conditioned_encoder(
        cfg,
        name=name,
        proprio=proprio,
        modes=modes,
        view_augmentations=view_augmentations,
        norm=norm,
        rgb_crop=rgb_crop,
        overlay_defaults=dict(overlay_defaults),
        backgrounds_dir=backgrounds_dir,
    )


POLICY_FAMILY_SPECS = {
    "global": Schema.GlobalPolicySpec,
}
# Policy-node keys the composed config consumes rather than the policy spec.
_NON_SPEC_POLICY_KEYS = ("name", "family", "wandb_project")


def _policy_family_values(cfg: Mapping[str, Any], family: str) -> dict:
    """The policy node's settings, checked against the family that consumes them.

    Hydra applies a primary config's ``_self_`` block after its group overrides,
    so a recipe patching ``policy.<key>`` keeps patching whichever policy is
    selected. Splatting the node would then either crash on a foreign key or
    silently drop one, depending on which family the override landed on.
    """
    spec = POLICY_FAMILY_SPECS.get(family)
    if spec is None:
        raise ValueError(f"unknown policy family: {family}")
    declared = {field.name for field in dataclasses.fields(spec)}
    values = {
        key: value
        for key, value in cfg.items()
        if key not in _NON_SPEC_POLICY_KEYS
    }
    foreign = sorted(set(values) - declared)
    if foreign:
        raise ValueError(
            f"policy {cfg['name']!r} of family {family!r} does not declare "
            f"{foreign}; those keys belong to another policy family, so the "
            "config setting them must select that family too"
        )
    return _tuple_sequences(values)


def resolve_policy(cfg):
    cfg = dict(_plain(cfg))
    name = str(cfg["name"])
    family = str(cfg["family"])
    values = _policy_family_values(cfg, family)
    observation_feature_dim = int(values.pop("observation_feature_dim"))
    generator_cfg = dict(values.pop("generator") or {})
    return Schema.GlobalPolicySpec(
        name,
        Schema.GeneratorSpec(kind=str(generator_cfg.pop("kind")), **generator_cfg),
        observation_feature_dim=observation_feature_dim,
        **values,
    )


def resolve_normalizer(*, dataset_path: str, task) -> str:
    """Automatically select global or per-robot linear normalization."""
    robots = dataset_robot_ids(dataset_path, task.name)
    return "multi_robot_linear" if len(robots) > 1 else "linear"


def _rgb_load_resolutions(encoder) -> tuple[tuple[str, int], ...]:
    if isinstance(encoder, Schema.PointCloudEncoderSpec):
        return ()
    if isinstance(encoder, Schema.RgbEncoderSpec):
        resolution = encoder.random_crop.input_res
        return tuple((key, resolution) for key in encoder.rgb_keys)
    if isinstance(encoder, Schema.FocusRefineEncoderSpec):
        resolution = (
            encoder.input_res
            if encoder.random_crop is None
            else encoder.random_crop.input_res
        )
        return tuple((key, resolution) for key in encoder.rgb_keys)
    if isinstance(encoder, Schema.VoxelEncoderSpec):
        resolution = (
            Schema.SOURCE_RGB_SHAPE[-1]
            if encoder.rgb_random_crop is None
            else encoder.rgb_random_crop.input_res
        )
        return tuple((key, resolution) for key in encoder.rgb_keys)
    if isinstance(encoder, Schema.FocusConditionedEncoderSpec):
        if encoder.source.name == "seeker":
            resolution = Schema.SOURCE_RGB_SHAPE[-1]
            return tuple((f"rgb_{view}", resolution) for view, _ in encoder.view_modes)
        low_resolution = encoder.random_crop.input_res
        return tuple(
            (
                f"rgb_{view}",
                low_resolution
                if mode in {"pass_through", "random_overlay"}
                else encoder.vit_in,
            )
            for view, mode in encoder.view_modes
        )
    raise TypeError(f"unknown encoder spec {type(encoder).__name__}")


def _observation_contract(
    input_spec: Schema.InputSpec,
    rgb_load_resolutions: tuple[tuple[str, int], ...],
) -> Schema.ObservationContract:
    resolutions = dict(rgb_load_resolutions)
    fields = [
        Schema.ObsFieldSpec(
            f"rgb_{view}",
            "rgb",
            (3, resolutions[f"rgb_{view}"], resolutions[f"rgb_{view}"]),
        )
        for view in input_spec.rgb_views
    ]
    for voxel in input_spec.voxels:
        fields.append(
            Schema.ObsFieldSpec(
                voxel.key,
                "voxel",
                (len(voxel.channels),) + voxel.resolution,
            )
        )
    if input_spec.point_cloud is not None:
        point_cloud = input_spec.point_cloud
        fields.append(
            Schema.ObsFieldSpec(
                point_cloud.key,
                "point_cloud",
                (point_cloud.num_points, len(point_cloud.channels)),
            )
        )
    fields.extend(
        Schema.ObsFieldSpec(key, "low_dim", Schema.PROPRIO_SHAPES[key])
        for key in input_spec.proprio
    )
    return Schema.ObservationContract(tuple(fields))


def resolve_augmentation_section(section: str, selections, pool) -> dict:
    selections = dict(_plain(selections) or {})
    pool = dict(_plain(pool) or {})
    unknown_methods = set(selections) - set(pool)
    if unknown_methods:
        raise ValueError(f"unknown {section} methods: {sorted(unknown_methods)}")
    missing_methods = set(pool) - set(selections)
    if missing_methods:
        raise ValueError(f"missing {section} selections: {sorted(missing_methods)}")

    resolved = {}
    for name, defaults in pool.items():
        defaults = dict(defaults or {})
        selection = selections[name]
        if isinstance(selection, str):
            method = selection
            overrides = {}
        elif isinstance(selection, Mapping):
            overrides = dict(selection)
            method = overrides.pop("method", None)
        else:
            raise TypeError(
                f"{section}.{name} must be a method string or override mapping"
            )
        if method not in ("enabled", "disabled"):
            raise ValueError(
                f"{section}.{name} method must be 'enabled' or 'disabled'"
            )
        unknown_options = set(overrides) - set(defaults)
        if unknown_options:
            raise ValueError(
                f"unknown {section}.{name} options: {sorted(unknown_options)}"
            )
        resolved[name] = {**defaults, **overrides, "enable": method == "enabled"}
    return resolved


def resolve_mirror_augmentation(cfg) -> Optional[MirrorAugmentationConfig]:
    return MirrorAugmentationConfig.from_config(cfg)


def voxel_grid_geometry(
    source_observation: Schema.SourceObservationSpec,
) -> SourceVoxelGeometry:
    """The one resolved source voxel grid, as the geometry the encoder also uses."""
    voxel_producers = [
        producer
        for producer in source_observation.producers
        if isinstance(producer, Spatial.VoxelProducerSpec)
    ]
    if len(voxel_producers) != 1:
        raise ValueError(
            "enabled scene_yaw augmentation requires exactly one resolved voxel workspace"
        )
    producer = voxel_producers[0]
    if producer.bounds_min is None or producer.bounds_max is None:
        raise ValueError(
            "enabled scene_yaw augmentation requires resolved voxel bounds"
        )
    return SourceVoxelGeometry(
        producer.bounds_min, producer.ws_size, producer.resolution
    )


def resolve_scene_yaw_augmentation(
    cfg, *, source_observation: Schema.SourceObservationSpec
) -> Optional[SceneYawAugmentationConfig]:
    """Resolve the yaw workspace onto the voxel array's own footprint.

    The augmentation rotates the grid about the array centre, so world points
    have to rotate about that same centre or obs and state shear apart; the
    array footprint is ``pitch * resolution``, which the producer's ``+1e-4``
    pitch makes slightly wider than ``ws_size``.
    """
    cfg = dict(cfg)
    enabled = bool(cfg["enable"])
    if enabled:
        geometry = voxel_grid_geometry(source_observation)
        center_xy = geometry.center[:2]
        extent_xy = geometry.extent[:2]
        if abs(extent_xy[0] - extent_xy[1]) > 1e-9:
            raise ValueError(
                f"scene_yaw augmentation requires a square voxel X/Y footprint, got {extent_xy}"
            )
        workspace = cfg.get("workspace")
        if workspace is None:
            cfg["workspace"] = {"center_xy": list(center_xy), "size": extent_xy[0]}
        else:
            configured = tuple(float(value) for value in workspace["center_xy"])
            if max(abs(a - b) for a, b in zip(configured, center_xy)) > 1e-6:
                raise ValueError(
                    "scene_yaw_augmentation.workspace.center_xy must be the voxel array "
                    f"centre {tuple(center_xy)}, got {configured}"
                )
            if float(workspace["size"]) > extent_xy[0] + 1e-6:
                raise ValueError(
                    "scene_yaw_augmentation.workspace.size must not exceed the voxel "
                    f"footprint {extent_xy[0]}, got {workspace['size']}"
                )
    return SceneYawAugmentationConfig.from_config(cfg)


def resolve_training(cfg: Mapping[str, Any]) -> Schema.TrainingSpec:
    training = dict(cfg["training"])
    n_demo = max(1, int(cfg["n_demo"]))
    optimizer = dict(cfg["optimizer"])

    def per_demo(total: int) -> int:
        return max(1, int(total) // n_demo)

    return Schema.TrainingSpec(
        device=str(training["device"]),
        seed=int(cfg["seed"]),
        debug=bool(training.get("debug", False)),
        resume=bool(training.get("resume", True)),
        num_epochs=per_demo(int(training["total_steps"])),
        lr=float(optimizer["lr"]),
        betas=tuple(float(value) for value in optimizer["betas"]),
        eps=float(optimizer["eps"]),
        weight_decay=float(optimizer["weight_decay"]),
        lr_scheduler=str(training.get("lr_scheduler", "cosine")),
        lr_warmup_steps=int(training.get("lr_warmup_steps", 500)),
        use_ema=bool(training.get("use_ema", True)),
        rollout_every=per_demo(int(training["rollout_interval_steps"])),
        checkpoint_every=per_demo(int(training["checkpoint_interval_steps"])),
        val_every=int(training.get("val_every", 10)),
        max_train_steps=training.get("max_train_steps"),
        max_val_steps=training.get("max_val_steps"),
        tqdm_interval_sec=float(training.get("tqdm_interval_sec", 1.0)),
    )


def _resolve_dataloader(cfg) -> Schema.DataLoaderSpec:
    cfg = dict(cfg)
    return Schema.DataLoaderSpec(
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle=bool(cfg["shuffle"]),
        pin_memory=bool(cfg["pin_memory"]),
        persistent_workers=bool(cfg["persistent_workers"]),
        drop_last=bool(cfg.get("drop_last", False)),
    )


def _resolve_visualization(cfg) -> Schema.VisualizationSpec:
    cfg = dict(cfg or {})
    save = dict(cfg.get("save") or {})
    upload = dict(cfg.get("upload") or {})
    return Schema.VisualizationSpec(
        enabled=bool(cfg.get("enabled", True)),
        num_samples=int(cfg.get("num_samples", 6)),
        augmentation_preview=bool(cfg.get("augmentation_preview", True)),
        save=Schema.MediaToggleSpec(
            images=bool(save.get("images", True)),
            videos=bool(save.get("videos", True)),
        ),
        upload=Schema.MediaToggleSpec(
            images=bool(upload.get("images", False)),
            videos=bool(upload.get("videos", False)),
        ),
    )


def resolve_policy_workspace(cfg) -> Schema.PolicyWorkspaceSpec:
    ema = dict(cfg["ema"])
    logging = dict(cfg["logging"])
    checkpoint = dict(cfg["checkpoint"])
    topk = dict(checkpoint["topk"])
    return Schema.PolicyWorkspaceSpec(
        train_loader=_resolve_dataloader(cfg["dataloader"]),
        val_loader=_resolve_dataloader(cfg["val_dataloader"]),
        ema=Schema.EmaSpec(
            update_after_step=int(ema["update_after_step"]),
            inv_gamma=float(ema["inv_gamma"]),
            power=float(ema["power"]),
            min_value=float(ema["min_value"]),
            max_value=float(ema["max_value"]),
        ),
        logging=Schema.LoggingSpec(
            project=str(logging["project"]),
            resume=bool(logging["resume"]),
            name=str(logging["name"]),
            group=str(logging["group"]),
            job_type=str(logging["job_type"]),
            tags=tuple(str(tag) for tag in logging.get("tags", ())),
            mode=logging.get("mode"),
            run_id=logging.get("id"),
        ),
        checkpoint=Schema.CheckpointSpec(
            topk=Schema.TopKCheckpointSpec(
                monitor_key=str(topk["monitor_key"]),
                mode=str(topk["mode"]),
                k=int(topk["k"]),
                format_str=str(topk["format_str"]),
            ),
            save_last=bool(checkpoint["save_last_ckpt"]),
            save_topk_full=bool(checkpoint.get("save_topk_full_ckpt", False)),
        ),
        visualization=_resolve_visualization(cfg.get("visualization")),
    )


def _resolve_logging(cfg) -> Schema.LoggingSpec:
    cfg = dict(cfg)
    return Schema.LoggingSpec(
        project=str(cfg["project"]),
        resume=bool(cfg["resume"]),
        name=str(cfg["name"]),
        group=str(cfg["group"]),
        job_type=str(cfg["job_type"]),
        tags=tuple(str(tag) for tag in cfg.get("tags", ())),
        mode=cfg.get("mode"),
        run_id=cfg.get("id"),
    )


def _resolve_sampler(cfg) -> Optional[Schema.SamplerSpec]:
    if not cfg:
        return None
    cfg = dict(cfg)
    return Schema.SamplerSpec(
        kind=str(cfg["type"]),
        samples_per_epoch=int(cfg["samples_per_epoch"]),
    )


def resolve_seeker_pretraining(cfg) -> Schema.SeekerPretrainingSpec:
    """Resolve Seeker pretraining before constructing runtime objects."""
    cfg = dict(_plain(cfg))
    n_demo = max(1, int(cfg["n_demo"]))
    trajectory = Schema.TrajectoryContract(
        action_dim=int(cfg["action_dim"]),
        prediction_horizon=int(cfg["horizon"]),
        observation_horizon=int(cfg["n_obs_steps"]),
        execution_horizon=int(cfg["n_action_steps"])
        + int(cfg.get("n_latency_steps", 0)),
        action_rep=str(cfg["action_rep"]),
    )
    input_spec = Schema.InputSpec(
        name="seeker_pretraining",
        rgb_views=("external", "wrist"),
        proprio=tuple(Schema.ABSOLUTE_PROPRIO_SHAPES),
    )
    source = resolve_source_observation(input_spec, task=None)
    policy_cfg = dict(cfg["policy"])
    scheduler_cfg = dict(policy_cfg["noise_scheduler"])
    generator = Schema.GeneratorSpec(
        kind="diffusion",
        unet_channels=tuple(int(value) for value in policy_cfg["unet_channels"]),
        kernel_size=int(policy_cfg["kernel_size"]),
        n_groups=int(policy_cfg["n_groups"]),
        cond_predict_scale=bool(policy_cfg["cond_predict_scale"]),
        num_inference_steps=int(policy_cfg["num_inference_steps"]),
        diffusion_step_embed_dim=int(policy_cfg["diffusion_step_embed_dim"]),
        num_train_timesteps=int(scheduler_cfg["num_train_timesteps"]),
        beta_start=float(scheduler_cfg["beta_start"]),
        beta_end=float(scheduler_cfg["beta_end"]),
        beta_schedule=str(scheduler_cfg["beta_schedule"]),
        variance_type=str(scheduler_cfg["variance_type"]),
        clip_sample=bool(scheduler_cfg["clip_sample"]),
        prediction_type=str(scheduler_cfg["prediction_type"]),
    )
    model = Schema.SeekerPretrainingModelSpec(
        visual_mode=str(cfg["visual_mode"]),
        image_size=int(cfg["image_size"]),
        obs_dropout=float(cfg["obs_dropout"]),
        stage_stride=max(1, int(cfg["stage_total_steps"]) // n_demo),
        weights=str(cfg["paths"]["weights_dir"]) + "/seeker.mimicgen.pth",
        background_path=str(cfg["paths"]["backgrounds_dir"]),
        generator=generator,
        num_refinement_iters=int(cfg["seeker_num_iters"]),
        disable_proprio=bool(cfg["seeker_disable_proprio"]),
        disable_head_gating=bool(cfg["seeker_disable_head_gating"]),
    )
    dataset = Schema.SeekerPretrainingDatasetSpec(
        path=str(cfg["dataset_path"]),
        source_observation=source,
        trajectory=trajectory,
        n_demo=n_demo,
        demo_count_mode=str(cfg["dataset_demo_count_mode"]),
        image_size=int(cfg["image_size"]),
        cache_dir=cfg.get("cache_dir"),
    )
    training_cfg = dict(cfg["training"])
    optimizer = dict(cfg["optimizer"])
    training = Schema.TrainingSpec(
        device=str(training_cfg["device"]),
        seed=int(cfg["seed"]),
        debug=bool(training_cfg["debug"]),
        resume=bool(training_cfg["resume"]),
        num_epochs=max(1, int(training_cfg["total_steps"]) // n_demo),
        lr=float(optimizer["lr"]),
        betas=tuple(float(value) for value in optimizer["betas"]),
        eps=float(optimizer["eps"]),
        weight_decay=float(optimizer["weight_decay"]),
        lr_scheduler=str(training_cfg["lr_scheduler"]),
        lr_warmup_steps=int(training_cfg["lr_warmup_steps"]),
        use_ema=bool(training_cfg["use_ema"]),
        checkpoint_every=int(training_cfg["checkpoint_every"]),
        max_train_steps=training_cfg.get("max_train_steps"),
        tqdm_interval_sec=float(training_cfg["tqdm_interval_sec"]),
    )
    loader_cfg = dict(cfg["dataloader"])
    ema_cfg = dict(cfg["ema"])
    checkpoint_cfg = dict(cfg["checkpoint"])
    return Schema.SeekerPretrainingSpec(
        model=model,
        dataset=dataset,
        training=training,
        workspace=Schema.SeekerPretrainingWorkspaceSpec(
            train_loader=_resolve_dataloader(loader_cfg),
            sampler=_resolve_sampler(loader_cfg.get("sampler")),
            ema=Schema.EmaSpec(
                update_after_step=int(ema_cfg["update_after_step"]),
                inv_gamma=float(ema_cfg["inv_gamma"]),
                power=float(ema_cfg["power"]),
                min_value=float(ema_cfg["min_value"]),
                max_value=float(ema_cfg["max_value"]),
            ),
            logging=_resolve_logging(cfg["logging"]),
            checkpoint=Schema.SeekerPretrainingCheckpointSpec(
                save_last=bool(checkpoint_cfg["save_last_ckpt"]),
                save_snapshot=bool(checkpoint_cfg["save_last_snapshot"]),
                seeker_light_path=checkpoint_cfg.get("seeker_light_path"),
            ),
            visualization=_resolve_visualization(cfg.get("visualization")),
        ),
    )


def resolve_rvt2_pretraining(cfg) -> Schema.Rvt2PretrainingSpec:
    """Resolve RVT2 pretraining and its model defaults at the Hydra boundary."""
    from visuomotor.perception.focus.rvt2 import config as Rvt2Config

    cfg = dict(_plain(cfg))
    n_demo = None if cfg.get("n_demo") is None else int(cfg["n_demo"])
    overrides = {
        key: cfg[key]
        for key in ("rvt2_heatmap", "query_composer")
        if isinstance(cfg.get(key), Mapping)
    }
    heatmap = Rvt2Config.load_rvt2_heatmap_config(overrides=overrides or None)
    query = Rvt2Config.load_rvt2_query_config(overrides=overrides or None)
    return Schema.Rvt2PretrainingSpec(
        model=Schema.Rvt2PretrainingModelSpec(
            heatmap=heatmap,
            query_composer=query,
            background_overlay=dict(cfg.get("background_overlay") or {}),
            stage_stride=max(
                1,
                int(cfg["stage_total_steps"]) // max(1, n_demo or 1),
            ),
        ),
        dataset=Schema.Rvt2PretrainingDatasetSpec(
            task_name=str(cfg["task_name"]),
            path=str(cfg["dataset_path"]),
            cache_dir=cfg.get("cache_dir"),
            n_demo=n_demo,
            demo_count_mode=str(cfg["dataset_demo_count_mode"]),
            skip_first_episodes=int(cfg["skip_first_episodes"]),
            val_ratio=float(cfg["val_ratio"]),
        ),
        training=Schema.Rvt2PretrainingTrainingSpec(
            epochs=int(cfg["epochs"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            device=str(cfg["device"]),
            resume=cfg.get("resume"),
            load_weights=cfg.get("load_weights"),
            resume_optimizer=not bool(cfg.get("no_resume_optimizer", False)),
            checkpoint_every=int(cfg["checkpoint_every"]),
            max_train_steps=cfg.get("max_train_steps"),
            max_val_steps=cfg.get("max_val_steps"),
            seed=int(cfg["seed"]),
        ),
        workspace=Schema.Rvt2PretrainingWorkspaceSpec(
            train_loader=Schema.DataLoaderSpec(
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                shuffle=True,
                pin_memory=True,
                persistent_workers=int(cfg["num_workers"]) > 0,
            ),
            val_loader=Schema.DataLoaderSpec(
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                shuffle=False,
                pin_memory=True,
                persistent_workers=int(cfg["num_workers"]) > 0,
            ),
            sampler=_resolve_sampler(cfg.get("sampler")),
            logging=_resolve_logging(cfg["logging"]) if cfg.get("logging") else None,
            print_epoch_metrics=bool(cfg["print_epoch_metrics"]),
            print_run_summary=bool(cfg["print_run_summary"]),
            show_label_progress=bool(cfg["show_label_progress"]),
            visualization=_resolve_visualization(cfg.get("visualization")),
            visualization_alpha=float(cfg.get("vis_alpha", 0.45)),
            visualization_sampling=str(cfg.get("vis_sampling", "even")),
        ),
    )


def _dataset_path(cfg, task, regime) -> str:
    root = str(cfg["paths"]["dataset_root"])
    return os.path.join(root, task.name, f"{task.name}_lmdb{regime.dataset_suffix}")


def _keypose_support_observation(source, requirements):
    """Widen the source with the pose fields keypose targets are derived from.

    Only a keypose-target policy reads these. Widening unconditionally would
    change every other policy's source observation, and with it what the
    dataset cache has to render.
    """
    if not requirements.keypose_targets:
        return source
    bound_sources = {
        field.source_key for field in source.fields + source.support_fields
    }
    support = []
    for canonical_key in ("eef_pos", "eef_rot6d"):
        source_key, source_shape = MimicgenObservations.source_proprio_field(
            canonical_key
        )
        if source_key not in bound_sources:
            support.append(
                Schema.SourceObsFieldSpec(source_key, "low_dim", source_shape)
            )
    if not support:
        return source
    return Schema.SourceObservationSpec(
        fields=source.fields,
        support_fields=source.support_fields + tuple(support),
        producers=source.producers,
    )


def resolve_policy_run(cfg) -> Schema.RunSpec:
    """Resolve and validate one experiment composition."""
    cfg = dict(_plain(cfg))
    task = get_task_spec(cfg["task"])
    regime = resolve_regime(cfg["regime"])
    training = resolve_training(cfg)
    workspace = resolve_policy_workspace(cfg)
    trajectory = Schema.TrajectoryContract(
        action_dim=int(cfg["action_dim"]),
        prediction_horizon=int(cfg["horizon"]),
        observation_horizon=int(cfg["n_obs_steps"]),
        execution_horizon=int(cfg["n_action_steps"]),
        action_rep=str(cfg["action_rep"]),
    )
    input_spec = resolve_input(cfg["input"])
    source = resolve_source_observation(input_spec, task=task)
    augmentation_defaults = dict(cfg.get("augmentation_defaults") or {})
    input_defaults = dict(augmentation_defaults.get("input") or {})
    overlay_defaults = {
        name: input_defaults.pop(name)
        for name in ("mask_guided_overlay", "random_background_overlay")
    }
    input_augmentation = resolve_augmentation_section(
        "input_augmentation",
        cfg.get("input_augmentation"),
        input_defaults,
    )
    # Normalization is part of the model architecture and must not change when
    # EMA is enabled or disabled for training.
    norm_layer = "groupnorm"
    encoder = resolve_encoder(
        cfg["encoder"],
        input_spec=input_spec,
        source_observation=source,
        regime=regime,
        backgrounds_dir=str(cfg["paths"]["backgrounds_dir"]),
        norm=norm_layer,
        rgb_crop=input_augmentation["rgb_crop"],
        overlay_defaults=overlay_defaults,
        voxel_crop=input_augmentation["voxel_crop"],
    )
    path = _dataset_path(cfg, task, regime)
    normalizer = resolve_normalizer(dataset_path=path, task=task)
    policy = resolve_policy(cfg["policy"])
    rgb_load_resolutions = _rgb_load_resolutions(encoder)
    observation = _observation_contract(input_spec, rgb_load_resolutions)
    model = Schema.ModelSpec(
        input_spec, observation, encoder, policy, normalizer, trajectory
    )

    mirror = resolve_mirror_augmentation(input_augmentation["mirror"])
    scene_yaw = resolve_scene_yaw_augmentation(
        input_augmentation["scene_yaw"], source_observation=source
    )
    requirements = model.data_requirements
    source = _keypose_support_observation(source, requirements)
    keypose = None
    if requirements.keypose_targets:
        keypose = Schema.KeyposeTargetSpec()
    common = dict(
        path=path,
        observation=observation,
        source_observation=source,
        trajectory=trajectory,
        rgb_load_resolutions=rgb_load_resolutions,
        n_demo=int(cfg["n_demo"]),
        val_ratio=float(cfg["val_ratio"]),
        cache_dir=cfg.get("cache_dir"),
        include_oracle_info=requirements.oracle_info,
        include_camera_matrices=False,
        mirror_augmentation=mirror,
        scene_yaw_augmentation=scene_yaw,
        keypose_targets=keypose,
    )
    dataset = Schema.DatasetSpec(**common)
    rollout = dict(cfg["rollout"])
    runner = Schema.RunnerSpec(
        dataset_path=path,
        observation=observation,
        source_observation=source,
        rgb_load_resolutions=rgb_load_resolutions,
        trajectory=trajectory,
        env_name=cfg.get("env_name"),
        cache_dir=cfg.get("cache_dir"),
        n_test=int(rollout["n_test"]),
        n_test_vis=int(rollout["n_test_vis"]),
        test_start_seed=int(rollout["test_start_seed"]),
        max_steps=task.max_steps,
        n_envs=int(rollout["n_envs"]),
        render_obs_key=MimicgenObservations.source_camera_key("external"),
        shuffle_table_texture=regime.shuffle_table_texture,
        enable_oracle_focus_info=requirements.oracle_info,
        mirror_augmentation=mirror,
        fps=int(rollout.get("fps", 10)),
        crf=int(rollout.get("crf", 28)),
        visualization=workspace.visualization,
    )
    exp_name = str(cfg["exp_name"])
    return Schema.validate(
        Schema.RunSpec(
            task, regime, model, dataset, runner, training, workspace, exp_name
        )
    )
