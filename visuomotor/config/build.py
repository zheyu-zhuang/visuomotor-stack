"""Build runtime objects exclusively from resolved typed specs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from visuomotor.config import schema as Schema
from visuomotor.data.core import spatial as Spatial


@dataclass(frozen=True)
class RolloutCheckpoint:
    """Constructed policy and runner recovered from any rollout-capable checkpoint."""

    policy: torch.nn.Module
    runner_spec: Schema.RunnerSpec
    checkpoint_format: str
    weights: str
    epoch: Optional[int] = None
    global_step: Optional[int] = None
    training_demo_count: Optional[int] = None


def build_dataloader(dataset, spec: Schema.DataLoaderSpec, *, sampler=None):
    """Construct a dataloader from one resolved loader specification."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        shuffle=spec.shuffle if sampler is None else False,
        sampler=sampler,
        pin_memory=spec.pin_memory,
        persistent_workers=spec.persistent_workers and len(dataset) > 0,
        drop_last=spec.drop_last,
    )


class _InfiniteSampler(torch.utils.data.Sampler):
    """Repeat the data source indefinitely, reshuffling between laps when enabled."""

    def __init__(self, data_source, shuffle: bool) -> None:
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            if self.shuffle:
                yield from torch.randperm(len(self.data_source)).tolist()
            else:
                yield from range(len(self.data_source))

    def __len__(self) -> int:
        return len(self.data_source)


def build_infinite_dataloader(dataset, spec: Schema.DataLoaderSpec):
    """Build an endless dataloader whose length reports one lap."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        sampler=_InfiniteSampler(dataset, shuffle=spec.shuffle),
        pin_memory=spec.pin_memory,
        persistent_workers=spec.persistent_workers and len(dataset) > 0,
        drop_last=spec.drop_last,
    )


def build_sampler(dataset, spec: Optional[Schema.SamplerSpec], *, seed: int):
    """Construct the configured task sampler without exposing config containers."""
    from visuomotor.data.core.sampler import build_task_balanced_sampler

    sampler_config = None
    if spec is not None:
        sampler_config = {
            "type": spec.kind,
            "samples_per_epoch": spec.samples_per_epoch,
        }
    return build_task_balanced_sampler(dataset, sampler_config, seed=seed)


def _proprio_dims(fields):
    return tuple(Schema.PROPRIO_SHAPES[key][0] for key in fields)


def _build_rgb_encoder(spec: Schema.RgbEncoderSpec):
    from visuomotor.perception.encoder.rgb import MultiViewRgbEncoder

    return MultiViewRgbEncoder(
        encoder_name=spec.name,
        rgb_keys=spec.rgb_keys,
        proprio_fields=spec.proprio_fields,
        proprio_dims=_proprio_dims(spec.proprio_fields),
        feature_dim=spec.feature_dim,
        random_crop=spec.random_crop,
        pretrained_imagenet=spec.pretrained_imagenet,
        norm=spec.norm,
    )


def _build_focus_conditioned_encoder(
    spec: Schema.FocusConditionedEncoderSpec, normalizer: str
):
    from visuomotor.perception.encoder.focus_conditioned import (
        FocusConditionedObsEncoder,
    )

    return FocusConditionedObsEncoder(spec, normalizer_kind=normalizer)


def _build_focus_refine_encoder(spec: Schema.FocusRefineEncoderSpec):
    from visuomotor.perception.encoder.focus_pool import FocusRefineEncoder

    return FocusRefineEncoder(
        spatial_rank=2,
        input_channels=spec.input_channels,
        input_res=spec.input_res,
        feature_dim=spec.feature_dim,
        rgb_keys=spec.rgb_keys,
        gripper_key=spec.gripper_key,
        proprio_fields=spec.proprio_fields,
        proprio_dims=_proprio_dims(spec.proprio_fields),
        random_crop=spec.random_crop,
        num_heads=spec.num_heads,
        num_iterations=spec.num_iterations,
        pool_stage=spec.pool_stage,
        pretrained_imagenet=spec.pretrained_imagenet,
        norm=spec.norm,
        attention_prior=spec.attention_prior,
        attention_prior_view=spec.attention_prior_view,
        attention_prior_raw_res=spec.attention_prior_raw_res,
    )


def _build_voxel_encoder(spec: Schema.VoxelEncoderSpec):
    from visuomotor.perception.encoder.voxel import VoxelObservationEncoder

    return VoxelObservationEncoder(
        encoder_name=spec.name,
        voxel_key=spec.voxel_key,
        source_shape=spec.voxel_shape,
        crop_size=spec.crop_size,
        voxel_architecture=spec.voxel_architecture,
        rgb_keys=spec.rgb_keys,
        rgb_architecture=spec.rgb_architecture,
        proprio_fields=spec.proprio_fields,
        proprio_dims=_proprio_dims(spec.proprio_fields),
        feature_dim=spec.feature_dim,
        rgb_feature_dim=spec.rgb_feature_dim,
        rgb_pretrained_imagenet=spec.rgb_pretrained_imagenet,
        rgb_norm=spec.rgb_norm,
        rgb_random_crop=spec.rgb_random_crop,
        coord_conv=spec.coord_conv,
        num_heads=spec.num_heads,
        num_iterations=spec.num_iterations,
        pool_stage=spec.pool_stage,
        attention_prior=spec.attention_prior,
        source_workspace_min=spec.workspace_min,
        source_workspace_size=spec.ws_size,
    )


def _build_point_cloud_encoder(spec: Schema.PointCloudEncoderSpec):
    from visuomotor.perception.encoder.point_cloud import PointCloudObservationEncoder

    return PointCloudObservationEncoder(
        encoder_name=spec.name,
        point_cloud_key=spec.point_cloud_key,
        source_shape=spec.point_cloud_shape,
        proprio_fields=spec.proprio_fields,
        proprio_dims=_proprio_dims(spec.proprio_fields),
        feature_dim=spec.feature_dim,
        state_mlp_dims=spec.state_mlp_dims,
        use_color=spec.use_color,
        use_layernorm=spec.use_layernorm,
        final_norm=spec.final_norm,
    )


def build_encoder(spec, *, normalizer: str = "multi_robot_linear"):
    if isinstance(spec, Schema.RgbEncoderSpec):
        return _build_rgb_encoder(spec)
    if isinstance(spec, Schema.FocusConditionedEncoderSpec):
        return _build_focus_conditioned_encoder(spec, normalizer)
    if isinstance(spec, Schema.FocusRefineEncoderSpec):
        return _build_focus_refine_encoder(spec)
    if isinstance(spec, Schema.VoxelEncoderSpec):
        return _build_voxel_encoder(spec)
    if isinstance(spec, Schema.PointCloudEncoderSpec):
        return _build_point_cloud_encoder(spec)
    raise TypeError(f"unknown encoder spec: {type(spec).__name__}")


def build_generator(
    spec: Schema.GeneratorSpec,
    *,
    trajectory: Schema.TrajectoryContract,
    condition_dim: int,
):
    shared = dict(
        action_dim=trajectory.action_dim,
        condition_dim=condition_dim,
        prediction_horizon=trajectory.prediction_horizon,
        unet_channels=spec.unet_channels,
        kernel_size=spec.kernel_size,
        n_groups=spec.n_groups,
        cond_predict_scale=spec.cond_predict_scale,
    )
    if spec.kind == "diffusion":
        from visuomotor.action_generation.diffusion import DiffusionActionGenerator

        if spec.scheduler == "ddim":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            scheduler = DDIMScheduler
        else:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            scheduler = DDPMScheduler

        scheduler_kwargs = {
            "num_train_timesteps": spec.num_train_timesteps,
            "beta_start": spec.beta_start,
            "beta_end": spec.beta_end,
            "beta_schedule": spec.beta_schedule,
            "clip_sample": spec.clip_sample,
            "prediction_type": spec.prediction_type,
        }
        if spec.scheduler == "ddpm":
            scheduler_kwargs["variance_type"] = spec.variance_type

        return DiffusionActionGenerator(
            noise_scheduler=scheduler(**scheduler_kwargs),
            num_inference_steps=spec.num_inference_steps,
            diffusion_step_embed_dim=spec.diffusion_step_embed_dim,
            **shared,
        )
    from visuomotor.action_generation.flow_matching import FlowMatchingActionGenerator

    return FlowMatchingActionGenerator(
        integration_steps=spec.integration_steps,
        time_embedding_dim=spec.time_embedding_dim,
        time_embedding_scale=spec.time_embedding_scale,
        clip_sample=spec.clip_sample,
        **shared,
    )


def build_normalizer(kind: str):
    from visuomotor.data.core.normalization import build_normalizer_module

    return build_normalizer_module(kind)


def build_policy(spec: Schema.ModelSpec):
    """Single construction path for policy construction and checkpoint restore."""
    encoder = build_encoder(spec.encoder, normalizer=spec.normalizer)
    observation_kinds = {field.key: field.kind for field in spec.observation.fields}
    if isinstance(spec.policy, Schema.GlobalPolicySpec):
        from visuomotor.policy.generative import GenerativePolicy

        return GenerativePolicy(
            encoder=encoder,
            generator=build_generator(
                spec.policy.generator,
                trajectory=spec.trajectory,
                condition_dim=spec.policy.observation_feature_dim
                * spec.trajectory.observation_horizon,
            ),
            observation_kinds=observation_kinds,
            observation_feature_dim=spec.policy.observation_feature_dim,
            action_normalizer=build_normalizer(spec.normalizer),
            execution_horizon=spec.trajectory.execution_horizon,
            model_spec=spec,
        )
    raise TypeError(f"unknown policy spec: {type(spec.policy).__name__}")


def build_dataset(spec: Schema.DatasetSpec, *, mode: Optional[str] = None):
    from visuomotor.data.mimicgen.dataset import MimicGenDataset

    voxel_specs, point_cloud_spec = sensor_specs(spec.source_observation)
    kwargs = dict(
        shape_meta=spec.source_observation.shape_meta(spec.trajectory.action_dim),
        dataset_path=spec.path,
        image_size=None,
        rgb_load_resolutions=dict(spec.rgb_load_resolutions),
        horizon=spec.trajectory.prediction_horizon,
        val_ratio=spec.val_ratio,
        n_demo=spec.n_demo,
        demo_count_mode=spec.demo_count_mode,
        n_obs_steps=spec.trajectory.observation_horizon,
        action_rep=spec.trajectory.action_rep,
        lmdb_readahead=spec.lmdb_readahead,
        cache_dir=spec.cache_dir,
        include_oracle_info=spec.include_oracle_info,
        include_camera_matrices=spec.include_camera_matrices,
        mirror_augmentation=spec.mirror_augmentation,
        scene_yaw_augmentation=spec.scene_yaw_augmentation,
        voxel_specs=voxel_specs,
        point_cloud_spec=point_cloud_spec,
    )
    if spec.keypose_targets is not None:
        kwargs.update(
            keypose_targets=True,
            keypose_gripper_motion_threshold=(
                spec.keypose_targets.gripper_motion_threshold
            ),
            keypose_gripper_valley_threshold=(
                spec.keypose_targets.gripper_valley_threshold
            ),
            keypose_gripper_valley_window=(spec.keypose_targets.gripper_valley_window),
        )
    dataset = MimicGenDataset(**kwargs)
    if mode is not None:
        dataset.set_mode(mode)
    return dataset


def build_seeker_pretraining_policy(spec: Schema.SeekerPretrainingSpec):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    from visuomotor.data.mimicgen import tasks as MimicgenTasks
    from visuomotor.policy.staged_seeker import SeekerTrainingPolicy

    model = spec.model
    generator = model.generator
    trajectory = spec.dataset.trajectory
    scheduler = DDPMScheduler(
        num_train_timesteps=generator.num_train_timesteps,
        beta_start=generator.beta_start,
        beta_end=generator.beta_end,
        beta_schedule=generator.beta_schedule,
        variance_type=generator.variance_type,
        clip_sample=generator.clip_sample,
        prediction_type=generator.prediction_type,
    )
    return SeekerTrainingPolicy(
        shape_meta=spec.dataset.source_observation.shape_meta(trajectory.action_dim),
        noise_scheduler=scheduler,
        horizon=trajectory.prediction_horizon,
        n_action_steps=trajectory.execution_horizon,
        n_obs_steps=trajectory.observation_horizon,
        image_size=model.image_size,
        num_robots=MimicgenTasks.NUM_ROBOTS,
        weights=model.weights,
        stage_stride=model.stage_stride,
        visual_mode=model.visual_mode,
        obs_dropout=model.obs_dropout,
        action_rep=trajectory.action_rep,
        num_inference_steps=generator.num_inference_steps,
        diffusion_step_embed_dim=generator.diffusion_step_embed_dim,
        unet_channels=generator.unet_channels,
        kernel_size=generator.kernel_size,
        n_groups=generator.n_groups,
        cond_predict_scale=generator.cond_predict_scale,
        background_path=model.background_path,
        seeker_overrides={
            "intent_refiner": {
                "num_refinement_iters": model.num_refinement_iters,
                "disable_head_gating": model.disable_head_gating,
            },
            "query_composer": {"disable_proprio": model.disable_proprio},
        },
    )


def build_seeker_pretraining_dataset(spec: Schema.SeekerPretrainingDatasetSpec):
    from visuomotor.data.mimicgen.dataset import MimicGenDataset

    trajectory = spec.trajectory
    voxel_specs, point_cloud_spec = sensor_specs(spec.source_observation)
    return MimicGenDataset(
        shape_meta=spec.source_observation.shape_meta(trajectory.action_dim),
        dataset_path=spec.path,
        n_demo=spec.n_demo,
        demo_count_mode=spec.demo_count_mode,
        horizon=trajectory.prediction_horizon,
        val_ratio=0.0,
        image_size=spec.image_size,
        action_rep=trajectory.action_rep,
        cache_dir=spec.cache_dir,
        voxel_specs=voxel_specs,
        point_cloud_spec=point_cloud_spec,
    )


@dataclass
class Rvt2PretrainingModel:
    patch_backbone: torch.nn.Module
    head: torch.nn.Module
    optimizer: torch.optim.Optimizer
    head_config: dict
    device: torch.device
    dino_checkpoint: Optional[str]


def build_rvt2_pretraining_dataset(spec: Schema.Rvt2PretrainingSpec):
    """Build the source dataset selected by an RVT2 pretraining spec."""
    from visuomotor.data.mimicgen.dataset import MimicGenDataset
    from visuomotor.data.mimicgen.rvt2 import rvt2_dataset as Rvt2Dataset

    dataset_spec = spec.dataset
    dataset_path = Rvt2Dataset.resolve_rvt2_heatmap_dataset_path(
        dataset_spec.task_name, dataset_spec.path
    )
    active_demo_count = None
    if dataset_spec.n_demo is not None:
        active_demo_count = int(dataset_spec.n_demo) + int(
            dataset_spec.skip_first_episodes
        )
    dataset = MimicGenDataset(
        shape_meta=Rvt2Dataset.RVT2_HEATMAP_SHAPE_META,
        dataset_path=str(dataset_path),
        image_size=None,
        horizon=1,
        val_ratio=0.0,
        n_demo=active_demo_count,
        demo_count_mode=dataset_spec.demo_count_mode,
        action_rep="absolute",
        cache_dir=dataset_spec.cache_dir,
        include_oracle_info=True,
    )
    return dataset_path, dataset


def build_rvt2_pretraining_model(
    spec: Schema.Rvt2PretrainingSpec,
) -> Rvt2PretrainingModel:
    """Build RVT2 trainable modules and optimizer from resolved configuration."""
    from visuomotor.data.mimicgen.rvt2 import rvt2_dataset as Rvt2Dataset
    from visuomotor.data.mimicgen.tasks import NUM_ROBOTS
    from visuomotor.perception.focus.rvt2 import model as Rvt2Heatmap

    model = dict(spec.model.heatmap)
    query = dict(spec.model.query_composer)
    training = spec.training
    device = torch.device(training.device)
    dino_checkpoint = (
        Rvt2Dataset.resolve_dino_checkpoint(model["dino_ckpt"])
        if model["patch_backbone"] == "dino"
        else None
    )
    patch_backbone = Rvt2Heatmap.PatchFeatureBackbone(
        backbone_type=model["patch_backbone"],
        dino_ckpt_path=dino_checkpoint,
        image_size=model["dino_image_size"],
        patch_size=model["patch_size"],
        conv_dim=model["conv_patch_dim"],
    ).to(device)
    head_config = {
        "patch_dim": int(patch_backbone.output_dim),
        "gripper_dim": 1,
        "hidden_dim": int(model["hidden_dim"]),
        "grid_size": int(model["dino_image_size"]) // int(model["patch_size"]),
        "task_emb_dim": int(query["task_emb_dim"]),
        "num_robots": int(NUM_ROBOTS),
        "query_hidden_mult": int(query["hidden_mult"]),
        "proprio_mode": str(query["proprio_mode"]),
        "proprio_dim": int(query["proprio_dim"]),
        "language_seq_len": 77,
        "transformer_depth": int(model["transformer_depth"]),
        "transformer_heads": int(model["transformer_heads"]),
        "transformer_dropout": float(model["transformer_dropout"]),
    }
    head = Rvt2Heatmap.PatchActivationHead(**head_config).to(device)
    trainable_params = list(head.parameters())
    if patch_backbone.backbone_type != "dino":
        trainable_params += list(patch_backbone.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training.lr,
        weight_decay=training.weight_decay,
    )
    return Rvt2PretrainingModel(
        patch_backbone=patch_backbone,
        head=head,
        optimizer=optimizer,
        head_config=head_config,
        device=device,
        dino_checkpoint=(None if dino_checkpoint is None else str(dino_checkpoint)),
    )


def build_rvt2_background_randomizer(
    overlay: Mapping,
    *,
    image_size: int,
):
    """Build the optional RVT2 background augmentation component."""
    if not bool(overlay.get("enabled", False)):
        return None
    background_path = overlay.get("background_path")
    if not background_path or not os.path.exists(str(background_path)):
        raise FileNotFoundError(
            "background_overlay.background_path must exist when RVT2 heatmap "
            f"pretraining overlay is enabled, got {background_path!r}"
        )
    from visuomotor.perception.common.augmentation import BackgroundRandomizer

    return BackgroundRandomizer(
        input_shape=(int(image_size), int(image_size)),
        background_path=str(background_path),
    )


def sensor_specs(observation: Schema.SourceObservationSpec):
    voxels = {}
    point_cloud = None
    for producer in observation.producers:
        if isinstance(producer, Spatial.VoxelProducerSpec):
            if producer.output_key in voxels:
                raise ValueError(
                    f"duplicate voxel producer key {producer.output_key!r}"
                )
            voxels[producer.output_key] = producer
        elif isinstance(producer, Spatial.PointCloudProducerSpec):
            if point_cloud is not None:
                raise ValueError("duplicate point-cloud producer")
            point_cloud = producer
        else:
            raise TypeError(f"unknown producer spec: {type(producer).__name__}")
    return voxels, point_cloud


def build_runner(
    spec: Schema.RunnerSpec,
    *,
    output_dir: Optional[str] = None,
):
    from visuomotor.data.mimicgen import observations as MimicgenObservations
    from visuomotor.environment.robomimic.robomimic_setup import (
        RobomimicRunnerRequest,
    )
    from visuomotor.environment.runner import SeekerRobomimicImageRunner

    voxel_specs, point_cloud_spec = sensor_specs(spec.source_observation)
    request = RobomimicRunnerRequest(
        output_dir=output_dir if output_dir is not None else spec.output_dir,
        dataset_path=spec.dataset_path,
        cache_dir=spec.cache_dir,
        action_rep=spec.trajectory.action_rep,
        n_test=spec.n_test,
        n_test_vis=spec.n_test_vis,
        test_start_seed=spec.test_start_seed,
        max_steps=spec.max_steps,
        terminate_on_success=spec.strict_task_success,
        n_obs_steps=spec.trajectory.observation_horizon,
        n_action_steps=spec.trajectory.execution_horizon,
        render_obs_key=spec.render_obs_key,
        fps=spec.fps,
        crf=spec.crf,
        n_envs=spec.n_envs,
        env_name=spec.env_name,
        shuffle_table_texture=spec.shuffle_table_texture,
        enable_oracle_subtask_info=spec.enable_oracle_subtask_info,
        oracle_projection_camera=spec.oracle_projection_camera,
        enable_oracle_focus_info=spec.enable_oracle_focus_info,
        oracle_focus_camera=spec.oracle_focus_camera,
        oracle_focus_patch_size=spec.oracle_focus_patch_size,
        oracle_focus_min_patch_area_fraction=(
            spec.oracle_focus_min_patch_area_fraction
        ),
        oracle_focus_min_mask_pixels=spec.oracle_focus_min_mask_pixels,
        enable_oracle_video_overlay=spec.enable_oracle_focus_info,
        oracle_overlay_zoom=4.0,
        voxel_specs=voxel_specs,
        point_cloud_spec=point_cloud_spec,
        rgb_load_resolutions=spec.rgb_load_resolutions,
        delta_history_source_keys=MimicgenObservations.delta_history_source_keys(
            spec.source_observation.shape_meta(spec.trajectory.action_dim)["obs"]
        ),
    )
    return SeekerRobomimicImageRunner(
        request,
        shape_meta=spec.source_observation.shape_meta(spec.trajectory.action_dim),
        past_action=spec.past_action,
        tqdm_interval_sec=spec.tqdm_interval_sec,
        mirror_augmentation=spec.mirror_augmentation,
        visualization_enabled=spec.visualization.enabled,
        save_images=spec.visualization.save.images,
        save_videos=spec.visualization.save.videos,
    )


def load_policy_checkpoint(path, *, map_location="cpu"):
    """Restore a release policy through the configuration-owned builder."""
    from visuomotor.policy import checkpoint as PolicyCheckpoint

    payload = PolicyCheckpoint.load_release_checkpoint_payload(
        path, map_location=map_location
    )
    policy = build_policy(Schema.from_dict(payload["model_spec"]))
    policy.load_state_dict(payload["state_dict"], strict=True)
    return policy


def load_runner_spec(path, *, map_location="cpu"):
    """Restore the typed runner spec recorded beside a release policy."""
    from visuomotor.policy import checkpoint as PolicyCheckpoint

    payload = PolicyCheckpoint.load_release_checkpoint_payload(
        path, map_location=map_location
    )
    runner_spec = payload.get("runner_spec")
    if not runner_spec:
        raise ValueError(f"policy checkpoint does not record a runner spec: {path}")
    return Schema.from_dict(runner_spec)


def _workspace_metadata(payload: Mapping, key: str) -> Optional[int]:
    metadata = payload.get("metadata", {})
    if isinstance(metadata, Mapping) and key in metadata:
        return int(metadata[key])

    pickles = payload.get("pickles", {})
    if not isinstance(pickles, Mapping) or key not in pickles:
        return None
    import dill

    return int(dill.loads(pickles[key]))


def _select_workspace_weights(state_dicts: Mapping, requested: str) -> str:
    available = tuple(key for key in ("ema_model", "model") if key in state_dicts)
    if requested == "auto":
        if available:
            return available[0]
    else:
        selected = {"ema": "ema_model", "model": "model"}[requested]
        if selected in state_dicts:
            return selected
    raise ValueError(
        f"checkpoint cannot provide {requested!r} rollout weights; "
        f"available policy states: {list(available)}"
    )


def load_rollout_checkpoint(path, *, map_location="cpu", weights: str = "auto"):
    """Build a rollout policy and runner from release or training checkpoints."""
    from visuomotor.policy import checkpoint as PolicyCheckpoint

    if weights not in ("auto", "ema", "model"):
        raise ValueError(f"unknown rollout weight selection {weights!r}")
    payload = PolicyCheckpoint.load_serialized_payload(path, map_location=map_location)

    if "model_spec" in payload and "state_dict" in payload:
        if weights != "auto":
            raise ValueError(
                "release checkpoints contain one preselected policy state; "
                "--weights only applies to training checkpoints"
            )
        version = payload.get("schema_version")
        if version != PolicyCheckpoint.RELEASE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported policy checkpoint schema {version!r}")
        model_spec = Schema.from_dict(payload["model_spec"])
        runner_spec = Schema.from_dict(payload.get("runner_spec"))
        if not isinstance(model_spec, Schema.ModelSpec):
            raise TypeError("release checkpoint model_spec is not a ModelSpec")
        if not isinstance(runner_spec, Schema.RunnerSpec):
            raise ValueError(f"policy checkpoint does not record a runner spec: {path}")
        state_dict = payload["state_dict"]
        checkpoint_format = "release"
        selected = "release"
        epoch = None
        global_step = None
        training_demo_count = None
    elif "cfg" in payload and "state_dicts" in payload:
        run_spec = Schema.from_dict(payload["cfg"])
        if not isinstance(run_spec, Schema.RunSpec):
            raise TypeError("training checkpoint cfg is not a policy RunSpec")
        state_dicts = payload["state_dicts"]
        if not isinstance(state_dicts, Mapping):
            raise TypeError("training checkpoint state_dicts must be a mapping")
        selected = _select_workspace_weights(state_dicts, weights)
        model_spec = run_spec.model
        runner_spec = run_spec.runner
        state_dict = state_dicts[selected]
        checkpoint_format = "training"
        epoch = _workspace_metadata(payload, "epoch")
        global_step = _workspace_metadata(payload, "global_step")
        training_demo_count = (
            None if run_spec.dataset.n_demo is None else int(run_spec.dataset.n_demo)
        )
    else:
        raise ValueError(
            "checkpoint is neither a rollout release nor a policy-training checkpoint"
        )

    policy = build_policy(model_spec)
    policy.load_state_dict(state_dict, strict=True)
    return RolloutCheckpoint(
        policy=policy,
        runner_spec=runner_spec,
        checkpoint_format=checkpoint_format,
        weights=selected,
        epoch=epoch,
        global_step=global_step,
        training_demo_count=training_demo_count,
    )
