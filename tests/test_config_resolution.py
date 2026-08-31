import itertools

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from visuomotor.config import tasks as ConfigTasks
from visuomotor.config.build import build_dataset, build_policy
from visuomotor.config.resolve import resolve_policy_run, voxel_grid_geometry
from visuomotor.config.schema import (
    FocusConditionedEncoderSpec,
    FocusRefineEncoderSpec,
    GlobalPolicySpec,
    PointCloudEncoderSpec,
    RgbEncoderSpec,
    VoxelEncoderSpec,
    from_dict,
    to_dict,
)
from visuomotor.data.mimicgen import observations as MimicgenObservations

INPUTS = (
    "rgb_external",
    "rgb_wrist",
    "rgb_external_wrist",
    "voxel",
    "voxel_wrist",
    "point_cloud",
)
RGB_ENCODERS = (
    "rgb_resnet18",
    "rgb_focus_pool2d",
    "seeker_resnet18",
    "rvt2_resnet18",
    "oracle_resnet18",
)
VOXEL_ENCODERS = ("voxel_simple", "voxel_resnet3d", "voxel_focus_pool3d")
POINT_CLOUD_ENCODERS = ("dp3",)
ENCODERS = RGB_ENCODERS + VOXEL_ENCODERS + POINT_CLOUD_ENCODERS
POLICIES = ("global_diffusion", "global_flow")


def _resolve(**overrides):
    """Resolve a composition from the neutral shared base.

    A ``train_*`` recipe patches its own policy node, so composing a different
    policy onto one now fails by design; the composition matrix belongs on the
    base every recipe is built from.
    """
    settings = {"input_augmentation.rgb_crop": "enabled"}
    settings.update(overrides)
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        cfg = compose(
            config_name="policy_training_defaults",
            overrides=[f"{key}={value}" for key, value in settings.items()],
        )
    OmegaConf.resolve(cfg)
    return resolve_policy_run(cfg)


def test_only_external_focus_presets_define_view_strategies():
    focus = {"seeker_resnet18", "rvt2_resnet18", "oracle_resnet18"}
    for encoder in focus.union(
        {"rgb_resnet18", "rgb_focus_pool2d"}, VOXEL_ENCODERS
    ):
        with initialize_config_module(
            config_module="visuomotor.config", version_base=None
        ):
            cfg = compose(
                config_name="train_rgb_diffusion", overrides=[f"encoder={encoder}"]
            )
        if encoder in focus:
            assert cfg.encoder.view_strategies != "disabled", encoder
        else:
            assert cfg.encoder.view_strategies == "disabled", encoder


def test_default_training_config_declares_coupled_augmentation_handles():
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        cfg = compose(config_name="train_rgb_diffusion")
    assert cfg.input_augmentation.rgb_crop == "enabled"
    assert cfg.input_augmentation.mirror == "disabled"
    assert cfg.input_augmentation.scene_yaw == "disabled"
    assert cfg.input_augmentation.voxel_crop == "disabled"


def test_policy_experiment_name_has_a_generated_default_and_simple_override():
    assert (
        _resolve().exp_name
        == "100d_absolute_s0"
    )
    assert _resolve(exp_name="my_experiment").exp_name == "my_experiment"


def test_policy_experiment_name_controls_hydra_output_directory():
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        cfg = compose(
            config_name="train_voxel_flow",
            overrides=["exp_name=my_experiment"],
            return_hydra_config=True,
        )
    assert OmegaConf.to_container(cfg.hydra.run, resolve=True)["dir"] == (
        "experiments/vm-global-flow/square_d0/voxel_simple/"
        "my_experiment"
    )


def test_visualization_defaults_are_local_only_and_shared_with_rollout():
    run = _resolve()
    visualization = run.workspace.visualization
    assert visualization.enabled
    assert visualization.save.images and visualization.save.videos
    assert not visualization.upload.images and not visualization.upload.videos
    assert run.runner.visualization == visualization
    assert (run.runner.n_test_vis, run.runner.fps, run.runner.crf) == (3, 10, 28)
    assert not run.runner.enable_oracle_subtask_info


def test_visualization_rejects_upload_when_local_save_is_disabled():
    with pytest.raises(ValueError, match="upload images"):
        _resolve(
            **{
                "visualization.save.images": "false",
                "visualization.upload.images": "true",
            }
        )


def test_first_class_training_examples_resolve():
    examples = (
        ("train_rgb_diffusion", "rgb_external_wrist", "rgb_resnet18", "global_diffusion"),
        (
            "train_seeker_diffusion",
            "rgb_external_wrist",
            "seeker_resnet18",
            "global_diffusion",
        ),
        ("train_voxel_diffusion", "voxel_wrist", "voxel_simple", "global_diffusion"),
        ("train_voxel_flow", "voxel_wrist", "voxel_simple", "global_flow"),
    )
    for config_name, input_name, encoder, policy in examples:
        with initialize_config_module(
            config_module="visuomotor.config", version_base=None
        ):
            cfg = compose(config_name=config_name)
        run = resolve_policy_run(cfg)
        assert run.model.input.name == input_name, config_name
        assert run.model.encoder.name == encoder, config_name
        assert run.model.policy.name == policy, config_name


def test_voxel_wrist_visual_streams_share_the_256d_contract():
    for encoder_name in VOXEL_ENCODERS:
        run = _resolve(input="voxel_wrist", encoder=encoder_name)
        encoder = run.model.encoder
        assert encoder.feature_dim == 256
        assert encoder.rgb_feature_dim == 256


def test_first_class_examples_match_the_source_generator_widths():
    expected = (
        ("train_voxel_diffusion", "diffusion", (256, 512, 1024), None),
        ("train_rgb_diffusion", "diffusion", (512, 1024, 2048), "rgb_resnet18"),
        ("train_voxel_flow", "flow", (256, 512, 1024), None),
        ("train_seeker_diffusion", "diffusion", (512, 1024, 2048), "seeker_resnet18"),
    )
    for config_name, kind, unet_channels, encoder in expected:
        with initialize_config_module(
            config_module="visuomotor.config", version_base=None
        ):
            cfg = compose(config_name=config_name)
        run = resolve_policy_run(cfg)

        assert run.model.policy.generator.kind == kind, config_name
        assert run.model.policy.generator.unet_channels == unet_channels, config_name
        if encoder is not None:
            assert run.model.encoder.name == encoder, config_name


def test_voxel_diffusion_matches_the_source_rot_aug_training_recipe():
    with initialize_config_module(
        config_module="visuomotor.config", version_base=None
    ):
        cfg = compose(
            config_name="train_voxel_diffusion",
            overrides=["task=stack_three_d1"],
        )
    run = resolve_policy_run(cfg)

    yaw = run.dataset.scene_yaw_augmentation
    assert yaw.enable
    assert yaw.min_deg == -180.0
    assert yaw.max_deg == 180.0
    assert yaw.max_attempts == 1000
    assert yaw.identity_probability == pytest.approx(1 / 64)
    assert run.model.encoder.crop_size == 58
    assert run.model.encoder.rgb_random_crop.input_res == 84
    assert run.model.encoder.rgb_random_crop.output_res == 76

    # Worker, env, and checkpoint-retention counts are host knobs, not the recipe.
    assert run.workspace.train_loader.batch_size == 64
    assert run.workspace.train_loader.shuffle
    assert run.workspace.train_loader.persistent_workers
    assert run.workspace.val_loader.batch_size == 64
    assert not run.workspace.val_loader.shuffle
    assert run.workspace.val_loader.persistent_workers

    assert run.training.num_epochs == 500
    assert run.training.rollout_every == 10
    assert run.training.checkpoint_every == 10
    assert run.training.val_every == 1
    assert run.training.use_ema
    assert run.runner.n_test == 50
    assert run.runner.n_test_vis == 4
    assert run.runner.crf == 22


def test_policy_training_defaults_roll_out_every_ten_epochs():
    run = _resolve(n_demo=100)

    assert run.training.rollout_every == 10
    assert run.training.checkpoint_every == 10


def test_all_voxel_training_examples_use_one_observation():
    for config_name in (
        "train_voxel_diffusion",
        "train_voxel_flow",
    ):
        with initialize_config_module(
            config_module="visuomotor.config", version_base=None
        ):
            cfg = compose(config_name=config_name)
        run = resolve_policy_run(cfg)
        assert run.dataset.trajectory is run.model.trajectory, config_name
        assert run.runner.trajectory is run.model.trajectory, config_name
        assert run.model.trajectory.observation_horizon == 1, config_name
        assert run.model.encoder.crop_size == 58, config_name
        assert run.model.encoder.rgb_norm == "groupnorm", config_name
        assert run.training.use_ema is True, config_name


def test_voxel_crop_is_selected_at_the_experiment_augmentation_boundary():
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        cfg = compose(
            config_name="train_voxel_flow",
            overrides=["input_augmentation.voxel_crop=disabled"],
        )
    assert resolve_policy_run(cfg).model.encoder.crop_size is None

    local_override = _resolve(
        input="voxel_wrist",
        encoder="voxel_simple",
        **{"input_augmentation.voxel_crop": "{method:enabled,output:60}"},
    )
    assert local_override.model.encoder.crop_size == 60


def test_rgb_crop_is_selected_at_the_input_augmentation_boundary():
    assert _resolve().model.encoder.random_crop.output_res == 76
    disabled = _resolve(**{"input_augmentation.rgb_crop": "disabled"})
    assert disabled.model.encoder.random_crop.enabled is False
    assert disabled.model.encoder.random_crop.output_res == 76


def test_augmentation_handles_reject_unsupported_selections():
    with pytest.raises(ValueError, match="requires voxel input"):
        _resolve(**{"input_augmentation.voxel_crop": "enabled"})

    with pytest.raises(ValueError, match="unknown input_augmentation.voxel_crop options"):
        _resolve(
            input="voxel_wrist",
            encoder="voxel_simple",
            **{"input_augmentation.voxel_crop": "{method:enabled,size:60}"},
        )


def test_training_augmentation_handles_resolve_on_the_data_boundary():
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        mirror_cfg = compose(config_name="train_rgb_diffusion")
        scene_cfg = compose(config_name="train_voxel_flow")
    mirror_cfg.input_augmentation.mirror = "enabled"
    scene_cfg.input_augmentation.scene_yaw = "enabled"
    mirror = resolve_policy_run(mirror_cfg)
    scene = resolve_policy_run(scene_cfg)
    assert mirror.dataset.mirror_augmentation.enable
    assert mirror.runner.mirror_augmentation.enable
    assert scene.dataset.scene_yaw_augmentation.enable


@pytest.mark.parametrize(
    "task,table_offset_xy",
    (("stack_three_d1", (0.0, 0.0)), ("kitchen_d1", (-0.2, 0.0))),
)
def test_voxel_scene_yaw_workspace_is_the_voxel_array_footprint(
    task, table_offset_xy, monkeypatch
):
    monkeypatch.setattr("visuomotor.config.resolve.dataset_robot_ids", lambda *_: (0,))
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        cfg = compose(
            config_name="train_voxel_flow",
            overrides=[f"task={task}", "input_augmentation.scene_yaw=enabled"],
        )
    run = resolve_policy_run(cfg)
    scene_yaw = run.dataset.scene_yaw_augmentation
    assert scene_yaw.enable
    # The yaw centre is the grid's own centre (what _rotate_voxels turns about),
    # which the producer's +1e-4 pitch offsets from the table by half a voxel.
    geometry = voxel_grid_geometry(run.dataset.source_observation)
    assert scene_yaw.workspace_center_xy == pytest.approx(geometry.center[:2])
    assert scene_yaw.workspace_size == pytest.approx(geometry.extent[0])
    assert scene_yaw.workspace_center_xy == pytest.approx(
        table_offset_xy, abs=geometry.pitch[0]
    )


def _compatible(input_name, encoder, policy):
    voxel = input_name.startswith("voxel")
    point_cloud = input_name == "point_cloud"
    external = "external" in input_name
    encoder_ok = encoder in (
        POINT_CLOUD_ENCODERS
        if point_cloud
        else VOXEL_ENCODERS
        if voxel
        else RGB_ENCODERS
    )
    if encoder in {"seeker_resnet18", "rvt2_resnet18", "oracle_resnet18"}:
        encoder_ok = encoder_ok and external
    return encoder_ok


def test_dp3_resolves_existing_point_cloud_transport():
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        cfg = compose(
            config_name="train_rgb_diffusion",
            overrides=["input=point_cloud", "encoder=dp3"],
        )
    run = resolve_policy_run(cfg)
    assert isinstance(run.model.encoder, PointCloudEncoderSpec)
    assert run.model.observation.field("point_cloud").shape == (1024, 6)
    assert run.dataset.source_observation.fields[0].kind == "point_cloud"
    producer = run.dataset.source_observation.producers[0]
    assert producer.table_margin == pytest.approx(0.005)
    assert producer.bounds_min[2] == pytest.approx(
        run.task.table_offset[2] + producer.table_margin
    )


@pytest.mark.parametrize(
    "task,table_height",
    [("square_d0", 0.82), ("stack_d1", 0.8), ("hammer_cleanup_d1", 0.9)],
)
def test_spatial_producers_separate_voxel_floor_from_table_crop(
    task, table_height, monkeypatch
):
    monkeypatch.setattr("visuomotor.config.resolve.dataset_robot_ids", lambda *_: (0,))
    voxel_run = _resolve(task=task, input="voxel", encoder="voxel_simple")
    voxel = voxel_run.dataset.source_observation.producers[0]
    assert voxel.bounds_min[2] == pytest.approx(0.7)
    assert voxel.bounds_max[2] == pytest.approx(0.7 + voxel.ws_size)
    assert voxel.reconstruction_resolution == 84

    point_run = _resolve(task=task, input="point_cloud", encoder="dp3")
    point_cloud = point_run.dataset.source_observation.producers[0]
    assert point_cloud.bounds_min[2] == pytest.approx(
        table_height + point_cloud.table_margin
    )
    assert point_cloud.bounds_max[2] == pytest.approx(
        table_height + point_cloud.ws_size
    )
    assert point_cloud.reconstruction_resolution == 84


def test_hdf5_environment_names_resolve_to_the_registered_task_contract():
    task = ConfigTasks.get_task_spec("StackThree_D1")
    assert task.name == "stack_three_d1"
    assert task.table_offset == (0.0, 0.0, 0.8)


def test_composition_matrix_is_validated_by_requirements():
    for input_name, encoder, policy in itertools.product(INPUTS, ENCODERS, POLICIES):
        combination = (input_name, encoder, policy)
        if not _compatible(input_name, encoder, policy):
            with pytest.raises(ValueError):
                _resolve(input=input_name, encoder=encoder, policy=policy)
            continue
        run = _resolve(input=input_name, encoder=encoder, policy=policy)
        assert run.model.input.name == input_name, combination
        assert run.model.encoder.name == encoder, combination
        assert run.model.policy.name == policy, combination


def test_input_and_source_observation_roles_are_separate():
    run = _resolve(
        input="voxel_wrist", encoder="voxel_focus_pool3d", policy="global_flow"
    )
    assert run.model.observation.keys() == (
        "rgb_wrist",
        "voxel",
        "eef_pos",
        "eef_rot6d",
        "gripper_qpos",
    )
    source = run.dataset.source_observation
    assert tuple(field.source_key for field in source.fields) == (
        "robot0_eye_in_hand_image",
        "voxel",
        "robot0_eef_pos",
        "robot0_eef_rot",
        "robot0_gripper_qpos",
    )
    assert run.runner.source_observation == source
    assert run.runner.render_obs_key == MimicgenObservations.source_camera_key(
        "external"
    )


@pytest.mark.parametrize("encoder", ("rgb_resnet18", "rgb_focus_pool2d"))
def test_low_resolution_rgb_consumers_resolve_84_transport(encoder):
    run = _resolve(input="rgb_external_wrist", encoder=encoder)
    assert run.dataset.rgb_load_resolutions == (
        ("rgb_external", 84),
        ("rgb_wrist", 84),
    )
    assert run.dataset.observation.field("rgb_external").shape == (3, 84, 84)
    assert run.dataset.observation.field("rgb_wrist").shape == (3, 84, 84)
    # Rollout loads at the resolution training loaded at, through the same codec.
    assert run.runner.rgb_load_resolutions == run.dataset.rgb_load_resolutions
    assert run.runner.observation.field("rgb_external").shape == (3, 84, 84)
    assert run.dataset.source_observation.fields[0].shape == (3, 256, 256)


def test_rvt2_resolves_selected_views_at_224_and_pass_through_at_84():
    run = _resolve(
        input="rgb_external_wrist", encoder="rvt2_resnet18", regime="in_domain"
    )
    assert run.dataset.rgb_load_resolutions == (
        ("rgb_external", 224),
        ("rgb_wrist", 84),
    )
    assert run.dataset.observation.field("rgb_external").shape == (3, 224, 224)
    assert run.dataset.observation.field("rgb_wrist").shape == (3, 84, 84)


def test_seeker_transports_views_at_native_resolution():
    run = _resolve(
        input="rgb_external_wrist", encoder="seeker_resnet18", regime="in_domain"
    )
    assert run.dataset.rgb_load_resolutions == (
        ("rgb_external", 256),
        ("rgb_wrist", 256),
    )
    assert run.dataset.observation.field("rgb_external").shape == (3, 256, 256)
    assert run.dataset.observation.field("rgb_wrist").shape == (3, 256, 256)
    assert run.runner.rgb_load_resolutions == run.dataset.rgb_load_resolutions


def test_focus_load_resolution_tracks_each_active_view_mode():
    seeker = _resolve(
        input="rgb_external_wrist", encoder="seeker_resnet18", regime="image_aug"
    )
    rvt2 = _resolve(
        input="rgb_external_wrist", encoder="rvt2_resnet18", regime="image_aug"
    )
    assert seeker.dataset.rgb_load_resolutions == (
        ("rgb_external", 256),
        ("rgb_wrist", 256),
    )
    assert rvt2.dataset.rgb_load_resolutions == (
        ("rgb_external", 224),
        ("rgb_wrist", 84),
    )


def test_pure_voxel_does_not_expose_an_rgb_model_branch():
    run = _resolve(input="voxel", encoder="voxel_resnet3d")
    assert isinstance(run.model.encoder, VoxelEncoderSpec)
    assert run.model.encoder.rgb_keys == ()
    assert run.model.observation.keys("rgb") == ()
    assert run.dataset.source_observation.producers[0].cameras


@pytest.mark.parametrize("views", ["[external]", "[wrist]", "[external,wrist]"])
def test_new_voxel_rgb_combinations_need_only_a_declarative_input_change(views):
    run = _resolve(
        input="voxel", encoder="voxel_resnet3d", **{"input.rgb_views": views}
    )
    expected = tuple(f"rgb_{view}" for view in views.strip("[]").split(","))
    assert run.model.encoder.rgb_keys == expected


def test_explicit_encoder_presets_resolve_to_matching_runtime_architectures():
    rgb = _resolve(input="rgb_external_wrist", encoder="rgb_resnet18")
    voxel = _resolve(input="voxel_wrist", encoder="voxel_resnet3d")
    assert isinstance(rgb.model.encoder, RgbEncoderSpec)
    assert isinstance(voxel.model.encoder, VoxelEncoderSpec)
    assert voxel.model.encoder.rgb_keys == ("rgb_wrist",)
    assert voxel.model.encoder.crop_size is None

    planar_focus = _resolve(input="rgb_wrist", encoder="rgb_focus_pool2d")
    voxel_focus = _resolve(input="voxel_wrist", encoder="voxel_focus_pool3d")
    assert isinstance(planar_focus.model.encoder, FocusRefineEncoderSpec)
    assert isinstance(voxel_focus.model.encoder, VoxelEncoderSpec)


def test_focus_sources_require_external_rgb_and_reject_voxel():
    for encoder in ("seeker_resnet18", "rvt2_resnet18", "oracle_resnet18"):
        run = _resolve(input="rgb_external_wrist", encoder=encoder)
        assert isinstance(run.model.encoder, FocusConditionedEncoderSpec)
        with pytest.raises(ValueError, match="external"):
            _resolve(input="rgb_wrist", encoder=encoder)
        with pytest.raises(ValueError, match="voxel"):
            _resolve(input="voxel_wrist", encoder=encoder)


def test_seeker_preset_uses_exact_input_views_and_declares_active_augmentations():
    external = _resolve(
        input="rgb_external", encoder="seeker_resnet18", regime="image_aug"
    ).model.encoder
    assert external.feature_architecture == "resnet18"
    assert external.source.name == "seeker"
    assert external.view_keys == (("external", "rgb_external"),)
    assert external.view_modes == (("external", "focus_mask_crop"),)
    assert external.view_augmentations == (("external", "mask_guided_overlay"),)
    assert external.guided_overlay is not None
    assert external.random_overlay is None

    both = _resolve(
        input="rgb_external_wrist", encoder="seeker_resnet18", regime="image_aug"
    ).model.encoder
    assert both.view_keys == (
        ("external", "rgb_external"),
        ("wrist", "rgb_wrist"),
    )
    assert both.view_modes == (
        ("external", "focus_mask_crop"),
        ("wrist", "focus_mask"),
    )
    assert both.view_augmentations == (
        ("external", "mask_guided_overlay"),
        ("wrist", "mask_guided_overlay"),
    )


def test_overlay_parameters_resolve_from_central_augmentation_defaults():
    encoder = _resolve(
        input="rgb_external",
        encoder="seeker_resnet18",
        regime="image_aug",
        **{"augmentation_defaults.input.mask_guided_overlay.probability": 0.25},
    ).model.encoder
    assert encoder.guided_overlay.prob == 0.25


def test_rvt2_wrist_branch_selects_its_corresponding_random_augmentation():
    encoder = _resolve(
        input="rgb_external_wrist", encoder="rvt2_resnet18", regime="image_aug"
    ).model.encoder
    assert encoder.view_augmentations == (
        ("external", "mask_guided_overlay"),
        ("wrist", "random_background_overlay"),
    )
    assert encoder.guided_overlay is not None
    assert encoder.random_overlay is not None


def test_oracle_preset_rejects_image_augmentation_regime_explicitly():
    with pytest.raises(ValueError, match="does not support"):
        _resolve(input="rgb_external", encoder="oracle_resnet18", regime="image_aug")


def test_policy_intrinsics_share_one_trajectory_contract():
    global_run = _resolve(
        input="rgb_external", encoder="rgb_resnet18", policy="global_flow"
    )
    assert isinstance(global_run.model.policy, GlobalPolicySpec)
    assert global_run.model.policy.generator.kind == "flow"
    assert global_run.dataset.trajectory is global_run.model.trajectory
    assert global_run.runner.trajectory is global_run.model.trajectory
    assert global_run.model.trajectory.observation_horizon == 2

def test_flow_configurations_fail_at_resolution_before_model_construction():
    rejected = (
        ({"policy": "global_flow", "horizon": 15}, "horizon 15 must be divisible"),
        (
            {"policy": "global_flow", "policy.generator.integration_steps": 0},
            "integration_steps",
        ),
    )
    for overrides, match in rejected:
        with pytest.raises(ValueError, match=match):
            _resolve(**overrides)


def test_builder_wires_the_policy_to_the_current_dataset(monkeypatch):
    from visuomotor.data.mimicgen.dataset import MimicGenDataset

    monkeypatch.setattr(
        MimicGenDataset,
        "__init__",
        lambda self, **kwargs: setattr(self, "construction_kwargs", kwargs),
    )

    global_dataset = build_dataset(_resolve().dataset)
    assert type(global_dataset) is MimicGenDataset
    assert "keypose_targets" not in global_dataset.construction_kwargs
    assert global_dataset.construction_kwargs["image_size"] is None
    assert global_dataset.construction_kwargs["rgb_load_resolutions"] == {
        "rgb_external": 84,
        "rgb_wrist": 84,
    }

def test_selected_proprioception_is_propagated_exactly():
    run = _resolve(
        input="rgb_wrist",
        encoder="rgb_resnet18",
        **{"input.proprio": "[gripper_qpos]"},
    )
    assert run.model.input.proprio == ("gripper_qpos",)
    assert run.model.encoder.proprio_fields == ("gripper_qpos",)
    assert run.model.observation.keys("low_dim") == ("gripper_qpos",)

def test_normalization_is_automatic_for_one_or_multiple_robots(monkeypatch):
    import visuomotor.config.resolve as resolve

    monkeypatch.setattr(resolve, "dataset_robot_ids", lambda *_: (0,))
    assert _resolve().model.normalizer == "linear"
    monkeypatch.setattr(resolve, "dataset_robot_ids", lambda *_: (0, 1))
    assert _resolve().model.normalizer == "multi_robot_linear"


def test_model_and_run_specs_round_trip_through_checkpoint_codec():
    run = _resolve(input="voxel_wrist", encoder="voxel_focus_pool3d")
    assert from_dict(to_dict(run)) == run


def test_conditioning_width_is_owned_by_the_policy_not_the_encoder():
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        voxel_flow = resolve_policy_run(compose(config_name="train_voxel_flow"))

    assert not hasattr(voxel_flow.model.encoder, "output_dim")
    assert voxel_flow.model.policy.observation_feature_dim == 256

    built = build_policy(voxel_flow.model)
    fused = 256 + 256 + 3 + 6 + 2
    assert built.encoder.output_dim == fused
    assert built.observation_projection.in_features == fused
    assert built.generator.condition_dim == 256

def test_voxel_flow_dataset_render_inputs_are_stable():
    """What the cache must render is pinned for the working voxel baseline.

    A changed source observation requires a full dataset re-render.
    """
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        cfg = compose(config_name="train_voxel_flow")
    dataset = resolve_policy_run(cfg).dataset

    assert dataset.keypose_targets is None
    assert not dataset.include_camera_matrices
    assert not dataset.include_oracle_info
    assert dataset.source_observation.support_fields == ()
    assert tuple(
        field.source_key for field in dataset.source_observation.fields
    ) == (
        "robot0_eye_in_hand_image",
        "voxel",
        "robot0_eef_pos",
        "robot0_eef_rot",
        "robot0_gripper_qpos",
    )
    assert dataset.rgb_load_resolutions == (("rgb_wrist", 84),)
