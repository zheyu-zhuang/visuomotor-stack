from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def _compose(config_name: str, overrides=()):
    with initialize_config_module(
        config_module="visuomotor.config", version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def test_policy_wandb_structure_uses_task_group_and_dimension_tags():
    cfg = _compose("train_voxel_flow")

    assert cfg.logging.project == "vm-global-flow"
    assert cfg.logging.group == "square_d0"
    assert cfg.logging.name == "voxel_simple"
    assert set(cfg.logging.tags) == {
        "task=square_d0",
        "stage=policy",
        "input=voxel_wrist",
        "encoder=voxel_simple",
        "policy=global_flow",
        "regime=in_domain",
        "action=absolute",
        "demonstrations=100",
        "seed=0",
    }


def test_policy_wandb_project_is_owned_by_the_selected_generator():
    for config_name, expected_project in (
        ("train_rgb_diffusion", "vm-global-diffusion"),
        ("train_voxel_flow", "vm-global-flow"),
    ):
        assert _compose(config_name).logging.project == expected_project


def test_policy_run_name_mentions_only_nonstandard_inputs():
    for input_name, expected_name in (
        ("voxel_wrist", "voxel_simple"),
        ("voxel", "voxel_simple__voxel-only"),
        ("rgb_external_wrist", "rgb_resnet18"),
        ("rgb_external", "rgb_resnet18__external-only"),
        ("rgb_wrist", "rgb_resnet18__wrist-only"),
        ("point_cloud", "dp3"),
        ("voxel_wrist_proprio_delta", "voxel_simple__proprio-delta"),
    ):
        encoder = "dp3" if input_name == "point_cloud" else None
        if input_name.startswith("rgb"):
            encoder = "rgb_resnet18"
        overrides = [f"input={input_name}"]
        if encoder is not None:
            overrides.append(f"encoder={encoder}")
        cfg = _compose("train_voxel_flow", overrides)

        assert cfg.logging.name == expected_name
        assert f"input={input_name}" in cfg.logging.tags


def test_pretraining_wandb_structure_uses_task_group_and_dimension_tags():
    for config_name, expected_name, expected_model in (
        (
            "pretrain/rvt2_heatmap",
            "rvt2_heatmap",
            "model=rvt2_heatmap",
        ),
        (
            "pretrain/seeker",
            "seeker",
            "model=seeker",
        ),
    ):
        cfg = _compose(config_name)

        assert cfg.logging.group == cfg.task_name
        assert cfg.logging.name == expected_name
        assert expected_model in cfg.logging.tags
        assert f"task={cfg.task_name}" in cfg.logging.tags
        assert "stage=visual_focus" in cfg.logging.tags
