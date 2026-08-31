import pytest
import torch

from visuomotor.config.schema import from_dict
from visuomotor.data.core.normalization import Normalizer
from visuomotor.paths import load_resource_paths
from visuomotor.policy.checkpoint import load_checkpoint, save_checkpoint


def test_shipped_composition_groups_are_discoverable_from_the_config_tree():
    from pathlib import Path

    import visuomotor.config as config_package

    inputs = {
        path.stem
        for path in (Path(config_package.__file__).parent / "input").glob("*.yaml")
    }
    encoders = {
        path.stem for path in (Path(config_package.__file__).parent / "encoder").glob("*.yaml")
    }
    policies = {
        path.stem for path in (Path(config_package.__file__).parent / "policy").glob("*.yaml")
    }
    assert inputs == {
        "point_cloud",
        "rgb_external",
        "rgb_wrist",
        "rgb_external_wrist",
        "voxel",
        "voxel_wrist",
        "voxel_wrist_proprio_delta",
    }
    assert encoders == {
        "rgb_resnet18",
        "rgb_focus_pool2d",
        "seeker_resnet18",
        "rvt2_resnet18",
        "oracle_resnet18",
        "dp3",
        "voxel_simple",
        "voxel_resnet3d",
        "voxel_focus_pool3d",
    }
    assert policies == {
        "global_diffusion",
        "global_flow",
    }


def test_first_class_policy_training_examples_are_discoverable():
    from pathlib import Path

    import visuomotor.config as config_package

    config_root = Path(config_package.__file__).parent
    names = {path.stem for path in config_root.glob("train*.yaml")}
    names.update(
        f"ablation/{path.stem}"
        for path in (config_root / "ablation").glob("train*.yaml")
    )
    assert names == {
        "train_rgb_diffusion",
        "train_seeker_diffusion",
        "train_voxel_diffusion",
        "train_voxel_flow",
    }
    assert not (config_root / "experiment").exists()


def test_training_examples_expose_representation_appropriate_augmentation_placeholders():
    from pathlib import Path

    from omegaconf import OmegaConf

    import visuomotor.config as config_package

    config_root = Path(config_package.__file__).parent
    for name in (
        "train_rgb_diffusion",
        "train_seeker_diffusion",
    ):
        cfg = OmegaConf.load(config_root / f"{name}.yaml")
        assert "mirror" in cfg.input_augmentation
        assert cfg.input_augmentation.rgb_crop == "enabled"
        assert cfg.input_augmentation.mirror == "disabled"
    cfg = OmegaConf.load(config_root / "train_voxel_diffusion.yaml")
    assert cfg.input_augmentation.rgb_crop == "enabled"
    assert cfg.input_augmentation.scene_yaw == {
        "method": "enabled",
        "max_attempts": 1000,
    }
    assert cfg.input_augmentation.voxel_crop == "enabled"
    # voxel_flow keeps scene yaw disabled while retaining the shared retry cap.
    cfg = OmegaConf.load(config_root / "train_voxel_flow.yaml")
    assert cfg.input_augmentation.rgb_crop == "enabled"
    assert cfg.input_augmentation.scene_yaw == {
        "method": "disabled",
        "max_attempts": 1000,
    }
    assert cfg.input_augmentation.voxel_crop == "enabled"


def test_augmentation_defaults_centralize_method_parameters():
    from pathlib import Path

    from omegaconf import OmegaConf

    import visuomotor.config as config_package

    defaults = OmegaConf.load(
        Path(config_package.__file__).parent / "augmentation" / "defaults.yaml"
    )
    assert defaults.input.rgb_crop == {
        "train": "random",
        "evaluation": "center",
        "resize": 84,
        "output": 76,
    }
    assert defaults.input.voxel_crop == {
        "train": "random",
        "evaluation": "center",
        "output": 58,
    }
    assert defaults.input.mask_guided_overlay.probability == 0.5
    assert defaults.input.random_background_overlay.alpha == [0.6, 0.6]


def test_train_command_defaults_to_the_named_rgb_diffusion_example():
    from visuomotor.cli import _build_parser

    args, overrides = _build_parser().parse_known_args(["train"])
    assert args.config_name == "train_rgb_diffusion"
    assert overrides == []


def test_rollout_command_defaults_to_ema_first_weight_selection():
    from visuomotor.cli import _build_parser

    args = _build_parser().parse_args(["rollout", "latest.ckpt"])
    assert args.weights == "auto"


def test_setup_downloads_assets_from_visuomotor_stack_by_default():
    from visuomotor.cli import _build_parser

    args = _build_parser().parse_args(["setup", "--assets-only"])
    assert args.repo == "zheyu-zhuang/visuomotor-stack"
    assert args.release_tag == "assets"


def test_setup_force_is_forwarded_to_dependencies_and_assets(monkeypatch):
    from visuomotor import cli

    calls = []

    class SetupDependencies:
        @staticmethod
        def install_suite_dependencies(**kwargs):
            calls.append(("dependencies", kwargs))

        @staticmethod
        def install_assets(**kwargs):
            calls.append(("assets", kwargs))
            return 0

    monkeypatch.setattr(cli, "_load_setup_dependencies", lambda: SetupDependencies)
    args = cli._build_parser().parse_args(["setup", "--force"])

    assert cli._run_setup(args) == 0
    assert calls[0] == (
        "dependencies", {"deps_root": None, "force": True}
    )
    assert calls[1][1]["force"] is True


def _run_spec():
    from visuomotor.config.schema import RegimeSpec, TaskSpec

    return {
        "task": TaskSpec("square_d0", 400, ("agentview",), (0.0, 0.0, 0.82)),
        "regime": RegimeSpec("in_domain"),
    }


def test_checkpoint_is_self_contained_and_versioned(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    normalizer = Normalizer()
    normalizer.update_samples("action", torch.tensor([[-1.0], [1.0]]))
    normalizer.finalize()
    path = tmp_path / "checkpoint.pt"
    run_spec = _run_spec()
    save_checkpoint(path, run_spec=run_spec, policy=model, ema=None,
                    optimizer=optimizer, scheduler=None, normalizer=normalizer,
                    epoch=2, global_step=7)
    payload = load_checkpoint(path)
    assert payload["global_step"] == 7
    assert from_dict(payload["run_spec"]) == run_spec
    payload.pop("rng")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="rng"):
        load_checkpoint(path)


def test_resource_environment_override(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    monkeypatch.setenv("VISUOMOTOR_DATASET_ROOT", str(dataset_root))
    assert load_resource_paths().dataset_root == dataset_root.resolve()
