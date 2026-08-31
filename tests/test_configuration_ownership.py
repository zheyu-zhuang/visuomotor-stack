from dataclasses import replace
from pathlib import Path

import torch
from hydra import compose, initialize_config_module

from visuomotor.config import schema as Schema
from visuomotor.config.resolve import (
    resolve_policy_run,
    resolve_rvt2_pretraining,
    resolve_seeker_pretraining,
)
from visuomotor.perception.focus.rvt2 import model as Rvt2Heatmap


def _compose(name):
    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        return compose(config_name=name)


def test_trajectory_contract_cannot_diverge_across_runtime_consumers():
    run = resolve_policy_run(_compose("train_rgb_diffusion"))
    incompatible = replace(
        run,
        runner=replace(
            run.runner,
            trajectory=replace(run.runner.trajectory, execution_horizon=3),
        ),
    )
    try:
        Schema.validate(incompatible)
    except ValueError as error:
        assert "trajectory contracts differ" in str(error)
    else:
        raise AssertionError("incompatible trajectory contracts must be rejected")


def test_pretraining_routes_resolve_to_typed_runtime_specs():
    seeker = resolve_seeker_pretraining(_compose("pretrain/seeker"))
    rvt2 = resolve_rvt2_pretraining(_compose("pretrain/rvt2_heatmap"))

    assert seeker.dataset.trajectory.action_dim == 10
    assert seeker.dataset.source_observation.shape_meta(10)["action"]["shape"] == [10]
    assert seeker.model.stage_stride == 50
    assert seeker.training.num_epochs == 300
    assert rvt2.model.stage_stride == 50
    assert rvt2.model.heatmap["patch_size"] == 16
    assert rvt2.workspace.train_loader.batch_size == 128
    assert rvt2.workspace.train_loader.shuffle is True
    assert rvt2.workspace.val_loader.shuffle is False
    assert Schema.from_dict(Schema.to_dict(seeker)) == seeker
    assert Schema.from_dict(Schema.to_dict(rvt2)) == rvt2


def test_spec_deserialization_uses_declared_defaults_for_omitted_fields():
    spec = Schema.InputSpec(
        name="voxel",
        proprio=("eef_pos",),
        voxel=Schema.VoxelInputSpec(),
    )
    payload = Schema.to_dict(spec)
    del payload["voxel"]["frame"]

    assert Schema.from_dict(payload) == spec


def test_policy_workspaces_do_not_read_raw_hydra_after_resolution():
    root = Path(__file__).parents[1] / "visuomotor" / "workspace"
    for name in ("policy_training.py", "seeker_pretraining.py", "rvt2_training.py"):
        text = (root / name).read_text()
        assert "self.cfg" not in text
        assert "OmegaConf" not in text
        assert "hydra.utils.instantiate" not in text


def test_workspaces_delegate_runtime_construction_to_config_builders():
    root = Path(__file__).parents[1] / "visuomotor" / "workspace"
    policy = (root / "policy_training.py").read_text()
    seeker = (root / "seeker_pretraining.py").read_text()
    rvt2 = (root / "rvt2_training.py").read_text()

    assert "Build.build_dataloader" in policy
    assert "Build.build_dataloader" in seeker
    assert "Build.build_dataloader" in rvt2
    assert "MimicGenDataset(" not in seeker
    assert "MimicGenDataset(" not in rvt2
    assert "PatchFeatureBackbone(" not in rvt2
    assert "PatchActivationHead(" not in rvt2
    assert 'model_state["' not in rvt2


def test_boundary_vocabulary_has_no_retired_ambiguous_names():
    package = Path(__file__).parents[1] / "visuomotor"
    text = "\n".join(
        path.read_text()
        for suffix in ("*.py", "*.yaml")
        for path in package.rglob(suffix)
    )

    assert "def compute_loss(" not in text
    assert "def validate_rgb(" not in text
    assert "def validate_voxel(" not in text
    assert "def canonicalize_modalities(" not in text
    assert "def get_config(" not in text
    assert "anchor_egocentric_dropout" not in text
    assert "pad_action_chunks_at_keypose_boundary" not in text
    assert "pad_action_chunks_at_boundary" not in text


def test_rvt2_inference_uses_checkpoint_owned_model_config(monkeypatch):
    model_config = {
        "patch_size": 4,
        "keypoint_box_zoom": 2.5,
        "conv_patch_dim": 16,
        "dino_ckpt": "unused",
    }
    backbone = Rvt2Heatmap.PatchFeatureBackbone(
        backbone_type="conv",
        dino_ckpt_path=None,
        image_size=8,
        patch_size=4,
        conv_dim=16,
    )
    head_config = {
        "patch_dim": backbone.output_dim,
        "num_robots": 2,
        "hidden_dim": 16,
        "grid_size": 2,
        "task_emb_dim": 8,
        "transformer_depth": 1,
        "transformer_heads": 4,
        "transformer_dropout": 0.0,
    }
    head = Rvt2Heatmap.PatchActivationHead(**head_config)
    payload = {
        "rvt2_heatmap_config": model_config,
        "patch_backbone": "conv",
        "patch_backbone_state_dict": backbone.state_dict(),
        "head_config": head_config,
        "head_state_dict": head.state_dict(),
        "gripper_mean": torch.zeros(1).numpy(),
        "gripper_std": torch.ones(1).numpy(),
    }
    monkeypatch.setattr(Rvt2Heatmap, "load_checkpoint_payload", lambda *_args, **_kwargs: payload)

    model = Rvt2Heatmap.RVT2Heatmap(checkpoint="model.pt", vit_in=8)

    assert model.patch_size == 4
    assert model.zoom == 2.5
