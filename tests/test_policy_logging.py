from pathlib import Path

from visuomotor.workspace.policy_training import (
    _format_metric_line,
    _format_rollout_line,
    _group_augmentation_runtime_config,
    _load_rollout_bests,
    _rollout_outputs,
)


def test_runtime_config_groups_encoder_crops_with_sample_augmentations():
    model_cfg = {
        "Voxel Encoder": {
            "Architecture": "voxel_simple",
            "Voxel Crop": "Enabled (64³→58³ cells)",
            "RGB Crop": "Disabled",
        }
    }

    augmentations = _group_augmentation_runtime_config(
        {"Scene Yaw": "Enabled ([-180, 180] deg)", "Mirror": "Disabled"},
        model_cfg,
    )

    assert augmentations == {
        "Scene Yaw": "Enabled ([-180, 180] deg)",
        "Mirror": "Disabled",
        "Voxel Crop": "Enabled (64³→58³ cells)",
        "RGB Crop": "Disabled",
    }
    assert model_cfg == {"Voxel Encoder": {"Architecture": "voxel_simple"}}


def test_metric_and_success_records_are_formatted_separately(tmp_path: Path):
    metrics = {"epoch": 3, "global_step": 20, "train_loss": 0.125, "lr": 1e-4}
    assert (
        _format_metric_line(metrics, "train")
        == "epoch 003 | step 20 | loss: 0.1250 | lr: 1.000e-04"
    )

    bests = {"test": 0.85}
    success = _rollout_outputs(
        {
            "test_mean_score": 0.8,
            "test_max_score": 0.9,
            "test_seed_0": 1.0,
            "performance/episodes_per_second": 2.5,
        },
        global_step=20,
        epoch=3,
        best_scores=bests,
    )
    assert success == {
        "global_step": 20,
        "epoch": 3,
        "test_mean_score": 0.8,
        "test_max_score": 0.9,
        "performance/episodes_per_second": 2.5,
    }
    line = "epoch 003 | step 20 | success: test 80.0% (best 90.0%)"
    assert _format_rollout_line(success) == line

    path = tmp_path / "rollout_success.log"
    path.write_text(line + "\n")
    assert _load_rollout_bests(path) == {"test": 0.9}
