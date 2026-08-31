from pathlib import Path
from types import SimpleNamespace

import dill
import torch

from visuomotor.workspace import rvt2_training
from visuomotor.workspace.base import BaseWorkspace


def test_base_workspace_snapshot_round_trip(tmp_path):
    workspace = BaseWorkspace({"name": "test"}, output_dir=tmp_path)

    path = Path(workspace.save_snapshot())

    restored = torch.load(path.open("rb"), pickle_module=dill)
    assert path == tmp_path / "snapshots" / "latest.pkl"
    assert restored.cfg == workspace.cfg
    assert restored.output_dir == tmp_path


def test_rvt2_augmentation_preview_uses_runtime_heatmap_config(
    monkeypatch, tmp_path
):
    visualization = SimpleNamespace(
        enabled=True,
        augmentation_preview=True,
        num_samples=2,
        save=SimpleNamespace(images=False, videos=False),
    )
    runtime = {
        "spec": SimpleNamespace(
            model=SimpleNamespace(stage_stride=50),
            training=SimpleNamespace(checkpoint_every=0, epochs=0),
            workspace=SimpleNamespace(
                visualization=visualization,
                visualization_sampling="even",
                visualization_alpha=0.45,
            ),
        ),
        "output_dir": tmp_path,
        "background_randomizer": None,
        "rvt2_heatmap_cfg": {
            "dino_image_size": 224,
            "patch_size": 16,
        },
    }
    model_state = SimpleNamespace(
        patch_backbone=object(),
        head=object(),
        device=torch.device("cpu"),
    )
    captured = {}
    monkeypatch.setattr(
        rvt2_training,
        "_make_visualization_loader",
        lambda *_args, **_kwargs: object(),
    )

    def save_visualization_grid(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        rvt2_training,
        "_save_visualization_grid",
        save_visualization_grid,
    )

    rvt2_training._run_training_loop(
        runtime=runtime,
        data={"train_set": object()},
        model_state=model_state,
        start_epoch=1,
        json_logger=object(),
    )

    assert captured["dino_image_size"] == 224
    assert captured["patch_size"] == 16
