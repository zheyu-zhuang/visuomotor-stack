import copy

import dill
import torch

from visuomotor.workspace.base import BaseWorkspace
from visuomotor.workspace.training_utils import (
    EMAModel,
    restore_checkpoint_with_optional_ema,
)


def _advance(model, ema, values):
    for value in values:
        with torch.no_grad():
            model.weight.fill_(value)
        ema.step(model)


def test_ema_resume_preserves_decay_progress():
    source = torch.nn.Linear(1, 1, bias=False)
    uninterrupted_model = copy.deepcopy(source)
    uninterrupted_average = copy.deepcopy(source)
    uninterrupted = EMAModel(uninterrupted_average, power=0.75)
    _advance(uninterrupted_model, uninterrupted, range(1, 7))

    resumed_model = copy.deepcopy(source)
    resumed_average = copy.deepcopy(source)
    before_resume = EMAModel(resumed_average, power=0.75)
    _advance(resumed_model, before_resume, range(1, 4))

    restored_average = copy.deepcopy(resumed_average)
    restored = EMAModel(restored_average, power=0.75)
    restored.load_state_dict(before_resume.state_dict())
    _advance(resumed_model, restored, range(4, 7))

    assert restored.optimization_step == uninterrupted.optimization_step
    assert restored.decay == uninterrupted.decay
    torch.testing.assert_close(restored_average.weight, uninterrupted_average.weight)


def test_ema_copies_model_buffers_and_stays_in_evaluation_mode():
    source = torch.nn.BatchNorm1d(2)
    average = copy.deepcopy(source)
    ema = EMAModel(average)
    average.train()
    source.running_mean.fill_(3)
    source.running_var.fill_(4)
    source.num_batches_tracked.fill_(5)

    ema.step(source)

    assert average.training is False
    torch.testing.assert_close(average.running_mean, source.running_mean)
    torch.testing.assert_close(average.running_var, source.running_var)
    torch.testing.assert_close(average.num_batches_tracked, source.num_batches_tracked)


def test_disabled_ema_ignores_ema_state_when_resuming(tmp_path):
    workspace = BaseWorkspace({}, output_dir=tmp_path)
    workspace.model = torch.nn.Linear(1, 1, bias=False)
    workspace.ema_model = None
    workspace.ema = None
    workspace.global_step = 4
    expected = torch.full_like(workspace.model.weight, 7)
    checkpoint = tmp_path / "checkpoint.ckpt"
    torch.save(
        {
            "cfg": {},
            "state_dicts": {
                "model": {"weight": expected},
                "ema_model": {"weight": torch.zeros_like(expected)},
                "ema": {"decay": 0.5, "optimization_step": 5},
            },
            "pickles": {},
        },
        checkpoint,
        pickle_module=dill,
    )

    restore_checkpoint_with_optional_ema(workspace, checkpoint)

    torch.testing.assert_close(workspace.model.weight, expected)
    assert workspace.ema_model is None
    assert workspace.ema is None
