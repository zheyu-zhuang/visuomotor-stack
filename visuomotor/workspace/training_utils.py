"""Shared training lifecycle utilities for Visuomotor Stack workspaces."""

from __future__ import annotations

import copy
import json
import numbers
import os
from typing import Any, Callable, Dict, Optional

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class TopKCheckpointManager:
    """Track top-k checkpoint paths by a scalar metric."""

    def __init__(
        self,
        save_dir,
        monitor_key: str,
        mode="min",
        k=1,
        format_str="epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt",
    ):
        assert mode in {"max", "min"}
        assert k >= 0
        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = {}

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None
        value = data[self.monitor_key]
        checkpoint_path = os.path.join(self.save_dir, self.format_str.format(**data))
        if len(self.path_value_map) < self.k:
            self.path_value_map[checkpoint_path] = value
            return checkpoint_path

        ranked = sorted(self.path_value_map.items(), key=lambda item: item[1])
        minimum_path, minimum_value = ranked[0]
        maximum_path, maximum_value = ranked[-1]
        replaced_path = None
        if self.mode == "max" and value > minimum_value:
            replaced_path = minimum_path
        elif self.mode == "min" and value < maximum_value:
            replaced_path = maximum_path
        if replaced_path is None:
            return None

        del self.path_value_map[replaced_path]
        self.path_value_map[checkpoint_path] = value
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if os.path.exists(replaced_path):
            os.remove(replaced_path)
        return checkpoint_path


class JsonLogger:
    """Append numeric training metrics and resume after the last complete line."""

    def __init__(
        self,
        path: str,
        filter_fn: Optional[Callable[[str, Any], bool]] = None,
    ):
        self.path = path
        self.filter_fn = filter_fn or (
            lambda _key, value: isinstance(value, numbers.Number)
        )
        self.file = None
        self.last_log = None

    def start(self):
        try:
            self.file = file = open(self.path, "r+", buffering=1)
        except FileNotFoundError:
            self.file = file = open(self.path, "w+", buffering=1)
        position = file.seek(0, os.SEEK_END)
        while position > 0 and file.read(1) != "\n":
            position -= 1
            file.seek(position)
        last_line_end = file.tell()
        position = max(0, position - 1)
        file.seek(position)
        while position > 0 and file.read(1) != "\n":
            position -= 1
            file.seek(position)
        if file.tell() < last_line_end:
            self.last_log = json.loads(file.readline())
        file.seek(last_line_end)
        file.truncate()

    def stop(self):
        self.file.close()
        self.file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def log(self, data: dict):
        filtered = {
            key: value for key, value in data.items() if self.filter_fn(key, value)
        }
        self.last_log = filtered
        for key, value in filtered.items():
            if isinstance(value, numbers.Integral):
                filtered[key] = int(value)
            elif isinstance(value, numbers.Number):
                filtered[key] = float(value)
        self.file.write(json.dumps(filtered).replace("\n", "") + "\n")

    def get_last_log(self):
        return copy.deepcopy(self.last_log)


class EMAModel:
    """Warm-started exponential moving average of model parameters."""

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        self.averaged_model = model.eval().requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.decay = 0.0
        self.optimization_step = 0

    def state_dict(self):
        return {
            "decay": self.decay,
            "optimization_step": self.optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.decay = float(state_dict["decay"])
        self.optimization_step = int(state_dict["optimization_step"])

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.averaged_model.eval()
        self.decay = self.get_decay(self.optimization_step)
        for module, averaged in zip(new_model.modules(), self.averaged_model.modules()):
            for parameter, average in zip(
                module.parameters(recurse=False),
                averaged.parameters(recurse=False),
            ):
                source = parameter.to(dtype=average.dtype).data
                if isinstance(module, _BatchNorm) or not parameter.requires_grad:
                    average.copy_(source)
                else:
                    average.mul_(self.decay).add_(source, alpha=1 - self.decay)
            for source, average in zip(
                module.buffers(recurse=False),
                averaged.buffers(recurse=False),
            ):
                average.copy_(source)
        self.optimization_step += 1


def restore_checkpoint_with_optional_ema(workspace, path):
    """Restore training state while tolerating disabled optional runtime objects."""
    exclude_keys = [] if workspace.ema is not None else ["ema_model", "ema"]
    payload = workspace.load_checkpoint(path=path, exclude_keys=exclude_keys)
    if workspace.ema is not None and "ema" not in payload["state_dicts"]:
        if "ema_model" not in payload["state_dicts"]:
            workspace.ema_model.load_state_dict(workspace.model.state_dict())
        workspace.ema.optimization_step = workspace.global_step + 1
    return payload


def get_scheduler(name, optimizer, num_warmup_steps=None, num_training_steps=None, **kwargs):
    """Build a Diffusers learning-rate scheduler with explicit step counts."""
    from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType

    name = SchedulerType(name)
    schedule = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule(optimizer, **kwargs)
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires num_warmup_steps")
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)
    if num_training_steps is None:
        raise ValueError(f"{name} requires num_training_steps")
    return schedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )


def init_wandb(logging, *, output_dir, config: dict, updates: Optional[dict] = None):
    """Initialize the shared experiment logger from a resolved logging spec."""
    if logging is None:
        return None
    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(
        dir=str(output_dir),
        config=config,
        settings=wandb.Settings(console="off"),
        project=logging.project,
        resume=logging.resume,
        mode=logging.mode,
        id=logging.run_id,
        name=logging.name,
        group=logging.group,
        job_type=logging.job_type,
        tags=logging.tags,
    )
    wandb.config.update({"output_dir": str(output_dir), **(updates or {})})
    return run
