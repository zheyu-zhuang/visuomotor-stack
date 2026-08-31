"""Deterministic sample selection and state-isolated diagnostic inference."""

from __future__ import annotations

import contextlib
import random
from typing import Iterator

import numpy as np
import torch


def evenly_spaced_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 0 or count < 1:
        raise ValueError("length must be non-negative and count positive")
    if length == 0:
        return ()
    count = min(length, count)
    return tuple(np.linspace(0, length - 1, num=count, dtype=np.int64).tolist())


@contextlib.contextmanager
def isolated_evaluation(module: torch.nn.Module, *, seed: int = 0) -> Iterator[None]:
    """Restore RNG and module mode after deterministic diagnostic inference."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    was_training = module.training
    devices = sorted(
        {
            parameter.device.index
            for parameter in module.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)
            module.eval()
            with torch.no_grad():
                yield
    finally:
        module.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
