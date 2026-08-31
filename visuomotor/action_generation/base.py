"""Shared action-generator interface."""

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Union

import torch


def conditioning_tensor(conditioning) -> torch.Tensor:
    """Flatten plain per-sample conditioning tensors."""
    if not torch.is_tensor(conditioning):
        raise TypeError("conditioning must be a tensor")
    return conditioning.reshape(conditioning.shape[0], -1)


class ActionGenerator(torch.nn.Module, ABC):
    @abstractmethod
    def loss(
        self,
        actions: torch.Tensor,
        condition: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        condition: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
