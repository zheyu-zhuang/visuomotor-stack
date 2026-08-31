"""Shared base interface for image-conditioned policies."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from visuomotor.data.core.normalization import Normalizer
from visuomotor.policy import checkpoint as PolicyCheckpoint


@dataclass
class PolicyDiagnostics:
    encoder: Optional[Any] = None
    predicted_actions: Optional[torch.Tensor] = None


class ModuleAttrMixin(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class BaseImagePolicy(ModuleAttrMixin):
    """Abstract policy interface used by training and inference policies."""

    def predict_action(
        self,
        canonical_obs: Dict[str, torch.Tensor],
        *,
        task_context: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run policy inference.

        Args:
            canonical_obs: Canonical observations with shape `[B, To, ...]` tensors.
            task_context: Task-constant tensors with shape `[B, ...]`.
        Returns:
            Dict containing at least `action` with shape `[B, Ta, Da]`.
        """
        raise NotImplementedError()

    def collect_diagnostics(
        self,
        canonical_obs: Dict[str, torch.Tensor],
        *,
        task_context: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs,
    ) -> PolicyDiagnostics:
        """Collect explicit evaluation diagnostics without model side state."""
        raise NotImplementedError()

    def reset(self):
        """Reset state for stateful policies."""
        pass

    def set_normalizer(self, normalizer: Normalizer):
        """Set data normalizer used by the policy."""
        raise NotImplementedError()

    def save_checkpoint(self, path, *, runner_spec=None) -> str:
        """Save a self-contained release checkpoint for either policy family."""
        from visuomotor.config.schema import to_dict

        model_spec = getattr(self, "model_spec", None)
        if model_spec is None:
            raise ValueError("saving a policy checkpoint requires a resolved ModelSpec")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": PolicyCheckpoint.RELEASE_CHECKPOINT_SCHEMA_VERSION,
                "model_spec": to_dict(model_spec),
                "runner_spec": None if runner_spec is None else to_dict(runner_spec),
                "state_dict": self.state_dict(),
            },
            path,
        )
        return str(path.resolve())
