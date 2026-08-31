"""Policy composition for perception-conditioned action generation."""

from __future__ import annotations

import inspect
from typing import Mapping, Optional

import torch
from torch import nn

from visuomotor.config.schema import ModelSpec
from visuomotor.data.core.normalization import Normalizer, normalize_obs, obs_robot_id
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.policy.base import BaseImagePolicy, PolicyDiagnostics


class GenerativePolicy(BaseImagePolicy):
    """Compose an encoder, action generator, and optional action normalizer."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        generator: nn.Module,
        observation_kinds: Mapping[str, str],
        observation_feature_dim: int,
        action_normalizer: Optional[nn.Module] = None,
        execution_horizon: Optional[int] = None,
        model_spec: Optional[ModelSpec] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        # Applied per observation step, so the generator's conditioning stays
        # `observation_feature_dim` per step once the time axis is flattened.
        self.observation_projection = nn.Linear(
            int(encoder.output_dim), int(observation_feature_dim)
        )
        self.observation_kinds = dict(observation_kinds)
        self.action_normalizer = action_normalizer
        prediction_horizon = int(getattr(generator, "prediction_horizon", 1))
        self.execution_horizon = int(
            prediction_horizon if execution_horizon is None else execution_horizon
        )
        if not 1 <= self.execution_horizon <= prediction_horizon:
            raise ValueError("execution_horizon must be within prediction_horizon")
        self.model_spec = model_spec
        # Focus caching and oracle boxes travel beside the observations; only the
        # encoders that declare them get them.
        parameters = inspect.signature(encoder.forward).parameters
        self.encoder_side_channels = tuple(
            name
            for name in (
                "obs_index",
                "oracle_info",
                "focus_target",
                "global_step",
                "canonical_obs",
                "task_context",
            )
            if name in parameters
        )

    @staticmethod
    def _robot_id(task_context: Optional[Mapping[str, torch.Tensor]]):
        if task_context is None:
            return None
        return obs_robot_id(task_context)

    def get_runtime_config(self, num_epochs: Optional[int] = None) -> dict:
        """Return the runtime configuration block shown at training startup."""
        generator_params = sum(p.numel() for p in self.generator.parameters()) / 1e6
        encoder_params = sum(
            p.numel()
            for module in (self.encoder, self.observation_projection)
            for p in module.parameters()
        ) / 1e6
        training: dict = {
            "Encoder": f"{type(self.encoder).__name__} ({encoder_params:.2f} M)",
            "Generator": f"{type(self.generator).__name__} ({generator_params:.2f} M)",
        }
        report: dict = {
            "Training": training,
        }
        if self.model_spec is not None:
            report["Normalizer"] = self.model_spec.normalizer
        encoder_runtime_config = getattr(self.encoder, "get_runtime_config", None)
        if callable(encoder_runtime_config):
            report.update(encoder_runtime_config())
        generator_runtime_config = getattr(self.generator, "get_runtime_config", None)
        if callable(generator_runtime_config):
            report.update(generator_runtime_config())
        return report

    def set_normalizer(self, normalizer: nn.Module) -> None:
        self._require_declared_normalizer(normalizer)
        self.action_normalizer = normalizer
        setter = getattr(self.encoder, "set_normalizer", None)
        if setter is not None:
            setter(normalizer)

    def _require_declared_normalizer(self, normalizer: nn.Module) -> None:
        """Keep the runtime normalizer consistent with the checkpointed spec.

        A mismatch would make the saved state dict unloadable by the policy that
        the recorded spec rebuilds, so it must fail here rather than at restore.
        """
        if self.model_spec is None:
            return
        expected = Normalizer
        if type(normalizer) is not expected:
            raise TypeError(
                f"policy declares normalizer {self.model_spec.normalizer!r} "
                f"({expected.__name__}) but received {type(normalizer).__name__}"
            )

    @staticmethod
    def _focus_target(*sources) -> Optional[dict]:
        """Group flat dataset focus-target fields for the encoder side channel."""
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            pos = source.get("focus_target_pos")
            valid = source.get("focus_target_valid")
            if pos is not None and valid is not None:
                return {"pos": pos, "valid": valid}
        return None

    def _side_channels(
        self,
        *sources,
        global_step: Optional[int] = None,
    ) -> dict:
        found = {}
        for name in self.encoder_side_channels:
            if name == "focus_target":
                value = self._focus_target(*sources)
            elif name == "global_step":
                value = global_step
            elif name == "task_context":
                value = None
            else:
                value = None
                for source in sources:
                    candidate = source.get(name) if isinstance(source, Mapping) else None
                    if candidate is not None:
                        value = candidate
                        break
            if value is not None:
                found[name] = value
        return found

    def normalize_obs(
        self,
        canonical_obs: Mapping[str, torch.Tensor],
        task_context: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> dict:
        """Canonical observation -> model input; this policy owns the transition."""
        if self.action_normalizer is None:
            if any(
                kind in ("rgb", "voxel") for kind in self.observation_kinds.values()
            ):
                raise ValueError("visual observations require a policy normalizer")
            return dict(canonical_obs)
        return normalize_obs(
            canonical_obs,
            self.action_normalizer,
            observation_kinds=self.observation_kinds,
            robot_id=self._robot_id(task_context),
        )

    def encode(
        self,
        canonical_obs: Mapping[str, torch.Tensor],
        *,
        task_context: Optional[Mapping[str, torch.Tensor]] = None,
        **side_channels,
    ) -> EncoderOutput:
        # Encoders that keep their own separately-owned model space (a released
        # Seeker and its checkpointed normalizer) declare `canonical_obs` and
        # normalize into that space themselves.
        if "canonical_obs" in self.encoder_side_channels:
            side_channels["canonical_obs"] = canonical_obs
        if "task_context" in self.encoder_side_channels:
            side_channels["task_context"] = task_context
        output = self.encoder(
            self.normalize_obs(canonical_obs, task_context), **side_channels
        )
        if torch.is_tensor(output):
            output = EncoderOutput(features=output)
        if not isinstance(output, EncoderOutput):
            raise TypeError("encoders must return a tensor or EncoderOutput")
        return output

    def _condition(self, encoded: EncoderOutput) -> torch.Tensor:
        return self.observation_projection(encoded.features)

    def _normalize(self, actions, task_context):
        if self.action_normalizer is None:
            return actions
        return self.action_normalizer.normalize_action(
            actions, robot_id=self._robot_id(task_context)
        )

    def _denormalize(self, actions, task_context):
        if self.action_normalizer is None:
            return actions
        return self.action_normalizer.denormalize_action(
            actions, robot_id=self._robot_id(task_context)
        )

    def loss(
        self,
        batch,
        actions: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        **kwargs,
    ):
        if actions is None:
            canonical_obs, actions = batch["obs"], batch["action"]
            task_context = batch.get("task_context")
            targets = batch.get("targets")
        else:
            canonical_obs = batch
            task_context = None
            targets = None
        encoded = self.encode(
            canonical_obs,
            task_context=task_context,
            **self._side_channels(
                targets,
                batch,
                canonical_obs,
                global_step=global_step,
            ),
        )
        result = self.generator.loss(
            self._normalize(actions, task_context), self._condition(encoded), **kwargs
        )
        if isinstance(result, Mapping):
            losses = dict(result)
            losses.update(encoded.auxiliary_losses)
            losses["loss"] = sum(losses.values())
            return losses
        return result + sum(encoded.auxiliary_losses.values(), result.new_zeros(()))

    @torch.no_grad()
    def _sample_with_encoder(
        self,
        canonical_obs: Mapping[str, torch.Tensor],
        *,
        task_context: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, EncoderOutput]:
        encoder_kwargs = {
            name: kwargs.pop(name)
            for name in tuple(kwargs)
            if name in self.encoder_side_channels
        }
        encoded = self.encode(
            canonical_obs,
            task_context=task_context,
            **encoder_kwargs,
            **self._side_channels(
                canonical_obs,
            ),
        )
        actions = self.generator.sample(self._condition(encoded), **kwargs)
        return self._denormalize(actions, task_context), encoded

    @torch.no_grad()
    def sample(
        self,
        canonical_obs: Mapping[str, torch.Tensor],
        *,
        task_context: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        prediction, _ = self._sample_with_encoder(
            canonical_obs, task_context=task_context, **kwargs
        )
        return prediction

    @torch.no_grad()
    def collect_diagnostics(self, canonical_obs, *, task_context=None, **kwargs):
        prediction, encoded = self._sample_with_encoder(
            canonical_obs, task_context=task_context, **kwargs
        )
        return PolicyDiagnostics(encoder=encoded, predicted_actions=prediction)

    def predict_action(self, canonical_obs, *, task_context=None, **kwargs) -> dict:
        prediction, encoded = self._sample_with_encoder(
            canonical_obs, task_context=task_context, **kwargs
        )
        return {
            "action": prediction[:, : self.execution_horizon],
            "action_pred": prediction,
            "diagnostics": {"focus": _rollout_focus_records(encoded.focus_records)},
        }


def _rollout_focus_records(records) -> tuple[dict, ...]:
    output = []
    for record in records:
        prediction = record.prediction
        item = {
            "source": record.source,
            "view": record.view,
            "source_size": int(record.image_size[0]),
            "box_px": prediction.box_px,
        }
        points = prediction.metadata.get("points_px")
        if points is not None:
            item["points_px"] = points
        output.append(item)
    return tuple(output)
