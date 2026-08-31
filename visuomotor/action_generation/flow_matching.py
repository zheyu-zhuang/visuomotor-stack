"""Conditional flow matching for action chunks, using the same UNet1D backbone as diffusion."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from torch.nn import functional as F

from visuomotor.action_generation.base import ActionGenerator, conditioning_tensor
from visuomotor.action_generation.unet1d import ConditionalUnet1D

CONDITIONAL_OT_PATH = AffineProbPath(scheduler=CondOTScheduler())


class FlowMatchingActionGenerator(ActionGenerator):
    """Generate fixed-horizon action chunks via conditional flow matching (CondOT)."""

    def __init__(
        self,
        *,
        action_dim: int,
        condition_dim: int,
        prediction_horizon: int,
        integration_steps: int = 100,
        time_embedding_dim: int = 128,
        time_embedding_scale: float = 100.0,
        unet_channels: Sequence[int] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        clip_sample: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.integration_steps = int(integration_steps)
        self.clip_sample = bool(clip_sample)
        unet_channels = tuple(int(width) for width in unet_channels)
        time_embedding_dim = int(time_embedding_dim)
        kernel_size = int(kernel_size)
        n_groups = int(n_groups)
        for name, value in (
            ("action_dim", self.action_dim),
            ("condition_dim", self.condition_dim),
            ("prediction_horizon", self.prediction_horizon),
            ("integration_steps", self.integration_steps),
            ("kernel_size", kernel_size),
            ("n_groups", n_groups),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if not unet_channels or any(width < 1 for width in unet_channels):
            raise ValueError(
                f"unet_channels must contain positive widths, got {unet_channels}"
            )
        if any(width % n_groups for width in unet_channels):
            raise ValueError(
                f"every unet_channels width must be divisible by n_groups={n_groups}, "
                f"got {unet_channels}"
            )
        downsample_factor = 2 ** (len(unet_channels) - 1)
        if self.prediction_horizon % downsample_factor:
            raise ValueError(
                f"prediction_horizon={self.prediction_horizon} must be divisible by "
                f"the UNet downsample factor {downsample_factor} for "
                f"unet_channels={unet_channels}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to preserve sequence length, got {kernel_size}")
        if time_embedding_dim < 4 or time_embedding_dim % 2:
            raise ValueError(
                "time_embedding_dim must be an even integer of at least 4, "
                f"got {time_embedding_dim}"
            )
        # Scaling [0, 1] path time restores a multi-scale sinusoidal time code.
        self.time_embedding_scale = float(time_embedding_scale)
        if (
            not math.isfinite(self.time_embedding_scale)
            or self.time_embedding_scale <= 0.0
        ):
            raise ValueError(
                "time_embedding_scale must be positive and finite, "
                f"got {self.time_embedding_scale}"
            )
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.condition_dim,
            diffusion_step_embed_dim=time_embedding_dim,
            down_dims=unet_channels,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def _validate_conditioning(
        self, conditioning: torch.Tensor, *, batch_size: Optional[int] = None
    ) -> None:
        expected_batch = conditioning.shape[0] if batch_size is None else int(batch_size)
        if conditioning.shape != (expected_batch, self.condition_dim):
            raise ValueError(
                f"expected conditioning [{expected_batch},{self.condition_dim}], got "
                f"{tuple(conditioning.shape)}"
            )

    def _velocity(
        self, trajectory: torch.Tensor, time, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the UNet as a continuous vector field at path time ``t`` in [0, 1]."""
        time = torch.as_tensor(time, device=trajectory.device, dtype=trajectory.dtype)
        time = time.reshape(-1).expand(trajectory.shape[0])
        return self.model(
            trajectory, time * self.time_embedding_scale, global_cond=conditioning
        )

    def _guided_velocity(
        self, trajectory: torch.Tensor, time: float, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Redirect the field at its clean-sample estimate, as DDPM's ``clip_sample`` does."""
        velocity = self._velocity(trajectory, time, conditioning)
        remaining = 1.0 - time
        if not self.clip_sample or remaining <= 0.0:
            return velocity
        endpoint = trajectory + remaining * velocity
        return (endpoint.clamp(-1.0, 1.0) - trajectory) / remaining

    def loss(
        self,
        actions: torch.Tensor,
        conditioning,
        *,
        sample_valid: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CondOT regression, optionally over a subset of the batch's rows.

        ``sample_valid`` marks the rows that carry a supervisable target; the
        rest are zeroed out of both the target and the conditioning so they
        cannot reach the UNet, and dropped from the mean.
        """
        if tuple(actions.shape[1:]) != (self.prediction_horizon, self.action_dim):
            raise ValueError("actions do not match the configured chunk shape")
        return self._flow_matching_loss(
            actions,
            conditioning,
            sample_valid=sample_valid,
            generator=generator,
            noise=noise,
        )

    def _flow_matching_loss(
        self,
        target: torch.Tensor,
        conditioning,
        *,
        sample_valid: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conditioning = conditioning_tensor(conditioning)
        self._validate_conditioning(conditioning, batch_size=target.shape[0])
        valid = None
        if sample_valid is not None:
            if sample_valid.shape != (target.shape[0],):
                raise ValueError(
                    f"sample_valid must have shape {(target.shape[0],)}, got "
                    f"{tuple(sample_valid.shape)}"
                )
            valid = sample_valid.bool()
            target = torch.where(
                valid.reshape(valid.shape[0], *([1] * (target.ndim - 1))),
                target,
                torch.zeros_like(target),
            )
            conditioning = torch.where(
                valid.reshape(valid.shape[0], *([1] * (conditioning.ndim - 1))),
                conditioning,
                torch.zeros_like(conditioning),
            )
        if noise is None:
            noise = torch.randn(
                target.shape,
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        time = torch.rand(
            target.shape[0],
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        path = CONDITIONAL_OT_PATH.sample(x_0=noise, x_1=target, t=time)
        predicted = self._velocity(path.x_t, path.t, conditioning)
        per_sample = F.mse_loss(predicted, path.dx_t, reduction="none").mean(
            dim=tuple(range(1, target.ndim))
        )
        if valid is None:
            return per_sample.mean()
        per_sample = torch.where(valid, per_sample, torch.zeros_like(per_sample))
        return per_sample.sum() / valid.sum().to(per_sample.dtype).clamp_min(1.0)

    def get_runtime_config(self) -> dict:
        return {"Flow": f"Midpoint RK2 · {self.integration_steps} steps · CondOT schedule"}

    @torch.no_grad()
    def sample(
        self,
        conditioning,
        *,
        steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Integrate the vector field over [0, 1] with fixed-step midpoint RK2."""
        return self._integrate(conditioning, steps=steps, generator=generator)

    def sample_with_grad(
        self,
        conditioning,
        *,
        steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """:meth:`sample`, keeping the solver graph so a downstream loss trains it."""
        return self._integrate(conditioning, steps=steps, generator=generator)

    def _integrate(
        self,
        conditioning,
        *,
        steps: Optional[int],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        conditioning = conditioning_tensor(conditioning)
        self._validate_conditioning(conditioning)
        state = torch.randn(
            (conditioning.shape[0], self.prediction_horizon, self.action_dim),
            device=conditioning.device,
            dtype=conditioning.dtype,
            generator=generator,
        )
        steps = self.integration_steps if steps is None else int(steps)
        if steps < 1:
            raise ValueError(f"integration needs at least one step, got {steps}")
        step = 1.0 / steps
        for index in range(steps):
            time = index * step
            slope = self._guided_velocity(state, time, conditioning)
            state = state + step * self._guided_velocity(
                state + (0.5 * step) * slope, time + 0.5 * step, conditioning
            )
        return state.clamp(-1.0, 1.0) if self.clip_sample else state
