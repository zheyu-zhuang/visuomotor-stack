"""Globally conditioned DDPM action generation."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.nn import functional as F

from visuomotor.action_generation.base import ActionGenerator, conditioning_tensor
from visuomotor.action_generation.unet1d import ConditionalUnet1D


class DiffusionActionGenerator(ActionGenerator):
    """Generate fixed-horizon action chunks with a conditional 1D UNet."""

    def __init__(
        self,
        *,
        action_dim: int,
        condition_dim: int,
        prediction_horizon: int,
        noise_scheduler,
        num_inference_steps: Optional[int] = None,
        diffusion_step_embed_dim: int = 128,
        unet_channels: Sequence[int] = (128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        scheduler_step_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.noise_scheduler = noise_scheduler
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.condition_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(unet_channels),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = int(num_inference_steps)
        self.scheduler_step_kwargs = dict(scheduler_step_kwargs or {})

    def _validate_actions(self, actions: torch.Tensor) -> None:
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B, horizon, action_dim]")
        if tuple(actions.shape[1:]) != (self.prediction_horizon, self.action_dim):
            raise ValueError(
                f"expected actions [B,{self.prediction_horizon},{self.action_dim}], got "
                f"{tuple(actions.shape)}"
            )

    def loss(
        self,
        actions: torch.Tensor,
        conditioning,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return the DDPM denoising objective for an action chunk."""
        self._validate_actions(actions)
        conditioning = conditioning_tensor(conditioning)
        if conditioning.shape != (actions.shape[0], self.condition_dim):
            raise ValueError(
                f"expected conditioning [B,{self.condition_dim}], got "
                f"{tuple(conditioning.shape)}"
            )
        noise = torch.randn(
            actions.shape,
            dtype=actions.dtype,
            device=actions.device,
            generator=generator,
        )
        timesteps = torch.randint(
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            generator=generator,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.model(noisy_actions, timesteps, global_cond=conditioning)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"unsupported prediction type: {prediction_type}")
        return F.mse_loss(prediction, target)

    @torch.no_grad()
    def sample(
        self,
        conditioning,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample one action chunk per conditioning row."""
        conditioning = conditioning_tensor(conditioning)
        if conditioning.shape[1] != self.condition_dim:
            raise ValueError(
                f"expected conditioning width {self.condition_dim}, got "
                f"{conditioning.shape[1]}"
            )
        trajectory = torch.randn(
            (conditioning.shape[0], self.prediction_horizon, self.action_dim),
            dtype=conditioning.dtype,
            device=conditioning.device,
            generator=generator,
        )
        try:
            self.noise_scheduler.set_timesteps(
                self.num_inference_steps, device=conditioning.device
            )
        except TypeError:
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(
                trajectory, timestep, global_cond=conditioning
            )
            trajectory = self.noise_scheduler.step(
                prediction,
                timestep,
                trajectory,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample
        return trajectory
