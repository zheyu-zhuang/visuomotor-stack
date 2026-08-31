from types import SimpleNamespace

import pytest
import torch

from visuomotor.action_generation.diffusion import DiffusionActionGenerator
from visuomotor.action_generation.flow_matching import FlowMatchingActionGenerator


class Scheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=4, prediction_type="epsilon")
        self.timesteps = []

    def add_noise(self, sample, noise, timesteps):
        return sample + noise

    def set_timesteps(self, count, device=None):
        self.timesteps = torch.arange(count - 1, -1, -1, device=device)

    def step(self, prediction, timestep, sample, **kwargs):
        return SimpleNamespace(prev_sample=sample - prediction * 0.01)


def test_diffusion_loss_and_sample_shapes():
    generator = DiffusionActionGenerator(
        action_dim=3,
        condition_dim=6,
        prediction_horizon=8,
        noise_scheduler=Scheduler(),
        num_inference_steps=2,
        diffusion_step_embed_dim=16,
        unet_channels=(8, 16),
        n_groups=4,
    )
    actions = torch.randn(2, 8, 3)
    condition = torch.randn(2, 6)
    assert generator.loss(actions, condition).ndim == 0
    assert generator.sample(condition).shape == actions.shape


def test_flow_matching_loss_and_sample_shapes():
    generator = FlowMatchingActionGenerator(
        action_dim=4,
        condition_dim=5,
        prediction_horizon=6,
        unet_channels=(8, 16),
        n_groups=4,
        integration_steps=2,
    )
    actions = torch.randn(3, 6, 4)
    condition = torch.randn(3, 5)
    assert generator.loss(actions, condition).ndim == 0
    assert generator.sample(condition).shape == actions.shape


def test_flow_matching_sample_with_grad_keeps_the_solver_graph():
    generator = FlowMatchingActionGenerator(
        action_dim=4,
        condition_dim=5,
        prediction_horizon=6,
        unet_channels=(8, 16),
        n_groups=4,
        integration_steps=2,
    )
    condition = torch.randn(3, 5, requires_grad=True)

    assert generator.sample(condition).grad_fn is None
    generator.sample_with_grad(condition).square().mean().backward()

    assert condition.grad is not None
    assert all(
        parameter.grad is not None for parameter in generator.model.parameters()
    )


def test_flow_matching_midpoint_solver_integrates_constant_velocity():
    generator = FlowMatchingActionGenerator(
        action_dim=2,
        condition_dim=3,
        prediction_horizon=4,
        unet_channels=(8,),
        n_groups=4,
        integration_steps=3,
    )

    class ConstantVelocity(torch.nn.Module):
        def forward(self, sample, timestep, global_cond):
            return torch.full_like(sample, 0.25)

    generator.model = ConstantVelocity()
    condition = torch.zeros(2, 3)
    expected_initial = torch.randn(
        2, 4, 2, generator=torch.Generator().manual_seed(7)
    )
    actual = generator.sample(
        condition, generator=torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(actual, expected_initial + 0.25)


class _Overshoot(torch.nn.Module):
    def forward(self, sample, timestep, global_cond):
        return torch.full_like(sample, 20.0)


def _overshooting_generator(*, clip_sample):
    generator = FlowMatchingActionGenerator(
        action_dim=2,
        condition_dim=3,
        prediction_horizon=4,
        unet_channels=(8,),
        n_groups=4,
        integration_steps=8,
        clip_sample=clip_sample,
    )
    generator.model = _Overshoot()
    return generator


def test_flow_matching_clips_the_chunk_only_when_asked():
    clipped, unclipped = (
        _overshooting_generator(clip_sample=clip).sample(
            torch.zeros(2, 3), generator=torch.Generator().manual_seed(11)
        )
        for clip in (True, False)
    )
    assert clipped.abs().max() <= 1.0 + 1e-5
    assert unclipped.abs().max() > 1.0


@pytest.mark.parametrize(
    "overrides,match",
    (
        ({"prediction_horizon": 5, "unet_channels": (8, 16)}, "downsample factor"),
        ({"integration_steps": 0}, "integration_steps"),
        ({"kernel_size": 4}, "kernel_size"),
        ({"time_embedding_dim": 3}, "time_embedding_dim"),
        ({"unet_channels": (6,), "n_groups": 4}, "divisible"),
    ),
)
def test_flow_matching_rejects_invalid_generator_contracts(overrides, match):
    kwargs = {
        "action_dim": 2,
        "condition_dim": 3,
        "prediction_horizon": 4,
        "unet_channels": (8,),
        "n_groups": 4,
        "integration_steps": 2,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        FlowMatchingActionGenerator(**kwargs)


def test_flow_matching_validates_condition_shape_before_the_unet():
    generator = FlowMatchingActionGenerator(
        action_dim=2,
        condition_dim=3,
        prediction_horizon=4,
        unet_channels=(8,),
        n_groups=4,
        integration_steps=1,
    )
    with pytest.raises(ValueError, match="expected conditioning"):
        generator.loss(torch.randn(2, 4, 2), torch.randn(2, 4))
