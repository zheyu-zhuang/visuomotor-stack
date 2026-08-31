"""Proprioceptive deltas: one definition, two schedulings that must agree.

The dataset differences a cached episode in one pass and indexes the result;
a rollout differences an extra frame retained by the step wrapper. Nothing
enforces that those two produce the same number except a test that runs both.
"""

from typing import Optional

import numpy as np
import pytest
import torch
from gym import spaces

from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.mimicgen.observations import (
    _previous_frame_index,
    delta_history_source_keys,
    derived_proprio_fields,
)
from visuomotor.environment.gym_wrappers.multistep_wrapper import (
    MultiStepWrapper,
    stack_last_n_obs,
)
from visuomotor.environment.runner import SeekerRobomimicImageRunner
from visuomotor.geometry import representation as Representation

FIELDS = ("eef_delta_pos", "eef_delta_rotvec", "gripper_qpos_delta")


def _trajectory(length: int, seed: int = 0) -> dict:
    """A smooth episode, at the per-step scale the cached demonstrations show."""
    generator = torch.Generator().manual_seed(seed)
    steps = torch.randn(length, 3, generator=generator) * 0.006
    position = torch.cumsum(steps, dim=0)
    turns = torch.randn(length, 3, generator=generator) * 0.01
    rotation = torch.eye(3).expand(length, 3, 3).clone()
    for index in range(1, length):
        rotation[index] = rotation[index - 1] @ Representation.rotvec_to_mat(
            turns[index]
        )
    gripper = torch.cumsum(
        torch.randn(length, 2, generator=generator) * 0.0005, dim=0
    )
    return {
        "eef_pos": position.to(torch.float32),
        "eef_rot6d": Representation.mat_to_rot6d(rotation).to(torch.float32),
        "gripper_qpos": gripper.to(torch.float32),
    }


def _dataset_deltas(trajectory: dict, episode_lengths) -> dict:
    """The adapter's scheduling: difference the whole cache once, then index."""
    previous_index = _previous_frame_index(episode_lengths)
    return CoreObservations.proprio_deltas(
        {key: value[previous_index] for key, value in trajectory.items()},
        trajectory,
    )


def _dataset_window(deltas: dict, episode_start: int, local_index: int, n_obs: int):
    """``MimicgenDataset.sampler``'s observation window, for one sample."""
    indices = episode_start + np.maximum(
        local_index + np.arange(-(n_obs - 1), 1), 0
    )
    return {key: value[indices] for key, value in deltas.items()}


def _rollout_window(trajectory: dict, control_step: int, n_obs: int):
    """The rollout's scheduling: the step wrapper's deque, then the runner's pairing."""
    frames = [
        {key: value[index].numpy() for key, value in trajectory.items()}
        for index in range(control_step + 1)
    ]
    retained = frames[-(n_obs + 1) :]
    stacked = {
        key: torch.from_numpy(
            stack_last_n_obs([frame[key] for frame in retained], n_obs + 1)
        ).unsqueeze(0)
        for key in trajectory
    }
    runner = _StubRunner(n_obs)
    windowed = SeekerRobomimicImageRunner._append_proprio_deltas(runner, stacked)
    return {key: windowed[key][0] for key in FIELDS}


class _StubRunner:
    """Only what :meth:`SeekerRobomimicImageRunner._append_proprio_deltas` reads."""

    def __init__(self, n_obs_steps: int) -> None:
        self.n_obs_steps = n_obs_steps
        self.derived_proprio_fields = FIELDS


class _SingleFrameEnv:
    action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    observation_space = spaces.Dict(
        {
            "eef_pos": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            "rgb": spaces.Box(0, 255, shape=(3, 4, 4), dtype=np.uint8),
        }
    )

    def reset(self):
        return {
            "eef_pos": np.zeros(3, dtype=np.float32),
            "rgb": np.zeros((3, 4, 4), dtype=np.uint8),
        }


def test_retained_history_matches_the_declared_observation_space():
    wrapper = MultiStepWrapper(
        _SingleFrameEnv(),
        n_obs_steps=1,
        n_action_steps=1,
        history_keys=("eef_pos",),
    )

    observation = wrapper.reset()

    assert observation["eef_pos"].shape == (2, 3)
    assert wrapper.observation_space["eef_pos"].shape == (2, 3)
    assert observation["rgb"].shape == (1, 3, 4, 4)
    assert wrapper.observation_space["rgb"].shape == (1, 3, 4, 4)


@pytest.mark.parametrize("n_obs", [1, 2, 3])
def test_dataset_and_rollout_schedulings_agree_at_every_step(n_obs):
    length = 12
    trajectory = _trajectory(length)
    deltas = _dataset_deltas(trajectory, [length])
    for control_step in range(length):
        expected = _dataset_window(deltas, 0, control_step, n_obs)
        actual = _rollout_window(trajectory, control_step, n_obs)
        for field in FIELDS:
            torch.testing.assert_close(
                actual[field],
                expected[field],
                atol=1e-6,
                rtol=0,
                msg=f"{field} disagrees at step {control_step}, n_obs={n_obs}",
            )


def test_reset_and_episode_start_both_zero_the_delta():
    length = 6
    trajectory = _trajectory(length)
    deltas = _dataset_deltas(trajectory, [length])
    for field in FIELDS:
        assert torch.count_nonzero(deltas[field][0]) == 0
    for n_obs in (1, 2):
        for field, value in _rollout_window(trajectory, 0, n_obs).items():
            assert torch.count_nonzero(value) == 0, field


def test_episode_boundaries_never_difference_across_demonstrations():
    first, second = 5, 7
    trajectory = _trajectory(first + second, seed=3)
    deltas = _dataset_deltas(trajectory, [first, second])
    for field in FIELDS:
        assert torch.count_nonzero(deltas[field][0]) == 0
        assert torch.count_nonzero(deltas[field][first]) == 0
    # The second episode's interior still differences against its own frames.
    within = _dataset_deltas(
        {key: value[first:] for key, value in trajectory.items()}, [second]
    )
    for field in FIELDS:
        torch.testing.assert_close(deltas[field][first:], within[field])


def test_delta_is_invariant_to_the_world_frame():
    """Body-frame is why scene-yaw augmentation may leave a cached delta alone."""
    trajectory = _trajectory(8, seed=5)
    deltas = _dataset_deltas(trajectory, [8])
    yaw = Representation.rotvec_to_mat(torch.tensor([0.0, 0.0, 0.7]))
    rotated = {
        "eef_pos": trajectory["eef_pos"] @ yaw.T + torch.tensor([0.3, -0.2, 0.1]),
        "eef_rot6d": Representation.mat_to_rot6d(
            yaw @ Representation.rot6d_to_mat(trajectory["eef_rot6d"])
        ),
        "gripper_qpos": trajectory["gripper_qpos"],
    }
    for field, value in _dataset_deltas(rotated, [8]).items():
        torch.testing.assert_close(value, deltas[field], atol=1e-6, rtol=0)


def test_rollout_requires_the_extra_retained_step():
    runner = _StubRunner(n_obs_steps=2)
    trajectory = _trajectory(4)
    short = {key: value[:2].unsqueeze(0) for key, value in trajectory.items()}
    with pytest.raises(ValueError, match="must carry 3 steps"):
        SeekerRobomimicImageRunner._append_proprio_deltas(runner, short)


def test_shape_meta_binds_the_source_fields_a_rollout_must_retain():
    shape_meta_obs = {
        "robot0_eef_pos": {"shape": [3]},
        "robot0_eef_rot": {"shape": [9]},
        "robot0_gripper_qpos": {"shape": [2]},
        "eef_delta_pos": {"shape": [3]},
        "gripper_qpos_delta": {"shape": [2]},
    }
    assert derived_proprio_fields(shape_meta_obs) == (
        "eef_delta_pos",
        "gripper_qpos_delta",
    )
    assert delta_history_source_keys(shape_meta_obs) == (
        "robot0_eef_pos",
        "robot0_eef_rot",
        "robot0_gripper_qpos",
    )


def _runner_request(config_name: str, *, input_name: Optional[str] = None):
    """The request `build_runner` hands the environment, without building one."""
    from unittest.mock import patch

    from hydra import compose, initialize_config_module

    from visuomotor.config import build as Build
    from visuomotor.config.resolve import resolve_policy_run

    with initialize_config_module(config_module="visuomotor.config", version_base=None):
        overrides = ["task=stack_three_d1"]
        if input_name is not None:
            overrides.append(f"input={input_name}")
        cfg = compose(config_name=config_name, overrides=overrides)
    spec = resolve_policy_run(cfg).runner
    captured = {}

    class _Stop(Exception):
        pass

    def _capture(request, **_):
        captured["request"] = request
        raise _Stop

    with patch(
        "visuomotor.environment.runner.SeekerRobomimicImageRunner", _capture
    ), pytest.raises(_Stop):
        Build.build_runner(spec)
    return captured["request"]


def test_the_environment_retains_history_only_when_the_input_selects_a_delta():
    """The env is built from `default_shape_meta`, which always carries every
    source proprio field, so what the rollout retains has to be decided by the
    resolved input instead."""
    assert _runner_request("train_voxel_flow").delta_history_source_keys == ()
    assert _runner_request(
        "train_voxel_flow", input_name="voxel_wrist_proprio_delta"
    ).delta_history_source_keys == (
        "robot0_eef_pos",
        "robot0_eef_rot",
        "robot0_gripper_qpos",
    )
