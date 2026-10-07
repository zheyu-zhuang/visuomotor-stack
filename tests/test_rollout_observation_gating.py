"""Camera gate behavior and rollout sampling-time parity."""

from types import SimpleNamespace

import gym
import numpy as np
import pytest
from gym import spaces

from visuomotor.data.core import images as CoreImages
from visuomotor.environment.gym_wrappers import multistep_wrapper as MultiStep
from visuomotor.environment.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)

SHAPE_META = {
    "obs": {
        "agentview_image": {"shape": [3, 4, 4], "type": "rgb"},
        "voxel": {"shape": [4, 2, 2, 2], "type": "voxel"},
        "robot0_eef_pos": {"shape": [3]},
    }
}


class _FakeRobosuiteEnv:
    def __init__(self):
        self.camera_names = ["agentview", "birdview"]
        self._observables = {
            "agentview_image": object(),
            "agentview_depth": object(),
            "birdview_image": object(),
            "birdview_depth": object(),
        }
        self.enabled = {name: True for name in self._observables}
        self.modify_calls = 0
        self.render_enabled = True

    def set_camera_render_enabled(self, enabled):
        self.render_enabled = bool(enabled)

    def modify_observable(self, observable_name, attribute, modifier):
        assert attribute == "enabled"
        self.enabled[observable_name] = bool(modifier)
        self.modify_calls += 1


class _FakeEnvRobosuite:
    """Mirrors the patched EnvRobosuite surface the image wrapper depends on."""

    def __init__(self):
        self.env = _FakeRobosuiteEnv()
        self._visual_obs_enabled = True
        self.tick = 0

    # Copied contract from the patched robomimic env.
    def set_visual_obs_enabled(self, enabled, keep_cameras=()):
        enabled = bool(enabled)
        keep = set(keep_cameras)
        for camera in self.env.camera_names:
            active = enabled or camera in keep
            for suffix in ("image", "depth"):
                name = "{}_{}".format(camera, suffix)
                if name not in self.env._observables:
                    continue
                if self.env.enabled[name] is active:
                    continue
                self.env.modify_observable(name, "enabled", active)
        self._visual_obs_enabled = enabled

    def observation(self):
        """Only the observables still enabled appear, as robosuite does."""
        self.tick += 1
        obs = {"robot0_eef_pos": np.full((3,), float(self.tick), dtype=np.float32)}
        if self.env.enabled["agentview_image"]:
            obs["agentview_image"] = np.full(
                (3, 4, 4), self.tick / 255.0, dtype=np.float32
            )
        if self._visual_obs_enabled:
            obs["voxel"] = np.full((4, 2, 2, 2), self.tick, dtype=np.uint8)
        return obs


def _canonical_rgb_for_tick(tick: int) -> np.ndarray:
    """What the shared cache codec makes of one fake frame."""
    frame = np.full((3, 4, 4), tick / 255.0, dtype=np.float32)
    source = np.ascontiguousarray(
        np.moveaxis(np.rint(frame * 255.0).astype(np.uint8), 0, -1)
    )
    return CoreImages.canonical_rgb_from_source(source, load_resolution=None)


def _wrapper():
    wrapper = object.__new__(RobomimicImageWrapper)
    wrapper.env = _FakeEnvRobosuite()
    wrapper.wrist_projection = None
    wrapper.shape_meta = SHAPE_META
    wrapper.render_obs_key = "agentview_image"
    wrapper.render_camera = "agentview"
    wrapper.render_cache = None
    wrapper._validated_rgb_keys = set()
    wrapper._validated_spatial_keys = set()
    wrapper._observation_needed = True
    wrapper._render_frame_needed = False
    wrapper._last_visual_obs = {}
    wrapper.skipped_observations = 0
    wrapper.produced_observations = 0
    wrapper.rgb_load_resolutions = {}
    observation_space = spaces.Dict()
    for key, field in SHAPE_META["obs"].items():
        kind = field.get("type")
        if kind in ("rgb", "voxel"):
            observation_space[key] = spaces.Box(
                low=0, high=255, shape=field["shape"], dtype=np.uint8
            )
        else:
            observation_space[key] = spaces.Box(
                low=-1, high=1, shape=field["shape"], dtype=np.float32
            )
    wrapper.observation_space = observation_space
    return wrapper


def test_skipped_steps_reuse_the_last_visuals_but_keep_proprio_fresh():
    wrapper = _wrapper()
    first = wrapper.get_observation(wrapper.env.observation())
    baseline_voxel = first["voxel"].copy()
    baseline_rgb = first["agentview_image"].copy()

    wrapper.set_observation_needed(False)
    skipped = wrapper.get_observation(wrapper.env.observation())

    np.testing.assert_array_equal(skipped["voxel"], baseline_voxel)
    np.testing.assert_array_equal(skipped["agentview_image"], baseline_rgb)
    # Proprio never stops: the executed-trajectory overlay reads it every step.
    assert skipped["robot0_eef_pos"][0] == pytest.approx(2.0)
    assert wrapper.skipped_observations == 1
    assert wrapper.produced_observations == 1


def test_a_needed_step_produces_fresh_visuals_again():
    wrapper = _wrapper()
    wrapper.get_observation(wrapper.env.observation())
    wrapper.set_observation_needed(False)
    wrapper.get_observation(wrapper.env.observation())

    wrapper.set_observation_needed(True)
    resumed = wrapper.get_observation(wrapper.env.observation())

    assert int(resumed["voxel"].flat[0]) == 3
    assert wrapper.produced_observations == 2


def test_discarded_steps_skip_fusion_and_rgb_preprocessing_without_changing_retained_obs(monkeypatch):
    wrapper = _wrapper()
    reference = _wrapper()
    wrapper.get_observation(wrapper.env.observation())
    reference.get_observation(reference.env.observation())
    encoded = []
    encode = wrapper._canonical_rgb

    def record_encode(key, value):
        encoded.append(key)
        return encode(key, value)

    monkeypatch.setattr(wrapper, "_canonical_rgb", record_encode)
    for index in range(8):
        needed = index == 7
        wrapper.set_observation_needed(needed)
        raw = wrapper.env.observation()
        assert ("voxel" in raw) == needed
        observed = wrapper.get_observation(raw)
        expected = reference.get_observation(reference.env.observation())

    for key in expected:
        np.testing.assert_array_equal(observed[key], expected[key])
    assert encoded == ["agentview_image"]
    assert wrapper.skipped_observations == 7
    assert wrapper.env.env.modify_calls == 0


@pytest.mark.parametrize("render_frame", [False, True])
def test_skipping_processing_keeps_all_cameras_enabled(render_frame):
    wrapper = _wrapper()
    wrapper.get_observation(wrapper.env.observation())

    wrapper.set_observation_needed(False, render_frame=render_frame)

    assert all(wrapper.env.env.enabled.values())
    assert wrapper.env.env.modify_calls == 0
    assert wrapper.env._visual_obs_enabled is False
    assert wrapper.env.env.render_enabled is bool(render_frame)
    skipped = wrapper.get_observation(wrapper.env.observation())
    np.testing.assert_array_equal(
        skipped["agentview_image"], _canonical_rgb_for_tick(1)
    )


def test_toggling_the_same_state_twice_does_not_touch_the_observables():
    wrapper = _wrapper()
    wrapper.set_observation_needed(False)
    calls = wrapper.env.env.modify_calls

    wrapper.set_observation_needed(False)

    # set_enabled() reallocates each observable's frame buffer, so repeats cost.
    assert wrapper.env.env.modify_calls == calls




class _RealObservableEnv:
    """Holds robosuite's own Observable objects, so the gate meets real semantics.

    ``Observable.set_enabled`` calls ``reset()``, which zeroes the cached value
    but leaves ``_sampled`` untouched -- only ``__init__`` clears it. Re-enabling
    a camera whose ``_sampled`` is still set makes its next ``update()`` skip the
    sensor and keep serving that zero, which reached rollout video as one black
    frame and reached the policy as a short-changed voxel grid.
    """

    CONTROL_FREQ = 20
    # One physics substep: enough to sample, short of the sampling period that
    # would clear _sampled again. This is the state a control step leaves behind
    # on a live env, measured on Square_D0.
    SUBSTEP = 0.002

    def __init__(self, depth=True):
        from robosuite.environments import robot_env as RobotEnv
        from robosuite.utils.observables import Observable

        self.camera_names = ["agentview", "birdview"]
        self.tick = 0
        self.render_calls = 0
        self.robot_env = object.__new__(RobotEnv.RobotEnv)
        self.robot_env.sim = SimpleNamespace(render=self.render)
        self.robot_env.set_camera_render_enabled(True)
        self._observables = {}
        for camera in self.camera_names:
            sensors, names = self.robot_env._create_camera_sensors(
                camera, 2, 2, depth, None
            )
            self._observables.update({
                name: Observable(name=name, sensor=sensor, sampling_rate=self.CONTROL_FREQ)
                for name, sensor in zip(names, sensors)
            })

    def render(self, **kwargs):
        self.render_calls += 1
        rgb = np.full((2, 2, 3), float(self.tick), dtype=np.float64)
        if kwargs.get("depth", False):
            return rgb, np.full((2, 2), float(self.tick), dtype=np.float64)
        return rgb

    def set_camera_render_enabled(self, enabled):
        self.robot_env.set_camera_render_enabled(enabled)

    def modify_observable(self, observable_name, attribute, modifier):
        assert attribute == "enabled"
        self._observables[observable_name].set_enabled(modifier)

    def sample_once(self):
        """Leave every observable in the sampled state a control step ends in."""
        self.tick += 1
        cache = {}
        for observable in self._observables.values():
            observable.update(timestep=self.SUBSTEP, obs_cache=cache)


def _gate(env, enabled, keep=()):
    """Invoke the patched EnvRobosuite.set_visual_obs_enabled under test."""
    from robomimic.envs.env_robosuite import EnvRobosuite

    holder = object.__new__(EnvRobosuite)
    holder.env = env
    holder._visual_obs_enabled = True
    EnvRobosuite.set_visual_obs_enabled(holder, enabled, keep_cameras=keep)


@pytest.mark.parametrize("depth", [False, True])
def test_render_skip_preserves_cached_pixels_and_does_not_block_video(depth):
    env = _RealObservableEnv(depth=depth)
    env.sample_once()
    initial = {key: obs.obs.copy() for key, obs in env._observables.items()}
    count = env.render_calls
    clocks = {
        key: (obs._time_since_last_sample, obs._sampled)
        for key, obs in env._observables.items()
    }

    env.set_camera_render_enabled(False)

    assert clocks == {
        key: (obs._time_since_last_sample, obs._sampled)
        for key, obs in env._observables.items()
    }
    for _ in range(25):
        env.sample_once()
    assert env.render_calls == count
    for key, obs in env._observables.items():
        assert obs.is_enabled()
        np.testing.assert_array_equal(obs.obs, initial[key])

    frame = env.robot_env.sim.render(camera_name="agentview", depth=False)
    assert frame.flat[0] == env.tick
    assert env.render_calls == count + 1

    env.set_camera_render_enabled(True)
    for _ in range(25):
        env.sample_once()
    assert env.render_calls == count + 3
    for key, obs in env._observables.items():
        assert obs.obs.flat[0] > initial[key].flat[0]


def test_disabling_a_camera_zeroes_its_cached_value_but_keeps_it_sampled():
    """The robosuite behaviour the gate has to compensate for."""
    env = _RealObservableEnv()
    env.sample_once()
    observable = env._observables["agentview_image"]
    assert float(np.asarray(observable._current_observed_value).mean()) == 1.0

    observable.set_enabled(False)

    assert float(np.asarray(observable._current_observed_value).mean()) == 0.0
    assert observable._sampled is True


def test_re_enabling_a_camera_clears_its_sample_flag_so_it_must_resample():
    env = _RealObservableEnv()
    env.sample_once()
    _gate(env, False)
    assert env._observables["agentview_image"]._sampled is True

    _gate(env, True)

    # Left set, the re-enabled camera skips its sensor and serves the zero.
    for suffix in ("image", "depth"):
        assert env._observables[f"agentview_{suffix}"]._sampled is False
        assert env._observables[f"birdview_{suffix}"]._sampled is False


def test_a_render_camera_held_alive_is_never_disabled_and_so_never_resets():
    env = _RealObservableEnv()
    env.sample_once()

    _gate(env, False, keep=("agentview",))

    agentview = env._observables["agentview_image"]
    assert agentview.is_enabled() is True
    # Untouched, so its value never went through reset()'s zeroing.
    assert float(np.asarray(agentview._current_observed_value).mean()) == 1.0
    assert env._observables["birdview_image"].is_enabled() is False


class _CameraClockEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            key: spaces.Box(
                0, np.inf, shape=(2, 2, 1 if key.endswith("depth") else 3), dtype=np.float64
            )
            for key in ("agentview_image", "agentview_depth", "proprio")
        })

    def reset(self):
        self.sim = _RealObservableEnv()
        for observable in self.sim._observables.values():
            observable.update(self.sim.SUBSTEP, {}, force=True)
        return self._observation()

    def set_observation_needed(self, needed):
        from robomimic.envs.env_robosuite import EnvRobosuite

        holder = object.__new__(EnvRobosuite)
        holder.env = self.sim
        wrapper = object.__new__(RobomimicImageWrapper)
        wrapper.env = holder
        wrapper.set_observation_needed(needed)

    def step(self, action):
        for _ in range(25):
            self.sim.sample_once()
        return self._observation(), 0.0, False, {}

    def _observation(self):
        return {
            "agentview_image": self.sim._observables["agentview_image"].obs.copy(),
            "agentview_depth": self.sim._observables["agentview_depth"].obs.copy(),
            "proprio": np.full((2, 2, 3), float(self.sim.tick)),
        }


@pytest.mark.parametrize("n_obs_steps", [1, 2])
@pytest.mark.parametrize("n_action_steps", [1, 8])
@pytest.mark.parametrize("history_keys", [(), ("proprio",)])
def test_rollout_camera_timestamps_match_continuous_sampling(
    n_obs_steps, n_action_steps, history_keys
):
    reference = _CameraClockEnv()
    rollout = MultiStep.MultiStepWrapper(
        _CameraClockEnv(),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=19,
        history_keys=history_keys,
    )
    history = [reference.reset()] * (n_obs_steps + bool(history_keys))
    rollout.reset()
    initial_renders = rollout.env.sim.render_calls
    expected_render_steps = 0
    action = np.zeros((n_action_steps, 1), dtype=np.float32)
    done = False
    steps = 0
    while not done:
        chunk_steps = min(n_action_steps, 19 - steps)
        expected_render_steps += min(chunk_steps, n_obs_steps + bool(history_keys))
        observed, _, done, _ = rollout.step(action)
        for _ in range(min(n_action_steps, 19 - steps)):
            history.append(reference.step(action[0])[0])
            steps += 1
        for key in observed:
            count = n_obs_steps + (key in history_keys)
            expected = np.stack([obs[key] for obs in history[-count:]])
            np.testing.assert_array_equal(observed[key], expected, err_msg=key)
    assert rollout.env.sim.render_calls - initial_renders == 2 * expected_render_steps
