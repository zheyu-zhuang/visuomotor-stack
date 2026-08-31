"""Control steps whose observation the caller discards must not render.

``MultiStepWrapper`` reads only the last ``n_obs_steps`` observations of an
action chunk. On the rest, camera rendering and RGB-D fusion are skipped in the
worker, and the wrapper hands back the last produced visual arrays.
"""

import numpy as np
import pytest
from gym import spaces

from visuomotor.data.core import images as CoreImages
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
    wrapper.rgb_jpeg_quality = CoreImages.JPEG_QUALITY_DEFAULT
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


def test_a_recording_lane_keeps_only_the_render_camera_alive():
    wrapper = _wrapper()
    wrapper.get_observation(wrapper.env.observation())

    wrapper.set_observation_needed(False, render_frame=True)

    assert wrapper.env.env.enabled["agentview_image"] is True
    assert wrapper.env.env.enabled["birdview_image"] is False
    # The render camera stays fresh so the encoded frame is not a repeat.
    frame = wrapper.get_observation(wrapper.env.observation())
    assert wrapper.render_cache is not None
    np.testing.assert_array_equal(
        frame["agentview_image"], _canonical_rgb_for_tick(2)
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

    def __init__(self):
        from robosuite.utils.observables import Observable, sensor

        self.camera_names = ["agentview", "birdview"]
        self.tick = 0

        def make(name):
            @sensor(modality="image")
            def read(obs_cache):
                return np.full((2, 2, 3), float(self.tick), dtype=np.float64)

            return Observable(
                name=name, sensor=read, sampling_rate=self.CONTROL_FREQ
            )

        self._observables = {
            f"{camera}_{suffix}": make(f"{camera}_{suffix}")
            for camera in self.camera_names
            for suffix in ("image", "depth")
        }

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
