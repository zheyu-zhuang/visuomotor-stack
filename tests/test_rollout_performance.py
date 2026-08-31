import gym
import numpy as np

from visuomotor.data.core import mirror as CoreMirror
from visuomotor.environment.gym_wrappers.async_vector_env import (
    AsyncState,
    AsyncVectorEnv,
)
from visuomotor.environment.gym_wrappers.multistep_wrapper import MultiStepWrapper
from visuomotor.environment.gym_wrappers.video_recording_wrapper import (
    VideoRecordingWrapper,
)
from visuomotor.environment.runner import SeekerRobomimicImageRunner


class _NullRecorder:
    """Stands in for the h264 recorder; frames are never actually encoded here."""

    def is_ready(self):
        return True

    def start(self, file_path):
        pass

    def write_frame(self, img, frame_time=None):
        pass

    def stop(self):
        pass


class _RewardSequenceEnv(gym.Env):
    def __init__(self, rewards):
        self.rewards = list(rewards)
        self.steps = 0
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self):
        self.steps = 0
        return np.zeros((1,), dtype=np.float32)

    def step(self, action):
        reward = self.rewards[self.steps]
        self.steps += 1
        return np.asarray([self.steps], dtype=np.float32), reward, False, {}


class _Pipe:
    def __init__(self, result):
        self.result = result
        self.sent = []

    def send(self, value):
        self.sent.append(value)

    def recv(self):
        return self.result, True


def test_multistep_strict_success_stops_remaining_actions():
    base = _RewardSequenceEnv([0.0, 1.0, 0.0, 0.0])
    env = MultiStepWrapper(
        base,
        n_obs_steps=2,
        n_action_steps=4,
        max_episode_steps=10,
        terminate_on_success=True,
    )
    env.reset()

    _, reward, done, _ = env.step(np.zeros((4, 1), dtype=np.float32))

    assert base.steps == 2
    assert reward == 1.0
    assert bool(done)


def test_borrowed_observation_preprocessing_does_not_mutate_shared_arrays():
    runner = object.__new__(SeekerRobomimicImageRunner)
    runner.pos_key = "eef_pos"
    runner.rot_key = "eef_rot"
    runner.mirror_augmentation = CoreMirror.MirrorAugmentationConfig(enable=True)
    obs = {
        "rgb": np.zeros((2, 3, 4, 4), dtype=np.uint8),
        "eef_pos": np.zeros((2, 2, 3), dtype=np.float32),
        "eef_rot": np.broadcast_to(
            np.eye(3, dtype=np.float32).reshape(1, 1, 9), (2, 2, 9)
        ).copy(),
    }
    original_pos = obs["eef_pos"].copy()
    original_rot = obs["eef_rot"].copy()

    prepared = runner.preprocess_obs(obs)

    np.testing.assert_array_equal(obs["eef_pos"], original_pos)
    np.testing.assert_array_equal(obs["eef_rot"], original_rot)
    assert prepared["rgb"] is obs["rgb"]
    assert prepared["eef_pos"] is not obs["eef_pos"]
    assert prepared["eef_rot"] is not obs["eef_rot"]


def test_selected_worker_calls_do_not_contact_unrecorded_environments():
    env = object.__new__(AsyncVectorEnv)
    env.closed = False
    env.num_envs = 3
    env._state = AsyncState.DEFAULT
    env.parent_pipes = [_Pipe(0), _Pipe(1), _Pipe(2)]

    results = env.call_each_at("set_diagnostics", [0, 2], [("a",), ("c",)])

    assert results == [0, 2]
    assert len(env.parent_pipes[0].sent) == 1
    assert env.parent_pipes[1].sent == []
    assert len(env.parent_pipes[2].sent) == 1
    env.viewer = None
    env.closed = True


class _ObservationGateEnv(gym.Env):
    """Records the observation-need flag declared before each inner step."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.declared = []
        self.needed = True
        self.render_frames = []
        self.steps = 0
        self.file_path = None

    def set_observation_needed(self, needed, render_frame=False):
        self.needed = bool(needed)
        self.render_frames.append(bool(render_frame))

    def reset(self):
        self.steps = 0
        return np.zeros((1,), dtype=np.float32)

    def step(self, action):
        self.declared.append(self.needed)
        self.steps += 1
        return np.asarray([self.steps], dtype=np.float32), 0.0, False, {}

    def render(self, mode="rgb_array", **kwargs):
        return np.zeros((4, 4, 3), dtype=np.uint8)


def test_only_the_read_tail_of_an_action_chunk_needs_its_observation():
    # _get_obs only reads the last n_obs_steps of the chunk.
    for n_obs_steps in (1, 3):
        base = _ObservationGateEnv()
        env = MultiStepWrapper(
            base, n_obs_steps=n_obs_steps, n_action_steps=8, max_episode_steps=100
        )
        env.reset()

        env.step(np.zeros((8, 1), dtype=np.float32))

        assert base.declared == [False] * (8 - n_obs_steps) + [True] * n_obs_steps


def test_only_recording_lanes_keep_the_render_camera_on_frame_due_steps():
    for file_path, expected in (
        ("/dev/null", [True, False] * 4),
        (None, [False] * 8),
    ):
        base = _ObservationGateEnv()
        recorder = VideoRecordingWrapper(
            base,
            video_recoder=_NullRecorder(),
            file_path=file_path,
            steps_per_render=2,
        )
        env = MultiStepWrapper(
            recorder, n_obs_steps=1, n_action_steps=8, max_episode_steps=100
        )
        env.reset()

        env.step(np.zeros((8, 1), dtype=np.float32))

        assert base.render_frames == expected, file_path
