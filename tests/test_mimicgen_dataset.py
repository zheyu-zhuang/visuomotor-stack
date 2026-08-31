import numpy as np
import pytest

from visuomotor.data.mimicgen import action as MimicgenAction
from visuomotor.data.mimicgen import dataset as MimicgenDataset


def _bare_dataset(episode_lengths=(20, 12, 10), val_ratio=1 / 3):
    dataset = MimicgenDataset.MimicGenDataset.__new__(
        MimicgenDataset.MimicGenDataset
    )
    episode_lengths = list(episode_lengths)
    n_episodes = len(episode_lengths)
    dataset.n_demo_all = n_episodes
    dataset.demo_count_mode = "total"
    dataset.episode_lengths_all = episode_lengths
    dataset.cum_lengths_all = np.cumsum([0] + episode_lengths).astype(np.int64)
    dataset.task_embedding_episode_all = np.zeros((n_episodes, 1), dtype=np.float32)
    dataset.task_language_tokens_episode_all = None
    dataset.task_id_episode_all = np.arange(n_episodes, dtype=np.int64)
    dataset.robot_id_episode_all = np.zeros(n_episodes, dtype=np.int64)
    dataset.task_instructions_all = None
    dataset.val_ratio = val_ratio
    dataset.n_obs_steps = 1
    dataset.horizon = 16
    dataset.set_active_demos()
    dataset.mode_set = False
    dataset.start_idx = 0
    dataset.end_idx = 0
    return dataset


def test_every_episode_exposes_t_minus_eight_samples():
    dataset = _bare_dataset()

    np.testing.assert_array_equal(
        dataset.episode_sample_lengths_active,
        np.array([12, 4, 2]),
    )
    np.testing.assert_array_equal(
        dataset.cum_sample_lengths_active,
        np.array([0, 12, 16, 18]),
    )
    assert dataset.n_frames_active == 42
    assert dataset.n_samples_active == 18
    assert dataset.train_length == 16
    assert dataset.eval_length == 2

    episode_idx, obs_indices, action_indices = dataset.sampler(15)
    assert episode_idx == 1
    assert obs_indices == [23]
    assert action_indices[:9] == list(range(23, 32))
    assert action_indices[9:] == [31] * 7

    with pytest.raises(IndexError, match=r"\[0, 18\)"):
        dataset.sampler(18)


def test_zero_sample_episodes_do_not_break_episode_mapping():
    dataset = _bare_dataset((8, 9, 7), val_ratio=0.0)

    np.testing.assert_array_equal(
        dataset.episode_sample_lengths_active,
        np.array([0, 1, 0]),
    )
    episode_idx, obs_indices, action_indices = dataset.sampler(0)
    assert episode_idx == 1
    assert obs_indices == [8]
    assert action_indices[:9] == list(range(8, 17))
    assert action_indices[9:] == [16] * 7


def test_sample_indices_translate_back_to_physical_frame_indices():
    dataset = _bare_dataset()
    dataset.set_mode("all")
    dataset.get_obs = lambda episode_idx, obs_indices: {}
    dataset._get_action_window = lambda obs_idx, action_indices: np.asarray(
        [obs_idx], dtype=np.float32
    )
    dataset.get_task_context = lambda episode_idx: {}
    dataset.target_adapter = None
    dataset.oracle_info = None

    sample = dataset[12]

    np.testing.assert_array_equal(sample["obs_index"], np.array([20]))
    np.testing.assert_array_equal(sample["action"], np.array([20], dtype=np.float32))


def test_keypose_action_window_repeats_anchor_endpoint_after_boundary():
    dataset = _bare_dataset((20,), val_ratio=0.0)
    dataset.action = np.arange(20, dtype=np.float32).reshape(-1, 1)
    dataset.action_adapter = MimicgenAction.MimicGenActionAdapter(
        cache_dir="unused",
        meta={},
        horizon=16,
        action_dim=1,
        action_rep="absolute",
    )
    last_indices = np.full(20, 19, dtype=np.int64)
    last_indices[:11] = 10
    dataset.target_adapter = type(
        "TargetAdapter", (), {"last_indices": last_indices}
    )()

    _, obs_indices, action_indices = dataset.sampler(4)
    window = dataset._get_action_window(obs_indices[-1], action_indices)

    expected_indices = np.minimum(np.arange(4, 20), 10)
    np.testing.assert_array_equal(window[:, 0], expected_indices.astype(np.float32))

    last_indices[:] = 19
    window = dataset._get_action_window(obs_indices[-1], action_indices)
    np.testing.assert_array_equal(window[:, 0], np.arange(4, 20, dtype=np.float32))


def test_non_keypose_action_window_retains_cross_boundary_trajectory():
    dataset = _bare_dataset((20,), val_ratio=0.0)
    dataset.action = np.arange(20, dtype=np.float32).reshape(-1, 1)
    dataset.action_adapter = MimicgenAction.MimicGenActionAdapter(
        cache_dir="unused",
        meta={},
        horizon=16,
        action_dim=1,
        action_rep="absolute",
    )
    dataset.target_adapter = None

    _, obs_indices, action_indices = dataset.sampler(4)
    window = dataset._get_action_window(obs_indices[-1], action_indices)

    np.testing.assert_array_equal(window[:, 0], np.arange(4, 20, dtype=np.float32))


def test_task_sample_ranges_use_trimmed_offsets():
    dataset = _bare_dataset()
    dataset.task_id_episode = np.array([3, 3, 7], dtype=np.int64)

    dataset.set_mode("all")
    assert dataset.task_sample_ranges() == {
        3: [(0, 12), (12, 16)],
        7: [(16, 18)],
    }

    dataset.set_mode("eval")
    assert dataset.task_sample_ranges() == {7: [(0, 2)]}
