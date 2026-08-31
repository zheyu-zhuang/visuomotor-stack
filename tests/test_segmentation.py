import numpy as np

from visuomotor.data.core.segmentation import (
    build_keypose_segments,
    build_keypose_segments_from_keyframes,
    segment_settled_gripper_commands,
    segment_velocity_scipy,
)


def test_segment_velocity_scipy_finds_gripper_change_and_boundaries():
    # A single 10-step episode: gripper opens at step 3 and closes at step 7.
    gripper = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1], dtype=np.float32)
    keyframes = segment_velocity_scipy(gripper, episode_ends=[10], use_gripper=True)
    assert len(keyframes) == 1
    assert keyframes[0].tolist() == [0, 3, 7, 9]


def test_segment_velocity_scipy_handles_multiple_episodes_independently():
    gripper = np.array([1, 1, -1, -1, 1, 1, 1, -1], dtype=np.float32)
    keyframes = segment_velocity_scipy(gripper, episode_ends=[4, 8], use_gripper=True)
    assert len(keyframes) == 2
    assert keyframes[0].tolist() == [0, 2, 3]
    assert keyframes[1].tolist() == [
        0,
        3,
    ]  # local indices within the second episode (change lands on the last frame)


def test_build_keypose_segments_progress_and_targets():
    gripper = np.array([1, 1, 1, -1, -1, -1], dtype=np.float32)
    first_index, last_index, progress, valid = build_keypose_segments(
        gripper, episode_ends=[6], segment_kwargs={"use_gripper": True}
    )
    assert valid.all()
    # Keyframes are {0, 3, 5}; steps 0-3 target segment (0,3), steps 3-5 target (3,5).
    np.testing.assert_array_equal(first_index, [0, 0, 0, 3, 3, 3])
    np.testing.assert_array_equal(last_index, [3, 3, 3, 5, 5, 5])
    np.testing.assert_allclose(progress, [0.0, 1 / 3, 2 / 3, 0.0, 0.5, 1.0], atol=1e-6)


def test_build_keypose_segments_marks_degenerate_single_frame_episodes_invalid():
    gripper = np.array([1, 1, 1, 0], dtype=np.float32)
    first_index, last_index, progress, valid = build_keypose_segments(
        gripper,
        episode_ends=[3, 4],
        segment_kwargs={"use_gripper": True},
    )
    assert valid.tolist() == [True, True, True, False]
    assert first_index[3] == 3
    assert last_index[3] == 3
    assert progress[3] == 0.0


def test_build_keypose_segments_progress_is_always_in_unit_interval():
    rng = np.random.default_rng(0)
    gripper = rng.choice([-1.0, 1.0], size=40)
    ends = [15, 40]
    _, _, progress, valid = build_keypose_segments(
        gripper, episode_ends=ends, segment_kwargs={"use_gripper": True}
    )
    assert (progress[valid] >= 0.0).all()
    assert (progress[valid] <= 1.0).all()


def test_settled_gripper_commands_keep_open_and_close_keyposes():
    command = np.array(
        [
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ],
        dtype=np.float32,
    )
    opening = np.array(
        [
            0.08,
            0.08,
            0.08,
            0.08,
            0.07,
            0.055,
            0.04,
            0.0399,
            0.03985,
            0.0398,
            0.03978,
            0.03976,
            0.045,
            0.06,
            0.075,
            0.0790,
            0.07905,
            0.0791,
            0.0791,
            0.0791,
        ],
        dtype=np.float32,
    )
    qpos = np.stack([opening / 2, -opening / 2], axis=-1)

    keyframes, command_keyframes = segment_settled_gripper_commands(
        command, qpos, episode_ends=[len(command)]
    )

    assert command_keyframes[0].tolist() == [7, 16]
    assert keyframes[0].tolist() == [0, 7, 16, 19]
    assert len(command_keyframes[0]) == np.count_nonzero(np.diff(command))


def test_settled_gripper_command_falls_back_without_dropping_event():
    command = np.array([-1, -1, 1, 1, -1, -1], dtype=np.float32)
    qpos = np.tile(np.array([[0.04, -0.04]], dtype=np.float32), (6, 1))

    keyframes, command_keyframes = segment_settled_gripper_commands(
        command, qpos, episode_ends=[6]
    )

    assert command_keyframes[0].tolist() == [3, 5]
    assert keyframes[0].tolist() == [0, 3, 5]


def test_preselected_keyposes_advance_at_the_keyframe():
    keyframes = [np.array([0, 4, 8], dtype=np.int64)]
    segments = build_keypose_segments_from_keyframes(keyframes, episode_ends=[9])

    np.testing.assert_array_equal(keyframes[0], [0, 4, 8])
    assert segments[1][3] == 4
    assert segments[1][4] == 8
