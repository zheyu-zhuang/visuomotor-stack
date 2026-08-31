"""Convert robomimic actions between controller representations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from visuomotor.geometry import representation as Representation


def convert_actions(
    env,
    states: np.ndarray,
    actions: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Convert source actions into absolute controller action space.

    Args:
        env: Robomimic environment whose controller interprets ``actions``.
        states: Simulator states aligned one-to-one with ``actions``.
        actions: Controller actions with final dimension ``7 * num_robots``.

    Returns:
        ``(converted_actions, robot_eef_rots)`` where ``converted_actions`` has
        an ``"absolute"`` entry in the same shape as ``actions``, and
        ``robot_eef_rots`` is ``[T, num_robots, 9]``.
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"single-arm controller actions must have shape [T,7], got {actions.shape}"
        )
    if states.shape[0] != actions.shape[0]:
        raise ValueError("states and actions must have the same trajectory length")
    actions = actions.copy()

    stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7).astype(np.float64)
    num_frames, num_robots = stacked_actions.shape[:2]
    if num_robots != 1:
        raise NotImplementedError(
            f"Multi-robot action conversion is not supported: {stacked_actions.shape}"
        )

    abs_goal_pos = np.zeros((num_frames, num_robots, 3), dtype=np.float64)
    abs_goal_ori = np.zeros((num_frames, num_robots, 3), dtype=np.float64)
    action_gripper = stacked_actions[..., [-1]]

    robot_eef_rots = np.zeros((num_frames, num_robots, 9), dtype=np.float64)

    for i in range(num_frames):
        env.reset_to({"states": states[i]})
        for idx, robot in enumerate(env.env.robots):
            controller = robot.controller
            # robot.control's OSC solve is dead weight here -- the sim is never
            # stepped, so only set_goal's goals are read. update(force=True) and
            # the flag restore stand in for run_controller: without them set_goal's
            # own update() no-ops and silently reuses a stale eef pose.
            controller.update(force=True)
            controller.set_goal(stacked_actions[i, idx][: controller.control_dim])
            controller.new_update = True

            goal_pos, goal_ori = controller.goal_pos, controller.goal_ori
            ee_ori_mat = controller.ee_ori_mat

            abs_goal_pos[i, idx] = goal_pos
            abs_goal_ori[i, idx] = Representation.mat_to_rotvec(
                torch.as_tensor(np.asarray(goal_ori), dtype=torch.float64)
            ).numpy()

            robot_eef_rots[i, idx] = ee_ori_mat.flatten()

    abs_actions = np.concatenate(
        [abs_goal_pos, abs_goal_ori, action_gripper], axis=-1
    ).reshape(actions.shape)
    converted_actions = {
        "absolute": abs_actions.astype(np.float32),
    }
    return converted_actions, robot_eef_rots


def convert_dataset_file(dataset: str, output: str, *, overwrite: bool = False) -> Path:
    """Copy one dataset and add absolute-controller actions in place."""
    source = Path(dataset).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    import h5py
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(source))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    with h5py.File(destination, "r+") as output:
        for demo_name in sorted(output["data"]):
            demo = output["data"][demo_name]
            converted, rotations = convert_actions(
                env, demo["states"][()], demo["actions"][()]
            )
            for key, value in (
                ("actions_absolute", converted["absolute"]),
                ("robot_eef_rotations", rotations),
            ):
                if key in demo:
                    del demo[key]
                demo.create_dataset(key, data=value)
    print(f"[converted] {destination}")
    return destination
