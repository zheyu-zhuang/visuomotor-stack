"""MimicGen dataset backed by LMDB image cache + NumPy low-dimensional arrays."""

import os

import numpy as np
import torch

from visuomotor.data.core import actions as CoreActions
from visuomotor.data.core import mirror as CoreMirror
from visuomotor.data.core import scene_augmentation as CoreSceneAugmentation
from visuomotor.data.mimicgen import action as MimicgenAction
from visuomotor.data.mimicgen import cache as MimicgenCache
from visuomotor.data.mimicgen import normalization as MimicgenNormalization
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.data.mimicgen import targets as MimicgenTargets
from visuomotor.data.mimicgen.oracle.oracle_cache import load_oracle_info

TERMINAL_SAMPLE_TRIM = 8


class MimicGenDataset(torch.utils.data.Dataset):
    """Torch dataset for Seeker training/evaluation on cached MimicGen trajectories."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        image_size=None,
        rgb_load_resolutions=None,
        horizon=16,
        val_ratio=0.02,
        n_demo=None,
        demo_count_mode="total",
        n_obs_steps=1,
        action_rep="absolute",
        lmdb_readahead=False,
        cache_dir=None,
        include_oracle_info=False,
        include_camera_matrices=False,
        mirror_augmentation=None,
        scene_yaw_augmentation=None,
        keypose_targets=False,
        keypose_gripper_motion_threshold=5e-4,
        keypose_gripper_valley_threshold=2e-4,
        keypose_gripper_valley_window=4,
        voxel_spec=None,
        voxel_specs=None,
        point_cloud_spec=None,
    ):
        if keypose_targets and str(action_rep) != "absolute":
            raise ValueError("keypose targets require action_rep='absolute'")
        self.n_obs_steps = int(n_obs_steps)
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1")

        self.image_size = image_size
        self.rgb_load_resolutions = dict(rgb_load_resolutions or {})
        self.horizon = int(horizon)
        self.val_ratio = float(val_ratio)
        self.action_rep = CoreActions.validate_action_rep(action_rep)
        self.mirror_augmentation = CoreMirror.MirrorAugmentationConfig.from_config(
            mirror_augmentation
        )
        self.scene_yaw_augmentation = (
            CoreSceneAugmentation.SceneYawAugmentationConfig.from_config(
                scene_yaw_augmentation
            )
        )
        self.lmdb_readahead = bool(lmdb_readahead)
        self.include_oracle_info = bool(include_oracle_info)
        self.include_camera_matrices = bool(include_camera_matrices)
        self.demo_count_mode = str(demo_count_mode).strip().lower()
        if self.demo_count_mode not in ("total", "per_task"):
            raise ValueError(
                "demo_count_mode must be one of ['total', 'per_task'], "
                f"got {demo_count_mode!r}"
            )
        self.action_dim = shape_meta["action"]["shape"][0]

        self.cache_dir = str(MimicgenCache.resolve_cache_dir(dataset_path, cache_dir))
        MimicgenCache.check_cache(cache_dir=self.cache_dir, dataset_path=dataset_path)
        self.observation_adapter = MimicgenObservations.MimicGenObservationAdapter(
            shape_meta=shape_meta,
            cache_dir=self.cache_dir,
            image_size=self.image_size,
            rgb_load_resolutions=self.rgb_load_resolutions,
            lmdb_readahead=self.lmdb_readahead,
            voxel_spec=voxel_spec,
            voxel_specs=voxel_specs,
            point_cloud_spec=point_cloud_spec,
        )
        self.rgb_keys = self.observation_adapter.rgb_keys
        self.lowdim_keys = self.observation_adapter.lowdim_keys
        self.derived_keys = self.observation_adapter.derived_keys
        self.derived_lowdim = self.observation_adapter.derived_lowdim
        self.voxel_keys = self.observation_adapter.voxel_keys
        self.point_cloud_keys = self.observation_adapter.point_cloud_keys
        self.meta = self.observation_adapter.meta
        self.episode_lengths_all = self.observation_adapter.episode_lengths
        self.lowdim = self.observation_adapter.lowdim
        self.lmdb_path = self.observation_adapter.lmdb_path
        source_keys = self.observation_adapter.source_keys
        self.pos_key = source_keys["eef_pos"]
        self.rot_key = source_keys["eef_rot"]
        self.gripper_key = source_keys["gripper_qpos"]
        self._load_cache_arrays()
        self.target_adapter = (
            None
            if not keypose_targets
            else MimicgenTargets.MimicGenKeyposeTargetAdapter.from_dataset(
                self,
                gripper_motion_threshold=keypose_gripper_motion_threshold,
                gripper_valley_threshold=keypose_gripper_valley_threshold,
                gripper_valley_window=keypose_gripper_valley_window,
            )
        )
        raw_oracle_info = (
            load_oracle_info(self.cache_dir)
            if self.include_oracle_info or self.include_camera_matrices
            else None
        )
        cached_oracle_info = (
            MimicgenObservations.canonicalize_oracle_info(raw_oracle_info)
            if raw_oracle_info is not None
            else None
        )
        self.oracle_info = cached_oracle_info if self.include_oracle_info else None
        self.camera_matrices = (
            {
                key: value
                for key, value in cached_oracle_info.items()
                if key.startswith("camera_matrix_")
            }
            if cached_oracle_info is not None and self.include_camera_matrices
            else None
        )

        self.set_active_demos(n_demo)
        self.mode_set = False
        self.start_idx = 0
        self.end_idx = 0

    def set_active_demos(self, n_demo=None):
        """Select active episodes and recompute split/sample bookkeeping."""
        if n_demo is None:
            selected_episode_indices = np.arange(self.n_demo_all, dtype=np.int64)
        elif self.demo_count_mode == "per_task":
            n = int(n_demo)
            if n < 1:
                raise ValueError(f"n_demo must be >= 1, got {n}")
            task_groups = self._get_task_episode_groups()
            selected_episode_indices = []
            for group in task_groups:
                if len(group) < n:
                    raise ValueError(
                        "n_demo exceeds demos available for one task group: "
                        f"requested {n}, got {len(group)}"
                    )
                selected_episode_indices.extend(group[:n].tolist())
            selected_episode_indices = np.asarray(
                selected_episode_indices, dtype=np.int64
            )
        else:
            n = int(n_demo)
            if n < 1 or n > self.n_demo_all:
                raise ValueError(f"n_demo must be in [1, {self.n_demo_all}], got {n}")
            selected_episode_indices = np.arange(n, dtype=np.int64)

        self.episode_indices_active = np.asarray(
            selected_episode_indices, dtype=np.int64
        )
        self.n_demo_active = int(len(self.episode_indices_active))
        if self.n_demo_active < 1:
            raise ValueError("Active demo selection is empty")

        lens_all = np.asarray(self.episode_lengths_all, dtype=np.int64)
        self.episode_lengths_active = (
            lens_all[self.episode_indices_active].astype(int).tolist()
        )
        self.cum_lengths_active = np.cumsum([0] + self.episode_lengths_active).astype(
            np.int64
        )
        step_ranges = [
            np.arange(
                self.cum_lengths_all[ep_idx],
                self.cum_lengths_all[ep_idx + 1],
                dtype=np.int64,
            )
            for ep_idx in self.episode_indices_active.tolist()
        ]
        self.active_step_indices = np.concatenate(step_ranges, axis=0)
        self.n_frames_active = int(self.active_step_indices.shape[0])
        self.episode_sample_lengths_active = np.maximum(
            np.asarray(self.episode_lengths_active, dtype=np.int64)
            - TERMINAL_SAMPLE_TRIM,
            0,
        )
        self.cum_sample_lengths_active = np.cumsum(
            np.concatenate(
                (np.zeros(1, dtype=np.int64), self.episode_sample_lengths_active)
            )
        )
        self.n_samples_active = int(self.cum_sample_lengths_active[-1])

        self.task_embedding_episode = self.task_embedding_episode_all[
            self.episode_indices_active
        ].astype(np.float32, copy=False)
        self.task_language_tokens_episode = (
            None
            if self.task_language_tokens_episode_all is None
            else self.task_language_tokens_episode_all[
                self.episode_indices_active
            ].astype(np.float32, copy=False)
        )
        self.task_id_episode = self.task_id_episode_all[
            self.episode_indices_active
        ].astype(np.int64, copy=False)
        self.robot_id_episode = self.robot_id_episode_all[
            self.episode_indices_active
        ].astype(
            np.int64,
            copy=False,
        )
        if self.task_instructions_all is not None:
            self.task_instructions = [
                self.task_instructions_all[int(ep_idx)]
                for ep_idx in self.episode_indices_active.tolist()
            ]
        else:
            self.task_instructions = None

        self.n_train_episodes = int(self.n_demo_active * (1.0 - self.val_ratio))
        self.n_eval_episodes = self.n_demo_active - self.n_train_episodes
        self.train_length = int(self.cum_sample_lengths_active[self.n_train_episodes])
        self.eval_length = int(self.n_samples_active - self.train_length)

    def set_mode(self, mode: str):
        """Select sampling range: ``train``, ``eval``, or ``all``."""
        ranges = {
            "train": (0, self.train_length),
            "eval": (self.train_length, self.train_length + self.eval_length),
            "all": (0, self.n_samples_active),
        }
        if mode not in ranges:
            raise ValueError(
                f"mode must be one of ['train','eval','all'], got {mode!r}"
            )
        self.start_idx, self.end_idx = map(int, ranges[mode])
        self.mode_set = True

    def __len__(self):
        return int(self.end_idx - self.start_idx)

    def __getitem__(self, idx):
        if not self.mode_set:
            raise RuntimeError("Dataset mode not set. set_mode('mode').")
        obs_idx = int(self.start_idx + idx)
        eps_index, obs_indices, action_indices = self.sampler(obs_idx)
        return self._make_sample(
            eps_index=eps_index,
            obs=self.get_obs(eps_index, obs_indices),
            action=self._get_action_window(
                obs_idx=obs_indices[-1], action_indices=action_indices
            ),
            obs_indices=obs_indices,
        )

    def get_normalizer(self, kind: str = "multi_robot_linear"):
        """Fit a model normalizer from the active data slice."""
        return MimicgenNormalization.build_normalizer(self, kind=kind)

    def active_to_global_indices(self, active_indices) -> np.ndarray:
        active_indices = np.asarray(active_indices, dtype=np.int64)
        return self.active_step_indices[active_indices]

    def episode_active_indices(self, episode_idx: int) -> np.ndarray:
        episode_start = int(self.cum_lengths_active[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        return np.arange(episode_start, episode_start + ep_len, dtype=np.int64)

    def episode_global_indices(self, episode_idx: int) -> np.ndarray:
        return self.active_to_global_indices(self.episode_active_indices(episode_idx))

    def optional_episode_lowdim(self, episode_idx: int, key: str):
        global_indices = self.episode_global_indices(episode_idx)
        if key in self.lowdim:
            return self.lowdim[key][global_indices].astype(np.float32, copy=False), key
        try:
            arr = MimicgenCache.load_numpy_array(
                self.cache_dir, os.path.join("lowdim", f"{key}.npy")
            )
        except FileNotFoundError:
            return None, None
        return arr[global_indices].astype(np.float32, copy=False), key

    def task_sample_ranges(self):
        if not self.mode_set:
            raise RuntimeError("Dataset mode not set. set_mode('mode').")
        task_to_ranges = {}
        for ep_idx, task_id in enumerate(self.task_id_episode.tolist()):
            ep_start = int(self.cum_sample_lengths_active[ep_idx])
            ep_end = int(self.cum_sample_lengths_active[ep_idx + 1])
            lo = max(ep_start, int(self.start_idx))
            hi = min(ep_end, int(self.end_idx))
            if hi > lo:
                task_to_ranges.setdefault(int(task_id), []).append(
                    (lo - int(self.start_idx), hi - int(self.start_idx))
                )
        return task_to_ranges

    def get_obs(self, eps_index, obs_indices):
        """Fetch canonical observations for a list of active frame indices."""
        _ = eps_index
        active_obs_indices = np.asarray(obs_indices, dtype=np.int64)
        global_obs_indices = self.active_to_global_indices(active_obs_indices)
        return self.observation_adapter.read(global_obs_indices)

    def get_task_context(self, eps_index: int) -> dict[str, np.ndarray]:
        """Return task-constant model context without a temporal dimension."""
        context = {
            "task_embedding": self.task_embedding_episode[eps_index].astype(
                np.float32, copy=False
            ),
            "task_id": np.asarray(self.task_id_episode[eps_index], dtype=np.int64),
            "robot_id": np.asarray(self.robot_id_episode[eps_index], dtype=np.int64),
        }
        if self.task_language_tokens_episode is not None:
            context["task_language_tokens"] = self.task_language_tokens_episode[
                eps_index
            ].astype(
                np.float32,
                copy=False,
            )
        return context

    def sampler(self, idx: int):
        """Map a trimmed sample index to episode, observation, and action frames."""
        total = int(self.cum_sample_lengths_active[-1])
        if idx < 0 or idx >= total:
            raise IndexError(f"idx {idx} out of range [0, {total})")

        episode_idx = int(
            np.searchsorted(self.cum_sample_lengths_active, idx, side="right") - 1
        )
        episode_start = int(self.cum_lengths_active[episode_idx])
        sample_start = int(self.cum_sample_lengths_active[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        local_idx = int(idx - sample_start)

        obs_indices = (
            episode_start
            + np.maximum(local_idx + np.arange(-(self.n_obs_steps - 1), 1), 0)
        ).tolist()
        action_indices = (
            episode_start + np.minimum(local_idx + np.arange(self.horizon), ep_len - 1)
        ).tolist()
        return episode_idx, obs_indices, action_indices

    def get_trajectory(self, episode_idx: int):
        """Return the full trajectory for one active episode."""
        assert 0 <= episode_idx < self.n_demo_active, "Invalid episode index."
        episode_start = int(self.cum_lengths_active[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        obs_indices = list(range(episode_start, episode_start + ep_len))
        global_obs_indices = self.active_to_global_indices(obs_indices)
        return self._make_sample(
            eps_index=episode_idx,
            obs=self.get_obs(episode_idx, obs_indices),
            action=self.action[global_obs_indices],
            obs_indices=obs_indices,
        )

    def _make_sample(self, eps_index, obs, action, obs_indices):
        sample = {
            "obs": obs,
            "action": action.astype(np.float32, copy=False),
            "obs_index": np.array(obs_indices, dtype=np.int32),
            "task_context": self.get_task_context(eps_index),
        }
        if self.target_adapter is not None:
            active_indices = np.asarray(obs_indices, dtype=np.int64)
            global_indices = self.active_to_global_indices(active_indices)
            sample["targets"] = self.target_adapter.fields(self, global_indices)
        if self.task_instructions is not None:
            sample["task_instruction"] = self.task_instructions[eps_index]
        if self.oracle_info is not None:
            sample["oracle_info"] = self._get_oracle_info(obs_indices)
        if getattr(self, "camera_matrices", None) is not None:
            sample["camera_matrices"] = self._slice_frame_metadata(
                self.camera_matrices, obs_indices
            )
        return sample

    def _get_oracle_info(self, obs_indices) -> dict[str, np.ndarray]:
        return self._slice_frame_metadata(self.oracle_info, obs_indices)

    def _slice_frame_metadata(self, value, obs_indices):
        active_obs_indices = np.asarray(obs_indices, dtype=np.int64)
        global_obs_indices = self.active_to_global_indices(active_obs_indices)
        return self._slice_oracle_info(value, global_obs_indices)

    def _slice_oracle_info(self, value, indices):
        if isinstance(value, dict):
            out = {}
            for key, subvalue in value.items():
                sliced = self._slice_oracle_info(subvalue, indices)
                if sliced is not None:
                    out[key] = sliced
            return out
        out = np.asarray(value)[indices]
        if out.dtype.kind in ("O", "U", "S"):
            return None
        return out.astype(np.float32, copy=False)

    def _get_action_window(self, obs_idx: int, action_indices) -> np.ndarray:
        observation_index = int(self.active_to_global_indices([obs_idx])[0])
        global_action_indices = self.active_to_global_indices(action_indices)
        if self.target_adapter is not None:
            anchor_index = int(self.target_adapter.last_indices[observation_index])
            global_action_indices = np.minimum(global_action_indices, anchor_index)
        window = self.action_adapter.sample_window(
            self.action,
            observation_index=observation_index,
            action_indices=global_action_indices,
        )
        return window

    def _load_cache_arrays(self) -> None:
        """Load action and episode-level arrays used for dataset sampling."""
        self.n_demo_all = len(self.episode_lengths_all)
        self.cum_lengths_all = np.cumsum([0] + self.episode_lengths_all).astype(
            np.int64
        )
        meta = dict(self.meta)
        meta.setdefault("lowdim_keys", list(self.lowdim_keys))
        self.action_adapter = MimicgenAction.MimicGenActionAdapter(
            cache_dir=self.cache_dir,
            meta=meta,
            horizon=self.horizon,
            action_dim=self.action_dim,
            action_rep=self.action_rep,
        )
        self.action = self.action_adapter.load()
        self.task_embedding_episode_all = MimicgenCache.load_numpy_array(
            self.cache_dir, os.path.join("lowdim", "task_embedding.npy")
        ).astype(np.float32, copy=False)
        try:
            self.task_language_tokens_episode_all = MimicgenCache.load_numpy_array(
                self.cache_dir, os.path.join("lowdim", "task_language_tokens.npy")
            ).astype(np.float32, copy=False)
        except FileNotFoundError:
            self.task_language_tokens_episode_all = None
        try:
            self.task_id_episode_all = (
                np.asarray(
                    MimicgenCache.load_numpy_array(
                        self.cache_dir, os.path.join("lowdim", "task_id.npy")
                    )
                )
                .reshape(-1)
                .astype(np.int64, copy=False)
            )
        except FileNotFoundError:
            self.task_id_episode_all = np.zeros((self.n_demo_all,), dtype=np.int64)
        self.robot_id_episode_all = (
            np.asarray(
                MimicgenCache.load_numpy_array(
                    self.cache_dir, os.path.join("lowdim", "robot_id.npy")
                )
            )
            .reshape(-1)
            .astype(np.int64, copy=False)
        )
        self.task_instructions_all = MimicgenCache.load_task_instructions(
            self.cache_dir
        )

    def _get_task_episode_groups(self):
        """Return contiguous episode groups that define one task each."""
        source_counts = self.meta.get("source_episode_counts")
        if not isinstance(source_counts, list) or len(source_counts) == 0:
            raise ValueError(
                "Merged cache is missing required source_episode_counts metadata. "
                "Rebuild the merged cache with vmstack data merge."
            )
        counts = [int(x) for x in source_counts]
        if sum(counts) != self.n_demo_all:
            raise ValueError(
                "Invalid source_episode_counts in cache metadata: "
                f"sum={sum(counts)} != n_demo_all={self.n_demo_all}"
            )
        groups = []
        start = 0
        for count in counts:
            if count < 1:
                raise ValueError(
                    "Invalid source_episode_counts in cache metadata: "
                    f"got non-positive count {count}"
                )
            stop = start + count
            groups.append(np.arange(start, stop, dtype=np.int64))
            start = stop
        return groups
