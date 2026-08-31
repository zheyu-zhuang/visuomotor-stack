import collections
import math
import pathlib
import time
from typing import Optional

import dill
import numpy as np
import torch
import tqdm
from torch.utils._pytree import tree_map

from visuomotor.data.core import mirror as CoreMirror
from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.data.mimicgen import tasks as MimicgenTasks
from visuomotor.environment.gym_wrappers.video_recording_wrapper import (
    VideoRecordingWrapper,
)
from visuomotor.environment.robomimic.robomimic_setup import (
    RobomimicRunnerRequest,
    build_robomimic_runner_setup,
)
from visuomotor.geometry import representation as Representation
from visuomotor.geometry import rigid as Rigid
from visuomotor.policy.base import BaseImagePolicy
from visuomotor.visualization import rollout_media as RolloutMedia
from visuomotor.visualization.artifacts import ArtifactStore


class SeekerRobomimicImageRunner:
    def __init__(
        self,
        request: RobomimicRunnerRequest,
        *,
        shape_meta: dict,
        past_action=False,
        tqdm_interval_sec=5.0,
        mirror_augmentation=None,
        visualization_enabled=True,
        save_images=True,
        save_videos=True,
    ):
        MimicgenTasks.setup_task_embedding_cache()
        runner_setup = build_robomimic_runner_setup(request)
        rotation_transformer = Representation.RotationTransformer(
            "axis_angle", "rotation_6d"
        )
        delta_rotation_transformer = Representation.RotationTransformer(
            "rotation_6d", "matrix"
        )

        self.env_meta = runner_setup.env_meta
        self.action_rep = runner_setup.action_rep
        self.env = runner_setup.env
        self.env_fns = runner_setup.env_fns
        self.env_seeds = runner_setup.env_seeds
        self.env_prefixes = runner_setup.env_prefixes
        self.env_init_fn_dills = runner_setup.env_init_fn_dills
        self.env_video_enabled = runner_setup.env_video_enabled
        self.fps = request.fps
        self.crf = request.crf
        self.n_obs_steps = request.n_obs_steps
        self.derived_proprio_fields = MimicgenObservations.derived_proprio_fields(
            shape_meta["obs"]
        )
        self.n_action_steps = request.n_action_steps
        self.past_action = past_action
        self.max_steps = request.max_steps
        self.rotation_transformer = rotation_transformer
        self.delta_rotation_transformer = delta_rotation_transformer
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_rewards = {}
        self._device_task_context = {}
        self.shape_meta = shape_meta
        self.default_shape_meta = runner_setup.default_shape_meta
        self.output_dir = request.output_dir
        self.visualization_enabled = bool(visualization_enabled)
        self.artifact_store = ArtifactStore(
            request.output_dir,
            save_images=self.visualization_enabled and bool(save_images),
            save_videos=self.visualization_enabled and bool(save_videos),
        )
        self.render_obs_key = request.render_obs_key
        self.strict_task_success = request.terminate_on_success
        self.enable_oracle_subtask_info = bool(request.enable_oracle_subtask_info)
        self.oracle_projection_camera = runner_setup.oracle_projection_camera
        self.enable_oracle_focus_info = bool(request.enable_oracle_focus_info)
        self.oracle_focus_camera = (
            runner_setup.oracle_projection_camera
            if request.oracle_focus_camera is None
            else str(request.oracle_focus_camera)
        )
        self.mirror_augmentation = CoreMirror.MirrorAugmentationConfig.from_config(
            mirror_augmentation
        )
        source_keys = MimicgenObservations.source_proprio_keys(
            self.default_shape_meta["obs"].keys()
        )
        self.pos_key = source_keys["eef_pos"]
        self.rot_key = source_keys["eef_rot"]
        env_name = self.env_meta["env_name"]
        self.task_embedding = (
            MimicgenTasks.env_name_to_task_embedding(env_name).cpu().numpy()
        )
        self.task_embedding = self.task_embedding.astype(np.float32, copy=False)
        instruction = MimicgenTasks.env_name_to_instruction(env_name)
        self.task_language_tokens = (
            MimicgenTasks.instruction_to_task_language_tokens(instruction).cpu().numpy()
        )
        self.task_language_tokens = self.task_language_tokens.astype(
            np.float32,
            copy=False,
        )
        self.robot_id = int(MimicgenTasks.env_name_to_robot_id(env_name))
        for prefix in self.env_prefixes:
            self.max_rewards[prefix] = 0
        self.max_rewards["total/"] = 0

    def _chunk_initializers(self, start, end, n_envs, epoch):
        init_fns = []
        recording_local_indices = []
        for global_idx in range(start, end):
            base_init_fn = self.env_init_fn_dills[global_idx]
            video_path = None
            if self.env_video_enabled[global_idx] and self.artifact_store.save_videos:
                video_path = self.artifact_store.rollout_dir(epoch).joinpath(
                    RolloutMedia.rollout_video_filename(
                        prefix=self.env_prefixes[global_idx],
                        seed=self.env_seeds[global_idx],
                    )
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path = str(video_path)
                recording_local_indices.append(global_idx - start)

            def init_fn(env, base_init_fn=base_init_fn, video_path=video_path):
                fn = dill.loads(base_init_fn)
                fn(env)
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = video_path

            init_fns.append(dill.dumps(init_fn))
        if len(init_fns) < n_envs:
            init_fns.extend([init_fns[0]] * (n_envs - len(init_fns)))
        assert len(init_fns) == n_envs
        return init_fns, recording_local_indices

    def _prepare_policy_input(
        self,
        obs,
        past_action,
        oracle_info,
        device,
        n_envs,
    ):
        base_pos = obs[self.pos_key][:, -1].astype(np.float32)
        base_rot = obs[self.rot_key][:, -1].astype(np.float32)
        np_obs_dict = self.preprocess_obs(obs)
        if self.past_action and past_action is not None:
            np_obs_dict["past_action"] = past_action[
                :, -(self.n_obs_steps - 1) :
            ].astype(np.float32)
        source_obs = tree_map(
            lambda value: torch.from_numpy(value).to(device=device)
            if isinstance(value, np.ndarray)
            else value,
            np_obs_dict,
        )
        task_context = self._task_context_on_device(n_envs, device)
        side_channels = {}
        if oracle_info is not None:
            side_channels["oracle_info"] = tree_map(
                lambda value: torch.from_numpy(value).to(device=device)
                if isinstance(value, np.ndarray)
                else value,
                oracle_info,
            )
        canonical_obs = CoreObservations.canonicalize_obs(
            source_obs,
            self.shape_meta["obs"],
            canonicalize_rgb=CoreObservations.canonicalize_rgb_from_uint8,
            source_proprio_keys=MimicgenObservations.source_proprio_keys,
            source_camera_keys=MimicgenObservations.source_camera_keys,
            validate_values=False,
        )
        canonical_obs = self._append_proprio_deltas(canonical_obs)
        return canonical_obs, task_context, side_channels, base_pos, base_rot

    def _append_proprio_deltas(self, canonical_obs: dict) -> dict:
        """Difference the extra retained frame, as the dataset differences the cache.

        The environment carries ``n_obs_steps + 1`` steps of every field a delta
        reads, so each observation step is paired with its own predecessor and
        the window matches the dataset's frame-to-frame difference exactly. At
        reset the wrapper pads with the first frame, which zeroes the delta the
        same way an episode's first frame does in the cache.
        """
        if not self.derived_proprio_fields:
            return canonical_obs
        sources = CoreObservations.derived_proprio_sources(
            self.derived_proprio_fields
        )
        missing = [key for key in sources if key not in canonical_obs]
        if missing:
            raise ValueError(f"proprio deltas need unavailable fields {missing}")
        for key in sources:
            if canonical_obs[key].shape[1] != self.n_obs_steps + 1:
                raise ValueError(
                    f"{key} must carry {self.n_obs_steps + 1} steps to difference, "
                    f"got {canonical_obs[key].shape[1]}"
                )
        deltas = CoreObservations.proprio_deltas(
            {key: canonical_obs[key][:, :-1] for key in sources},
            {key: canonical_obs[key][:, 1:] for key in sources},
        )
        for key in sources:
            canonical_obs[key] = canonical_obs[key][:, 1:]
        canonical_obs.update(
            {field: deltas[field] for field in self.derived_proprio_fields}
        )
        return canonical_obs

    def _publish_diagnostics(
        self,
        diagnostics,
        env_action,
        base_pos,
        recording_local_indices,
        n_envs,
        start,
    ) -> None:
        diagnostic_dict = {"diagnostics": diagnostics or {}}
        focus_diagnostics = RolloutMedia.extract_focus_diagnostics(
            diagnostic_dict, n_envs=n_envs
        )
        if recording_local_indices and any(focus_diagnostics):
            self.env.call_each_at(
                "set_focus_diagnostics",
                recording_local_indices,
                args_list=[
                    (focus_diagnostics[index],)
                    for index in recording_local_indices
                ],
            )
        anchor_payloads = RolloutMedia.extract_rollout_diagnostics(
            diagnostic_dict,
            action_positions=env_action[..., :3],
            eef_positions=base_pos,
            n_envs=n_envs,
        )
        for local_index, payload in enumerate(anchor_payloads):
            if payload is not None and start + local_index < len(self.env_seeds):
                payload["seed"] = self.env_seeds[start + local_index]
        if recording_local_indices:
            self.env.call_each_at(
                "set_rollout_diagnostics",
                recording_local_indices,
                args_list=[
                    (anchor_payloads[index],)
                    for index in recording_local_indices
                ],
            )

    def _finalize_rollouts(
        self, all_video_paths, all_rewards, all_start_frames, epoch
    ):
        max_rewards = collections.defaultdict(list)
        total_scores = []
        log_data = {}
        success_flags = []
        summary_labels = []
        artifact_records = []
        for index, rewards in enumerate(all_rewards):
            seed = self.env_seeds[index]
            prefix = self.env_prefixes[index]
            max_reward = np.max(rewards)
            episode_score = float(max_reward)
            if self.strict_task_success:
                episode_score = float(max_reward >= 1.0)
            max_rewards[prefix].append(episode_score)
            total_scores.append(episode_score)
            success = bool(episode_score >= 1.0)
            success_flags.append(success)
            label = f"{prefix.strip('/')} {seed} {'SUCCESS' if success else 'FAIL'}"
            summary_labels.append(f"{prefix.strip('/')}  ·  seed {seed}")
            video_path = all_video_paths[index]
            if video_path and pathlib.Path(video_path).is_file():
                outcome_path = self._finalize_video_path(
                    pathlib.Path(video_path), prefix, seed, success
                )
                record = self.artifact_store.video_record(
                    outcome_path,
                    key="media/rollout/video",
                    caption=label,
                )
                if record is not None:
                    artifact_records.append(record)
            if self.strict_task_success:
                log_data[prefix + f"sim_task_success_{seed}"] = episode_score

        snapshot_items = [
            item
            for item in zip(all_start_frames, success_flags, summary_labels)
            if item[0] is not None
        ]
        if snapshot_items:
            frames, flags, labels = zip(*snapshot_items)
            grid = RolloutMedia.make_image_grid(
                list(frames), success_flags=list(flags), labels=list(labels)
            )
            record = self.artifact_store.save_image(
                grid,
                self.artifact_store.rollout_summary(epoch=epoch),
                key="media/rollout/summary",
                caption=f"rollout epoch {epoch}",
            )
            if record is not None:
                artifact_records.append(record)

        for prefix, values in max_rewards.items():
            value = float(np.mean(values))
            log_data[prefix + "mean_score"] = value
            if prefix == "test/":
                self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
                log_data[prefix + "max_score"] = self.max_rewards[prefix]
        if total_scores:
            total_mean = float(np.mean(total_scores))
            log_data["total/mean_score"] = total_mean
            self.max_rewards["total/"] = max(self.max_rewards["total/"], total_mean)
            log_data["total/max_score"] = self.max_rewards["total/"]
        return log_data, artifact_records

    @staticmethod
    def _finalize_video_path(closed_path, prefix, seed, success):
        outcome_path = closed_path.with_name(
            RolloutMedia.rollout_video_filename(
                prefix=prefix, seed=seed, outcome=success
            )
        )
        if outcome_path.exists():
            for number in range(1, 10000):
                candidate = outcome_path.with_name(
                    f"{outcome_path.stem}_run_{number:04d}{outcome_path.suffix}"
                )
                if not candidate.exists():
                    outcome_path = candidate
                    break
        closed_path.replace(outcome_path)
        return outcome_path

    def _performance_metrics(
        self, *, started, n_inits, policy_calls, phase_seconds
    ):
        elapsed = time.perf_counter() - started
        metrics = {
            "performance/rollout_seconds": elapsed,
            "performance/episodes_per_second": (
                float(n_inits) / elapsed if elapsed > 0.0 else 0.0
            ),
            "performance/policy_calls": float(policy_calls),
        }
        metrics.update(
            {
                f"performance/{phase}_seconds": float(seconds)
                for phase, seconds in phase_seconds.items()
            }
        )
        return metrics

    def run(self, policy: BaseImagePolicy, *, epoch: Optional[int] = None):
        run_started = time.perf_counter()
        phase_seconds = collections.defaultdict(float)
        policy_calls = 0
        device = policy.device
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_start_frames = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns, recording_local_indices = self._chunk_initializers(
                start, end, n_envs, epoch
            )

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            start_frames = RolloutMedia.extract_start_frames(obs, self.render_obs_key)
            all_start_frames[this_global_slice] = list(start_frames[this_local_slice])
            past_action = None
            policy.reset()
            oracle_info = self._current_oracle_focus_info(env, n_envs)

            env_name = self.env_meta["env_name"]

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name} Image {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False

            while not done:
                prepare_started = time.perf_counter()
                (
                    canonical_obs,
                    task_context,
                    policy_side_channels,
                    base_pos,
                    base_rot,
                ) = self._prepare_policy_input(
                    obs,
                    past_action,
                    oracle_info,
                    device,
                    n_envs,
                )
                phase_seconds["prepare"] += time.perf_counter() - prepare_started

                policy_started = time.perf_counter()
                with torch.inference_mode():
                    action_dict = policy.predict_action(
                        canonical_obs,
                        task_context=task_context,
                        **policy_side_channels,
                    )
                action = action_dict["action"].detach().to("cpu").numpy()
                diagnostics = None
                if recording_local_indices and action_dict.get("diagnostics"):
                    diagnostics = tree_map(
                        lambda x: x.detach().to("cpu").numpy()
                        if torch.is_tensor(x)
                        else x,
                        action_dict["diagnostics"],
                    )
                phase_seconds["policy"] += time.perf_counter() - policy_started
                policy_calls += 1
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                env_action_input = action
                if self.mirror_augmentation.enable and self.action_rep == "absolute":
                    env_action_input = CoreMirror.action_mirror_frame_to_world(
                        env_action_input,
                        self.mirror_augmentation,
                        action_rep=self.action_rep,
                        action_dim=self.shape_meta["action"]["shape"][0],
                    )
                env_action = self.undo_transform_action(
                    env_action_input, base_pos=base_pos, base_rot=base_rot
                )
                self._publish_diagnostics(
                    diagnostics,
                    env_action,
                    base_pos,
                    recording_local_indices,
                    n_envs,
                    start,
                )

                env_started = time.perf_counter()
                obs, reward, done, info = env.step(env_action)
                phase_seconds["environment"] += time.perf_counter() - env_started
                oracle_info = self._oracle_focus_info_from_step_infos(info)
                if oracle_info is None:
                    oracle_info = self._current_oracle_focus_info(env, n_envs)
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
        log_data, artifact_records = self._finalize_rollouts(
            all_video_paths, all_rewards, all_start_frames, epoch
        )
        log_data.update(
            self._performance_metrics(
                started=run_started,
                n_inits=n_inits,
                policy_calls=policy_calls,
                phase_seconds=phase_seconds,
            )
        )
        return log_data, tuple(artifact_records)

    def undo_transform_action(self, action, *, base_pos=None, base_rot=None):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            raise NotImplementedError("Dual-arm not supported yet.")
        if action.shape[-1] != 10:
            raise ValueError(
                "rollout actions must use the 10D xyz+rot6d+gripper-command "
                f"contract, got shape {action.shape}"
            )

        rot_dim = action.shape[-1] - 4  # pos(3) + gripper(1)
        pos = action[..., :3]
        rot = action[..., 3 : 3 + rot_dim]
        gripper = action[..., [-1]]
        if self.action_rep == "delta":
            B, T = pos.shape[:2]
            base_rot = base_rot.reshape(B, 1, 3, 3)
            rot_mat = self.delta_rotation_transformer.forward(rot.reshape(B * T, 6))
            rot_mat = rot_mat.reshape(B, T, 3, 3)
            base_rot_t, rot_mat_t, pos_t = map(
                torch.from_numpy, (base_rot, rot_mat, pos)
            )
            zero_t = pos_t.new_zeros(B, 1, 3)
            # Convert chunked EE-frame deltas into world-frame absolute targets.
            rot = (
                Rigid.transform_rotation(base_rot_t, rot_mat_t)
                .numpy()
                .reshape(B, T, 9)[..., :6]
            )
            pos = (
                base_pos[:, None, :]
                + Rigid.transform(base_rot_t, zero_t, pos_t).numpy()
            )

        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        return uaction

    def preprocess_obs(self, obs):
        prepared = dict(obs)
        if self.mirror_augmentation.enable:
            prepared[self.pos_key] = obs[self.pos_key].copy()
            prepared[self.rot_key] = obs[self.rot_key].copy()
            CoreMirror.center_lowdim_observations(
                prepared, self.pos_key, self.rot_key, self.mirror_augmentation
            )
        return prepared

    def _task_context(self, batch_size: int) -> dict[str, np.ndarray]:
        """Task-constant policy context, batched without an observation horizon."""
        return {
            "task_embedding": np.broadcast_to(
                self.task_embedding, (batch_size,) + self.task_embedding.shape
            ).copy(),
            "task_language_tokens": np.broadcast_to(
                self.task_language_tokens,
                (batch_size,) + self.task_language_tokens.shape,
            ).copy(),
            "robot_id": np.full((batch_size,), self.robot_id, dtype=np.int64),
        }

    def _task_context_on_device(self, batch_size: int, device) -> dict[str, torch.Tensor]:
        key = (int(batch_size), str(device))
        context = self._device_task_context.get(key)
        if context is None:
            context = tree_map(
                lambda value: torch.from_numpy(value).to(device=device),
                self._task_context(batch_size),
            )
            self._device_task_context[key] = context
        return context

    def _current_oracle_focus_info(self, env, n_envs: int):
        if not self.enable_oracle_focus_info:
            return None
        infos = env.call("get_oracle_focus_info")
        if len(infos) != int(n_envs):
            raise RuntimeError(
                f"Expected {n_envs} oracle focus info entries, got {len(infos)}"
            )
        return self._stack_oracle_focus_infos(infos, repeat=self.n_obs_steps)

    def _oracle_focus_info_from_step_infos(self, infos):
        if not self.enable_oracle_focus_info:
            return None
        return self._stack_oracle_focus_infos(infos, repeat=None)

    def _stack_oracle_focus_infos(self, infos, repeat: Optional[int]):
        infos = list(infos)
        if not infos:
            return None
        keys = sorted(
            {
                key
                for info in infos
                if isinstance(info, dict)
                for key in info
                if str(key).startswith("oracle_target_")
            }
        )
        if not keys:
            return None

        out = {}
        for key in keys:
            values = []
            for info in infos:
                if key not in info:
                    return None
                arr = np.asarray(info[key])
                if repeat is not None:
                    arr = np.repeat(arr[None], int(repeat), axis=0)
                else:
                    arr = self._pad_oracle_temporal(arr)
                values.append(arr)
            out[key] = np.stack(values, axis=0)
        return MimicgenObservations.canonicalize_oracle_info(out)

    def _pad_oracle_temporal(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape[0] >= self.n_obs_steps:
            return arr[-self.n_obs_steps :]
        pad_count = int(self.n_obs_steps) - int(arr.shape[0])
        pad = np.repeat(arr[:1], pad_count, axis=0)
        return np.concatenate([pad, arr], axis=0)
