"""Robomimic environment construction for rollout runners."""

from __future__ import annotations

import collections
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dill
import mimicgen  # noqa
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from visuomotor.data.core import actions as CoreActions
from visuomotor.data.core import images as CoreImages
from visuomotor.data.core import spatial as Spatial
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.data.mimicgen.cache import load_metadata, resolve_cache_dir
from visuomotor.environment.gym_wrappers import (
    video_recording_wrapper as VideoWrapper,
)
from visuomotor.environment.gym_wrappers.async_vector_env import AsyncVectorEnv
from visuomotor.environment.gym_wrappers.multistep_wrapper import MultiStepWrapper
from visuomotor.environment.robomimic import robomimic as RobomimicEnv
from visuomotor.environment.robomimic.mjcf_texture import list_texture_files
from visuomotor.environment.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
from visuomotor.paths import TEXTURES_DIR

_SYNTHETIC_OBS_TYPES = frozenset({"voxel", "point_cloud"})

RobomimicEnv.install_table_texture_hook()


@dataclass
class RobomimicRunnerSetup:
    env_meta: dict
    action_rep: str
    env: AsyncVectorEnv
    env_fns: list
    env_seeds: list[int]
    env_prefixes: list[str]
    env_init_fn_dills: list[bytes]
    env_video_enabled: list[bool]
    default_shape_meta: dict
    oracle_projection_camera: str


@dataclass(frozen=True)
class RobomimicRunnerRequest:
    output_dir: str
    dataset_path: str
    cache_dir: Optional[str]
    action_rep: str
    n_test: int
    n_test_vis: int
    test_start_seed: int
    max_steps: int
    terminate_on_success: bool
    n_obs_steps: int
    n_action_steps: int
    render_obs_key: str
    fps: int
    crf: int
    n_envs: int
    env_name: Optional[str]
    shuffle_table_texture: bool
    enable_oracle_subtask_info: bool
    oracle_projection_camera: Optional[str]
    enable_oracle_focus_info: bool
    oracle_focus_camera: Optional[str]
    oracle_focus_patch_size: int
    oracle_focus_min_patch_area_fraction: float
    oracle_focus_min_mask_pixels: int
    enable_oracle_video_overlay: bool
    oracle_overlay_zoom: float
    voxel_specs: dict[str, Spatial.VoxelProducerSpec]
    point_cloud_spec: Optional[Spatial.PointCloudProducerSpec]
    rgb_load_resolutions: tuple[tuple[str, int], ...]
    # Source fields the step wrapper must retain one extra frame of, so the
    # runner can difference them. Decided by the resolved model input, not by
    # `default_shape_meta`, which always carries every source proprio field.
    delta_history_source_keys: tuple[str, ...] = ()


def assign_textures(texture_dir, eval_split=True, n_envs=8, seed=0):
    split_dir = Path(texture_dir) / ("eval" if eval_split else "train")
    files = list_texture_files(split_dir)
    if len(files) == 0:
        raise RuntimeError(f"No textures found in {split_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    shuffled = [files[i] for i in order]

    return [shuffled[i % len(shuffled)] for i in range(n_envs)]


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type in _SYNTHETIC_OBS_TYPES:
            # voxel/point_cloud obs are synthesized directly by the patched robomimic
            # env (bypassing ObsUtils.process_obs entirely) and must never be
            # registered as an rgb/depth/low_dim modality.
            continue
        modality_mapping[obs_type].append(key)

    # Synthetic spatial observations are built inside EnvRobosuite from raw
    # RGB-D streams. Those streams are intentionally absent from shape_meta,
    # but Robomimic still requires them in its modality registry before it will
    # expose them to the voxel / point-cloud producer.
    env_kwargs = env_meta.get("env_kwargs", {})
    spatial_cameras = []
    for enabled_key, cameras_key in (
        ("use_voxel_obs", "voxel_cameras"),
        ("use_point_cloud_obs", "point_cloud_cameras"),
    ):
        if not bool(env_kwargs.get(enabled_key, False)):
            continue
        cameras = env_kwargs.get(cameras_key) or env_kwargs.get("camera_names", ())
        spatial_cameras.extend(str(camera) for camera in cameras)
    for producer in env_kwargs.get("voxel_producers") or ():
        spatial_cameras.extend(str(camera) for camera in producer.get("cameras", ()))
    for camera in dict.fromkeys(spatial_cameras):
        modality_mapping["rgb"].append(f"{camera}_image")
        modality_mapping["depth"].append(f"{camera}_depth")
    modality_mapping = {
        modality: list(dict.fromkeys(keys))
        for modality, keys in modality_mapping.items()
    }
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )


def default_rollout_shape_meta(
    default_res: int = 256,
    voxel_spec: Optional[Spatial.VoxelProducerSpec] = None,
    voxel_specs: Optional[dict[str, Spatial.VoxelProducerSpec]] = None,
    point_cloud_spec: Optional[Spatial.PointCloudProducerSpec] = None,
    rgb_load_resolutions=None,
) -> dict:
    obs = MimicgenObservations.default_source_observation_meta(
        default_res, rgb_resolutions=rgb_load_resolutions
    )
    voxel_specs = dict(voxel_specs or {})
    if voxel_spec is not None:
        if voxel_specs:
            raise ValueError("use either voxel_spec or voxel_specs")
        voxel_specs[voxel_spec.output_key] = voxel_spec
    for key, spec in voxel_specs.items():
        obs[key] = {"shape": list(spec.resolution), "type": "voxel"}
        obs[key]["shape"].insert(0, len(spec.channels))
    if point_cloud_spec is not None:
        obs["point_cloud"] = {
            "shape": [point_cloud_spec.num_points, len(point_cloud_spec.channels)],
            "type": "point_cloud",
        }
    return {
        "obs": obs,
        "action": {
            "shape": [10],
        },
    }


def _resolve_oracle_projection_camera(
    *, oracle_projection_camera: Optional[str], render_obs_key: str
) -> str:
    if oracle_projection_camera is not None:
        return oracle_projection_camera
    return MimicgenObservations.source_camera_name_for_key(render_obs_key)


def _wrap_robomimic_env(
    *,
    history_keys=(),
    robomimic_env,
    shape_meta: dict,
    render_obs_key: str,
    fps: int,
    crf: int,
    steps_per_render: int,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    terminate_on_success: bool,
    enable_oracle_subtask_info: bool,
    oracle_task_name: Optional[str] = None,
    oracle_projection_camera: Optional[str] = None,
    oracle_projection_height: Optional[int] = None,
    oracle_projection_width: Optional[int] = None,
    enable_oracle_focus_info: bool = False,
    oracle_focus_camera: Optional[str] = None,
    oracle_focus_patch_size: int = 16,
    oracle_focus_min_patch_area_fraction: float = 0.05,
    oracle_focus_min_mask_pixels: int = 16,
    enable_oracle_video_overlay: bool = False,
    oracle_overlay_zoom: float = 4.0,
    rgb_load_resolutions=None,
    rgb_jpeg_quality: int = CoreImages.JPEG_QUALITY_DEFAULT,
):
    return MultiStepWrapper(
        VideoWrapper.VideoRecordingWrapper(
            RobomimicImageWrapper(
                env=robomimic_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key=render_obs_key,
                enable_oracle_subtask_info=enable_oracle_subtask_info,
                oracle_task_name=oracle_task_name,
                oracle_projection_camera=oracle_projection_camera,
                oracle_projection_height=oracle_projection_height,
                oracle_projection_width=oracle_projection_width,
                enable_oracle_focus_info=enable_oracle_focus_info,
                oracle_focus_camera=oracle_focus_camera,
                oracle_focus_resolution=oracle_projection_height,
                oracle_focus_patch_size=oracle_focus_patch_size,
                oracle_focus_min_patch_area_fraction=(
                    oracle_focus_min_patch_area_fraction
                ),
                oracle_focus_min_mask_pixels=oracle_focus_min_mask_pixels,
                enable_oracle_video_overlay=enable_oracle_video_overlay,
                oracle_overlay_zoom=oracle_overlay_zoom,
                rgb_load_resolutions=rgb_load_resolutions,
                rgb_jpeg_quality=rgb_jpeg_quality,
            ),
            video_recoder=VideoWrapper.VideoRecorder.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=crf,
                thread_type="FRAME",
                thread_count=1,
            ),
            file_path=None,
            steps_per_render=steps_per_render,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        history_keys=history_keys,
        terminate_on_success=terminate_on_success,
    )


def _build_vector_env(
    *,
    env_meta: dict,
    default_shape_meta: dict,
    history_keys=(),
    n_envs: int,
    render_obs_key: str,
    fps: int,
    crf: int,
    steps_per_render: int,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    terminate_on_success: bool,
    shuffle_table_texture: bool,
    enable_oracle_subtask_info: bool,
    oracle_projection_camera: str,
    enable_oracle_focus_info: bool,
    oracle_focus_camera: str,
    oracle_focus_patch_size: int,
    oracle_focus_min_patch_area_fraction: float,
    oracle_focus_min_mask_pixels: int,
    default_res: int,
    enable_oracle_video_overlay: bool,
    oracle_overlay_zoom: float,
    rgb_load_resolutions,
    rgb_jpeg_quality: int,
):
    assigned = (
        assign_textures(str(TEXTURES_DIR), eval_split=True, n_envs=n_envs, seed=0)
        if shuffle_table_texture
        else [None] * n_envs
    )

    def make_env_fn(idx: int):
        tex = assigned[idx]

        def env_fn():
            with RobomimicEnv.table_texture(tex):
                robomimic_env = create_env(
                    env_meta=env_meta, shape_meta=default_shape_meta
                )
                robomimic_env.env.hard_reset = False

            return _wrap_robomimic_env(
                robomimic_env=robomimic_env,
                shape_meta=default_shape_meta,
                history_keys=history_keys,
                render_obs_key=render_obs_key,
                fps=fps,
                crf=crf,
                steps_per_render=steps_per_render,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_steps=max_steps,
                terminate_on_success=terminate_on_success,
                enable_oracle_subtask_info=enable_oracle_subtask_info,
                oracle_task_name=env_meta["env_name"],
                oracle_projection_camera=oracle_projection_camera,
                oracle_projection_height=default_res,
                oracle_projection_width=default_res,
                enable_oracle_focus_info=enable_oracle_focus_info,
                oracle_focus_camera=oracle_focus_camera,
                oracle_focus_patch_size=oracle_focus_patch_size,
                oracle_focus_min_patch_area_fraction=(
                    oracle_focus_min_patch_area_fraction
                ),
                oracle_focus_min_mask_pixels=oracle_focus_min_mask_pixels,
                enable_oracle_video_overlay=enable_oracle_video_overlay,
                oracle_overlay_zoom=oracle_overlay_zoom,
                rgb_load_resolutions=rgb_load_resolutions,
                rgb_jpeg_quality=rgb_jpeg_quality,
            )

        return env_fn

    def dummy_env_fn():
        robomimic_env = create_env(
            env_meta=env_meta, shape_meta=default_shape_meta, enable_render=False
        )
        print(
            f"[env] {n_envs}x {env_meta['env_name']} "
            f"(action_dim={robomimic_env.action_dimension})"
        )
        return _wrap_robomimic_env(
            robomimic_env=robomimic_env,
            shape_meta=default_shape_meta,
            history_keys=history_keys,
            render_obs_key=render_obs_key,
            fps=fps,
            crf=crf,
            steps_per_render=steps_per_render,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_steps=max_steps,
            terminate_on_success=terminate_on_success,
            enable_oracle_subtask_info=False,
            enable_oracle_focus_info=False,
            rgb_load_resolutions=rgb_load_resolutions,
            rgb_jpeg_quality=rgb_jpeg_quality,
        )

    env_fns = [make_env_fn(i) for i in range(n_envs)]
    return AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn, copy=False), env_fns


def _build_rollout_init_specs(
    *,
    output_dir: str,
    n_test: int,
    n_test_vis: int,
    test_start_seed: int,
) -> tuple[list[int], list[str], list[bytes], list[bool]]:
    env_seeds: list[int] = []
    env_prefixes: list[str] = []
    env_init_fn_dills: list[bytes] = []
    env_video_enabled: list[bool] = []

    for i in range(n_test):
        seed = test_start_seed + i
        enable_render = i < n_test_vis

        def init_fn(
            env,
            seed=seed,
            enable_render=enable_render,
        ):
            assert isinstance(env.env, VideoWrapper.VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            if enable_render:
                filename = Path(output_dir).joinpath(
                    "media",
                    f"test_seed_{seed}.mp4",
                )
                filename.parent.mkdir(parents=False, exist_ok=True)
                env.env.file_path = str(filename)

            assert isinstance(env.env.env, RobomimicImageWrapper)
            env.env.env.init_state = None
            env.seed(seed)

        env_seeds.append(seed)
        env_prefixes.append("test/")
        env_init_fn_dills.append(dill.dumps(init_fn))
        env_video_enabled.append(enable_render)

    return env_seeds, env_prefixes, env_init_fn_dills, env_video_enabled


def _load_cache_rgb_codec(
    *, dataset_path: str, cache_dir: Optional[str]
) -> tuple[int, int]:
    """The cache's camera render resolution and JPEG quality.

    Rollout renders at the resolution the cache was built from and replays its
    codec, so an observation the policy sees in evaluation went through the
    same reconstruction and the same compression it did in training.
    """
    cache_path = resolve_cache_dir(dataset_path, cache_dir)
    meta, _ = load_metadata(str(cache_path))
    try:
        render_resolution = int(meta["image_size"])
    except KeyError as error:
        raise ValueError(
            "rollout cache is missing 'image_size'; rebuild it so rollout can "
            "render at the resolution the dataset was rendered at"
        ) from error
    quality = int(meta.get("jpeg_quality", CoreImages.JPEG_QUALITY_DEFAULT))
    return render_resolution, quality


def _load_env_meta_from_cache(
    *,
    dataset_path: str,
    cache_dir: Optional[str],
    env_name: Optional[str],
) -> dict:
    cache_path = resolve_cache_dir(dataset_path, cache_dir)
    meta, _ = load_metadata(str(cache_path))
    if isinstance(meta.get("env_meta"), dict):
        env_meta = copy.deepcopy(meta["env_meta"])
        if env_name is None or str(env_meta.get("env_name", "")) == str(env_name):
            return env_meta

    source_env_metas = meta.get("source_env_metas", [])
    matches = []
    for item in source_env_metas:
        if not isinstance(item, dict):
            continue
        if env_name is None or str(item.get("env_name", "")) == str(env_name):
            matches.append(item)

    if len(matches) == 1:
        return copy.deepcopy(matches[0])

    available = [
        str(item.get("env_name", ""))
        for item in source_env_metas
        if isinstance(item, dict)
    ]
    if env_name is None:
        raise ValueError(
            "Merged cache contains multiple env_metas; set env_name to select "
            f"one. Available envs: {available}"
        )
    raise ValueError(
        f"Cache metadata does not contain env_meta for env_name={env_name!r}. "
        f"Available envs: {available}"
    )


def _validate_spatial_cache_metadata(
    *,
    dataset_path: str,
    cache_dir: Optional[str],
    voxel_specs: Optional[dict[str, Spatial.VoxelProducerSpec]],
    point_cloud_spec: Optional[Spatial.PointCloudProducerSpec],
) -> None:
    cache_path = resolve_cache_dir(dataset_path, cache_dir)
    meta, _ = load_metadata(str(cache_path))
    voxel_specs = dict(voxel_specs or {})
    if voxel_specs:
        metadata_by_key = meta.get("voxel_specs")
        if metadata_by_key is None:
            if len(voxel_specs) != 1 or meta.get("voxel_spec") is None:
                raise ValueError("rollout cache is missing per-key voxel metadata")
            metadata_by_key = {next(iter(voxel_specs)): meta["voxel_spec"]}
        if set(metadata_by_key) != set(voxel_specs):
            raise ValueError("rollout cache voxel keys do not match producer specs")
        for key, spec in voxel_specs.items():
            spec.validate_metadata(metadata_by_key[key])
    if point_cloud_spec is not None:
        try:
            metadata = meta["point_cloud_spec"]
        except KeyError as error:
            raise ValueError(
                "rollout cache is missing point_cloud_spec metadata"
            ) from error
        point_cloud_spec.validate_metadata(metadata)


def build_robomimic_runner_setup(
    request: RobomimicRunnerRequest,
) -> RobomimicRunnerSetup:
    output_dir = request.output_dir
    dataset_path = request.dataset_path
    cache_dir = request.cache_dir
    action_rep = request.action_rep
    n_test = request.n_test
    n_test_vis = request.n_test_vis
    test_start_seed = request.test_start_seed
    max_steps = request.max_steps
    terminate_on_success = request.terminate_on_success
    n_obs_steps = request.n_obs_steps
    n_action_steps = request.n_action_steps
    render_obs_key = request.render_obs_key
    fps = request.fps
    crf = request.crf
    n_envs = request.n_envs
    env_name = request.env_name
    shuffle_table_texture = request.shuffle_table_texture
    enable_oracle_subtask_info = request.enable_oracle_subtask_info
    oracle_projection_camera = request.oracle_projection_camera
    enable_oracle_focus_info = request.enable_oracle_focus_info
    oracle_focus_camera = request.oracle_focus_camera
    oracle_focus_patch_size = request.oracle_focus_patch_size
    oracle_focus_min_patch_area_fraction = (
        request.oracle_focus_min_patch_area_fraction
    )
    oracle_focus_min_mask_pixels = request.oracle_focus_min_mask_pixels
    enable_oracle_video_overlay = request.enable_oracle_video_overlay
    oracle_overlay_zoom = request.oracle_overlay_zoom
    voxel_specs = request.voxel_specs
    point_cloud_spec = request.point_cloud_spec
    dataset_path = os.path.expanduser(dataset_path)
    action_rep = CoreActions.validate_action_rep(action_rep)

    _validate_spatial_cache_metadata(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        voxel_specs=voxel_specs,
        point_cloud_spec=point_cloud_spec,
    )
    voxel_specs = dict(voxel_specs or {})

    env_meta = _load_env_meta_from_cache(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        env_name=env_name,
    )
    env_meta = RobomimicEnv.update_env_controller(env_meta, action_rep)
    env_meta["env_kwargs"]["use_object_obs"] = False
    if env_name is not None:
        env_meta["env_name"] = env_name

    default_res, rgb_jpeg_quality = _load_cache_rgb_codec(
        dataset_path=dataset_path, cache_dir=cache_dir
    )
    env_meta["env_kwargs"]["camera_heights"] = default_res
    env_meta["env_kwargs"]["camera_widths"] = default_res
    rgb_load_resolutions = {
        MimicgenObservations.source_camera_key_for_canonical(key): int(value)
        for key, value in dict(request.rgb_load_resolutions).items()
    }
    spatial_specs = [
        spec for spec in (*voxel_specs.values(), point_cloud_spec) if spec is not None
    ]
    if spatial_specs:
        # The producer specs name the cameras fused for 3D reconstruction; extra
        # cameras (birdview, sideview) are rendered here but never added to the
        # RGB observation contract, and they match what dataset rerendering used
        # at train time so train/eval coverage stays consistent.
        spatial_camera_names = sorted(
            {camera for spec in spatial_specs for camera in spec.cameras}
        )
        env_meta["env_kwargs"]["camera_names"] = sorted(
            set(env_meta["env_kwargs"]["camera_names"]) | set(spatial_camera_names)
        )
        reconstruction_resolutions = {
            int(spec.reconstruction_resolution) for spec in spatial_specs
        }
        if len(reconstruction_resolutions) != 1:
            raise ValueError(
                "voxel and point-cloud reconstruction resolutions must match"
            )
        env_meta["env_kwargs"]["spatial_recon_resolution"] = (
            reconstruction_resolutions.pop()
        )
    if voxel_specs:
        env_meta["env_kwargs"]["use_voxel_obs"] = True
        env_meta["env_kwargs"]["voxel_producers"] = [
            spec.declaration() for spec in voxel_specs.values()
        ]
        if len(voxel_specs) == 1:
            single = next(iter(voxel_specs.values()))
            env_meta["env_kwargs"]["voxel_resolution"] = list(single.resolution)
            env_meta["env_kwargs"]["voxel_bounds_min"] = (
                list(single.bounds_min) if single.bounds_min is not None else None
            )
            env_meta["env_kwargs"]["voxel_bounds_max"] = (
                list(single.bounds_max) if single.bounds_max is not None else None
            )
            env_meta["env_kwargs"]["voxel_ws_size"] = single.ws_size
            env_meta["env_kwargs"]["voxel_cameras"] = list(single.cameras)
    if point_cloud_spec is not None:
        env_meta["env_kwargs"]["use_point_cloud_obs"] = True
        env_meta["env_kwargs"]["point_cloud_num_points"] = point_cloud_spec.num_points
        env_meta["env_kwargs"]["point_cloud_bounds_min"] = (
            list(point_cloud_spec.bounds_min)
            if point_cloud_spec.bounds_min is not None
            else None
        )
        env_meta["env_kwargs"]["point_cloud_bounds_max"] = (
            list(point_cloud_spec.bounds_max)
            if point_cloud_spec.bounds_max is not None
            else None
        )
        env_meta["env_kwargs"]["point_cloud_ws_size"] = point_cloud_spec.ws_size
        env_meta["env_kwargs"]["point_cloud_table_margin"] = (
            point_cloud_spec.table_margin
        )
        env_meta["env_kwargs"]["point_cloud_cameras"] = list(point_cloud_spec.cameras)
    default_shape_meta = default_rollout_shape_meta(
        default_res,
        voxel_specs=voxel_specs,
        point_cloud_spec=point_cloud_spec,
        rgb_load_resolutions=rgb_load_resolutions,
    )
    oracle_projection_camera = _resolve_oracle_projection_camera(
        oracle_projection_camera=oracle_projection_camera,
        render_obs_key=render_obs_key,
    )
    oracle_focus_camera = _resolve_oracle_projection_camera(
        oracle_projection_camera=oracle_focus_camera or oracle_projection_camera,
        render_obs_key=render_obs_key,
    )

    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)
    env, env_fns = _build_vector_env(
        env_meta=env_meta,
        default_shape_meta=default_shape_meta,
        history_keys=tuple(request.delta_history_source_keys),
        n_envs=n_envs,
        render_obs_key=render_obs_key,
        fps=fps,
        crf=crf,
        steps_per_render=steps_per_render,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        terminate_on_success=terminate_on_success,
        shuffle_table_texture=shuffle_table_texture,
        enable_oracle_subtask_info=enable_oracle_subtask_info,
        oracle_projection_camera=oracle_projection_camera,
        enable_oracle_focus_info=enable_oracle_focus_info,
        oracle_focus_camera=oracle_focus_camera,
        oracle_focus_patch_size=oracle_focus_patch_size,
        oracle_focus_min_patch_area_fraction=oracle_focus_min_patch_area_fraction,
        oracle_focus_min_mask_pixels=oracle_focus_min_mask_pixels,
        default_res=default_res,
        enable_oracle_video_overlay=enable_oracle_video_overlay,
        oracle_overlay_zoom=oracle_overlay_zoom,
        rgb_load_resolutions=rgb_load_resolutions,
        rgb_jpeg_quality=rgb_jpeg_quality,
    )
    env_seeds, env_prefixes, env_init_fn_dills, env_video_enabled = (
        _build_rollout_init_specs(
            output_dir=output_dir,
            n_test=n_test,
            n_test_vis=n_test_vis,
            test_start_seed=test_start_seed,
        )
    )
    return RobomimicRunnerSetup(
        env_meta=env_meta,
        action_rep=action_rep,
        env=env,
        env_fns=env_fns,
        env_seeds=env_seeds,
        env_prefixes=env_prefixes,
        env_init_fn_dills=env_init_fn_dills,
        env_video_enabled=env_video_enabled,
        default_shape_meta=default_shape_meta,
        oracle_projection_camera=oracle_projection_camera,
    )
