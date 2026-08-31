"""Simulator-bound episode rendering for MimicGen datasets."""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import mimicgen  # noqa: F401
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from visuomotor.data.core import sparse_voxels as SparseVoxels
from visuomotor.data.core import spatial as Spatial
from visuomotor.data.core.images import encode_rgb_to_jpg_bytes
from visuomotor.data.mimicgen import action as MgAction
from visuomotor.data.mimicgen.oracle import oracle_affordance as OracleAffordance
from visuomotor.data.mimicgen.oracle import oracle_cache as OracleCache
from visuomotor.environment._dataset_rendering import common as RenderingCommon
from visuomotor.environment.action_conversion import convert_actions
from visuomotor.environment.robomimic import mjcf_texture as MjcfTexture
from visuomotor.environment.robomimic import robomimic as RobomimicEnv
from visuomotor.environment.robomimic import (
    robomimic_image_wrapper as RobomimicImageWrapper,
)
from visuomotor.paths import TEXTURES_DIR


def build_oracle_context(
    *,
    env,
    env_meta: dict,
    camera_names: List[str],
    camera_name: str,
    resolution: int,
    patch_size: int,
) -> OracleCache.OracleContext:
    """Initialize simulator-bound MimicGen oracle helpers."""
    from mimicgen.env_interfaces.base import make_interface

    if camera_name not in camera_names:
        return OracleCache.OracleContext()
    try:
        task_name = RobomimicImageWrapper._normalize_mimicgen_task_name(
            env_meta["env_name"]
        )
        return OracleCache.OracleContext(
            task_spec=RobomimicImageWrapper._build_task_spec(task_name),
            interface=make_interface(
                name=RobomimicImageWrapper._default_interface_name(task_name),
                interface_type="robosuite",
                env=env.env,
            ),
            affordance=OracleAffordance.OracleAffordanceResolver(
                env=env,
                env_meta=env_meta,
                camera_name=camera_name,
                resolution=resolution,
                patch_size=patch_size,
            ),
        )
    except Exception as exc:
        print(f"[oracle] disabled: could not initialize MimicGen oracle ({exc})")
        return OracleCache.OracleContext()


class DatasetRenderer:
    def __init__(
        self,
        dataset_path: str,
        out_cache_dir: str,
        camera_resolution: int,
        table_texture_every: int = None,
        texture_rank_by_source_demo: Optional[Dict[int, int]] = None,
        oracle_camera: Optional[str] = "agentview",
        oracle_patch_size: int = 16,
        oracle_min_patch_area_fraction: float = 0.05,
        oracle_min_mask_pixels: int = 16,
        voxel_spec: Optional[Spatial.VoxelProducerSpec] = None,
        voxel_specs: Optional[Dict[str, Spatial.VoxelProducerSpec]] = None,
        point_cloud_spec: Optional[Spatial.PointCloudProducerSpec] = None,
        jpeg_quality: int = RenderingCommon.JPEG_QUALITY_DEFAULT,
        overwrite: bool = False,
    ):
        self.input_path = str(Path(dataset_path).expanduser().resolve())
        self.out_dir = str(Path(out_cache_dir).expanduser().resolve())
        self.res = int(camera_resolution)
        self.jpeg_quality = int(jpeg_quality)
        self.oracle_camera = None if oracle_camera is None else str(oracle_camera)
        self.oracle_patch_size = int(oracle_patch_size)
        self.oracle_min_patch_area_fraction = float(oracle_min_patch_area_fraction)
        self.oracle_min_mask_pixels = int(oracle_min_mask_pixels)
        self.voxel_specs = dict(voxel_specs or {})
        if voxel_spec is not None:
            if self.voxel_specs:
                raise ValueError("use either voxel_spec or voxel_specs")
            self.voxel_specs[voxel_spec.output_key] = voxel_spec
        if any(key != spec.output_key for key, spec in self.voxel_specs.items()):
            raise ValueError("voxel_specs keys must match each spec.output_key")
        self.voxel_spec = (
            next(iter(self.voxel_specs.values()))
            if len(self.voxel_specs) == 1
            else None
        )
        self.point_cloud_spec = point_cloud_spec
        self.enable_voxel = bool(self.voxel_specs)
        self.enable_point_cloud = self.point_cloud_spec is not None
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input dataset not found: {self.input_path}")

        RenderingCommon._remove_existing_output_dir(self.out_dir, overwrite=overwrite)
        os.makedirs(self.out_dir, exist_ok=True)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=self.input_path)

        self.source_env_meta = RobomimicEnv.require_upstream_controller_config(env_meta)
        source_controller = self.source_env_meta["env_kwargs"]["controller_configs"]
        if not bool(source_controller.get("control_delta", True)):
            raise ValueError(
                "MimicGen rerendering requires source actions recorded by a "
                "delta controller"
            )
        self.env_meta = RobomimicEnv.update_env_controller(
            env_meta, action_rep="absolute"
        )

        # Source-controller env converts raw data/<demo>/actions into absolute
        # targets before the rendering env executes them.
        self.action_conversion_env = EnvUtils.create_env_from_metadata(
            env_meta=self.source_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )

        # Create a rendering env with the absolute controller.
        rgb_camera_names = list(self.env_meta["env_kwargs"]["camera_names"])
        self.rgb_camera_names = tuple(rgb_camera_names)
        spatial_camera_names = list(
            dict.fromkeys(
                camera
                for spec in (*self.voxel_specs.values(), self.point_cloud_spec)
                if spec is not None
                for camera in spec.cameras
            )
        )
        render_camera_names = sorted(set(rgb_camera_names) | set(spatial_camera_names))
        self.env_meta["env_kwargs"]["camera_names"] = render_camera_names

        spatial_specs = [
            spec
            for spec in (*self.voxel_specs.values(), self.point_cloud_spec)
            if spec is not None
        ]
        reconstruction_resolutions = {
            int(spec.reconstruction_resolution) for spec in spatial_specs
        }
        if len(reconstruction_resolutions) > 1:
            raise ValueError(
                "voxel and point-cloud reconstruction resolutions must match"
            )
        if self.voxel_specs:
            self.env_meta["env_kwargs"]["use_voxel_obs"] = True
            self.env_meta["env_kwargs"]["voxel_producers"] = [
                spec.declaration() for spec in self.voxel_specs.values()
            ]
            if len(self.voxel_specs) == 1:
                single = next(iter(self.voxel_specs.values()))
                self.env_meta["env_kwargs"].update(
                    voxel_resolution=list(single.resolution),
                    voxel_bounds_min=(
                        list(single.bounds_min)
                        if single.bounds_min is not None
                        else None
                    ),
                    voxel_bounds_max=(
                        list(single.bounds_max)
                        if single.bounds_max is not None
                        else None
                    ),
                    voxel_ws_size=single.ws_size,
                    voxel_cameras=list(single.cameras),
                    spatial_recon_resolution=single.reconstruction_resolution,
                )
        if self.point_cloud_spec is not None:
            self.env_meta["env_kwargs"].update(
                use_point_cloud_obs=True,
                point_cloud_num_points=self.point_cloud_spec.num_points,
                point_cloud_bounds_min=(
                    list(self.point_cloud_spec.bounds_min)
                    if self.point_cloud_spec.bounds_min is not None
                    else None
                ),
                point_cloud_bounds_max=(
                    list(self.point_cloud_spec.bounds_max)
                    if self.point_cloud_spec.bounds_max is not None
                    else None
                ),
                point_cloud_ws_size=self.point_cloud_spec.ws_size,
                point_cloud_table_margin=self.point_cloud_spec.table_margin,
                point_cloud_cameras=list(self.point_cloud_spec.cameras),
                spatial_recon_resolution=(
                    self.point_cloud_spec.reconstruction_resolution
                ),
            )
        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=self.env_meta,
            camera_names=self.env_meta["env_kwargs"]["camera_names"],
            camera_height=self.res,
            camera_width=self.res,
            reward_shaping=False,
        )

        self.camera_names = list(self.env_meta["env_kwargs"]["camera_names"])
        self.rgb_keys = RenderingCommon._camera_obs_keys(rgb_camera_names)
        self.oracle = self._init_oracle_context()
        self.rerender_counter = 0
        self.table_texture_every = table_texture_every
        self.table_texture_files = None
        if self.table_texture_every is not None:
            texture_dir = Path(TEXTURES_DIR) / "train"
            self.table_texture_files = sorted(
                MjcfTexture.list_texture_files(str(texture_dir)),
                key=lambda p: Path(p).name,
            )
        self.texture_rank_by_source_demo = (
            None
            if texture_rank_by_source_demo is None
            else {int(k): int(v) for k, v in texture_rank_by_source_demo.items()}
        )

    def _init_oracle_context(self):
        if self.oracle_camera is None:
            return OracleCache.OracleContext()
        return build_oracle_context(
            env=self.env,
            env_meta=self.env_meta,
            camera_names=self.camera_names,
            camera_name=self.oracle_camera,
            resolution=self.res,
            patch_size=self.oracle_patch_size,
        )

    def _table_texture_for_rank(self, demo_rank: int) -> Optional[str]:
        if self.table_texture_files is None:
            return None
        texture_idx = (int(demo_rank) // int(self.table_texture_every)) % len(
            self.table_texture_files
        )
        return self.table_texture_files[texture_idx]

    def rerender_demonstration(
        self,
        h5_file: h5py.File,
        ep: str,
        texture_demo_rank: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> RenderingCommon.RenderedEpisode:
        """Rerender one episode into bounded compressed and sparse buffers."""
        is_robosuite_env = EnvUtils.is_robosuite_env(self.env_meta)
        if jpeg_quality is None:
            jpeg_quality = self.jpeg_quality

        ep_grp = h5_file[f"data/{ep}"]
        states = ep_grp["states"][()]
        if "actions" not in ep_grp:
            raise KeyError(
                f"Episode {ep} missing raw action dataset 'actions'. "
                "Rerendering replays data/<demo>/actions."
            )
        actions = ep_grp["actions"][()]

        initial_state = {"states": states[0]}
        if is_robosuite_env:
            init_state = h5_file[f"data/{ep}"].attrs["model_file"]
            if self.table_texture_every is not None:
                demo_rank = (
                    self.rerender_counter
                    if texture_demo_rank is None
                    else int(texture_demo_rank)
                )
                texture = self._table_texture_for_rank(demo_rank)
                if texture is None:
                    raise RuntimeError("table texture scheduler was not initialized")
                init_state = MjcfTexture.apply_table_texture(
                    init_state,
                    texture_file=texture,
                )
            initial_state["model"] = init_state

        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                "single-arm MimicGen source actions must have shape [T,7], "
                f"got {actions.shape}"
            )

        converted_actions, _ = convert_actions(
            self.action_conversion_env,
            states,
            actions,
        )
        actions = converted_actions["absolute"]
        absolute_action = MgAction.action_to_posmat(actions)

        collector = None
        if self.oracle_camera is not None:
            collector = OracleCache.OracleFrameCollector.build(
                oracle=self.oracle,
                horizon=int(actions.shape[0]),
                camera_name=self.oracle_camera,
                camera_names=self.rgb_camera_names,
                resolution=self.res,
                patch_size=self.oracle_patch_size,
                min_patch_area_fraction=self.oracle_min_patch_area_fraction,
                min_mask_pixels=self.oracle_min_mask_pixels,
            )

        lowdim_frames: Dict[str, List[np.ndarray]] = defaultdict(list)
        rgb_jpeg = {key: [] for key in self.rgb_keys}
        voxel_frames = {key: [] for key in self.voxel_specs}
        point_cloud_frames = []
        success = False

        self.env.reset()
        for t in range(states.shape[0]):
            state = initial_state if t == 0 else {"states": states[t]}
            obs = self.env.reset_to(state)
            state_dict = self.env.get_state()
            if collector is not None:
                collector(env=self.env, t=t, obs=obs, state_dict=state_dict)

            success = success or bool(self.env.is_success()["task"])
            for key, value in obs.items():
                if (
                    "image" in key
                    or "depth" in key
                    or key in RenderingCommon.SPATIAL_OBS_KEYS
                    or key in self.voxel_specs
                ):
                    continue
                lowdim_frames[key].append(np.array(value, dtype=np.float32, copy=True))

            for camera_key in self.rgb_keys:
                if camera_key not in obs:
                    available_keys = list(obs.keys())[:10]
                    raise KeyError(
                        f"Missing obs image key {camera_key!r}. "
                        f"Have keys: {available_keys} ..."
                    )
                image = np.asarray(obs[camera_key], dtype=np.uint8)
                if image.shape[:2] != (self.res, self.res):
                    image = cv2.resize(
                        image,
                        (self.res, self.res),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb_jpeg[camera_key].append(
                    encode_rgb_to_jpg_bytes(image, quality=jpeg_quality)
                )

            for key in self.voxel_specs:
                try:
                    voxel = np.asarray(obs[key], dtype=np.uint8)
                except KeyError as error:
                    raise KeyError(
                        f"Missing obs key {key!r}; is keyed voxel production "
                        "enabled on the patched robomimic env?"
                    ) from error
                voxel_frames[key].append(SparseVoxels.encode(voxel))

            if self.enable_point_cloud:
                try:
                    point_cloud = np.ascontiguousarray(
                        obs["point_cloud"], dtype=np.float32
                    )
                except KeyError as error:
                    raise KeyError(
                        "Missing obs key 'point_cloud'; is point-cloud production "
                        "enabled on the patched robomimic env?"
                    ) from error
                point_cloud_frames.append(point_cloud.tobytes())

        # Source states cover every recorded transition. Execute only the final
        # target when success is not already visible in those states, preserving
        # the old terminal-success check without rendering every next_obs.
        if not success:
            self.env.step(actions[-1])
            success = bool(self.env.is_success()["task"])

        lowdim = {
            key: np.stack(frames, axis=0).astype(np.float32, copy=False)
            for key, frames in lowdim_frames.items()
        }
        oracle_info = {} if collector is None else collector.as_arrays()
        return RenderingCommon.RenderedEpisode(
            absolute_action=absolute_action,
            lowdim=lowdim,
            rgb_jpeg=rgb_jpeg,
            voxel_frames=voxel_frames,
            point_cloud_frames=point_cloud_frames,
            oracle=oracle_info,
            success=success,
        )

    def close(self) -> None:
        """Release simulator and EGL resources before a worker exits."""
        for wrapped in (self.env, self.action_conversion_env):
            env = getattr(wrapped, "env", None)
            if env is not None:
                with suppress(Exception):
                    env.close()
