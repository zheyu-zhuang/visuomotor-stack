import re
from typing import Optional

import gym
import numpy as np
from gym import spaces
from mimicgen.configs import MG_TaskSpec, config_factory
from mimicgen.env_interfaces.base import make_interface
from robomimic.envs.env_robosuite import EnvRobosuite
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

from visuomotor.data.core import images as CoreImages
from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.mimicgen import observations as MimicgenObservations
from visuomotor.data.mimicgen.oracle.oracle_affordance import OracleAffordanceResolver
from visuomotor.visualization import rollout as RolloutOverlays


def _normalize_mimicgen_task_name(env_name: str) -> str:
    """Map env/task names like ``Square_D0`` or ``square_d2`` to ``square``."""
    name = str(env_name).strip()
    name = re.sub(r"(_)?(d\d+|v\d+)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.lower()


def _default_interface_name(task_name: str) -> str:
    """Return MimicGen's conventional robosuite interface class name."""
    return "MG_" + "".join(part.capitalize() for part in task_name.split("_"))


def _build_task_spec(task_name: str) -> MG_TaskSpec:
    """Build a MimicGen task spec from the registered robosuite config."""
    mg_config = config_factory(name=task_name, config_type="robosuite")
    return MG_TaskSpec.from_json(json_string=mg_config.task.task_spec.dump())


def _validate_spatial_observation(
    key: str,
    obs_type: str,
    value: np.ndarray,
    expected_shape,
) -> None:
    """Reject spatial producer failures before observations reach the policy."""
    if tuple(value.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"rollout {obs_type} observation {key!r} has shape {value.shape}; "
            f"expected {tuple(expected_shape)}"
        )
    if obs_type == "voxel":
        if not np.any(value[0]):
            raise RuntimeError(
                f"rollout voxel observation {key!r} has zero occupied cells; "
                "spatial RGB-D reconstruction failed (check camera modality "
                "registration and workspace bounds)"
            )
    elif obs_type == "point_cloud":
        if not np.isfinite(value).all():
            raise RuntimeError(
                f"rollout point-cloud observation {key!r} contains non-finite values"
            )
        if not np.any(value[..., :3]):
            raise RuntimeError(
                f"rollout point-cloud observation {key!r} has all-zero XYZ; "
                "spatial RGB-D reconstruction failed (check camera modality "
                "registration and workspace bounds)"
            )


# Observation kinds produced by camera rendering and 3D fusion, i.e. the ones
# that can be skipped on control steps whose observation the caller discards.
_VISUAL_OBS_TYPES = frozenset({"rgb", "voxel", "point_cloud"})


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key=None,
        enable_oracle_subtask_info: bool = False,
        oracle_task_name: Optional[str] = None,
        oracle_interface_name: Optional[str] = None,
        oracle_interface_type: str = "robosuite",
        oracle_projection_camera: Optional[str] = None,
        oracle_projection_height: Optional[int] = None,
        oracle_projection_width: Optional[int] = None,
        enable_oracle_focus_info: bool = False,
        oracle_focus_camera: Optional[str] = None,
        oracle_focus_resolution: Optional[int] = None,
        oracle_focus_patch_size: int = 16,
        oracle_focus_min_patch_area_fraction: float = 0.05,
        oracle_focus_min_mask_pixels: int = 16,
        enable_oracle_video_overlay: bool = False,
        oracle_overlay_zoom: float = 4.0,
        rgb_load_resolutions=None,
        rgb_jpeg_quality: int = CoreImages.JPEG_QUALITY_DEFAULT,
    ):

        self.env = env
        self.render_obs_key = (
            MimicgenObservations.source_camera_key("external")
            if render_obs_key is None
            else render_obs_key
        )
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.render_camera = self._camera_from_obs_key(self.render_obs_key)
        self.has_reset_before = False
        self._observation_needed = True
        self._render_frame_needed = False
        self._last_visual_obs = {}
        self.skipped_observations = 0
        self.produced_observations = 0
        self._validated_rgb_keys = set()
        self._validated_spatial_keys = set()
        self.rgb_load_resolutions = {
            str(key): int(value)
            for key, value in dict(rgb_load_resolutions or {}).items()
        }
        self.rgb_jpeg_quality = int(rgb_jpeg_quality)
        self.enable_oracle_subtask_info = bool(enable_oracle_subtask_info)
        self.oracle_projection_camera = oracle_projection_camera
        self.oracle_projection_height = oracle_projection_height
        self.oracle_projection_width = oracle_projection_width
        self.enable_oracle_focus_info = bool(enable_oracle_focus_info)
        self.oracle_focus_camera = oracle_focus_camera or oracle_projection_camera
        self.oracle_focus_resolution = oracle_focus_resolution
        self.oracle_focus_patch_size = int(oracle_focus_patch_size)
        self.oracle_focus_min_patch_area_fraction = float(
            oracle_focus_min_patch_area_fraction
        )
        self.oracle_focus_min_mask_pixels = int(oracle_focus_min_mask_pixels)
        self.enable_oracle_video_overlay = bool(enable_oracle_video_overlay)
        self.oracle_overlay_zoom = float(oracle_overlay_zoom)
        self.focus_diagnostics = []
        self.rollout_diagnostics = RolloutOverlays.RolloutDiagnosticState()
        self.oracle_env_interface = None
        self.oracle_task_spec = None
        self.oracle_affordance = None
        self.oracle_active_subtask_idx = 0
        self.oracle_prev_subtask_signals = {}
        self.oracle_cached_info = {}
        if self.enable_oracle_subtask_info or self.enable_oracle_focus_info:
            task_name = oracle_task_name or _normalize_mimicgen_task_name(env.name)
            task_name = _normalize_mimicgen_task_name(task_name)
            interface_name = oracle_interface_name or _default_interface_name(task_name)
            self.oracle_task_spec = _build_task_spec(task_name)
            self.oracle_env_interface = make_interface(
                name=interface_name,
                interface_type=oracle_interface_type,
                env=env.env,
            )
            if self.enable_oracle_focus_info:
                if self.oracle_focus_camera is None:
                    raise ValueError(
                        "oracle_focus_camera or oracle_projection_camera is required "
                        "when enable_oracle_focus_info=True"
                    )
                self.oracle_affordance = OracleAffordanceResolver(
                    env=env,
                    env_meta={"env_name": env.name},
                    camera_name=self.oracle_focus_camera,
                    resolution=self._oracle_focus_resolution(),
                    patch_size=self.oracle_focus_patch_size,
                )

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        # Unbounded: the controller runs absolute OSC targets, so actions carry
        # world-frame positions and rot6d, not the [-1,1] deltas robosuite's own
        # action_spec advertises. Declaring [-1,1] here would invite any clipping
        # wrapper to silently flatten every target.
        action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=action_shape, dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            if key in CoreObservations.DERIVED_PROPRIO_FIELDS:
                # Built by the runner from consecutive frames, not read from the
                # simulator, which has no such field to report.
                continue
            shape = value["shape"]
            obs_type = value.get("type")
            if obs_type == "voxel":
                # get_observation() passes the voxel key through as raw channel-first
                # uint8: the env's synthesized [occupancy,R,G,B] grid is already uint8.
                observation_space[key] = spaces.Box(
                    low=0, high=255, shape=shape, dtype=np.uint8
                )
                continue
            if obs_type == "rgb":
                observation_space[key] = spaces.Box(
                    low=0, high=255, shape=shape, dtype=np.uint8
                )
                continue

            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 1
            elif key.endswith("image_tcp_centered"):
                min_value, max_value = 0, 1
            elif key.endswith("depth"):
                min_value, max_value = 0, 1
            elif key.endswith("point_cloud"):
                min_value, max_value = -10, 10
            elif key.endswith("quat") or key.endswith("rot") or key.endswith("rot_6d"):
                min_value, max_value = -1, 1
            elif key.endswith("qpos"):
                min_value, max_value = -1, 1
            elif key.endswith("pos") or key.endswith("eef_z"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")

            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def set_observation_needed(self, needed: bool, render_frame: bool = False):
        """Declare whether the next control step's observation will be read.

        ``MultiStepWrapper`` discards every observation but the last
        ``n_obs_steps`` of an action chunk, so on the others the cameras never
        need to render and the RGB-D fusion never needs to run. The render
        camera is kept alive separately for the lanes recording video.
        """
        needed = bool(needed)
        render_frame = bool(render_frame)
        keep = () if needed else ((self.render_camera,) if render_frame else ())
        self.env.set_visual_obs_enabled(needed, keep_cameras=keep)
        self._observation_needed = needed
        self._render_frame_needed = render_frame

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        if self.render_obs_key in raw_obs:
            self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            field = self.shape_meta["obs"][key]
            obs_type = field.get("type")
            if obs_type in _VISUAL_OBS_TYPES and key not in raw_obs:
                # Rendering was skipped for this control step; the caller
                # discards the result, so reuse the last produced value.
                obs[key] = self._last_visual_obs[key]
                continue
            value = raw_obs[key]
            if obs_type == "rgb":
                validated_rgb_keys = getattr(self, "_validated_rgb_keys", set())
                if key not in validated_rgb_keys:
                    if (
                        value.dtype != np.float32
                        or not np.isfinite(value).all()
                        or float(value.min()) < 0.0
                        or float(value.max()) > 1.0
                    ):
                        raise RuntimeError(
                            f"rollout RGB observation {key!r} must be float32 in [0, 1]"
                        )
                    validated_rgb_keys.add(key)
                    self._validated_rgb_keys = validated_rgb_keys
                value = self._canonical_rgb(key, value)
            if (
                obs_type in ("voxel", "point_cloud")
                and key not in self._validated_spatial_keys
            ):
                _validate_spatial_observation(
                    key, obs_type, value, field["shape"]
                )
                self._validated_spatial_keys.add(key)
            obs[key] = value
            if obs_type in _VISUAL_OBS_TYPES:
                self._last_visual_obs[key] = value
        if self._observation_needed:
            self.produced_observations += 1
        else:
            self.skipped_observations += 1
        return obs

    def _canonical_rgb(self, key: str, value: np.ndarray) -> np.ndarray:
        """Compact one rendered frame exactly as the dataset cache stores it.

        Training never sees a rendered frame directly: it sees one that was
        JPEG-encoded into the cache and decoded back out at the encoder's load
        resolution. Replaying both halves here is what keeps a rollout
        observation bit-identical to the trained-on encoding of the same frame.
        """
        source = np.ascontiguousarray(
            np.moveaxis(np.rint(value * 255.0).astype(np.uint8), -3, -1)
        )
        return CoreImages.canonical_rgb_from_source(
            source,
            load_resolution=self.rgb_load_resolutions.get(key),
            quality=self.rgb_jpeg_quality,
        )

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        self.set_observation_needed(True)
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # Validate every episode's initial spatial reconstruction, then avoid
        # scanning large voxel grids on each control step.
        self._validated_rgb_keys.clear()
        self._validated_spatial_keys.clear()
        self._last_visual_obs = {}
        self.skipped_observations = 0
        self.produced_observations = 0
        obs = self.get_observation(raw_obs)
        self.oracle_active_subtask_idx = 0
        self.oracle_prev_subtask_signals = {}
        self.focus_diagnostics = []
        self.rollout_diagnostics.reset()
        self._update_oracle_cached_info(include_focus=self.enable_oracle_focus_info)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        position_key, _ = MimicgenObservations.source_proprio_field("eef_pos")
        self.rollout_diagnostics.observe(obs.get(position_key))
        self.rollout_diagnostics.advance()
        if self.enable_oracle_subtask_info or self.enable_oracle_focus_info:
            expose_oracle_info = self._observation_needed or self._render_frame_needed
            oracle_info = self._update_oracle_cached_info(
                include_focus=(
                    self.enable_oracle_focus_info and expose_oracle_info
                )
            )
            if expose_oracle_info:
                info = dict(info)
                info.update(oracle_info)
        return obs, reward, done, info

    def set_rollout_diagnostics(self, payload=None):
        """Set the action-trajectory payload for the next replan."""
        self.rollout_diagnostics.update(payload)

    def get_oracle_subtask_info(self, include_focus: Optional[bool] = None):
        """Return cached MimicGen oracle subtask object-ref and projection info."""
        _ = include_focus
        return dict(self.oracle_cached_info)

    def _update_oracle_cached_info(self, include_focus: Optional[bool] = None):
        """Advance oracle state once for the current env state and cache the result."""
        if self.oracle_env_interface is None or self.oracle_task_spec is None:
            self.oracle_cached_info = {}
            return {}
        if include_focus is None:
            include_focus = self.enable_oracle_focus_info

        datagen_info = self.oracle_env_interface.get_datagen_info(action=None)
        self.oracle_active_subtask_idx = self._advance_oracle_subtask(
            self.oracle_active_subtask_idx,
            self.oracle_prev_subtask_signals,
            datagen_info.subtask_term_signals,
        )
        self.oracle_prev_subtask_signals = self._scalar_signal_dict(
            datagen_info.subtask_term_signals
        )
        subtask_idx = self.oracle_active_subtask_idx
        subtask = self.oracle_task_spec[subtask_idx]
        object_ref = subtask["object_ref"]

        result = {
            "oracle_subtask_idx": np.asarray(subtask_idx, dtype=np.int64),
            "oracle_subtask_term_signal": subtask["subtask_term_signal"] or "",
            "oracle_object_ref": object_ref or "",
        }
        if object_ref is None:
            if include_focus:
                result.update(self._empty_oracle_focus_info())
            self.oracle_cached_info = result
            return result

        object_pose = datagen_info.object_poses[object_ref]
        xyz = np.asarray(object_pose[:3, 3], dtype=np.float32)
        result["oracle_object_xyz"] = xyz

        if self.oracle_projection_camera is not None:
            h, w = self._oracle_projection_size()
            world_to_pixel = get_camera_transform_matrix(
                sim=self.env.env.sim,
                camera_name=self.oracle_projection_camera,
                camera_height=h,
                camera_width=w,
            )
            row_col = project_points_from_world_to_camera(
                xyz[None],
                world_to_pixel,
                camera_height=h,
                camera_width=w,
            )[0].astype(np.int64)
            result["oracle_object_pixel"] = row_col
        if include_focus:
            result.update(
                self._oracle_focus_info(
                    object_ref=object_ref,
                    subtask_idx=subtask_idx,
                    object_xyz=xyz,
                )
            )
        self.oracle_cached_info = result
        return result

    def get_oracle_focus_info(self):
        """Return only live oracle focus arrays for the current env state."""
        info = self.oracle_cached_info
        return {
            key: value
            for key, value in info.items()
            if key.startswith("oracle_target_")
        }

    def _oracle_focus_resolution(self) -> int:
        if self.oracle_focus_resolution is not None:
            return int(self.oracle_focus_resolution)
        if self.oracle_focus_camera is not None:
            h, w = self._oracle_projection_size_for_camera(self.oracle_focus_camera)
            if h != w:
                raise ValueError(
                    f"Oracle focus expects square camera images, got {h}x{w}"
                )
            return int(h)
        return 256

    def _oracle_patch_grid_size(self) -> int:
        return int(
            np.ceil(
                float(self._oracle_focus_resolution())
                / float(max(self.oracle_focus_patch_size, 1))
            )
        )

    def _empty_oracle_focus_info(self) -> dict:
        if self.oracle_focus_camera is None:
            camera = MimicgenObservations.source_camera_name("external")
        else:
            camera = str(self.oracle_focus_camera)
        grid_size = self._oracle_patch_grid_size()
        return {
            f"oracle_target_box_{camera}": np.full((4,), np.nan, dtype=np.float32),
            f"oracle_target_patch_mask_{camera}": np.zeros(
                (grid_size, grid_size), dtype=np.uint8
            ),
            f"oracle_target_mask_area_{camera}": np.asarray(np.nan, dtype=np.float32),
            "oracle_target_xyz": np.full((3,), np.nan, dtype=np.float32),
        }

    def _oracle_focus_info(self, *, object_ref: str, subtask_idx: int, object_xyz):
        if self.oracle_affordance is None or self.oracle_focus_camera is None:
            return self._empty_oracle_focus_info()
        spec = self.oracle_affordance.affordance_spec(
            ref=object_ref,
            subtask_idx=int(subtask_idx),
        )
        points = self.oracle_affordance.affordance_points(
            ref=object_ref,
            subtask_idx=int(subtask_idx),
            object_xyz=np.asarray(object_xyz, dtype=np.float32),
            spec=spec,
        )
        out = self._empty_oracle_focus_info()
        out["oracle_target_xyz"] = np.mean(
            np.asarray(points, dtype=np.float32).reshape(-1, 3),
            axis=0,
        ).astype(np.float32)
        target_box, target_area, target_mask = self.oracle_affordance.segmentation_box(
            ref=object_ref,
            spec=spec,
            min_patch_area_fraction=self.oracle_focus_min_patch_area_fraction,
            min_mask_pixels=self.oracle_focus_min_mask_pixels,
        )
        camera = str(self.oracle_focus_camera)
        if target_box is not None:
            out[f"oracle_target_box_{camera}"] = np.asarray(
                target_box, dtype=np.float32
            ).reshape(4)
            out[f"oracle_target_mask_area_{camera}"] = np.asarray(
                float(target_area), dtype=np.float32
            )
        if target_mask is not None:
            out[f"oracle_target_patch_mask_{camera}"] = np.asarray(
                target_mask, dtype=np.uint8
            )
        return out

    def _advance_oracle_subtask(self, current_idx, prev_signals, subtask_term_signals):
        """Advance on 0->1 subtask completion transitions."""
        idx = int(current_idx)
        while idx < len(self.oracle_task_spec) - 1:
            signal = self.oracle_task_spec[idx]["subtask_term_signal"]
            if signal is None:
                break
            prev = int(prev_signals.get(signal, 0))
            cur = int(
                np.asarray(subtask_term_signals.get(signal, 0)).reshape(-1)[0]
            )
            if prev == 0 and cur == 1:
                idx += 1
                continue
            break
        return idx

    @staticmethod
    def _scalar_signal_dict(subtask_term_signals):
        return {
            key: int(np.asarray(value).reshape(-1)[0])
            for key, value in subtask_term_signals.items()
        }

    def _oracle_projection_size(self):
        return self._oracle_projection_size_for_camera(self.oracle_projection_camera)

    def _oracle_projection_size_for_camera(self, camera_name: str):
        h = self.oracle_projection_height
        w = self.oracle_projection_width
        if h is None or w is None:
            source_key = MimicgenObservations.source_camera_key_for_name(camera_name)
            shape = self.shape_meta["obs"][source_key]["shape"]
            h = int(shape[-2])
            w = int(shape[-1])
        return int(h), int(w)

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        if self.enable_oracle_video_overlay:
            img = self._draw_oracle_overlay(img)
        if self.focus_diagnostics:
            render_view = self._view_for_obs_key(self.render_obs_key)
            img = RolloutOverlays.draw_focus_overlay(
                img, self.focus_diagnostics, render_view=render_view
            )
        diagnostics = self.rollout_diagnostics.current()
        if diagnostics is not None:
            world_to_pixel = self._render_world_to_pixel(img.shape[:2])
            img = RolloutOverlays.draw_trajectory_overlay(img, diagnostics, world_to_pixel)
            img = RolloutOverlays.draw_hud(img, diagnostics)
        return img

    def set_focus_diagnostics(self, items):
        """Set explicit per-frame focus diagnostics for rollout composition."""
        self.focus_diagnostics = [] if items is None else list(items)

    def _render_world_to_pixel(self, frame_hw):
        """Camera matrix projecting world points into the rendered frame's pixels."""
        camera = self._camera_from_obs_key(self.render_obs_key)
        height, width = frame_hw
        return get_camera_transform_matrix(
            sim=self.env.env.sim,
            camera_name=camera,
            camera_height=int(height),
            camera_width=int(width),
        )

    @staticmethod
    def _camera_from_obs_key(obs_key: str) -> str:
        """Resolve a source-native RGB key through the MimicGen adapter."""
        return MimicgenObservations.source_camera_name_for_key(str(obs_key))

    @classmethod
    def _view_for_obs_key(cls, obs_key: str) -> str:
        """Canonical view name, to match the overlay labels the encoder records."""
        try:
            return MimicgenObservations.view_for_source_key(obs_key)
        except ValueError:
            return cls._camera_from_obs_key(obs_key)

    def _draw_oracle_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw an oracle object-ref box centered at its projected image point."""
        if self.oracle_projection_camera is None:
            return img
        info = self.get_oracle_subtask_info()
        pixel = info.get("oracle_object_pixel", None)
        if pixel is None:
            return img
        proj_h, proj_w = self._oracle_projection_size()
        return RolloutOverlays.draw_oracle_box_overlay(
            img,
            pixel_row_col=pixel,
            projection_height=proj_h,
            projection_width=proj_w,
            zoom=self.oracle_overlay_zoom,
        )
