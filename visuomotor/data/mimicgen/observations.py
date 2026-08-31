"""MimicGen/robomimic observation adapter.

Owns cache extraction and the benchmark's ``robotN_*`` proprio and simulator
camera conventions. Canonical observation keys are ``<modality>_<view>``: the
simulator's ``agentview`` is the default third-person camera
(``rgb_external``), and ``robot0_eye_in_hand`` is robot-mounted
(``rgb_wrist``).
"""

import os
from typing import Mapping

import lmdb
import numpy as np
import torch

from visuomotor.data.core import images as CoreImages
from visuomotor.data.core import observations as CoreObservations
from visuomotor.data.core import sparse_voxels as SparseVoxels
from visuomotor.data.mimicgen import cache as MimicgenCache

_SOURCE_PROPRIO_FIELDS = {
    "eef_pos": ("robot0_eef_pos", (3,)),
    "eef_rot6d": ("robot0_eef_rot", (9,)),
    "gripper_qpos": ("robot0_gripper_qpos", (2,)),
}
_VIEW_CAMERAS = {"external": "agentview", "wrist": "robot0_eye_in_hand"}
_VIEW_SOURCE_KEYS = {
    "external": "agentview_image",
    "wrist": "robot0_eye_in_hand_image",
}


def _previous_frame_index(episode_lengths) -> np.ndarray:
    """Index of each frame's predecessor, with every episode's first frame on itself."""
    lengths = np.asarray(list(episode_lengths), dtype=np.int64)
    index = np.arange(int(lengths.sum()), dtype=np.int64) - 1
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    index[starts] = starts
    return index


def source_proprio_field(canonical_key: str) -> tuple[str, tuple[int, ...]]:
    """MimicGen field name and physical source shape for canonical proprioception."""
    try:
        return _SOURCE_PROPRIO_FIELDS[canonical_key]
    except KeyError as error:
        raise ValueError(f"unknown canonical proprio field: {canonical_key}") from error


def derived_proprio_fields(shape_meta_obs) -> tuple:
    """Derived proprio fields a shape_meta selects, in canonical order."""
    selected = set(shape_meta_obs)
    return tuple(
        key
        for key in CoreObservations.DERIVED_PROPRIO_FIELDS
        if key in selected
    )


def delta_history_source_keys(shape_meta_obs) -> tuple:
    """Source fields a rollout must retain one extra step of, to difference."""
    sources = CoreObservations.derived_proprio_sources(
        derived_proprio_fields(shape_meta_obs)
    )
    return tuple(source_proprio_field(key)[0] for key in sources)


def source_proprio_keys(keys) -> dict:
    """Source-native proprio keys among ``keys``, keyed by source field name."""
    keys = set(keys)
    return {
        "eef_pos": _source_or_canonical_key("eef_pos", keys),
        "eef_rot": _source_key("eef_rot6d", keys),
        "gripper_qpos": _source_or_canonical_key("gripper_qpos", keys),
    }


def source_camera_keys(keys) -> dict:
    """Source-native camera keys among ``keys``, keyed by canonical observation key."""
    keys = set(keys)
    return {
        canonical_key: (
            source_key
            if source_key in keys
            else canonical_key
            if canonical_key in keys
            else None
        )
        for view, source_key in _VIEW_SOURCE_KEYS.items()
        for canonical_key in (f"rgb_{view}",)
    }


def canonical_camera_key(source_key: str) -> str:
    """Canonical observation key for one source camera key."""
    for canonical, matched in source_camera_keys([source_key]).items():
        if matched is not None:
            return canonical
    raise ValueError(f"unrecognized source camera key: {source_key}")


def view_for_source_key(source_key: str) -> str:
    """Canonical view behind a source camera key, i.e. the key minus its modality."""
    return canonical_camera_key(source_key).split("_", 1)[1]


def source_camera_name(view: str) -> str:
    """Simulator camera name behind a canonical view, for oracle projections."""
    if view not in _VIEW_CAMERAS:
        raise ValueError(f"unknown canonical view: {view}")
    return _VIEW_CAMERAS[view]


def canonicalize_oracle_info(info: Mapping) -> dict:
    """Rename camera-qualified oracle metadata to canonical view names."""
    canonical = {}
    for key, value in info.items():
        if isinstance(value, Mapping):
            value = canonicalize_oracle_info(value)
        canonical_key = str(key)
        for view, camera_name in _VIEW_CAMERAS.items():
            suffix = f"_{camera_name}"
            if canonical_key.endswith(suffix):
                canonical_key = canonical_key[: -len(suffix)] + f"_{view}"
                break
        if canonical_key in canonical:
            raise ValueError(f"duplicate canonical oracle field: {canonical_key}")
        canonical[canonical_key] = value
    return canonical


def source_camera_key(view: str) -> str:
    """Dataset RGB field behind one canonical view."""
    if view not in _VIEW_SOURCE_KEYS:
        raise ValueError(f"unknown canonical view: {view}")
    return _VIEW_SOURCE_KEYS[view]


def source_camera_key_for_canonical(canonical_key: str) -> str:
    """Dataset RGB field behind one canonical observation key."""
    key = str(canonical_key)
    modality, _, view = key.partition("_")
    if modality != "rgb" or not view:
        raise ValueError(f"not a canonical RGB observation key: {canonical_key}")
    return source_camera_key(view)


def source_camera_name_for_key(source_key: str) -> str:
    """Simulator camera name for a source-native RGB observation key."""
    try:
        return source_camera_name(view_for_source_key(source_key))
    except ValueError:
        if source_key.endswith("_image"):
            return source_key[: -len("_image")]
        raise ValueError(f"unrecognized source camera key: {source_key}")


def source_camera_key_for_name(camera_name: str) -> str:
    """Source-native RGB observation key for a simulator camera name."""
    for view, known_name in _VIEW_CAMERAS.items():
        if camera_name == known_name:
            return source_camera_key(view)
    return f"{camera_name}_image"


def default_source_observation_meta(
    image_size: int, rgb_resolutions=None
) -> dict[str, dict]:
    """Source-native observation metadata required by a MimicGen rollout.

    ``rgb_resolutions`` overrides per source camera key, for views the encoder
    loads at something other than the camera's render resolution.
    """
    rgb_resolutions = dict(rgb_resolutions or {})
    obs = {}
    for view in _VIEW_SOURCE_KEYS:
        source_key = source_camera_key(view)
        resolution = int(rgb_resolutions.get(source_key, image_size))
        obs[source_key] = {
            "shape": [3, resolution, resolution],
            "type": "rgb",
        }
    for canonical_key in _SOURCE_PROPRIO_FIELDS:
        source_key, shape = source_proprio_field(canonical_key)
        obs[source_key] = {"shape": list(shape)}
    return obs


def _source_key(canonical_key: str, keys: set[str]):
    source_key, _ = source_proprio_field(canonical_key)
    return source_key if source_key in keys else None


def _source_or_canonical_key(canonical_key: str, keys: set[str]):
    source_key = _source_key(canonical_key, keys)
    if source_key is not None:
        return source_key
    return canonical_key if canonical_key in keys else None


def fixed_camera_rgb_keys(obs_meta: Mapping[str, Mapping]) -> list:
    """RGB observation keys from world-fixed cameras; ``eye_in_hand`` is wrist-mounted."""
    return [
        key
        for key, field_spec in obs_meta.items()
        if field_spec.get("type", "low_dim") == "rgb"
        and not CoreObservations.is_matched_key("eye_in_hand", key)
    ]


class MimicGenObservationAdapter:
    """Read MimicGen source observations and emit canonical physical arrays."""

    def __init__(
        self,
        *,
        shape_meta: Mapping,
        cache_dir: str,
        image_size,
        lmdb_readahead: bool,
        rgb_load_resolutions=None,
        voxel_spec=None,
        voxel_specs=None,
        point_cloud_spec=None,
    ) -> None:
        (
            self.rgb_keys,
            self.lowdim_keys,
            self.voxel_keys,
            self.point_cloud_keys,
        ) = MimicgenCache.get_obs_keys(shape_meta)
        self.derived_keys = tuple(
            key
            for key in self.lowdim_keys
            if key in CoreObservations.DERIVED_PROPRIO_FIELDS
        )
        self.lowdim_keys = [
            key for key in self.lowdim_keys if key not in self.derived_keys
        ]
        self.source_keys = source_proprio_keys(self.lowdim_keys)
        self.cache_dir = str(cache_dir)
        self.image_size = image_size
        self.rgb_load_resolutions = {
            str(key): int(value)
            for key, value in dict(rgb_load_resolutions or {}).items()
        }
        if self.image_size is not None and self.rgb_load_resolutions:
            raise ValueError(
                "use either a global image size or per-view RGB load resolutions"
            )
        expected_rgb = tuple(canonical_camera_key(key) for key in self.rgb_keys)
        if (
            self.rgb_load_resolutions
            and tuple(self.rgb_load_resolutions) != expected_rgb
        ):
            raise ValueError(
                "per-view RGB load resolutions must exactly cover selected RGB views"
            )
        if any(resolution < 1 for resolution in self.rgb_load_resolutions.values()):
            raise ValueError("RGB load resolutions must be positive")
        self.lmdb_path = os.path.join(self.cache_dir, "images.lmdb")
        self.lmdb_readahead = bool(lmdb_readahead)
        self._lmdb_env = None
        self._lmdb_txn = None

        self.meta, self.episode_lengths = MimicgenCache.load_metadata(self.cache_dir)
        self.lowdim = MimicgenCache.load_lowdim(self.cache_dir, self.lowdim_keys)
        self.derived_lowdim = self._build_derived_lowdim()
        expected_voxel_specs = dict(voxel_specs or {})
        if voxel_spec is not None:
            if expected_voxel_specs:
                raise ValueError("use either voxel_spec or voxel_specs")
            expected_voxel_specs[voxel_spec.output_key] = voxel_spec
        self.voxel_resolution = None
        self.voxel_channels = None
        self.voxel_offsets = None
        self.voxel_index = None
        self.voxel_colour = None
        self.voxel_max_points = None
        self.voxel_resolutions = {}
        self.voxel_channels_by_key = {}
        self.voxel_offsets_by_key = {}
        self.voxel_index_by_key = {}
        self.voxel_colour_by_key = {}
        self.voxel_max_points_by_key = {}
        if self.voxel_keys:
            if set(expected_voxel_specs) != set(self.voxel_keys):
                raise ValueError(
                    "voxel observations require expected specs for exactly "
                    f"{self.voxel_keys}, got {sorted(expected_voxel_specs)}"
                )
            metadata_by_key = self.meta.get("voxel_specs")
            if metadata_by_key is None:
                if len(self.voxel_keys) != 1 or self.meta.get("voxel_spec") is None:
                    raise KeyError("cache is missing per-key voxel producer metadata")
                metadata_by_key = {self.voxel_keys[0]: self.meta["voxel_spec"]}
            storage_by_key = self.meta.get("voxel_storage")
            if not isinstance(storage_by_key, Mapping):
                storage_by_key = {key: storage_by_key for key in self.voxel_keys}
            max_points_by_key = self.meta.get("voxel_max_points")
            if not isinstance(max_points_by_key, Mapping):
                max_points_by_key = {
                    key: max_points_by_key for key in self.voxel_keys
                }
            for key in self.voxel_keys:
                try:
                    voxel_metadata = metadata_by_key[key]
                    max_points = int(max_points_by_key[key])
                except (KeyError, TypeError) as error:
                    raise KeyError(
                        f"cache is missing sparse metadata for voxel key {key!r}"
                    ) from error
                expected_voxel_specs[key].validate_metadata(voxel_metadata)
                if storage_by_key.get(key) != "sparse":
                    raise KeyError(
                        f"voxel key {key!r} does not use sparse occupied-cell storage"
                    )
                index_name, colour_name, offsets_name = SparseVoxels.array_names(key)
                self.voxel_resolutions[key] = tuple(
                    int(value) for value in voxel_metadata["resolution"]
                )
                self.voxel_channels_by_key[key] = len(voxel_metadata["channels"])
                self.voxel_offsets_by_key[key] = MimicgenCache.load_numpy_array(
                    self.cache_dir, offsets_name
                )
                self.voxel_index_by_key[key] = MimicgenCache.load_numpy_array(
                    self.cache_dir, index_name
                )
                self.voxel_colour_by_key[key] = MimicgenCache.load_numpy_array(
                    self.cache_dir, colour_name
                )
                self.voxel_max_points_by_key[key] = max_points
            if len(self.voxel_keys) == 1:
                key = self.voxel_keys[0]
                self.voxel_resolution = self.voxel_resolutions[key]
                self.voxel_channels = self.voxel_channels_by_key[key]
                self.voxel_offsets = self.voxel_offsets_by_key[key]
                self.voxel_index = self.voxel_index_by_key[key]
                self.voxel_colour = self.voxel_colour_by_key[key]
                self.voxel_max_points = self.voxel_max_points_by_key[key]

        self.point_cloud_shape = None
        if self.point_cloud_keys:
            point_cloud_metadata = self.meta.get("point_cloud_spec")
            if point_cloud_metadata is None:
                raise KeyError(
                    "shape_meta declares a point_cloud obs key but the cache's "
                    "meta.json has no 'point_cloud_spec' (rebuild the cache with "
                    "--enable-point-cloud)"
                )
            if point_cloud_spec is None:
                raise ValueError(
                    "point-cloud observations require an expected PointCloudProducerSpec"
                )
            point_cloud_spec.validate_metadata(point_cloud_metadata)
            table_margin = point_cloud_metadata.get("table_margin")
            if table_margin is None or float(table_margin) <= 0:
                raise KeyError(
                    "point-cloud cache predates tabletop removal; rebuild it with "
                    "--enable-point-cloud"
                )
            self.point_cloud_shape = (
                int(point_cloud_metadata["num_points"]),
                len(point_cloud_metadata["channels"]),
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lmdb_env"] = None
        state["_lmdb_txn"] = None
        return state

    def read(self, global_indices) -> dict[str, np.ndarray]:
        """Read one temporal window and convert it to canonical observations.

        Images stay in their cached uint8 form and voxels in their sparse
        occupied-cell form; see ``canonicalize_obs``.
        """
        global_indices = np.asarray(global_indices, dtype=np.int64)
        observations = {}
        transaction = self._transaction()
        for key in self.rgb_keys:
            observations[key] = np.stack(
                [
                    self._decode_image(transaction, key, index)
                    for index in global_indices
                ]
            )
        for key in self.voxel_keys:
            index_key, colour_key = SparseVoxels.sparse_keys(key)
            records = [
                self._decode_voxel(transaction, key, index) for index in global_indices
            ]
            observations[index_key] = np.stack([record[0] for record in records])
            observations[colour_key] = np.stack([record[1] for record in records])
        for key in self.point_cloud_keys:
            observations[key] = np.stack(
                [
                    self._decode_point_cloud(transaction, key, index)
                    for index in global_indices
                ]
            )
        observations.update(
            {
                key: self.lowdim[key][global_indices].astype(np.float32, copy=False)
                for key in self.lowdim_keys
            }
        )
        canonical = self.canonicalize_obs(observations)
        canonical.update(
            {
                key: self.derived_lowdim[key][global_indices]
                for key in self.derived_keys
            }
        )
        return canonical

    def _build_derived_lowdim(self) -> dict:
        """Finite-difference the derived proprio fields over the whole cache.

        The delta is defined on a pair of consecutive frames, which a single
        read window (one step wide by default) does not contain. Differencing
        here instead keeps the read path a plain index and costs one pass over
        arrays that are already resident.
        """
        if not self.derived_keys:
            return {}
        required = CoreObservations.derived_proprio_sources(self.derived_keys)
        canonical = self.canonicalize_obs(
            {
                # Copied off the read-only mmap: the canonical conversion wraps
                # these in tensors, which must not alias the cache.
                key: np.array(self.lowdim[key], dtype=np.float32)
                for key in dict.fromkeys(self.source_keys.values())
                if key is not None and key in self.lowdim
            }
        )
        missing = [field for field in required if field not in canonical]
        if missing:
            raise ValueError(
                f"derived proprioception {list(self.derived_keys)} requires "
                f"unselected canonical fields {missing}"
            )
        previous_index = _previous_frame_index(self.episode_lengths)
        deltas = CoreObservations.proprio_deltas(
            {
                field: torch.from_numpy(canonical[field][previous_index])
                for field in required
            },
            {field: torch.from_numpy(canonical[field]) for field in required},
        )
        return {
            key: deltas[key].numpy().astype(np.float32, copy=False)
            for key in self.derived_keys
        }

    def canonicalize_obs(self, observations: Mapping[str, np.ndarray]) -> dict:
        """Convert MimicGen source arrays to the canonical physical contract.

        Images keep their compact uint8 encoding, so collation, pinning, and the
        device transfer move a quarter of the bytes; the policy normalizer
        converts them directly to model space. Voxels go further and cross as
        occupied cells only (see :mod:`visuomotor.data.core.sparse_voxels`),
        which is where the dense grid's ~98% padding stops being paid for.
        """
        canonical = CoreObservations.canonicalize_numpy_obs(
            observations,
            rgb_source_keys=self.rgb_keys,
            source_proprio_keys=source_proprio_keys,
            source_camera_keys=source_camera_keys,
        )
        self._validate_sparse_voxels(canonical)
        return canonical

    def _validate_sparse_voxels(
        self, observations: Mapping[str, np.ndarray]
    ) -> None:
        for key in self.voxel_keys:
            index_key, colour_key = SparseVoxels.sparse_keys(key)
            if index_key not in observations:
                continue
            cell_index, colour = observations[index_key], observations[colour_key]
            if cell_index.dtype != np.int32:
                raise ValueError("MimicGen voxel cell indices must be int32")
            if colour.dtype != np.uint8:
                raise ValueError("MimicGen voxel cell colours must be uint8")
            max_points = getattr(self, "voxel_max_points_by_key", {}).get(
                key, self.voxel_max_points
            )
            resolution = getattr(self, "voxel_resolutions", {}).get(
                key, self.voxel_resolution
            )
            if cell_index.shape[-1] != max_points:
                raise ValueError(
                    "canonical sparse voxels must be padded to the cache's "
                    f"voxel_max_points ({max_points}), got "
                    f"{cell_index.shape[-1]}"
                )
            if colour.shape[:-1] != cell_index.shape or colour.shape[-1] != 3:
                raise ValueError(
                    f"voxel colour {colour.shape} does not match cell index "
                    f"{cell_index.shape}"
                )
            cells = SparseVoxels.cell_count(resolution)
            if cell_index.max(initial=-1) >= cells:
                raise ValueError("voxel cell index is outside the grid")

    def _transaction(self):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=self.lmdb_readahead,
                meminit=False,
                subdir=False,
                max_readers=2048,
            )
        if self._lmdb_txn is None:
            self._lmdb_txn = self._lmdb_env.begin(write=False, buffers=True)
        return self._lmdb_txn

    def _decode_image(self, transaction, key: str, index: int) -> np.ndarray:
        value = self._read_value(transaction, key, index)
        image_size = self.rgb_load_resolutions.get(
            canonical_camera_key(key), self.image_size
        )
        return CoreImages.decode_jpg_bytes(
            value, image_size=image_size, to_float=False, fmt="CHW"
        )

    def _decode_voxel(self, transaction, key: str, index: int):
        """Return one padded sparse voxel record."""
        index = int(index)
        _ = transaction
        offsets = getattr(self, "voxel_offsets_by_key", {}).get(
            key, self.voxel_offsets
        )
        cell_index = getattr(self, "voxel_index_by_key", {}).get(
            key, self.voxel_index
        )
        colour = getattr(self, "voxel_colour_by_key", {}).get(
            key, self.voxel_colour
        )
        max_points = getattr(self, "voxel_max_points_by_key", {}).get(
            key, self.voxel_max_points
        )
        start, end = int(offsets[index]), int(offsets[index + 1])
        return SparseVoxels.decode(
            cell_index[start:end],
            colour[start:end],
            max_points,
        )

    def _decode_point_cloud(self, transaction, key: str, index: int) -> np.ndarray:
        value = self._read_value(transaction, key, index)
        return np.frombuffer(bytes(value), dtype=np.float32).reshape(
            self.point_cloud_shape
        )

    @staticmethod
    def _read_value(transaction, key: str, index: int):
        lmdb_key = f"{key}/{int(index):08d}".encode("ascii")
        value = transaction.get(lmdb_key)
        if value is None:
            raise KeyError(f"Missing LMDB key: {lmdb_key!r}")
        return value
