"""Shared voxel and point-cloud producer contracts."""

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Tuple, Union


def _plain(data: Mapping) -> dict:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in data.items()
    }


def _validate_bounds(label, bounds_min, bounds_max) -> None:
    if (bounds_min is None) != (bounds_max is None):
        raise ValueError(f"{label} bounds_min/bounds_max must both be set or both be None")
    if bounds_min is None:
        return
    if len(bounds_min) != 3 or len(bounds_max) != 3:
        raise ValueError(f"{label} bounds must be 3D")
    if any(low >= high for low, high in zip(bounds_min, bounds_max)):
        raise ValueError(f"each {label} lower bound must be below its upper bound")


@dataclass(frozen=True)
class VoxelProducerSpec:
    cameras: Tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    output_key: str = "voxel"
    frame: str = "world"
    resolution: Tuple[int, int, int] = (64, 64, 64)
    channels: Tuple[str, ...] = ("occupancy", "R", "G", "B")
    ws_size: float = 0.6
    bounds_min: Optional[Tuple[float, float, float]] = None
    bounds_max: Optional[Tuple[float, float, float]] = None
    reconstruction_resolution: int = 84

    def __post_init__(self) -> None:
        if not self.output_key:
            raise ValueError("voxel output_key cannot be empty")
        if self.frame not in ("world", "eef_centered"):
            raise ValueError(f"unknown voxel frame {self.frame!r}")
        if len(self.resolution) != 3 or any(
            int(value) < 2 for value in self.resolution
        ):
            raise ValueError("voxel resolution must contain three values >= 2")
        if self.channels != ("occupancy", "R", "G", "B"):
            raise ValueError("voxel producers require occupancy/R/G/B channels")
        if float(self.ws_size) <= 0:
            raise ValueError("voxel ws_size must be positive")
        _validate_bounds("voxel", self.bounds_min, self.bounds_max)
        if not self.cameras:
            raise ValueError("at least one voxel camera is required")
        if int(self.reconstruction_resolution) < 1:
            raise ValueError("voxel reconstruction_resolution must be positive")
        if self.frame == "eef_centered" and self.bounds_min is not None:
            half = self.ws_size / 2.0
            if tuple(self.bounds_min) != (-half, -half, -half) or tuple(
                self.bounds_max
            ) != (half, half, half):
                raise ValueError(
                    "EEF-centered voxel bounds must be symmetric about the tool origin"
                )

    @property
    def observation_shape(self) -> Tuple[int, int, int, int]:
        return (len(self.channels),) + self.resolution

    def declaration(self) -> dict:
        """Return the complete keyed semantic producer declaration."""
        return _plain(
            {
                "output_key": self.output_key,
                "frame": self.frame,
                "resolution": self.resolution,
                "bounds_min": self.bounds_min,
                "bounds_max": self.bounds_max,
                "ws_size": self.ws_size,
                "cameras": self.cameras,
                "channels": self.channels,
                "reconstruction_resolution": self.reconstruction_resolution,
            }
        )

    def metadata(self) -> dict:
        """Return the persistent cache metadata, preserving the legacy world grid."""
        data = {
            "output_key": self.output_key,
            "frame": self.frame,
            "resolution": self.resolution,
            "bounds_min": self.bounds_min,
            "bounds_max": self.bounds_max,
            "ws_size": self.ws_size,
            "cameras": self.cameras,
            "channels": self.channels,
            "reconstruction_resolution": self.reconstruction_resolution,
        }
        if self.output_key == "voxel" and self.frame == "world":
            data.pop("output_key")
            data.pop("frame")
        return _plain(data)

    def validate_metadata(self, metadata: Mapping) -> None:
        actual = _plain(metadata)
        if actual not in (self.metadata(), self.declaration()):
            raise ValueError(
                "voxel metadata mismatch: expected "
                f"{self.metadata()} (or keyed {self.declaration()}), got {actual}"
            )


@dataclass(frozen=True)
class PointCloudProducerSpec:
    cameras: Tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    output_key: str = "point_cloud"
    num_points: int = 1024
    channels: Tuple[str, ...] = ("x", "y", "z", "R", "G", "B")
    ws_size: float = 0.6
    table_margin: float = 0.005
    bounds_min: Optional[Tuple[float, float, float]] = None
    bounds_max: Optional[Tuple[float, float, float]] = None
    reconstruction_resolution: int = 84

    def __post_init__(self) -> None:
        if self.output_key != "point_cloud":
            raise ValueError("point-cloud output_key must be 'point_cloud'")
        if int(self.num_points) < 1:
            raise ValueError("point cloud num_points must be >= 1")
        if self.channels != ("x", "y", "z", "R", "G", "B"):
            raise ValueError("point-cloud producers require XYZRGB channels")
        if float(self.ws_size) <= 0:
            raise ValueError("point cloud ws_size must be positive")
        if float(self.table_margin) <= 0:
            raise ValueError("point cloud table_margin must be positive")
        _validate_bounds("point cloud", self.bounds_min, self.bounds_max)
        if not self.cameras:
            raise ValueError("at least one point cloud camera is required")
        if int(self.reconstruction_resolution) < 1:
            raise ValueError("point cloud reconstruction_resolution must be positive")

    @property
    def observation_shape(self) -> Tuple[int, int]:
        return (self.num_points, len(self.channels))

    def declaration(self) -> dict:
        """Return the complete semantic producer declaration."""
        return _plain(asdict(self))

    def metadata(self) -> dict:
        """Return persistent cache metadata without the canonical output key."""
        return _plain(
            {
                "num_points": self.num_points,
                "bounds_min": self.bounds_min,
                "bounds_max": self.bounds_max,
                "ws_size": self.ws_size,
                "table_margin": self.table_margin,
                "cameras": self.cameras,
                "channels": self.channels,
                "reconstruction_resolution": self.reconstruction_resolution,
            }
        )

    def validate_metadata(self, metadata: Mapping) -> None:
        actual = _plain(metadata)
        expected = self.metadata()
        if actual != expected:
            raise ValueError(
                f"point cloud metadata mismatch: expected {expected}, got {actual}"
            )


SpatialProducerSpec = Union[VoxelProducerSpec, PointCloudProducerSpec]
