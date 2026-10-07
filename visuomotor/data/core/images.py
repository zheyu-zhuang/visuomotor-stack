"""Image tensor conversion, byte codecs, normalization, and resizing."""

from dataclasses import dataclass
from typing import Literal, Mapping, Optional

import cv2
import imagecodecs
import numpy as np
import torch
import torch.nn.functional as F

from visuomotor.data.core import normalization as CoreNormalization

IMAGE_SOURCE_MODES = {"raw", "uint8", "float01", "imagenet"}


@dataclass(frozen=True)
class RGBCodec:
    """Lossless Blosc Zstd encoding for LMDB RGB frames."""

    name: Literal["blosc_zstd"] = "blosc_zstd"

    def __post_init__(self):
        if self.name != "blosc_zstd":
            raise ValueError(
                "RGB caches must declare rgb_codec='blosc_zstd'; "
                "re-render JPEG or unversioned caches from simulator states"
            )

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "RGBCodec":
        return cls(name=metadata.get("rgb_codec"))

    def metadata(self) -> dict:
        return {"rgb_codec": self.name}

    def encode(self, image: np.ndarray) -> bytes:
        """Encode one HWC RGB uint8 frame for LMDB."""
        _validate_rgb_pixels(image)
        return imagecodecs.blosc_encode(
            np.ascontiguousarray(image), level=3, compressor="zstd",
            typesize=1, shuffle=False, numthreads=1,
        )

    def decode(
        self, buf: bytes, *, render_resolution: int,
        load_resolution: Optional[int] = None,
    ) -> np.ndarray:
        """Decode a cache frame to canonical CHW RGB uint8."""
        pixels = imagecodecs.blosc_decode(buf, numthreads=1)
        resolution = int(render_resolution)
        if resolution < 1 or len(pixels) != resolution * resolution * 3:
            raise ValueError("RGB frame byte count does not match cache image_size")
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(resolution, resolution, 3)
        return _prepare_lossless_rgb(image, load_resolution)


def _validate_rgb_pixels(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB pixels must be HWC uint8 with three channels")


def _prepare_lossless_rgb(image: np.ndarray, load_resolution: Optional[int]) -> np.ndarray:
    _validate_rgb_pixels(image)
    if load_resolution is not None:
        resolution = int(load_resolution)
        if resolution < 1 or resolution > min(image.shape[:2]):
            raise ValueError("RGB load resolution must be positive and cannot upsample")
        if image.shape[:2] != (resolution, resolution):
            image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.moveaxis(image, -1, 0))


def canonical_rgb_from_source(
    image: np.ndarray,
    *,
    load_resolution: Optional[int],
) -> np.ndarray:
    """Prepare direct RGB pixels identically to a decoded lossless cache frame."""
    return _prepare_lossless_rgb(image, load_resolution)


def _range(x: torch.Tensor) -> tuple[float, float]:
    if x.numel() == 0:
        return 0.0, 0.0
    return float(x.min().item()), float(x.max().item())


def _require_source(source: str) -> str:
    source = str(source)
    if source not in IMAGE_SOURCE_MODES:
        raise ValueError(f"Invalid image source {source!r}; expected {IMAGE_SOURCE_MODES}")
    return source


def _require_float01(x: torch.Tensor, *, source: str) -> torch.Tensor:
    if not x.is_floating_point():
        raise TypeError(f"Expected {source} image tensor to be floating point")
    min_v, max_v = _range(x)
    if min_v < 0.0 or max_v > 1.0:
        raise ValueError(
            f"Expected {source} image tensor in [0, 1], got range "
            f"[{min_v:.4g}, {max_v:.4g}]"
        )
    return x


def _uint_to_float01(x: torch.Tensor, *, source: str) -> torch.Tensor:
    if x.is_floating_point():
        raise TypeError(f"Expected {source} image tensor to be integer/uint8")
    min_v, max_v = _range(x)
    if min_v < 0.0 or max_v > 255.0:
        raise ValueError(
            f"Expected {source} image tensor in [0, 255], got range "
            f"[{min_v:.4g}, {max_v:.4g}]"
        )
    return x.float().div(255.0)


def _raw_to_float01(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point():
        return _uint_to_float01(x, source="raw")
    return _require_float01(x, source="raw")


def image_to_float01(x: torch.Tensor, *, source: str = "raw") -> torch.Tensor:
    """Convert image tensors to float [0, 1] for visualization/augmentation."""
    source = _require_source(source)
    if source == "raw":
        x = _raw_to_float01(x)
    elif source == "uint8":
        x = _uint_to_float01(x, source=source)
    elif source == "float01":
        x = _require_float01(x, source=source)
    elif source == "imagenet":
        if not x.is_floating_point():
            raise TypeError("Expected imagenet image tensor to be floating point")
        x = CoreNormalization.Normalizer.denormalize_rgb(x)
    return x.clamp(0.0, 1.0)


def resize_image(image: torch.Tensor, out_res: int) -> torch.Tensor:
    """
    Resize image to out_res x out_res if needed.

    image: [N, 3, H, W]
    """
    _, _, H, W = image.shape
    if (H, W) != (out_res, out_res):
        image = F.interpolate(
            image,
            size=(out_res, out_res),
            mode="bilinear",
            align_corners=False,
        )
    return image
