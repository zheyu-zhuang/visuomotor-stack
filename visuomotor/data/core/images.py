"""Image tensor conversion, byte codecs, normalization, and resizing."""

import io
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from visuomotor.data.core import normalization as CoreNormalization

IMAGE_SOURCE_MODES = {"raw", "uint8", "float01", "imagenet"}
ImageFormat = Literal["HWC", "CHW"]

# The cache's RGB codec setting. Rollout replays it inline, so it is the one
# place either path can change it from.
JPEG_QUALITY_DEFAULT = 90


def decode_jpg_bytes(
    buf: bytes,
    image_size: Optional[int] = None,
    *,
    bgr_to_rgb: bool = False,
    to_float: bool = True,
    fmt: ImageFormat = "CHW",
) -> np.ndarray:
    """Decode JPEG bytes, optionally resize, recolor, and change layout."""
    native_decode = image_size is None
    if image_size is not None:
        try:
            with Image.open(io.BytesIO(buf)) as decoded:
                native_decode = decoded.size == (image_size, image_size)
                if not native_decode:
                    decoded.draft("RGB", (image_size, image_size))
                    if decoded.size != (image_size, image_size):
                        decoded = decoded.resize(
                            (image_size, image_size), resample=Image.Resampling.BOX
                        )
                    image = np.asarray(decoded)
        except (OSError, ValueError) as error:
            raise ValueError("PIL JPEG decode failed (buffer may be corrupted)") from error

    if native_decode:
        image = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode failed (buffer may be corrupted)")
        if bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        if not bgr_to_rgb:
            image = image[..., ::-1].copy()
    if to_float:
        image = image.astype(np.float32) / 255.0
    if fmt == "CHW":
        image = np.moveaxis(image, -1, 0)
    elif fmt != "HWC":
        raise ValueError(f"unsupported image format: {fmt}")
    return image


def encode_rgb_to_jpg_bytes(
    image: np.ndarray, quality: int = JPEG_QUALITY_DEFAULT
) -> bytes:
    """Encode an HWC uint8 RGB image as JPEG bytes."""
    image = image.astype(np.uint8, copy=False)
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buffer.tobytes()


def canonical_rgb_from_source(
    image: np.ndarray,
    *,
    load_resolution: Optional[int],
    quality: int = JPEG_QUALITY_DEFAULT,
) -> np.ndarray:
    """Run an HWC uint8 source frame through the cache codec to canonical CHW uint8.

    Dataset loading splits these two halves across the cache write and the
    cache read; a rollout has no cache in between and runs both here. Sharing
    the call is what makes a rendered frame reach the policy through the same
    JPEG quantization and the same decode-time resampling either way.
    """
    return decode_jpg_bytes(
        encode_rgb_to_jpg_bytes(image, quality=quality),
        image_size=load_resolution,
        to_float=False,
        fmt="CHW",
    )


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
