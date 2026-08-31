"""Compressed local media storage and optional publishing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import numpy as np
from PIL import Image

ArtifactKind = Literal["image", "video"]


@dataclass(frozen=True)
class ArtifactRecord:
    key: str
    kind: ArtifactKind
    path: Path
    caption: str = ""


def allocate_rollout_output(checkpoint: str | Path, output_dir: Optional[str | Path] = None) -> Path:
    """Allocate a standalone rollout directory without overwriting prior runs."""
    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    checkpoint = Path(checkpoint).expanduser().resolve()
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    for number in range(1, 10000):
        candidate = run_dir.parent / f"{run_dir.name}_rollout_{number:04d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate rollout output beside {run_dir}")


def _image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"image must be HWC RGB/RGBA, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class ArtifactStore:
    """Own stable artifact paths and encode each local artifact exactly once."""

    def __init__(
        self,
        root: str | Path,
        *,
        save_images: bool = True,
        save_videos: bool = True,
        webp_quality: int = 85,
    ) -> None:
        self.root = Path(root).expanduser().resolve() / "media"
        self.save_images = bool(save_images)
        self.save_videos = bool(save_videos)
        self.webp_quality = int(webp_quality)
        if not 1 <= self.webp_quality <= 100:
            raise ValueError("WebP quality must be in [1, 100]")

    def training_image(self, name: str = "augmentation_preview") -> Path:
        return self.root / "training" / f"{name}.webp"

    def eval_image(self, category: str, *, epoch: int, step: int) -> Path:
        return self.root / "eval" / category / f"epoch_{epoch:04d}_step_{step:08d}.webp"

    def rollout_dir(self, epoch: Optional[int]) -> Path:
        label = "unknown" if epoch is None else f"{int(epoch):04d}"
        return self.root / "rollout" / f"epoch_{label}"

    def rollout_video(self, *, epoch: Optional[int], split: str, seed: int) -> Path:
        split = str(split).strip("/") or "rollout"
        return self.rollout_dir(epoch) / f"{split}_seed_{int(seed)}.mp4"

    def rollout_summary(self, *, epoch: Optional[int]) -> Path:
        return self.rollout_dir(epoch) / "summary.webp"

    def save_image(
        self,
        image,
        path: str | Path,
        *,
        key: str,
        caption: str = "",
    ) -> Optional[ArtifactRecord]:
        if not self.save_images:
            return None
        path = Path(path)
        if path.suffix.lower() != ".webp":
            raise ValueError("diagnostic images must use .webp")
        path.parent.mkdir(parents=True, exist_ok=True)
        _image(image).save(path, format="WEBP", quality=self.webp_quality, method=6)
        return ArtifactRecord(key=key, kind="image", path=path.resolve(), caption=caption)

    def video_record(
        self,
        path: str | Path,
        *,
        key: str,
        caption: str = "",
    ) -> Optional[ArtifactRecord]:
        if not self.save_videos:
            return None
        path = Path(path)
        if path.suffix.lower() != ".mp4":
            raise ValueError("rollout videos must use .mp4")
        if not path.is_file():
            raise FileNotFoundError(path)
        return ArtifactRecord(key=key, kind="video", path=path.resolve(), caption=caption)


def publish_artifacts(
    run,
    records: Iterable[ArtifactRecord],
    *,
    upload_images: bool,
    upload_videos: bool,
    step: Optional[int] = None,
    image_factory: Optional[Callable] = None,
    video_factory: Optional[Callable] = None,
) -> dict:
    """Publish compressed local files, choosing one stable rollout video."""
    records = tuple(records)
    if not upload_images and not upload_videos:
        return {}
    if image_factory is None or video_factory is None:
        import wandb

        image_factory = image_factory or wandb.Image
        video_factory = video_factory or wandb.Video

    payload = {}
    if upload_images:
        for record in records:
            if record.kind == "image":
                payload[record.key] = image_factory(str(record.path), caption=record.caption)
    if upload_videos:
        videos = sorted(
            (record for record in records if record.kind == "video"),
            key=lambda record: (record.key, str(record.path)),
        )
        if videos:
            record = videos[0]
            payload[record.key] = video_factory(
                str(record.path), format="mp4", caption=record.caption
            )
    if payload:
        run.log(payload, step=step)
    return payload
