"""Rollout media extraction, naming, and contact-sheet rendering."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch


def extract_start_frames(obs: dict, source_key: str) -> np.ndarray:
    """Return reset-time render frames as HWC uint8."""
    images = obs[source_key]
    if images.ndim != 5:
        raise ValueError(
            f"Expected {source_key} to have shape [B, T, C, H, W], "
            f"got {images.shape}"
        )

    # MultiStepWrapper repeats the reset observation across the time axis when needed.
    frames = images[:, 0]
    frames = np.moveaxis(frames, 1, -1)
    if frames.dtype == np.uint8:
        return frames.copy()
    if not np.issubdtype(frames.dtype, np.floating):
        raise ValueError(f"Expected uint8 or float RGB frames, got {frames.dtype}")
    return (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _annotate_rollout_tile(
    image: np.ndarray,
    *,
    success: bool,
    label: str,
) -> np.ndarray:
    frame = image.copy()
    color = (60, 180, 75) if success else (220, 80, 80)
    h, w = frame.shape[:2]
    border = max(2, min(h, w) // 100)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness=border)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.55, max(0.35, w / 640.0))
    thickness = 1 if scale < 0.5 else 2
    horizontal_pad = max(6, w // 50)
    status = "SUCCESS" if success else "FAIL"
    (status_w, text_h), baseline = cv2.getTextSize(
        status, font, scale, thickness
    )
    badge_pad = max(5, w // 64)
    badge_w = status_w + 2 * badge_pad
    max_text_w = max(w - 2 * horizontal_pad, 1)
    display_label = str(label)
    while display_label:
        text = display_label if display_label == label else display_label.rstrip() + "..."
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_text_w:
            display_label = text
            break
        display_label = display_label[:-1]
    if not display_label:
        display_label = "..."
        (tw, _), _ = cv2.getTextSize(display_label, font, scale, thickness)

    vertical_pad = max(4, h // 100)
    footer_h = text_h + baseline + 2 * vertical_pad
    out = np.empty((h + footer_h, w, image.shape[2]), dtype=np.uint8)
    out[:h] = frame
    out[h:] = (26, 29, 34)
    text_y = h + vertical_pad + text_h
    cv2.putText(
        out,
        display_label,
        (horizontal_pad, text_y),
        font,
        scale,
        (235, 238, 242),
        thickness,
        lineType=cv2.LINE_AA,
    )
    badge_x = w - border - badge_w
    badge_top = border
    badge_bottom = border + text_h + baseline + 2 * vertical_pad
    cv2.rectangle(
        out,
        (badge_x, badge_top),
        (w - border - 1, badge_bottom),
        color,
        thickness=-1,
    )
    cv2.putText(
        out,
        status,
        (badge_x + badge_pad, badge_top + vertical_pad + text_h),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return out


def make_image_grid(
    images: list[np.ndarray],
    pad: int = 2,
    *,
    success_flags: Optional[list[bool]] = None,
    labels: Optional[list[str]] = None,
) -> Optional[np.ndarray]:
    """Build a near-square HWC uint8 RGB contact sheet."""
    if len(images) == 0:
        return None

    first = images[0]
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {first.shape}")
    if (success_flags is None) != (labels is None):
        raise ValueError("success_flags and labels must be provided together")
    if success_flags is not None and (
        len(success_flags) != len(images) or len(labels) != len(images)
    ):
        raise ValueError("success_flags and labels must match images length")
    for idx, image in enumerate(images):
        if image.shape != first.shape:
            raise ValueError(
                f"All grid images must have the same shape. "
                f"Expected {first.shape}, got {image.shape} at index {idx}."
            )

    annotated = None
    if success_flags is not None and labels is not None:
        annotated = [
            _annotate_rollout_tile(image, success=bool(success), label=str(label))
            for image, success, label in zip(images, success_flags, labels)
        ]

    tiles = annotated if annotated is not None else images
    h, w, c = tiles[0].shape
    if len(images) == 50:
        rows, cols = 5, 10
    else:
        cols = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / cols)

    grid_h = rows * h + pad * max(rows - 1, 0)
    grid_w = cols * w + pad * max(cols - 1, 0)
    grid = np.full((grid_h, grid_w, c), (26, 29, 34), dtype=np.uint8)

    for idx, image in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = image

    return grid


def rollout_video_filename(*, prefix: str, seed: int, outcome: Optional[bool] = None) -> str:
    """Return a stable rollout filename, adding outcome only after closure."""
    split = str(prefix).strip("/") or "rollout"
    suffix = "" if outcome is None else ("_success" if outcome else "_fail")
    return f"{split}_seed_{int(seed)}{suffix}.mp4"


def extract_focus_diagnostics(action_dict: dict, *, n_envs: int) -> list[list[dict]]:
    """Return per-env focus overlays from explicit prediction diagnostics."""
    diagnostics = action_dict.get("diagnostics") or {}
    last_video_items = diagnostics.get("focus") or ()
    per_env = [[] for _ in range(n_envs)]
    if not last_video_items:
        return per_env

    for item in last_video_items:
        if not isinstance(item, dict):
            continue
        box_px = item.get("box_px")
        points_px = item.get("points_px")
        if box_px is None and points_px is None:
            continue

        box_arr = _to_numpy_or_none(box_px)
        points_arr = _to_numpy_or_none(points_px)
        batch_count = [
            int(arr.shape[0])
            for arr in (box_arr, points_arr)
            if arr is not None and arr.ndim > 0
        ]
        if not batch_count:
            continue

        count = min(n_envs, *batch_count)
        for env_idx in range(count):
            overlay = {
                "source": str(item.get("source", "pred")),
                "view": str(item.get("view", "external")),
                "source_size": int(item.get("source_size", 224)),
            }
            if box_arr is not None:
                overlay["box_px"] = (
                    box_arr[env_idx].astype(np.float32, copy=False).tolist()
                )
            if points_arr is not None:
                overlay["points_px"] = (
                    points_arr[env_idx].astype(np.float32, copy=False).tolist()
                )
                if "mean_point_index" in item:
                    overlay["mean_point_index"] = int(item["mean_point_index"])
            per_env[env_idx].append(overlay)
    return per_env


def extract_rollout_diagnostics(
    action_dict: dict,
    *,
    action_positions,
    eef_positions,
    n_envs: int,
) -> list[Optional[dict]]:
    """Build per-environment trajectory payloads."""
    positions = _to_numpy_or_none(action_positions)
    eef = _to_numpy_or_none(eef_positions)
    count = min(n_envs, int(positions.shape[0]), int(eef.shape[0]))
    payloads: list[Optional[dict]] = [None] * n_envs
    for env_idx in range(count):
        payload = {
            "action_positions": positions[env_idx],
            "eef_position": eef[env_idx],
        }
        payloads[env_idx] = payload
    return payloads


def _to_numpy_or_none(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)
