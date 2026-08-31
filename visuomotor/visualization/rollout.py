"""Centralized rollout-video composition: focus, trajectory, and HUD.

All functions here are pure (frame in, frame out) and operate on HWC uint8 RGB
frames plus plain dict/array payloads -- no coupling to a specific env wrapper,
policy, or config system, so any rollout/render path can compose them.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import cv2
import numpy as np
import torch

from visuomotor.geometry import projection as Projection

# ---------------------------------------------------------------------------
# Shared world -> pixel projection
# ---------------------------------------------------------------------------

_TRAJECTORY_PAYLOAD_KEYS = {
    "action_positions",
    "eef_position",
}


def project_world_to_image(points, world_to_pixel, image_height, image_width):
    """Project world points to pixel (x, y), rejecting invalid/offscreen ones.

    ``points`` ends in 3 world coordinates, ``world_to_pixel`` is a ``(4, 4)``
    camera matrix (e.g. ``robosuite.utils.camera_utils.get_camera_transform_matrix``).
    Points that are non-finite, behind the camera, or land outside the image
    come back as NaN so callers can skip them without extra bookkeeping.
    """
    points = np.asarray(points, dtype=np.float64)
    shape = points.shape[:-1]
    if points.shape[-1:] != (3,):
        raise ValueError("points must end in three world coordinates")
    transform = np.asarray(world_to_pixel, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_to_pixel must have shape (4, 4)")

    flat = points.reshape(-1, 3)
    row_col = Projection.world_xyz_to_pixel_row_col(
        torch.from_numpy(flat),
        torch.from_numpy(transform),
        (image_height, image_width),
        clamp=False,
    ).numpy()
    pixels = row_col[:, (1, 0)]
    valid = (
        np.isfinite(pixels).all(axis=-1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image_width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image_height)
    )
    pixels[~valid] = np.nan
    return pixels.reshape(shape + (2,)), valid.reshape(shape)


def _point(xy):
    return tuple(np.rint(xy).astype(int))


# ---------------------------------------------------------------------------
# Focus-crop overlay: oracle target box + predicted box/points
# ---------------------------------------------------------------------------


def draw_oracle_box_overlay(
    frame: np.ndarray,
    *,
    pixel_row_col,
    projection_height: int,
    projection_width: int,
    zoom: float,
) -> np.ndarray:
    """Draw a box centered on the oracle object-ref's projected image point."""
    if pixel_row_col is None or zoom <= 0:
        return frame

    out = frame.copy()
    frame_h, frame_w = out.shape[:2]
    row = int(round(float(pixel_row_col[0]) * frame_h / max(projection_height, 1)))
    col = int(round(float(pixel_row_col[1]) * frame_w / max(projection_width, 1)))

    box_h = max(2, int(round(frame_h / zoom)))
    box_w = max(2, int(round(frame_w / zoom)))
    r0 = max(0, row - box_h // 2)
    r1 = min(frame_h - 1, row + box_h // 2)
    c0 = max(0, col - box_w // 2)
    c1 = min(frame_w - 1, col + box_w // 2)

    color = np.asarray([0, 255, 80], dtype=np.uint8)
    thickness = max(2, int(round(min(frame_h, frame_w) / 128)))
    out[r0 : min(r0 + thickness, frame_h), c0 : c1 + 1] = color
    out[max(r1 - thickness + 1, 0) : r1 + 1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0 : min(c0 + thickness, frame_w)] = color
    out[r0 : r1 + 1, max(c1 - thickness + 1, 0) : c1 + 1] = color

    dot = max(2, thickness + 1)
    out[
        max(0, row - dot) : min(frame_h, row + dot + 1),
        max(0, col - dot) : min(frame_w, col + dot + 1),
    ] = color
    return out


_FOCUS_SOURCE_COLORS = {
    "seeker": (0, 220, 255),
    "rvt2_heatmap": (255, 210, 0),
    "oracle": (0, 255, 80),
    "focus_refiner": (0, 220, 255),
}
_FOCUS_POINT_COLORS = [
    (255, 64, 64),
    (64, 192, 255),
    (255, 220, 64),
    (96, 255, 128),
    (224, 96, 255),
    (255, 144, 64),
    (64, 255, 224),
    (160, 160, 255),
]


def _scale_overlay_point(point, *, x_scale, y_scale, frame_w, frame_h):
    x = int(round(float(point[0]) * x_scale))
    y = int(round(float(point[1]) * y_scale))
    return int(np.clip(x, 0, frame_w - 1)), int(np.clip(y, 0, frame_h - 1))


def _draw_square(out, *, x, y, radius, color):
    frame_h, frame_w = out.shape[:2]
    cv2.rectangle(
        out,
        (max(0, x - radius), max(0, y - radius)),
        (min(frame_w - 1, x + radius), min(frame_h - 1, y + radius)),
        color,
        thickness=-1,
    )


def _draw_focus_box(out, *, box_px, label, color, thickness, x_scale, y_scale):
    frame_h, frame_w = out.shape[:2]
    box = np.asarray(box_px, dtype=np.float32).reshape(-1)
    if box.shape[0] != 4 or not np.all(np.isfinite(box)):
        return

    x0, y0 = _scale_overlay_point(box[:2], x_scale=x_scale, y_scale=y_scale, frame_w=frame_w, frame_h=frame_h)
    x1, y1 = _scale_overlay_point(box[2:], x_scale=x_scale, y_scale=y_scale, frame_w=frame_w, frame_h=frame_h)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness=thickness)
    cv2.putText(
        out,
        label,
        (x0, max(12, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _draw_focus_points(out, *, points_px, mean_point_index, radius, x_scale, y_scale):
    frame_h, frame_w = out.shape[:2]
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    mean_idx = None if mean_point_index is None else int(mean_point_index)
    for idx, point in enumerate(points):
        if not np.all(np.isfinite(point)):
            continue
        x, y = _scale_overlay_point(point, x_scale=x_scale, y_scale=y_scale, frame_w=frame_w, frame_h=frame_h)
        if mean_idx is not None and idx == mean_idx:
            _draw_square(out, x=x, y=y, radius=radius + 2, color=(0, 0, 0))
            _draw_square(out, x=x, y=y, radius=radius, color=(255, 255, 255))
            continue
        _draw_square(
            out, x=x, y=y, radius=radius, color=_FOCUS_POINT_COLORS[idx % len(_FOCUS_POINT_COLORS)]
        )


def draw_focus_overlay(frame: np.ndarray, items, *, render_view: str) -> np.ndarray:
    """Draw predicted/oracle focus boxes and points in render-frame pixels.

    ``items`` is a list of dicts, each optionally carrying ``view`` (skipped if
    it doesn't match ``render_view``), ``source`` (color key), ``source_size``
    (the square resolution the box/points were computed in), and ``box_px``
    and/or ``points_px`` (plus ``mean_point_index`` for points).
    """
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]
    thickness = max(2, int(round(min(frame_h, frame_w) / 128)))

    for item in items:
        if not isinstance(item, dict):
            continue
        item_view = item.get("view")
        if item_view is not None and str(item_view) != render_view:
            continue

        source = str(item.get("source", "pred"))
        color = _FOCUS_SOURCE_COLORS.get(source, (255, 80, 80))
        source_size = max(float(item.get("source_size", max(frame_h, frame_w))), 1.0)
        x_scale = float(frame_w - 1) / max(source_size - 1.0, 1.0)
        y_scale = float(frame_h - 1) / max(source_size - 1.0, 1.0)

        if "box_px" in item:
            _draw_focus_box(
                out,
                box_px=item["box_px"],
                label=source.upper(),
                color=color,
                thickness=thickness,
                x_scale=x_scale,
                y_scale=y_scale,
            )
        if "points_px" in item:
            _draw_focus_points(
                out,
                points_px=item["points_px"],
                mean_point_index=item.get("mean_point_index"),
                radius=max(2, thickness + 1),
                x_scale=x_scale,
                y_scale=y_scale,
            )
    return out


# ---------------------------------------------------------------------------
# Action-trajectory overlay
# ---------------------------------------------------------------------------

_AMBER = (255, 205, 70)
_SLATE = (135, 150, 165)
_PLANNED = (45, 225, 195)
_WHITE = (245, 248, 250)


class RolloutDiagnosticState:
    """Chunk-stable trajectory diagnostics for one rollout environment.

    A producer calls ``update`` once per replan with ``action_positions`` and
    ``eef_position``. The environment wrapper records actual positions and
    advances the action-chunk cursor between renders.
    """

    _REQUIRED_KEYS = _TRAJECTORY_PAYLOAD_KEYS

    def __init__(self):
        self.reset()

    def reset(self):
        self.payload = None
        self.action_index = 0
        self.eef_history = []
        self.replan_index = 0

    def update(self, payload=None):
        self.action_index = 0
        if payload is None or not self._REQUIRED_KEYS.issubset(payload):
            self.payload = None
            self.eef_history = []
            return
        payload = deepcopy(payload)
        payload["replan_index"] = self.replan_index
        self.replan_index += 1
        self.payload = payload
        eef = np.asarray(payload["eef_position"], dtype=np.float64)
        self.eef_history = [eef.copy()] if np.isfinite(eef).all() else []

    def observe(self, eef_position=None):
        """Record the actual post-step EEF position for trajectory comparison."""
        if self.payload is None or eef_position is None:
            return
        eef = np.asarray(eef_position, dtype=np.float64).reshape(-1)
        if eef.shape == (3,) and np.isfinite(eef).all():
            self.eef_history.append(eef.copy())

    def advance(self):
        if self.payload is not None:
            self.action_index += 1

    def current(self):
        if self.payload is None:
            return None
        payload = dict(self.payload)
        payload["action_index"] = self.action_index
        payload["executed_eef_positions"] = np.asarray(
            self.eef_history, dtype=np.float64
        ).reshape(-1, 3)
        actions = np.asarray(payload["action_positions"])
        if self.eef_history and len(actions):
            index = min(self.action_index, len(actions) - 1)
            payload["tracking_error"] = float(
                np.linalg.norm(self.eef_history[-1] - actions[index])
            )
        return payload


def draw_trajectory_overlay(frame: np.ndarray, payload: Optional[dict], world_to_pixel) -> np.ndarray:
    """Overlay the planned action-chunk path and the executed EEF path."""
    if payload is None or world_to_pixel is None or not _TRAJECTORY_PAYLOAD_KEYS.issubset(payload):
        return frame
    out = frame.copy()
    source_height, source_width = out.shape[:2]
    action_index = int(payload.get("action_index", 0))

    actions = np.asarray(payload["action_positions"])
    action_pixels, action_valid = project_world_to_image(
        actions, world_to_pixel, source_height, source_width
    )
    for index in range(max(len(action_pixels) - 1, 0)):
        if not (action_valid[index] and action_valid[index + 1]):
            continue
        color = _SLATE if index < action_index else _PLANNED
        cv2.line(out, _point(action_pixels[index]), _point(action_pixels[index + 1]), color, 2, cv2.LINE_AA)
    if len(action_pixels) and action_valid[min(action_index, len(action_pixels) - 1)]:
        current = action_pixels[min(action_index, len(action_pixels) - 1)]
        cv2.circle(out, _point(current), 4, _AMBER, -1, cv2.LINE_AA)

    executed = np.asarray(payload.get("executed_eef_positions", ()))
    if executed.size:
        executed = executed.reshape(-1, 3)
        executed_pixels, executed_valid = project_world_to_image(
            executed, world_to_pixel, source_height, source_width
        )
        for index in range(max(len(executed_pixels) - 1, 0)):
            if executed_valid[index] and executed_valid[index + 1]:
                cv2.line(
                    out,
                    _point(executed_pixels[index]),
                    _point(executed_pixels[index + 1]),
                    _WHITE,
                    1,
                    cv2.LINE_AA,
                )

    current_eef = executed[-1] if executed.size else np.asarray(payload["eef_position"])
    eef_pixel, eef_valid = project_world_to_image(
        current_eef[None], world_to_pixel, source_height, source_width
    )
    if eef_valid[0]:
        cv2.drawMarker(out, _point(eef_pixel[0]), _WHITE, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, payload: Optional[dict]) -> np.ndarray:
    """Append compact rollout diagnostics below the camera frame."""
    if payload is None:
        return frame
    height, width = frame.shape[:2]
    replan = int(payload.get("replan_index", 0))
    action = int(payload.get("action_index", 0))
    tracking_cm = 100 * float(payload.get("tracking_error", 0.0))
    values = [f"plan {replan}:{action}  |  track {tracking_cm:.1f} cm"]
    if "translation_spread_p90" in payload:
        spread_cm = 100 * float(payload["translation_spread_p90"])
        spread_deg = np.degrees(float(payload.get("rotation_spread_p90", 0.0)))
        values.append(f"spread p90  {spread_cm:.1f} cm  |  {spread_deg:.0f} deg")
    elif "anchor_jump" in payload:
        jump_cm = 100 * float(payload["anchor_jump"])
        values.append(f"jump  {jump_cm:.1f} cm")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.4, max(0.28, width / 640.0))
    thickness = 1
    text_sizes = [cv2.getTextSize(value, font, scale, thickness)[0] for value in values]
    line_height = max(size[1] for size in text_sizes) + 5
    footer_height = line_height * len(values) + 5
    out = np.empty((height + footer_height, width, frame.shape[2]), dtype=np.uint8)
    out[:height] = frame
    out[height:] = (16, 20, 24)
    for index, value in enumerate(values):
        cv2.putText(
            out,
            value,
            (5, height + line_height * (index + 1)),
            font,
            scale,
            _WHITE,
            thickness,
            cv2.LINE_AA,
        )
    return out
