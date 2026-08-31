"""Pure renderers for experiment diagnostics."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from visuomotor.data.core import images as CoreImages
from visuomotor.geometry import representation as Representation
from visuomotor.geometry import rigid as Rigid
from visuomotor.visualization import rollout as RolloutOverlays


def to_rgb(value, *, source: str = "raw") -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim != 3:
            raise ValueError(f"RGB tensor must be CHW, got {tuple(value.shape)}")
        value = CoreImages.image_to_float01(value, source=source)
        array = value.permute(1, 2, 0).numpy()
    else:
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"RGB array must be HWC, got {array.shape}")
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
    return Image.fromarray(
        np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    )


def image_grid(
    panels: Sequence[Image.Image | np.ndarray],
    *,
    labels: Optional[Sequence[str]] = None,
    columns: Optional[int] = None,
    padding: int = 6,
    background=(26, 29, 34),
    footer: Optional[str] = None,
) -> Image.Image:
    if not panels:
        raise ValueError("image grid needs at least one panel")
    panels = [panel.convert("RGB") if isinstance(panel, Image.Image) else to_rgb(panel) for panel in panels]
    if labels is not None and len(labels) != len(panels):
        raise ValueError("labels must match panels")
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    label_height = 0
    if labels is not None:
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        label_height = max(
            measure.multiline_textbbox((0, 0), str(label), spacing=2)[3]
            for label in labels
        ) + 8
    columns = columns or min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    footer_height = 0
    if footer:
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        footer_height = measure.textbbox((0, 0), footer)[3] + 2 * padding
    canvas = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * padding,
            rows * (height + label_height) + (rows + 1) * padding + footer_height,
        ),
        color=background,
    )
    draw = ImageDraw.Draw(canvas)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x = padding + column * (width + padding)
        y = padding + row * (height + label_height + padding)
        canvas.paste(panel, (x, y))
        if labels is not None:
            draw.multiline_text(
                (x + 3, y + height + 3),
                str(labels[index]),
                fill=(240, 240, 240),
                spacing=2,
            )
    if footer:
        footer_width = draw.textbbox((0, 0), footer)[2]
        draw.text(
            ((canvas.width - footer_width) // 2, canvas.height - footer_height + padding),
            footer,
            fill=(190, 196, 204),
        )
    return canvas


def render_rgb_observations(
    observations: Mapping[str, torch.Tensor], *, num_samples: int, source: str = "raw"
) -> Image.Image:
    rows = []
    for key in sorted(key for key in observations if key.startswith("rgb_")):
        tensor = observations[key]
        if tensor.ndim == 4:
            tensor = tensor[:, None]
        panels = []
        for sample in range(min(num_samples, tensor.shape[0])):
            for time in range(tensor.shape[1]):
                panels.append(to_rgb(tensor[sample, time], source=source))
        rows.append((key.removeprefix("rgb_").replace("_", " "), panels))

    return _labelled_rows(rows, empty_message="RGB observation grid needs at least one camera view")


def _labelled_rows(rows, *, empty_message: str) -> Image.Image:
    if not rows:
        raise ValueError(empty_message)
    tile_width = max(panel.width for _, panels in rows for panel in panels)
    tile_height = max(panel.height for _, panels in rows for panel in panels)
    columns = max(len(panels) for _, panels in rows)
    padding = 6
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_width = max(measure.textbbox((0, 0), label)[2] for label, _ in rows) + 12
    canvas = Image.new(
        "RGB",
        (
            label_width + columns * tile_width + (columns + 1) * padding,
            len(rows) * tile_height + (len(rows) + 1) * padding,
        ),
        (26, 29, 34),
    )
    draw = ImageDraw.Draw(canvas)
    for row, (label, panels) in enumerate(rows):
        y = padding + row * (tile_height + padding)
        label_box = draw.textbbox((0, 0), label)
        draw.text(
            (6, y + (tile_height - (label_box[3] - label_box[1])) // 2),
            label,
            fill=(190, 196, 204),
        )
        for column, panel in enumerate(panels):
            x = label_width + padding + column * (tile_width + padding)
            canvas.paste(
                panel,
                (x + (tile_width - panel.width) // 2, y + (tile_height - panel.height) // 2),
            )
    return canvas


def stack_sections(
    sections: Sequence[tuple[str, Image.Image]],
    *,
    padding: int = 8,
    background=(26, 29, 34),
) -> Image.Image:
    """Stack naturally sized diagnostic sections under compact headings."""
    if not sections:
        raise ValueError("section stack needs at least one image")
    heading_height = 18
    width = max(image.width for _, image in sections) + 2 * padding
    height = padding + sum(
        heading_height + image.height + padding for _, image in sections
    )
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    y = padding
    for title, image in sections:
        draw.text((padding, y + 2), title, fill=(220, 224, 230))
        y += heading_height
        canvas.paste(image, (padding, y))
        y += image.height + padding
    return canvas


def render_observation_strip(
    observations: Mapping[str, torch.Tensor],
    *,
    num_samples: int,
    rgb_source: str = "raw",
    tile_size: int = 96,
) -> Image.Image:
    """Render one unlabeled sample row for augmentation comparison."""
    panels = []
    rgb_keys = sorted(key for key in observations if key.startswith("rgb_"))
    if rgb_keys:
        tensor = observations[rgb_keys[0]]
        if tensor.ndim == 4:
            tensor = tensor[:, None]
        panels = [
            to_rgb(tensor[sample, -1], source=rgb_source)
            for sample in range(min(num_samples, tensor.shape[0]))
        ]
    elif "voxel" in observations:
        array = observations["voxel"].detach().cpu().numpy()
        if array.ndim == 5:
            array = array[:, None]
        panels = [
            to_rgb(np.rot90(_voxel_projection(array[sample, -1], (0, 1)), k=1))
            for sample in range(min(num_samples, array.shape[0]))
        ]
    elif "point_cloud" in observations:
        array = observations["point_cloud"].detach().cpu().numpy()
        if array.ndim == 3:
            array = array[:, None]
        panels = [
            _point_projection(array[sample, -1], (0, 1), size=tile_size)
            for sample in range(min(num_samples, array.shape[0]))
        ]
    if not panels:
        raise ValueError("no visual observation found")
    panels = [panel.resize((tile_size, tile_size), Image.Resampling.NEAREST) for panel in panels]
    return image_grid(panels, columns=len(panels), padding=5)


def sample_row_report(
    rows: Sequence[tuple[str, Image.Image]],
    *,
    padding: int = 7,
    background=(26, 29, 34),
) -> Image.Image:
    """Lay out stage headers at left and unlabeled samples across three rows."""
    if not rows:
        raise ValueError("sample-row report needs at least one row")
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    header_width = max(measure.textbbox((0, 0), title)[2] for title, _ in rows) + 14
    content_width = max(image.width for _, image in rows)
    width = header_width + content_width + 3 * padding
    height = padding + sum(image.height + padding for _, image in rows)
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    y = padding
    for title, image in rows:
        bounds = draw.textbbox((0, 0), title)
        draw.text(
            (padding, y + (image.height - (bounds[3] - bounds[1])) // 2),
            title,
            fill=(220, 224, 230),
        )
        canvas.paste(image, (header_width + 2 * padding, y))
        y += image.height + padding
    return canvas


def _voxel_projection(voxel: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    occupancy = voxel[0] > 0
    colors = np.moveaxis(voxel[1:4], 0, -1).astype(np.float32)
    if colors.max(initial=0) > 1.0:
        colors /= 255.0
    depth_axis = ({0, 1, 2} - set(axes)).pop()
    weights = occupancy.astype(np.float32)
    count = weights.sum(axis=depth_axis)
    color = (colors * weights[..., None]).sum(axis=depth_axis)
    color /= np.maximum(count[..., None], 1.0)
    return np.flipud(np.clip(color, 0.0, 1.0))


def render_voxel_observations(voxel: torch.Tensor, *, num_samples: int) -> Image.Image:
    array = voxel.detach().cpu().numpy()
    if array.ndim == 5:
        array = array[:, None]
    rows = []
    for axes, name in (((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")):
        panels = []
        for sample in range(min(num_samples, array.shape[0])):
            for time in range(array.shape[1]):
                panels.append(to_rgb(_voxel_projection(array[sample, time], axes)))
        rows.append((name, panels))
    return _labelled_rows(rows, empty_message="voxel grid needs at least one sample")


def _point_projection(points: np.ndarray, axes: tuple[int, int], size: int = 256) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (18, 21, 25))
    draw = ImageDraw.Draw(canvas)
    xyz = points[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    xyz, points = xyz[finite], points[finite]
    if not len(xyz):
        return canvas
    coordinates = xyz[:, axes]
    low, high = coordinates.min(0), coordinates.max(0)
    normalized = (coordinates - low) / np.maximum(high - low, 1e-6)
    pixels = np.rint(normalized * (size - 9) + 4).astype(int)
    pixels[:, 1] = size - 1 - pixels[:, 1]
    colors = points[:, 3:6] if points.shape[1] >= 6 else np.ones((len(points), 3))
    if colors.max(initial=0) <= 1.0:
        colors = colors * 255.0
    for pixel, color in zip(pixels, colors.astype(np.uint8)):
        draw.point(tuple(pixel), fill=tuple(int(value) for value in color))
    return canvas


def render_point_cloud_observations(points: torch.Tensor, *, num_samples: int) -> Image.Image:
    array = points.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[:, None]
    rows = []
    for axes, name in (((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")):
        panels = []
        for sample in range(min(num_samples, array.shape[0])):
            for time in range(array.shape[1]):
                panels.append(_point_projection(array[sample, time], axes))
        rows.append((name, panels))
    return _labelled_rows(rows, empty_message="point-cloud grid needs at least one sample")


def render_observations(
    observations: Mapping[str, torch.Tensor], *, num_samples: int = 6, rgb_source: str = "raw"
) -> Image.Image:
    result = None
    if any(key.startswith("rgb_") for key in observations):
        result = render_rgb_observations(
            observations, num_samples=num_samples, source=rgb_source
        )
    elif "voxel" in observations:
        result = render_voxel_observations(
            observations["voxel"], num_samples=num_samples
        )
    elif "point_cloud" in observations:
        result = render_point_cloud_observations(
            observations["point_cloud"], num_samples=num_samples
        )
    if result is None:
        raise ValueError("no visual observation found")
    context = []
    for key in ("eef_pos", "gripper_qpos"):
        if key in observations:
            source = observations[key]
            value = source.detach().cpu().reshape(
                source.shape[0], -1, source.shape[-1]
            )[0, -1]
            context.append(
                f"{key}=" + ",".join(f"{float(item):.3f}" for item in value)
            )
    if context:
        canvas = Image.new("RGB", (result.width, result.height + 24), (18, 21, 25))
        ImageDraw.Draw(canvas).text((6, 6), " | ".join(context), fill="white")
        canvas.paste(result, (0, 24))
        result = canvas
    return result


def draw_patch_diagnostic(
    *, image: torch.Tensor, prob_grid: torch.Tensor, target_patch: int,
    pred_patch: int, source_frame: int, target_frame: int, alpha: float,
) -> Image.Image:
    base = CoreImages.image_to_float01(image.detach().cpu(), source="float01")
    height, width = map(int, base.shape[-2:])
    heat = F.interpolate(prob_grid.detach().float().cpu()[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    heat = heat / heat.max().clamp_min(1e-12)
    overlay = base.clone()
    overlay[0] = overlay[0] * (1 - alpha * heat) + alpha * heat
    overlay[1:] *= 1 - alpha * heat[None]
    panel = to_rgb(overlay, source="float01")
    draw = ImageDraw.Draw(panel)
    grid_size = int(prob_grid.shape[0])
    for patch, color in ((target_patch, "lime"), (pred_patch, "red")):
        row, column = divmod(int(patch), grid_size)
        box = (column * width / grid_size, row * height / grid_size, (column + 1) * width / grid_size - 1, (row + 1) * height / grid_size - 1)
        draw.rectangle(box, outline=color, width=2)
    draw.text((4, 4), f"{source_frame}->{target_frame}", fill="white")
    return panel


def _as_action_batch(value) -> np.ndarray:
    value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if value.ndim < 3:
        raise ValueError(f"actions must have shape [...,T,D], got {value.shape}")
    return value.reshape(-1, value.shape[-2], value.shape[-1])


def _action_errors(predicted: np.ndarray, target: np.ndarray):
    position = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=-1)
    rotation = None
    if predicted.shape[-1] >= 9 and target.shape[-1] >= 9:
        pred_rotation = Representation.rot6d_to_mat(
            torch.from_numpy(predicted[:, 3:9]).float()
        )
        target_rotation = Representation.rot6d_to_mat(
            torch.from_numpy(target[:, 3:9]).float()
        )
        rotation = torch.rad2deg(
            Rigid.geodesic_angle(pred_rotation, target_rotation)
        ).numpy()
    gripper = None
    if predicted.shape[-1] >= 10 and target.shape[-1] >= 10:
        gripper = np.abs(predicted[:, -1] - target[:, -1])
    return position, rotation, gripper


def _plot_series(draw, box, series, colors, *, zero_floor=False):
    left, top, right, bottom = box
    finite = np.concatenate(
        [np.asarray(values, dtype=np.float32)[np.isfinite(values)] for values in series]
    )
    if not len(finite):
        return
    low = min(0.0, float(finite.min())) if zero_floor else float(finite.min())
    high = float(finite.max())
    if high - low < 1e-8:
        padding = max(abs(high) * 0.05, 1e-3)
        low, high = low - padding, high + padding
    draw.line((left, bottom, right, bottom), fill=(70, 76, 84), width=1)
    for values, color in zip(series, colors):
        values = np.asarray(values, dtype=np.float32)
        denominator = max(len(values) - 1, 1)
        points = [
            (
                round(left + index / denominator * (right - left)),
                round(bottom - (float(value) - low) / (high - low) * (bottom - top)),
            )
            for index, value in enumerate(values)
            if np.isfinite(value)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)


def _action_timeline_panel(predicted, target, *, width=180, height=96):
    canvas = Image.new("RGB", (width, height), (18, 21, 25))
    draw = ImageDraw.Draw(canvas)
    position, rotation, _ = _action_errors(predicted, target)
    bands = [("position error", position * 1000, ((110, 205, 255),), "mm")]
    if rotation is not None:
        bands.append(("rotation error", rotation, ((235, 130, 255),), "deg"))
    if predicted.shape[-1] >= 10 and target.shape[-1] >= 10:
        bands.append(
            (
                "gripper command",
                (target[:, -1], predicted[:, -1]),
                ((80, 220, 100), (255, 105, 90)),
                "",
            )
        )
    band_height = height // len(bands)
    for index, (label, values, colors, unit) in enumerate(bands):
        values = values if isinstance(values, tuple) else (values,)
        top = index * band_height
        peak = max(float(np.nanmax(np.abs(value))) for value in values)
        suffix = f"  max {peak:.0f}{unit}" if unit else ""
        draw.text((4, top + 1), f"{label}{suffix}", fill=(190, 196, 204))
        _plot_series(
            draw,
            (4, top + 13, width - 5, min(top + band_height - 3, height - 3)),
            values,
            colors,
            zero_floor=bool(unit),
        )
    return canvas


def _action_summary(predicted, target, sample):
    position, rotation, gripper = _action_errors(predicted, target)
    lines = [
        f"sample {sample}",
        f"position  {position.mean() * 100:.1f} / {position.max() * 100:.1f} cm  mean / max",
    ]
    if rotation is not None:
        lines.append(
            f"rotation  {rotation.mean():.0f} / {rotation.max():.0f} deg  mean / max"
        )
    elif gripper is not None:
        lines.append(f"gripper  {gripper.mean():.3f} MAE")
    return "\n".join(lines)


def _render_action_components(predicted, target, *, num_samples, action_rep):
    panels, labels = [], []
    count = min(num_samples, len(predicted), len(target))
    for sample in range(count):
        position_names = (
            ("dx", "dy", "dz") if action_rep == "delta" else ("x", "y", "z")
        )
        rows = [
            (
                f"{position_names[axis]} (m)",
                (target[sample, :, axis], predicted[sample, :, axis]),
                ((80, 220, 100), (255, 105, 90)),
                False,
            )
            for axis in range(min(3, predicted.shape[-1]))
        ]
        _, rotation, _ = _action_errors(predicted[sample], target[sample])
        if rotation is not None:
            rows.append(("rot err (deg)", (rotation,), ((235, 130, 255),), True))
        if predicted.shape[-1] >= 10 and target.shape[-1] >= 10:
            rows.append(
                (
                    "gripper",
                    (target[sample, :, -1], predicted[sample, :, -1]),
                    ((80, 220, 100), (255, 105, 90)),
                    False,
                )
            )
        row_height = 42
        canvas = Image.new(
            "RGB", (360, 10 + len(rows) * row_height + 18), (18, 21, 25)
        )
        draw = ImageDraw.Draw(canvas)
        for row, (name, values, colors, zero_floor) in enumerate(rows):
            top = 8 + row * row_height
            draw.text((6, top + 13), name, fill=(190, 196, 204))
            _plot_series(
                draw,
                (62, top, 352, top + 33),
                values,
                colors,
                zero_floor=zero_floor,
            )
        draw.text((62, canvas.height - 15), "horizon  0 ->", fill=(150, 158, 168))
        panels.append(canvas)
        labels.append(_action_summary(predicted[sample], target[sample], sample))
    return image_grid(
        panels,
        labels=labels,
        columns=2,
        footer=f"{action_rep} action components  |  target green  |  prediction coral",
    )


def render_action_comparison(
    predicted,
    target,
    *,
    num_samples: int = 6,
    action_rep: str = "absolute",
    observations: Optional[Mapping[str, torch.Tensor]] = None,
    voxel_geometry=None,
) -> Image.Image:
    """Compare physical action trajectories without plotting rot6d as coordinates."""
    predicted = _as_action_batch(predicted)
    target = _as_action_batch(target)
    if predicted.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"predicted and target actions must match, got {predicted.shape} and {target.shape}"
        )
    if (
        action_rep == "absolute"
        and predicted.shape[-1] >= 3
        and observations is not None
        and "voxel" in observations
        and voxel_geometry is not None
    ):
        return _render_action_pose_overlay(
            predicted,
            target,
            observations,
            voxel_geometry,
            num_samples=num_samples,
        )
    return _render_action_components(
        predicted, target, num_samples=num_samples, action_rep=action_rep
    )


_FOCUS_HEAD_COLORS = (
    (255, 64, 64),
    (255, 220, 64),
    (96, 255, 128),
    (224, 96, 255),
    (255, 144, 64),
    (255, 96, 192),
    (176, 255, 64),
    (192, 96, 64),
)


def _focus_top_p_centers(attention, centers, top_p=0.7):
    probability = attention.reshape(attention.shape[0], -1).clamp_min(0)
    probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)
    sorted_probability, sorted_indices = probability.sort(-1, descending=True)
    keep = sorted_probability.cumsum(-1) <= float(top_p)
    keep[..., 0] = True
    truncated = torch.zeros_like(probability)
    truncated.scatter_(
        -1,
        sorted_indices,
        sorted_probability * keep.to(sorted_probability.dtype),
    )
    truncated = truncated / truncated.sum(-1, keepdim=True).clamp_min(1e-8)
    points = centers.to(attention).reshape(-1, 3)
    if points.shape[0] != probability.shape[-1]:
        raise ValueError("feature grid does not match voxel attention shape")
    return truncated @ points


def _focus_rasterize(values, centers, output_shape):
    result = values
    for axis, size in enumerate(output_shape):
        output_axis = torch.linspace(-1, 1, size, dtype=values.dtype)
        center_index = [0, 0, 0, axis]
        center_index[axis] = slice(None)
        feature_axis = centers[tuple(center_index)].to(dtype=values.dtype)
        selected = (output_axis[:, None] - feature_axis[None]).abs().argmin(-1)
        result = result.index_select(axis, selected)
    return result


def _focus_project(volume, view):
    _, projection, row_axis, _, rows_up, _ = view
    result = volume.amax(projection)
    kept = [axis for axis in range(3) if axis != projection]
    if row_axis == kept[1]:
        result = result.transpose(0, 1)
    if rows_up:
        result = result.flip(0)
    return result


def _focus_voxel(voxel):
    voxel = voxel.float()
    occupancy = voxel[:1]
    colour = voxel[1:4]
    if colour.max() > 1.0:
        colour = colour / 255.0
    return torch.cat((occupancy, colour * occupancy), dim=0)


def _focus_last(value, *, event_rank):
    if value is None:
        return None
    value = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    return value[:, -1] if value.ndim == event_rank + 2 else value


def _focus_point_pixel(point, view, size):
    _, _, row_axis, column_axis, rows_up, _ = view
    unit = (torch.as_tensor(point).float() + 1) / 2
    row, column = float(unit[row_axis]), float(unit[column_axis])
    if rows_up:
        row = 1 - row
    return round(column * (size - 1)), round(row * (size - 1))


def _focus_marker(draw, point, view, size, *, color, shape, radius=3):
    x, y = _focus_point_pixel(point, view, size)
    if shape == "cross":
        draw.line((x - radius, y, x + radius, y), fill=color, width=2)
        draw.line((x, y - radius, x, y + radius), fill=color, width=2)
    elif shape == "diamond":
        draw.polygon(
            ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
            fill=color,
            outline=(0, 0, 0),
        )
    else:
        draw.rectangle(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(0, 0, 0),
        )
    return x, y


def _focus_triview(
    attention,
    feature_geometry,
    crop_transform,
    source_geometry,
    observations,
    targets,
    *,
    num_samples,
):
    voxel = observations.get("voxel")
    eef = observations.get("eef_pos")
    if voxel is None or eef is None:
        return None
    voxel = voxel.detach().cpu()
    observation_steps = voxel.shape[1] if voxel.ndim == 6 else 1
    if voxel.ndim == 6:
        voxel = voxel[:, -1]
    eef = _focus_last(eef, event_rank=1)
    count = min(num_samples, attention.shape[0], voxel.shape[0], eef.shape[0])
    if count == 0:
        return None

    starts = crop_transform.starts.detach().cpu()
    if starts.shape[0] != voxel.shape[0]:
        indices = torch.arange(voxel.shape[0]) * observation_steps + observation_steps - 1
        starts = starts.index_select(0, indices)
    crop_shape = tuple(int(size) for size in crop_transform.crop_shape)
    feature_centers = feature_geometry.centers.detach().cpu()
    target_pos = _focus_last(
        None if targets is None else targets.get("focus_target_pos"), event_rank=1
    )
    target_valid = _focus_last(
        None if targets is None else targets.get("focus_target_valid"), event_rank=0
    )
    eef_source = source_geometry.world_to_grid(eef.float())
    target_source = (
        None if target_pos is None else source_geometry.world_to_grid(target_pos.float())
    )

    states = []
    for sample in range(count):
        start = starts[sample].to(torch.int64)
        slices = tuple(
            slice(int(start[axis]), int(start[axis]) + crop_shape[axis])
            for axis in range(3)
        )
        cropped = _focus_voxel(voxel[(sample, slice(None), *slices)])
        heat = attention[sample].mean(0).clamp_min(0)
        heat = heat / heat.max().clamp_min(1e-8)
        per_head = _focus_top_p_centers(attention[sample], feature_centers)
        aggregate = _focus_top_p_centers(
            attention[sample].mean(0, keepdim=True), feature_centers
        )[0]
        transform_index = sample * observation_steps + observation_steps - 1
        if crop_transform.starts.shape[0] == voxel.shape[0]:
            transform_index = sample
        offset = crop_transform.offset[transform_index].detach().cpu()
        scale = crop_transform.scale.detach().cpu()
        eef_crop = eef_source[sample] * scale + offset
        target_crop = None
        target_outside = False
        if (
            target_source is not None
            and target_valid is not None
            and bool(target_valid[sample])
            and bool(torch.isfinite(target_source[sample]).all())
        ):
            target_crop = target_source[sample] * scale + offset
            target_outside = not bool((target_crop.abs() <= 1).all())
            if target_outside:
                crop_size = target_crop.new_tensor(crop_shape)
                target_index = ((target_crop.clamp(-1, 1) + 1) / 2 * (crop_size - 1)).round()
                target_crop = target_index / (crop_size - 1) * 2 - 1
        states.append(
            {
                "voxel": cropped,
                "heat": _focus_rasterize(heat, feature_centers, crop_shape),
                "heads": per_head,
                "mean": aggregate,
                "eef": eef_crop,
                "eef_inside": bool(torch.isfinite(eef_crop).all() and (eef_crop.abs() <= 1).all()),
                "target": target_crop,
                "target_outside": target_outside,
            }
        )

    tile = crop_shape[0] * 3
    gutter, row_label, header, legend = 4, 24, 16, 18
    width = row_label + count * tile + (count - 1) * gutter
    height = header + 3 * tile + 2 * gutter + legend
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    canvas_draw = ImageDraw.Draw(canvas)
    for sample in range(count):
        x = row_label + sample * (tile + gutter)
        canvas_draw.text((x + tile // 2 - 8, 3), f"S{sample}", fill=(230, 230, 230))

    for row, view in enumerate(_POSE_VIEWS):
        y = header + row * (tile + gutter)
        canvas_draw.text((4, y + tile // 2 - 5), view[0], fill=(230, 230, 230))
        for sample, state in enumerate(states):
            panel = _voxel_surface(state["voxel"], *view[1:]).resize(
                (tile, tile), Image.Resampling.NEAREST
            )
            heat = _focus_project(state["heat"], view).numpy()
            heat = np.asarray(
                Image.fromarray(np.rint(heat * 255).astype(np.uint8)).resize(
                    (tile, tile), Image.Resampling.NEAREST
                ),
                dtype=np.float32,
            )[..., None] / 255.0
            pixels = np.asarray(panel, dtype=np.float32)
            pixels = pixels * (1 - 0.45 * heat) + np.array((0, 255, 255)) * (0.45 * heat)
            panel = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(panel)
            mean_px = _focus_marker(
                draw, state["mean"], view, tile, color=(255, 255, 255), shape="square"
            )
            if state["target"] is not None:
                target_px = _focus_point_pixel(state["target"], view, tile)
                draw.line((*mean_px, *target_px), fill=(255, 150, 235), width=1)
            if state["eef_inside"]:
                _focus_marker(
                    draw, state["eef"], view, tile, color=(245, 248, 250), shape="cross"
                )
            else:
                draw.rectangle((0, 0, tile - 1, tile - 1), outline=(196, 48, 48), width=2)
                draw.text((4, 3), "EEF OUT", fill=(196, 48, 48), stroke_width=1, stroke_fill=(0, 0, 0))
            for head, point in enumerate(state["heads"]):
                _focus_marker(
                    draw,
                    point,
                    view,
                    tile,
                    color=_FOCUS_HEAD_COLORS[head % len(_FOCUS_HEAD_COLORS)],
                    shape="square",
                    radius=2,
                )
            _focus_marker(
                draw, state["mean"], view, tile, color=(255, 255, 255), shape="square"
            )
            if state["target"] is not None:
                _focus_marker(
                    draw, state["target"], view, tile, color=(255, 48, 220), shape="diamond", radius=4
                )
                if state["target_outside"]:
                    draw.text((4, 15), "GT EDGE", fill=(255, 48, 220), stroke_width=1, stroke_fill=(0, 0, 0))
            x = row_label + sample * (tile + gutter)
            canvas.paste(panel, (x, y))

    legend_y = height - legend + 3
    entries = (
        ("mean", (255, 255, 255)),
        ("EEF", (245, 248, 250)),
        ("GT", (255, 48, 220)),
        ("attention", (0, 255, 255)),
    )
    x = 5
    canvas_draw.text((x, legend_y), "heads", fill=(230, 230, 230))
    x += 35
    for color in _FOCUS_HEAD_COLORS[:4]:
        canvas_draw.rectangle((x, legend_y + 2, x + 6, legend_y + 8), fill=color)
        x += 9
    for label, color in entries:
        x += 5
        canvas_draw.rectangle((x, legend_y + 2, x + 6, legend_y + 8), fill=color)
        x += 9
        canvas_draw.text((x, legend_y), label, fill=(230, 230, 230))
        x += int(canvas_draw.textlength(label)) + 4
    return canvas


def render_focus_diagnostics(
    encoder_output,
    *,
    num_samples: int = 6,
    observations: Optional[Mapping[str, torch.Tensor]] = None,
    targets: Optional[Mapping[str, torch.Tensor]] = None,
) -> Optional[Image.Image]:
    panels, labels = [], []
    voxel_triview = None
    for record in encoder_output.focus_records:
        key = f"rgb_{record.view}"
        images = encoder_output.prepared_inputs.get(key)
        if images is None:
            continue
        if images.ndim == 4:
            images = images[:, None]
        boxes = record.prediction.box_px.detach().cpu().numpy()
        for sample in range(min(num_samples, images.shape[0], boxes.shape[0])):
            panel = to_rgb(images[sample, -1], source="imagenet")
            draw = ImageDraw.Draw(panel)
            draw.rectangle(tuple(float(value) for value in boxes[sample]), outline="cyan", width=3)
            panels.append(panel)
            labels.append(f"{record.source} {record.view} head focus")
    attention = encoder_output.attention
    if attention is not None:
        attention = attention.detach().float().cpu()
        geometry = encoder_output.attention_geometry
        spatial_rank = (
            int(geometry.centers.shape[-1])
            if geometry is not None and hasattr(geometry, "centers")
            else max(2, attention.ndim - 3)
        )
        target_rank = spatial_rank + 2
        while attention.ndim > target_rank:
            attention = attention[:, -1]
        if spatial_rank == 2 and attention.ndim == 4:
            for sample in range(min(num_samples, attention.shape[0])):
                for head in range(attention.shape[1]):
                    heat = attention[sample, head]
                    weights = heat.clamp_min(0)
                    rows = torch.arange(heat.shape[0], dtype=heat.dtype)[:, None]
                    columns = torch.arange(heat.shape[1], dtype=heat.dtype)[None]
                    denominator = weights.sum().clamp_min(1e-12)
                    center = (
                        float((weights * columns).sum() / denominator),
                        float((weights * rows).sum() / denominator),
                    )
                    heat = (heat - heat.min()) / (
                        heat.max() - heat.min()
                    ).clamp_min(1e-12)
                    colored = torch.stack((heat, torch.zeros_like(heat), 1 - heat))
                    panels.append(
                        to_rgb(colored, source="float01").resize((256, 256))
                    )
                    labels.append(
                        f"attention sample {sample} head {head} center=({center[0]:.1f},{center[1]:.1f})"
                    )
        elif spatial_rank == 3 and attention.ndim == 5:
            if (
                observations is not None
                and encoder_output.voxel_crop_transform is not None
                and encoder_output.voxel_crop_geometry is not None
                and geometry is not None
            ):
                voxel_triview = _focus_triview(
                    attention,
                    geometry,
                    encoder_output.voxel_crop_transform,
                    encoder_output.voxel_crop_geometry,
                    observations,
                    targets,
                    num_samples=num_samples,
                )
            else:
                for sample in range(min(num_samples, attention.shape[0])):
                    for head in range(attention.shape[1]):
                        volume = attention[sample, head].numpy()
                        for axes, name in (
                            ((0, 1), "XY"),
                            ((0, 2), "XZ"),
                            ((1, 2), "YZ"),
                        ):
                            axis = ({0, 1, 2} - set(axes)).pop()
                            heat = volume.max(axis=axis)
                            heat = (heat - heat.min()) / max(
                                float(heat.max() - heat.min()), 1e-12
                            )
                            rgb = np.stack(
                                (heat, np.zeros_like(heat), 1 - heat), axis=-1
                            )
                            panels.append(to_rgb(rgb).resize((256, 256)))
                            labels.append(
                                f"voxel attention sample {sample} head {head} {name}"
                            )
    if voxel_triview is not None:
        if not panels:
            return voxel_triview
        return stack_sections(
            (("Focus heads", image_grid(panels, labels=labels)), ("Voxel attention", voxel_triview))
        )
    if not panels:
        return None
    return image_grid(panels, labels=labels)


_POSE_VIEWS = (
    ("XY", 2, 0, 1, False, True),
    ("XZ", 1, 2, 0, True, False),
    ("YZ", 0, 2, 1, True, True),
)
_POSE_AXIS_COLORS = ((255, 80, 80), (120, 255, 120), (120, 170, 255))


def _voxel_surface(voxel, projection, row_axis, column_axis, rows_up, from_high):
    occupancy = voxel[0] > 0
    colour = voxel[1:4].clamp(0, 1)
    if from_high:
        occupancy = occupancy.flip(projection)
        colour = colour.flip(projection + 1)
    index = occupancy.float().argmax(projection)
    gather = index.unsqueeze(0).unsqueeze(projection + 1).expand(3, *index.shape[:projection], 1, *index.shape[projection:])
    surface = torch.take_along_dim(colour, gather, dim=projection + 1).squeeze(projection + 1)
    surface = surface * occupancy.any(projection).unsqueeze(0)
    kept = [axis for axis in range(3) if axis != projection]
    if row_axis == kept[1]:
        surface = surface.transpose(1, 2)
    if rows_up:
        surface = surface.flip(1)
    return to_rgb(surface, source="float01")


def _action_pose_panel(
    voxel, geometry, view, predicted, target, *, size=180
):
    _, projection, row_axis, column_axis, rows_up, from_high = view
    panel = _voxel_surface(
        voxel, projection, row_axis, column_axis, rows_up, from_high
    ).resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel, "RGBA")

    def pixels(points):
        points = torch.as_tensor(points, dtype=torch.float32)
        unit = ((geometry.world_to_grid(points) + 1) / 2).numpy()
        rows = unit[:, row_axis]
        if rows_up:
            rows = 1 - rows
        return [
            (round(float(point[column_axis]) * (size - 1)), round(float(row) * (size - 1)))
            for point, row in zip(unit, rows)
        ]

    target_pixels = pixels(target[:, :3])
    predicted_pixels = pixels(predicted[:, :3])
    stride = max(1, math.ceil(len(target_pixels) / 8))
    for step in range(0, len(target_pixels), stride):
        draw.line(
            (*target_pixels[step], *predicted_pixels[step]),
            fill=(225, 230, 235, 85),
            width=1,
        )
    if len(target_pixels) > 1:
        draw.line(target_pixels, fill=(80, 220, 100, 255), width=4, joint="curve")
        draw.line(predicted_pixels, fill=(255, 105, 90, 255), width=3, joint="curve")
    for point in target_pixels:
        x, y = point
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(80, 220, 100, 220))
    for point in predicted_pixels:
        x, y = point
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 105, 90, 230))
    x, y = target_pixels[0]
    draw.ellipse(
        (x - 6, y - 6, x + 6, y + 6),
        outline=(80, 220, 100, 255),
        width=2,
    )
    x, y = predicted_pixels[0]
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 105, 90, 255))
    x, y = target_pixels[-1]
    draw.polygon(
        ((x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)),
        outline=(80, 220, 100, 255),
        width=2,
    )
    x, y = predicted_pixels[-1]
    draw.rectangle(
        (x - 4, y - 4, x + 4, y + 4),
        fill=(255, 105, 90, 255),
        outline=(0, 0, 0, 255),
    )
    return panel


def _render_action_pose_overlay(
    predicted,
    target,
    observations,
    geometry,
    *,
    num_samples,
):
    voxel = observations["voxel"].detach().cpu()
    if voxel.ndim == 6:
        voxel = voxel[:, -1]
    if voxel.dtype == torch.uint8:
        occupancy = voxel[:, :1].float()
        colour = voxel[:, 1:4].float().div(255).mul(occupancy)
        voxel = torch.cat((occupancy, colour), dim=1)
    count = min(num_samples, len(predicted), len(target), len(voxel))
    if count < 1:
        raise ValueError("action diagnostics need at least one sample")

    tile = 180
    timeline_height = 96
    gutter = 5
    row_label = 48
    header = 43
    legend = 32
    width = row_label + count * tile + max(count - 1, 0) * gutter
    height = (
        header
        + len(_POSE_VIEWS) * tile
        + len(_POSE_VIEWS) * gutter
        + timeline_height
        + legend
    )
    canvas = Image.new("RGB", (width, height), (26, 29, 34))
    draw = ImageDraw.Draw(canvas)

    for sample in range(count):
        x = row_label + sample * (tile + gutter) + 4
        draw.multiline_text(
            (x, 2),
            _action_summary(predicted[sample], target[sample], sample),
            fill=(235, 238, 242),
            spacing=1,
        )
    view_labels = ("top\nXY", "front\nXZ", "side\nYZ")
    for row, (view, label) in enumerate(zip(_POSE_VIEWS, view_labels)):
        y = header + row * (tile + gutter)
        bounds = draw.multiline_textbbox((0, 0), label, spacing=1)
        draw.multiline_text(
            (5, y + (tile - (bounds[3] - bounds[1])) // 2),
            label,
            fill=(190, 196, 204),
            spacing=1,
            align="center",
        )
        for sample in range(count):
            x = row_label + sample * (tile + gutter)
            canvas.paste(
                _action_pose_panel(
                    voxel[sample],
                    geometry,
                    view,
                    predicted[sample],
                    target[sample],
                    size=tile,
                ),
                (x, y),
            )

    timeline_y = header + len(_POSE_VIEWS) * (tile + gutter)
    draw.multiline_text(
        (5, timeline_y + 33),
        "error\nvs step",
        fill=(190, 196, 204),
        spacing=1,
        align="center",
    )
    for sample in range(count):
        x = row_label + sample * (tile + gutter)
        canvas.paste(
            _action_timeline_panel(
                predicted[sample], target[sample], width=tile, height=timeline_height
            ),
            (x, timeline_y),
        )

    legend_y = height - legend + 2
    draw.text(
        (row_label, legend_y),
        "target green ring / diamond  |  prediction coral dot / square",
        fill=(190, 196, 204),
    )
    draw.text(
        (row_label, legend_y + 13),
        "faint ties: same timestep  |  bottom: errors across the horizon",
        fill=(150, 158, 168),
    )
    return canvas


def _pose_overlay_panel(voxel, geometry, view, selected, particles, target, eef, scores, size=180):
    _, projection, row_axis, column_axis, rows_up, from_high = view
    panel = _voxel_surface(
        voxel, projection, row_axis, column_axis, rows_up, from_high
    ).resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel, "RGBA")

    def unit(point):
        point = torch.as_tensor(point, dtype=torch.float32)
        return ((geometry.world_to_grid(point) + 1) / 2).numpy()

    def pixel(point):
        value = unit(point)
        row, column = float(value[row_axis]), float(value[column_axis])
        if rows_up:
            row = 1 - row
        return round(column * (size - 1)), round(row * (size - 1))

    def inside(point):
        value = unit(point)
        return bool(np.isfinite(value).all() and (value >= 0).all() and (value <= 1).all())

    def pose_axes(matrix):
        center = matrix[:3, 3]
        length = float(geometry.workspace_size) * 0.09
        return center, [center + matrix[:3, axis] * length for axis in range(3)]

    if scores is None:
        quality = np.ones(len(particles), dtype=np.float32)
    else:
        values = np.asarray(scores, dtype=np.float32)
        order = values.argsort(kind="stable")
        quality = np.empty_like(values)
        quality[order] = np.linspace(1.0, 0.25, len(values), dtype=np.float32)
    for matrix, alpha in zip(particles, quality):
        center = matrix[:3, 3]
        if inside(center):
            x, y = pixel(center)
            radius = 2
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(135, 190, 235, round(45 + 150 * float(alpha))),
            )

    if eef is not None and inside(eef):
        x, y = pixel(eef)
        draw.line((x - 4, y, x + 4, y), fill=(245, 248, 250, 255), width=2)
        draw.line((x, y - 4, x, y + 4), fill=(245, 248, 250, 255), width=2)

    target_center = None
    if target is not None:
        target_center, target_axes = pose_axes(target)
        if inside(target_center):
            center_px = pixel(target_center)
            for endpoint, color in zip(target_axes, _POSE_AXIS_COLORS):
                if inside(endpoint):
                    end = pixel(endpoint)
                    steps = max(abs(end[0] - center_px[0]), abs(end[1] - center_px[1]), 1)
                    for step in range(0, steps + 1, 4):
                        ratio = step / steps
                        point = (
                            round(center_px[0] + ratio * (end[0] - center_px[0])),
                            round(center_px[1] + ratio * (end[1] - center_px[1])),
                        )
                        draw.ellipse((*point, *point), fill=(*color, 180))
            x, y = center_px
            draw.polygon(((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)), fill=(255, 48, 220, 255))

    selected_center, selected_axes = pose_axes(selected)
    if inside(selected_center):
        center_px = pixel(selected_center)
        if target_center is not None and inside(target_center):
            draw.line((*pixel(target_center), *center_px), fill=(255, 150, 235, 180), width=1)
        for endpoint, color in zip(selected_axes, _POSE_AXIS_COLORS):
            if inside(endpoint):
                draw.line((*center_px, *pixel(endpoint)), fill=(*color, 255), width=2)
        x, y = center_px
        draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=(255, 205, 70, 255), outline=(0, 0, 0, 255))
    return panel


def render_seeker_stages(stages: Mapping[str, Sequence[object]], *, num_samples: int = 6) -> Optional[Image.Image]:
    panels, labels = [], []
    for view, outputs in stages.items():
        for output in outputs:
            images = output.image.detach().cpu()
            masks = output.mask.detach().float().cpu()
            boxes = output.tight_box.detach().cpu().numpy()
            for sample in range(min(num_samples, images.shape[0])):
                image = CoreImages.image_to_float01(images[sample], source="imagenet")
                mask = F.interpolate(
                    masks[sample : sample + 1],
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                mask = mask / mask.max().clamp_min(1e-12)
                overlay = image.clone()
                overlay[0] = overlay[0] * (1 - 0.45 * mask) + 0.45 * mask
                overlay[1:] *= 1 - 0.45 * mask[None]
                panel = to_rgb(overlay, source="float01")
                ImageDraw.Draw(panel).rectangle(
                    tuple(float(value) for value in boxes[sample]),
                    outline="cyan",
                    width=3,
                )
                panels.append(panel)
                labels.append(f"Seeker {view} {output.stage} sample {sample}")
    if not panels:
        return None
    return image_grid(panels, labels=labels)
