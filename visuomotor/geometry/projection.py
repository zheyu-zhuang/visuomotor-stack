"""Camera projection / unprojection."""

from __future__ import annotations

import torch


def world_xyz_to_pixel_row_col(
    xyz: torch.Tensor,
    world_to_pixel: torch.Tensor,
    image_size: int | tuple[int, int],
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Project world-space points to pixel ``(row, col)`` via a per-sample camera matrix.

    ``xyz`` is ``[..., 3]`` and ``world_to_pixel`` is ``[..., 4, 4]``.
    Points behind the camera or with a non-finite projection come back as NaN;
    valid points are clamped to the image bounds unless ``clamp=False``.
    """
    if xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be [...,3], got {tuple(xyz.shape)}")
    if world_to_pixel.shape[-2:] != (4, 4):
        raise ValueError(f"world_to_pixel must be [...,4,4], got {tuple(world_to_pixel.shape)}")
    ones = xyz.new_ones(xyz.shape[:-1] + (1,))
    homogeneous = torch.cat([xyz, ones], dim=-1).unsqueeze(-1)
    projected = torch.matmul(world_to_pixel, homogeneous).squeeze(-1)
    depth = projected[..., 2]
    invalid = ~torch.isfinite(depth) | (depth <= 1e-8)
    safe_depth = torch.where(invalid, depth.new_ones(()), depth)
    col = projected[..., 0] / safe_depth
    row = projected[..., 1] / safe_depth
    height, width = (
        (int(image_size), int(image_size))
        if isinstance(image_size, int)
        else (int(image_size[0]), int(image_size[1]))
    )
    if clamp:
        col = col.clamp(0, width - 1)
        row = row.clamp(0, height - 1)
    invalid = invalid | ~torch.isfinite(row) | ~torch.isfinite(col)
    nan = row.new_full((), float("nan"))
    row = torch.where(invalid, nan, row)
    col = torch.where(invalid, nan, col)
    return torch.stack([row, col], dim=-1)
