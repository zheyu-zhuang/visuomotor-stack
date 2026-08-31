"""Fixed non-learned positional encodings for spatial feature grids.

Ported from ``seeker-dev``'s ``focus-pool`` branch
(``seeker/model/common/position_encoding.py``).
"""

from __future__ import annotations

import math

import torch


def get_normalized_grid_coordinates(grid_h: int, grid_w: int) -> torch.Tensor:
    """``[H*W, 2]`` grid cell centers, each row ``(x, y)`` in ``[-1, 1]``."""
    y = torch.linspace(-1.0, 1.0, grid_h)
    x = torch.linspace(-1.0, 1.0, grid_w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


def get_freq_position_embedding(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """``[H*W, dim]`` fixed sin/cos Fourier features of the grid coordinates."""
    num_bands = max(1, math.ceil(dim / 4))
    freqs = torch.exp(torch.linspace(0.0, math.log(16.0), num_bands))
    coords = get_normalized_grid_coordinates(grid_h, grid_w).view(grid_h, grid_w, 2)
    angles = coords[..., :, None] * freqs
    enc = torch.cat([angles.sin(), angles.cos()], dim=-2).flatten(-2)
    return enc[..., :dim].reshape(-1, dim)


def get_normalized_grid_coordinates_3d(
    grid_d: int, grid_h: int, grid_w: int
) -> torch.Tensor:
    """``[D*H*W, 3]`` cell centers as ``(x, y, z)`` in ``[-1, 1]``."""
    z = torch.linspace(-1.0, 1.0, grid_d)
    y = torch.linspace(-1.0, 1.0, grid_h)
    x = torch.linspace(-1.0, 1.0, grid_w)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).view(-1, 3)


def get_freq_position_embedding_3d(
    grid_d: int, grid_h: int, grid_w: int, dim: int
) -> torch.Tensor:
    """``[D*H*W, dim]`` fixed Fourier features of 3D grid coordinates."""
    num_bands = max(1, math.ceil(dim / 6))
    freqs = torch.exp(torch.linspace(0.0, math.log(16.0), num_bands))
    coords = get_normalized_grid_coordinates_3d(grid_d, grid_h, grid_w).view(
        grid_d, grid_h, grid_w, 3
    )
    angles = coords[..., :, None] * freqs
    enc = torch.cat([angles.sin(), angles.cos()], dim=-2).flatten(-2)
    return enc[..., :dim].reshape(-1, dim)
