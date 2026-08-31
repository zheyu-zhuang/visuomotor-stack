"""O(3) reflection math, kept distinct from :mod:`visuomotor.geometry.rigid`'s SE(3) algebra."""

from __future__ import annotations

import torch


def reflection_matrix(axis: int = 0, *, device=None, dtype=None) -> torch.Tensor:
    """A 3x3 reflection across the plane orthogonal to ``axis`` (flips that one coordinate)."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    signs = [1.0, 1.0, 1.0]
    signs[axis] = -1.0
    return torch.diag(torch.tensor(signs, device=device, dtype=dtype))


def reflect_point(reflection: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", reflection, p)


def reflect_rotation(reflection: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Reflect a rotation across ``reflection``'s plane (conjugation restores O(3))."""
    return reflection @ R @ reflection
