"""Rigid-frame (SE(3)) algebra: compose, invert, transform points/rotations/poses.

Conventions (see ``docs/architecture.md``, "Geometry boundary")::

    R_AB, t_AB  -- rotation/translation of frame B expressed in frame A
    p_A = R_AB @ p_B + t_AB

``(R, t)`` is the computational representation; ``to_homogeneous``/
``from_homogeneous`` cross to a 4x4 matrix only at boundaries that genuinely
need one (a stored/serialized frame, an external API).

Batch dims broadcast the way ``torch.einsum`` broadcasts its ``...``: a
caller whose frame has one fewer batch dim than its points/poses (e.g. one
frame shared across a horizon) should ``unsqueeze`` the frame before calling.
"""

from __future__ import annotations

from typing import Tuple

import pytorch3d.transforms as p3t
import torch

from visuomotor.geometry.representation import mat_to_rot6d, rot6d_to_mat


def _check_rotation(R: torch.Tensor, name: str) -> None:
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must be [...,3,3], got {tuple(R.shape)}")


def _check_points(p: torch.Tensor, name: str) -> None:
    if p.shape[-1] != 3:
        raise ValueError(f"{name} must be [...,3], got {tuple(p.shape)}")


def _check_frame(R: torch.Tensor, t: torch.Tensor, R_name: str, t_name: str) -> None:
    """Validate one ``(R, t)`` frame: shapes, plus agreeing batch dims.

    Batch agreement is required rather than broadcast so that a rank mismatch
    fails here instead of silently producing a frame the caller did not mean;
    share one frame across a horizon by unsqueezing it, as the module docstring
    describes.
    """
    _check_rotation(R, R_name)
    _check_points(t, t_name)
    if R.shape[:-2] != t.shape[:-1]:
        raise ValueError(
            f"{R_name} batch dims {tuple(R.shape[:-2])} do not match "
            f"{t_name} batch dims {tuple(t.shape[:-1])}"
        )


def inv_rotation(R_AB: torch.Tensor) -> torch.Tensor:
    """Inverse of a pure rotation: ``R_BA`` such that ``p_B = R_BA @ p_A``."""
    _check_rotation(R_AB, "R_AB")
    return R_AB.transpose(-1, -2)


def inv(R_AB: torch.Tensor, t_AB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_frame(R_AB, t_AB, "R_AB", "t_AB")
    R_BA = inv_rotation(R_AB)
    t_BA = -torch.einsum("...ij,...j->...i", R_BA, t_AB)
    return R_BA, t_BA


def compose(
    R_AB: torch.Tensor, t_AB: torch.Tensor, R_BC: torch.Tensor, t_BC: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_frame(R_AB, t_AB, "R_AB", "t_AB")
    _check_frame(R_BC, t_BC, "R_BC", "t_BC")
    R_AC = R_AB @ R_BC
    t_AC = torch.einsum("...ij,...j->...i", R_AB, t_BC) + t_AB
    return R_AC, t_AC


def relative(
    R_WA: torch.Tensor,
    t_WA: torch.Tensor,
    R_WB: torch.Tensor,
    t_WB: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``T_AB = T_WA^-1 T_WB`` for two poses in a shared world frame."""
    R_AW, t_AW = inv(R_WA, t_WA)
    return compose(R_AW, t_AW, R_WB, t_WB)


def transform(R_AB: torch.Tensor, t_AB: torch.Tensor, p_B: torch.Tensor) -> torch.Tensor:
    _check_frame(R_AB, t_AB, "R_AB", "t_AB")
    _check_points(p_B, "p_B")
    return torch.einsum("...ij,...j->...i", R_AB, p_B) + t_AB


def transform_rotation(R_AB: torch.Tensor, R_BC: torch.Tensor) -> torch.Tensor:
    _check_rotation(R_AB, "R_AB")
    _check_rotation(R_BC, "R_BC")
    return R_AB @ R_BC


def transform_pose(R_AB: torch.Tensor, t_AB: torch.Tensor, pose_B: torch.Tensor) -> torch.Tensor:
    """Transform ``[position(3), rotation6d(6), extras...]`` pose/action chunks.

    Trailing non-spatial dims (e.g. gripper) pass through unchanged. ``pose_B``
    may carry exactly one extra leading (horizon) dim relative to ``R_AB``/
    ``t_AB`` -- one frame shared across a horizon of poses -- in which case
    the frame is broadcast across that dim.
    """
    _check_frame(R_AB, t_AB, "R_AB", "t_AB")
    if pose_B.shape[-1] < 9:
        raise ValueError("pose must have shape [...,D] with D >= 9 (pos(3)+rot6d(6)+extras)")
    extra_dims = pose_B.ndim - R_AB.ndim + 1  # R_AB has 2 trailing dims, pose_B has 1
    if extra_dims == 1:
        R_AB, t_AB = R_AB.unsqueeze(-3), t_AB.unsqueeze(-2)
    elif extra_dims != 0:
        raise ValueError(
            "pose_B must share R_AB's batch rank, or have exactly one extra (horizon) dim"
        )
    position_A = transform(R_AB, t_AB, pose_B[..., :3])
    rotation_A = transform_rotation(R_AB, rot6d_to_mat(pose_B[..., 3:9]))
    return torch.cat((position_A, mat_to_rot6d(rotation_A), pose_B[..., 9:]), dim=-1)


def to_homogeneous(R_AB: torch.Tensor, t_AB: torch.Tensor) -> torch.Tensor:
    _check_frame(R_AB, t_AB, "R_AB", "t_AB")
    X_AB = t_AB.new_zeros(R_AB.shape[:-2] + (4, 4))
    X_AB[..., :3, :3] = R_AB
    X_AB[..., :3, 3] = t_AB
    X_AB[..., 3, 3] = 1
    return X_AB


def from_homogeneous(X_AB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if X_AB.shape[-2:] != (4, 4):
        raise ValueError("X_AB must end in [4,4]")
    return X_AB[..., :3, :3], X_AB[..., :3, 3]


def geodesic_angle(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Angle in radians between two batches of rotation matrices."""
    relative = R_a.transpose(-1, -2) @ R_b
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0))

def project_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Project matrices onto SO(3) with the nearest SVD-orthogonal rotation."""
    _check_rotation(rotation, "rotation")
    u, _, v_t = torch.linalg.svd(rotation)
    det = torch.linalg.det(u @ v_t)
    diagonal = torch.ones(
        rotation.shape[:-2] + (3,), dtype=rotation.dtype, device=rotation.device
    )
    diagonal[..., -1] = det
    return u @ torch.diag_embed(diagonal) @ v_t


def rotation_chordal_mean(rotations: torch.Tensor, dim: int) -> torch.Tensor:
    """SVD-projected mean rotation (chordal/L2 mean on SO(3))."""
    return project_rotation_matrix(rotations.mean(dim=dim))


def geometric_median(
    points: torch.Tensor, dim: int, iterations: int = 16, eps: float = 1e-6
) -> torch.Tensor:
    """Weiszfeld's algorithm: the point minimizing summed distance to ``points``."""
    median = points.mean(dim=dim, keepdim=True)
    for _ in range(iterations):
        weights = 1.0 / (points - median).norm(dim=-1, keepdim=True).clamp_min(eps)
        median = (points * weights).sum(dim=dim, keepdim=True) / weights.sum(dim=dim, keepdim=True)
    return median.squeeze(dim)


def rotation_geometric_median(
    rotations: torch.Tensor, dim: int, iterations: int = 16, eps: float = 1e-6
) -> torch.Tensor:
    """Riemannian (tangent-space Weiszfeld) geometric median on SO(3)."""
    output_dtype = rotations.dtype
    if rotations.dtype in (torch.float16, torch.bfloat16):
        rotations = rotations.float()
    rotations = project_rotation_matrix(rotations)
    median = rotation_chordal_mean(rotations, dim)
    for _ in range(iterations):
        relative = median.unsqueeze(dim).transpose(-1, -2) @ rotations
        flat_shape = relative.shape[:-2]
        tangent = p3t.matrix_to_axis_angle(relative.reshape(-1, 3, 3)).reshape(
            *flat_shape, 3
        )
        weights = 1.0 / tangent.norm(dim=-1, keepdim=True).clamp_min(eps)
        mean_tangent = (tangent * weights).sum(dim=dim, keepdim=True) / weights.sum(
            dim=dim, keepdim=True
        )
        mean_tangent = mean_tangent.squeeze(dim)
        update = p3t.so3_exp_map(mean_tangent.reshape(-1, 3)).reshape(*mean_tangent.shape[:-1], 3, 3)
        median = project_rotation_matrix(median @ update)
    return median.to(dtype=output_dtype)
