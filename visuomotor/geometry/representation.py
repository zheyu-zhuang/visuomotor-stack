"""Rotation representation conversions. Delegates directly to pytorch3d.transforms."""

from __future__ import annotations

import functools
from typing import Union

import numpy as np
import pytorch3d.transforms as p3t
import torch


def rot6d_to_mat(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-1] != 6:
        raise ValueError("6D rotations must end in six values")
    return p3t.rotation_6d_to_matrix(rotation)


def mat_to_rot6d(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in [3,3]")
    return p3t.matrix_to_rotation_6d(rotation)


def rotvec_to_mat(rotvec: torch.Tensor) -> torch.Tensor:
    if rotvec.shape[-1] != 3:
        raise ValueError("axis-angle rotations must end in three values")
    return p3t.axis_angle_to_matrix(rotvec)


def canonicalize_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """Re-express an axis-angle vector on the shortest arc: angle in ``[0, pi]``.

    ``v`` and ``-v * (2*pi - |v|) / |v|`` are the same rotation; only the latter
    is bounded by pi.
    """
    if rotvec.shape[-1] != 3:
        raise ValueError("axis-angle rotations must end in three values")
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    scale = torch.where(
        angle > torch.pi,
        (angle - 2 * torch.pi) / angle.clamp_min(1e-12),
        torch.ones_like(angle),
    )
    return rotvec * scale


def mat_to_rotvec(rotation: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector, canonicalized to the shortest arc.

    pytorch3d returns angles above pi for some rotations. Cached MimicGen
    actions and the robosuite controllers both assume the shortest-arc form, so
    the convention is enforced here rather than at each call site.
    """
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in [3,3]")
    return canonicalize_rotvec(p3t.matrix_to_axis_angle(rotation))


def quat_to_mat(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternions must end in four values")
    return p3t.quaternion_to_matrix(quaternion)


def mat_to_quat(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in [3,3]")
    return p3t.matrix_to_quaternion(rotation)


def quat_xyzw_to_mat(quaternion: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from a robomimic-convention XYZW quaternion."""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternions must end in four values")
    return quat_to_mat(quaternion[..., [3, 0, 1, 2]])


class RotationTransformer:
    """Convert rigid rotations through a matrix intermediate representation."""

    valid_reps = ("axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix")

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        if from_rep == to_rep or from_rep not in self.valid_reps or to_rep not in self.valid_reps:
            raise ValueError(f"unsupported rotation conversion: {from_rep} -> {to_rep}")
        forward, inverse = [], []
        for representation, convention, to_matrix in (
            (from_rep, from_convention, True),
            (to_rep, to_convention, False),
        ):
            if representation == "matrix":
                continue
            if representation == "euler_angles" and convention is None:
                raise ValueError("Euler conversions require a convention")
            encode = getattr(p3t, f"{representation}_to_matrix")
            # Axis-angle decodes through the shortest-arc convention, the one
            # cached actions and the robosuite controllers are written against;
            # pytorch3d's raw decode returns the 2*pi complement for some
            # rotations.
            decode = (
                mat_to_rotvec
                if representation == "axis_angle"
                else getattr(p3t, f"matrix_to_{representation}")
            )
            if convention is not None:
                encode = functools.partial(encode, convention=convention)
                decode = functools.partial(decode, convention=convention)
            forward.append(encode if to_matrix else decode)
            inverse.append(decode if to_matrix else encode)
        self.forward_funcs = forward
        self.inverse_funcs = inverse[::-1]

    @staticmethod
    def _apply(values, functions):
        numpy_input = isinstance(values, np.ndarray)
        result = torch.from_numpy(values) if numpy_input else values
        for function in functions:
            result = function(result)
        return result.numpy() if numpy_input else result

    def forward(self, values: Union[np.ndarray, torch.Tensor]):
        return self._apply(values, self.forward_funcs)

    def inverse(self, values: Union[np.ndarray, torch.Tensor]):
        return self._apply(values, self.inverse_funcs)
