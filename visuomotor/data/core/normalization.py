"""The single owner of model-space numeric normalization.

Normalization sits after spatial augmentation and geometry, and before the
model: geometry/augmentation define the final physical value, and this module
alone defines how that value enters model space.

Datasets own *what* gets fitted (field selection, robot partitioning, action
representation, augmentation bounds); this module owns *how* fitted state is
accumulated, stored, and applied:

    normalizer.update_bounds("eef_pos", lo, hi, robot_id)
    normalizer.finalize()

    x = normalizer.normalize("eef_pos", x, robot_id)
    action = normalizer.normalize_action(action, robot_id)
    rgb = normalizer.normalize("rgb", rgb)
    voxel = normalizer.normalize("voxel", voxel)
    point_cloud = normalizer.normalize("point_cloud", point_cloud)

:func:`normalize_obs` is the whole-observation entry point that model boundaries
call; it is the only place a canonical observation becomes a model input.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# action = [position(3), rotation_6d(6), gripper(>=1)]
_ACTION_POS_SLICE = slice(0, 3)
_ACTION_ROT_SLICE = slice(3, 9)
_ACTION_GRIPPER_SLICE = slice(9, None)
_ACTION_POS_FIELD = "action_pos"
_ACTION_GRIPPER_FIELD = "action_gripper"
# Canonical low-dimensional fields, by the normalization scheme each one takes.
FITTED_LOW_DIM_FIELDS = (
    "eef_pos",
    "gripper_qpos",
    "eef_delta_pos",
    "eef_delta_rotvec",
    "gripper_qpos_delta",
)
# Rotations are already in [-1, 1] by construction and are never fitted, exactly
# as the rot6d block of an action passes through `normalize_action` unchanged.
PASSTHROUGH_LOW_DIM_FIELDS = ("eef_rot6d",)


class _AffineState(nn.Module):
    """A fitted per-axis affine map: ``scale``/``offset`` only (no input stats)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("offset", torch.zeros(dim))

    def normalize(self, x: Union[torch.Tensor, np.ndarray], *, forward: bool) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.shape[-1] != self.scale.shape[-1]:
            raise ValueError(
                f"expected trailing dim {self.scale.shape[-1]}, got {x.shape[-1]}"
            )
        x = x.to(dtype=self.scale.dtype, device=self.scale.device)
        if forward:
            return x * self.scale + self.offset
        return (x - self.offset) / self.scale

def _fit_affine(
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    output_min: float = -1.0,
    output_max: float = 1.0,
    range_eps: float = 1e-4,
) -> _AffineState:
    """Fit a min/max affine mapping ``[lo, hi] -> [output_min, output_max]``."""
    lo = torch.as_tensor(lo).float()
    hi = torch.as_tensor(hi).float()
    input_range = hi - lo
    ignore = input_range < range_eps
    safe_range = torch.where(ignore, torch.full_like(input_range, output_max - output_min), input_range)
    scale = (output_max - output_min) / safe_range
    offset = torch.where(ignore, (output_max + output_min) / 2 - lo, output_min - scale * lo)
    state = _AffineState(int(lo.shape[-1]))
    with torch.no_grad():
        state.scale.copy_(scale)
        state.offset.copy_(offset)
    return state


def _reduce_last_dim(x) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    return x.reshape(-1, x.shape[-1])


class Normalizer(nn.Module):
    """The runtime normalizer: fitted numeric state plus fixed RGB/voxel schemes.

    Multi-robot state is one ``Normalizer`` instance holding robot-specific
    fitted state per field; pass ``robot_id=None`` for single-robot fitting.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self) -> None:
        super().__init__()
        self._fields = nn.ModuleDict()
        self._pending_samples: Dict[str, list] = {}
        self._pending_bounds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _key(field: str, robot_id) -> str:
        if robot_id is None:
            return field
        return f"{field}::{int(robot_id)}"

    # ------------------------------------------------------------ fitting

    def update_samples(self, field: str, values, robot_id: Optional[int] = None) -> None:
        """Accumulate raw/derived samples for ``field``, pooled over every leading dim."""
        key = self._key(field, robot_id)
        self._pending_samples.setdefault(key, []).append(_reduce_last_dim(values))

    def update_bounds(self, field: str, lo, hi, robot_id: Optional[int] = None) -> None:
        """Accumulate conservative per-sample ``(lo, hi)`` bounds for ``field``."""
        lo, hi = _reduce_last_dim(lo).min(dim=0).values, _reduce_last_dim(hi).max(dim=0).values
        key = self._key(field, robot_id)
        if key in self._pending_bounds:
            prev_lo, prev_hi = self._pending_bounds[key]
            lo, hi = torch.minimum(lo, prev_lo), torch.maximum(hi, prev_hi)
        self._pending_bounds[key] = (lo, hi)

    def finalize(self) -> None:
        """Build fitted state from every accumulated sample/bound, then clear them."""
        for key in set(self._pending_samples) | set(self._pending_bounds):
            lo = hi = None
            if key in self._pending_samples:
                data = torch.cat(self._pending_samples[key], dim=0)
                lo, hi = data.min(dim=0).values, data.max(dim=0).values
            if key in self._pending_bounds:
                b_lo, b_hi = self._pending_bounds[key]
                lo = b_lo if lo is None else torch.minimum(lo, b_lo)
                hi = b_hi if hi is None else torch.maximum(hi, b_hi)
            self._fields[key] = _fit_affine(lo, hi)
        self._pending_samples.clear()
        self._pending_bounds.clear()

    # ------------------------------------------------------------ runtime

    def has(self, field: str, robot_id: Optional[int] = None) -> bool:
        return self._key(field, robot_id) in self._fields

    def has_field(self, field: str) -> bool:
        """Whether ``field`` was fitted, bare or under any robot-suffixed key."""
        if field in self._fields:
            return True
        prefix = field + "::"
        return any(key.startswith(prefix) for key in self._fields)

    def fitted_fields(self) -> list:
        """Every fitted key, robot suffix included."""
        return sorted(self._fields)

    def require_fitted(self, field: str, robot_id=None) -> None:
        """Raise unless ``field`` can actually be normalized as called.

        :meth:`normalize` returns its input unchanged for a field it cannot
        route, which would put raw physical values into model space; callers
        that must not degrade that way check here first.
        """
        if field in self._fields:
            return
        if not self.has_field(field):
            raise KeyError(
                f"no fitted normalization for field {field!r}; "
                f"fitted fields: {self.fitted_fields()}"
            )
        if robot_id is None:
            raise KeyError(
                f"field {field!r} was fitted per robot, but no robot_id was given"
            )

    def normalize_field(self, field: str, x, robot_id=None):
        """:meth:`normalize`, but failing loudly instead of passing raw values through."""
        self.require_fitted(field, robot_id)
        return self.normalize(field, x, robot_id)

    def denormalize_field(self, field: str, x, robot_id=None):
        self.require_fitted(field, robot_id)
        return self.denormalize(field, x, robot_id)

    def normalize(self, field: str, x, robot_id=None):
        return self._dispatch(field, x, robot_id, forward=True)

    def denormalize(self, field: str, x, robot_id=None):
        return self._dispatch(field, x, robot_id, forward=False)

    def normalize_action(self, x, robot_id=None):
        return self.normalize("action", x, robot_id)

    def denormalize_action(self, x, robot_id=None):
        return self.denormalize("action", x, robot_id)

    @classmethod
    def _imagenet_stats(
        cls, x: torch.Tensor, *, channel_dim: int = -3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = [1] * x.ndim
        shape[channel_dim] = 3
        mean = x.new_tensor(cls.IMAGENET_MEAN).reshape(shape)
        std = x.new_tensor(cls.IMAGENET_STD).reshape(shape)
        return mean, std

    @classmethod
    def normalize_rgb(
        cls, rgb: torch.Tensor, *, channel_dim: int = -3
    ) -> torch.Tensor:
        """ImageNet-normalize float ``[0, 1]`` RGB at ``channel_dim``."""
        if not rgb.is_floating_point():
            raise TypeError("normalize_rgb expects float input in [0, 1]")
        mean, std = cls._imagenet_stats(rgb, channel_dim=channel_dim)
        return (rgb - mean) / std

    @classmethod
    def normalize_canonical_rgb(
        cls, rgb: torch.Tensor, *, channel_dim: int = -3
    ) -> torch.Tensor:
        """Convert canonical uint8 RGB to ImageNet-normalized model space."""
        if rgb.dtype != torch.uint8:
            raise TypeError("canonical RGB must be uint8")
        return cls.normalize_rgb(rgb.to(torch.float32).div(255.0), channel_dim=channel_dim)

    @classmethod
    def denormalize_rgb(
        cls, rgb: torch.Tensor, *, channel_dim: int = -3
    ) -> torch.Tensor:
        mean, std = cls._imagenet_stats(rgb, channel_dim=channel_dim)
        return rgb * std + mean

    @classmethod
    def normalize_voxel(cls, voxel: torch.Tensor) -> torch.Tensor:
        """Convert uint8 occupancy/RGB to binary occupancy and masked RGB01."""
        if voxel.dtype != torch.uint8:
            raise TypeError("canonical voxel must be uint8")
        if voxel.shape[-4] != 4:
            raise ValueError(
                "voxel must have 4 channels [occupancy,R,G,B], "
                f"got {voxel.shape[-4]}"
            )
        model = voxel.to(torch.float32)
        occupancy = model.narrow(-4, 0, 1)
        colour = model.narrow(-4, 1, 3)
        colour.div_(255.0).mul_(occupancy)
        return model

    def _dispatch(self, field: str, x, robot_id, forward: bool):
        if field == "rgb":
            return (
                self.normalize_canonical_rgb(x)
                if forward
                else self.denormalize_rgb(x).mul(255.0).round().to(torch.uint8)
            )
        if field == "voxel":
            if not forward:
                raise NotImplementedError("voxel denormalization is not supported")
            return self.normalize_voxel(x)
        if field == "action":
            return self._apply_action(x, robot_id, forward)
        return self._apply_plain(field, x, robot_id, forward)

    def _apply_plain(self, field: str, x, robot_id, forward: bool):
        state = self._fields[field] if field in self._fields else None
        if state is not None:
            return state.normalize(x, forward=forward)
        if robot_id is not None:
            return self._apply_routed(field, x, robot_id, forward)
        return x

    def _apply_action(self, x: torch.Tensor, robot_id, forward: bool) -> torch.Tensor:
        """Split into pos(3)/rot6d(6)/gripper(rest) only when those fields were fitted.

        Lets ``normalize_action`` degrade to a single plain fitted field for
        non-pose action spaces (e.g. lightweight policy tests), while the real
        MimicGen pose-action contract fits ``action_pos``/``action_gripper``.
        """
        if not (self.has_field(_ACTION_POS_FIELD) or self.has_field(_ACTION_GRIPPER_FIELD)):
            return self._apply_plain("action", x, robot_id, forward)
        if x.shape[-1] < 10:
            raise ValueError(
                f"action needs pos(3)+rot6d(6)+gripper(>=1) dims, got {x.shape[-1]}"
            )
        position = self._apply_plain(_ACTION_POS_FIELD, x[..., _ACTION_POS_SLICE], robot_id, forward)
        rotation = x[..., _ACTION_ROT_SLICE]
        gripper = self._apply_plain(_ACTION_GRIPPER_FIELD, x[..., _ACTION_GRIPPER_SLICE], robot_id, forward)
        return torch.cat((position, rotation, gripper), dim=-1)

    def _apply_routed(self, field: str, x, robot_id, forward: bool):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if torch.is_tensor(robot_id):
            robot_ids = robot_id.reshape(-1).long()
        else:
            robot_ids = torch.full((x.shape[0],), int(robot_id), dtype=torch.long)
        if robot_ids.shape[0] != x.shape[0]:
            raise ValueError("x and robot_id must share the leading batch dimension")

        if not self.has_field(field):
            return x

        out = None
        for rid in torch.unique(robot_ids).tolist():
            routed_key = self._key(field, int(rid))
            if routed_key not in self._fields:
                raise KeyError(
                    f"no fitted normalization for field {field!r} robot_id={rid}; "
                    f"have {sorted(self._fields.keys())}"
                )
            state = self._fields[routed_key]
            idx = (robot_ids == rid).nonzero(as_tuple=False).squeeze(1)
            y = state.normalize(x.index_select(0, idx), forward=forward)
            if out is None:
                out = torch.empty(x.shape, device=y.device, dtype=y.dtype)
            out.index_copy_(0, idx, y)
        return out

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        marker = prefix + "_fields."
        for full_key in list(state_dict):
            if not full_key.startswith(marker):
                continue
            field_key = full_key[len(marker) :].split(".", 1)[0]
            if field_key not in self._fields:
                state_prefix = f"{marker}{field_key}."
                if state_prefix + "scale" in state_dict:
                    dim = int(state_dict[state_prefix + "scale"].shape[-1])
                    self._fields[field_key] = _AffineState(dim)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def obs_robot_id(observations: Mapping[str, torch.Tensor], robot_id=None):
    """Per-sample robot ids for field routing, with any time dimension dropped.

    Robot identity is constant along a sample's observation window, so routing
    only needs the leading batch dimension -- which is also the dimension the
    fields being normalized still carry at the model boundary.
    """
    if robot_id is None:
        robot_id = observations.get("robot_id")
    if not torch.is_tensor(robot_id):
        return robot_id
    while robot_id.ndim > 1:
        robot_id = robot_id[:, 0]
    return robot_id


def _normalize_obs(
    observations, normalizer, robot_id, observation_kinds, *, forward: bool
) -> dict:
    if normalizer is None:
        raise ValueError("normalize_obs requires a normalizer")
    robot_id = obs_robot_id(observations, robot_id)
    out = dict(observations)
    for field, value in observations.items():
        kind = observation_kinds.get(field)
        if kind == "voxel":
            if not forward:
                raise NotImplementedError("voxel denormalization is not supported")
            out[field] = normalizer.normalize("voxel", value)
        elif kind == "point_cloud":
            out[field] = (
                normalizer.normalize_field(field, value, robot_id)
                if forward
                else normalizer.denormalize_field(field, value, robot_id)
            )
        elif kind == "rgb":
            out[field] = (
                normalizer.normalize_canonical_rgb(value)
                if forward
                else normalizer.denormalize_rgb(value).mul(255.0).round().to(torch.uint8)
            )
        elif field in FITTED_LOW_DIM_FIELDS:
            out[field] = (
                normalizer.normalize_field(field, value, robot_id)
                if forward
                else normalizer.denormalize_field(field, value, robot_id)
            )
    return out


def normalize_obs(
    observations: Mapping[str, torch.Tensor],
    normalizer: "Normalizer",
    *,
    observation_kinds: Mapping[str, str],
    robot_id=None,
) -> dict:
    """Canonical physical observation -> model input, at the model boundary.

    This is the one transition between the canonical representation that
    datasets and rollout environments agree on and the model space an encoder
    consumes, so training and rollout reach the network through it alone.
    Visual fields take their fixed schemes and the fitted low-dim fields take
    their affine state; rotations pass through untouched. Task context and
    training targets travel outside this observation mapping. A fitted field
    that cannot be routed raises rather than reaching the model unnormalized.
    """
    return _normalize_obs(
        observations, normalizer, robot_id, observation_kinds, forward=True
    )


def denormalize_obs(
    observations: Mapping[str, torch.Tensor],
    normalizer: "Normalizer",
    *,
    observation_kinds: Mapping[str, str],
    robot_id=None,
) -> dict:
    """Inverse of :func:`normalize_obs`, for inspecting model inputs."""
    return _normalize_obs(
        observations, normalizer, robot_id, observation_kinds, forward=False
    )


NORMALIZER_KINDS = ("linear", "multi_robot_linear")


def build_normalizer_module(kind: str) -> Normalizer:
    """Construct an empty normalizer of the kind a resolved run declares.

    Both kinds build the same :class:`Normalizer`; ``kind`` only tells the
    dataset whether to fit per-robot state.
    """
    if str(kind) not in NORMALIZER_KINDS:
        raise ValueError(f"unknown normalizer kind {kind!r}")
    return Normalizer()
