"""RVT2Heatmap configuration defaults, shared across training and inference."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from visuomotor.paths import WEIGHTS_DIR
from visuomotor.perception.focus.seeker.config import DINOV3_CHECKPOINT_NAME


@dataclass(frozen=True)
class RVT2QueryComposerSpec:
    """Query-composer defaults; gripper_only matches the heuristic progress signal."""

    proprio_mode: str = "gripper_only"
    task_emb_dim: int = 512
    hidden_mult: int = 4
    proprio_dim: int = 1


@dataclass(frozen=True)
class RVT2HeatmapSpec:
    """RVT2Heatmap defaults shared across training and inference.

    ``patch_backbone`` selects frozen DINOv3 patch tokens ("dino") or a
    trainable MVT-style patch stem ("conv").
    """

    camera: str = "external"
    patch_backbone: str = "dino"
    conv_patch_dim: int = 384
    dino_image_size: int = 224
    patch_size: int = 16
    hidden_dim: int = 256
    transformer_depth: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    target_sigma_patches: float = 0.75
    joint_vel_atol: float = 0.1
    stopped_buffer_len: int = 4
    include_final: bool = True
    mute_initial_gripper_open: bool = True
    keypoint_box_zoom: float = 4.0


def _apply_overrides(spec, overrides, name: str):
    if not overrides:
        return spec
    overrides = {str(key): value for key, value in overrides.items()}
    unknown = sorted(set(overrides).difference({entry.name for entry in fields(spec)}))
    if unknown:
        raise ValueError(f"unknown {name} overrides: {unknown}")
    return replace(spec, **overrides)


def load_rvt2_query_config(overrides=None) -> dict:
    section = (overrides or {}).get("query_composer")
    spec = _apply_overrides(RVT2QueryComposerSpec(), section, "query_composer")
    return {
        "proprio_mode": str(spec.proprio_mode),
        "task_emb_dim": int(spec.task_emb_dim),
        "hidden_mult": int(spec.hidden_mult),
        "proprio_dim": int(spec.proprio_dim),
    }


def load_rvt2_heatmap_config(overrides=None) -> dict:
    section = (overrides or {}).get("rvt2_heatmap")
    spec = _apply_overrides(RVT2HeatmapSpec(), section, "rvt2_heatmap")
    config = {entry.name: getattr(spec, entry.name) for entry in fields(spec)}
    config["patch_backbone"] = str(spec.patch_backbone).lower()
    config["dino_ckpt"] = str((WEIGHTS_DIR / DINOV3_CHECKPOINT_NAME).resolve())
    return config
