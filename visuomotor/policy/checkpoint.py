"""Checkpoint payload serialization, validation, and weight stripping."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import dill
import torch

from visuomotor.config.schema import to_dict

CHECKPOINT_SCHEMA_VERSION = 3
RELEASE_CHECKPOINT_SCHEMA_VERSION = 3
REQUIRED_KEYS = {
    "schema_version",
    "run_spec",
    "policy",
    "ema",
    "optimizer",
    "scheduler",
    "normalizer",
    "epoch",
    "global_step",
    "rng",
}


def save_checkpoint(
    path,
    *,
    run_spec,
    policy,
    ema,
    optimizer,
    scheduler,
    normalizer,
    epoch,
    global_step,
):
    """Save complete training and rollout state in the versioned schema."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_spec": to_dict(run_spec),
        "policy": policy.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "normalizer": normalizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path) -> Mapping:
    """Validate and return a complete training checkpoint."""
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {payload['schema_version']}")
    return payload


def strip_seeker_backbone(state_dict: dict, *, prefix: str = "vit.") -> dict:
    """Drop frozen DINOv3 backbone tensors from a flat Seeker state_dict.

    ``Seeker.load_pretrained_weights`` loads the backbone separately (see
    ``Seeker._init_dinov3``) and excludes ``vit.*`` keys from its strict
    compatibility check, so these tensors are redundant in a released
    checkpoint and are only safe to drop if the backbone was frozen
    throughout training.
    """
    kept = {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
    removed = len(state_dict) - len(kept)
    print(f"[strip_backbone] Removed {removed} '{prefix}*' tensors, kept {len(kept)}")
    return kept


def strip_rvt2_heatmap_backbone(payload: dict) -> dict:
    """Drop ``patch_backbone_state_dict`` from an RVT2Heatmap checkpoint payload.

    ``RVT2Heatmap.__init__`` tolerates a missing ``patch_backbone_state_dict``
    for a "dino" backbone, since ``PatchFeatureBackbone`` already loads the
    same frozen weights from ``dino_ckpt_path`` during construction. A
    "conv" backbone is trained, so its weights must not be dropped.
    """
    if "patch_backbone_state_dict" not in payload:
        print("[strip_backbone] No patch_backbone_state_dict present; nothing to strip")
        return payload
    if str(payload.get("patch_backbone")) != "dino":
        raise ValueError(
            "Refusing to strip patch_backbone_state_dict: patch_backbone is "
            f"{payload.get('patch_backbone')!r}, not 'dino' (only the frozen "
            "DINOv3 backbone is safe to drop; a 'conv' backbone is trained)."
        )
    stripped = dict(payload)
    removed = len(stripped.pop("patch_backbone_state_dict"))
    print(f"[strip_backbone] Removed patch_backbone_state_dict ({removed} tensors)")
    return stripped


def strip_backbone(payload) -> dict:
    """Remove frozen DINOv3 backbone weights from a Seeker or RVT2Heatmap checkpoint.

    The removed tensors must be verified byte-identical to the separately
    distributed ``dinov3.vits16plus.pth`` before stripping is safe; this
    function does not re-verify that per call.
    """
    if isinstance(payload, dict) and "patch_backbone_state_dict" in payload:
        return strip_rvt2_heatmap_backbone(payload)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        stripped = dict(payload)
        stripped["state_dict"] = strip_seeker_backbone(payload["state_dict"])
        return stripped
    return strip_seeker_backbone(payload)


def load_release_checkpoint_payload(path, *, map_location="cpu") -> Mapping:
    """Validate and return a release-checkpoint payload without construction."""
    payload = load_serialized_payload(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("policy checkpoint must contain a mapping")
    version = payload.get("schema_version")
    if version != RELEASE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported policy checkpoint schema {version!r}; "
            f"expected {RELEASE_CHECKPOINT_SCHEMA_VERSION}"
        )
    missing = [key for key in ("model_spec", "state_dict") if key not in payload]
    if missing:
        raise ValueError(f"policy checkpoint is missing required fields: {missing}")
    return payload


def load_serialized_payload(path, *, map_location="cpu") -> Mapping:
    """Load either repository-owned checkpoint container without interpreting it."""
    payload = torch.load(Path(path), map_location=map_location, pickle_module=dill)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    return payload
