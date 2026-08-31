"""Raw DINOv3 ViT loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from visuomotor.perception.backbone.dinov3_core.make_dinov3_vits import (
    dinov3_vits16plus,
)


def load_frozen_dinov3_vits16plus(
    checkpoint: Union[str, Path],
    *,
    device: Optional[Union[torch.device, str]] = None,
) -> nn.Module:
    """Load a frozen DINOv3 ViT-S/16+ backbone from a local checkpoint."""
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")
    logging.getLogger("dinov3").setLevel(logging.WARNING)
    backbone = dinov3_vits16plus(pretrained=False)
    backbone.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    if device is not None:
        backbone.to(device)
    backbone.eval().requires_grad_(False)
    return backbone
