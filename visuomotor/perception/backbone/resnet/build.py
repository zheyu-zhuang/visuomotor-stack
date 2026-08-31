"""Raw ResNet-18 construction and stage-decomposition helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

GROUPNORM_DIVISOR = 16


@dataclass(frozen=True)
class ResNetStageShapes:
    stage_channels: int
    stage_grid_h: int
    stage_grid_w: int


def build_resnet18_backbone(
    *,
    pretrained_imagenet: bool = True,
    norm: str = "groupnorm",
) -> nn.Module:
    """A torchvision ResNet-18, optionally replacing every BatchNorm2d with GroupNorm.

    Policy EMA uses GroupNorm to avoid moving averages of BatchNorm statistics, at the
    cost of degrading the pretrained RGB representation. ``norm="batchnorm"`` keeps
    torchvision's default BatchNorm2d. FocusPool prefers vanilla ResNet with pretrained
    weights.
    """
    if norm not in ("groupnorm", "batchnorm"):
        raise ValueError(f"norm must be 'groupnorm' or 'batchnorm', got {norm!r}")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
    backbone = resnet18(weights=weights)
    if norm == "batchnorm":
        return backbone

    matches = [
        key.split(".")
        for key, module in backbone.named_modules(remove_duplicate=True)
        if isinstance(module, nn.BatchNorm2d)
    ]
    for *parent, key in matches:
        parent_module = backbone
        if parent:
            parent_module = backbone.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            bn = parent_module[int(key)]
            channels = bn.num_features
            parent_module[int(key)] = nn.GroupNorm(
                max(1, channels // GROUPNORM_DIVISOR),
                channels,
            )
        else:
            bn = getattr(parent_module, key)
            channels = bn.num_features
            setattr(
                parent_module,
                key,
                nn.GroupNorm(max(1, channels // GROUPNORM_DIVISOR), channels),
            )

    assert not any(isinstance(module, nn.BatchNorm2d) for module in backbone.modules())
    return backbone


def build_resnet18_stages(
    *,
    pretrained_imagenet: bool = True,
    norm: str = "groupnorm",
) -> nn.Module:
    """A :func:`build_resnet18_backbone` with a synthesized ``.stem`` attribute.

    The single path to a ResNet-18 that can be consumed either as a complete
    backbone or stage-by-stage (see :func:`get_resnet18_stage_modules` and
    :func:`run_resnet18_until`).
    """
    backbone = build_resnet18_backbone(pretrained_imagenet=pretrained_imagenet, norm=norm)
    backbone.stem = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
    )
    return backbone


def get_resnet18_stage_modules(backbone: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """Named ``(stem, l1, l2, l3, l4)`` stages of a :func:`build_resnet18_stages` backbone."""
    return (
        ("stem", backbone.stem),
        ("l1", backbone.layer1),
        ("l2", backbone.layer2),
        ("l3", backbone.layer3),
        ("l4", backbone.layer4),
    )


def run_resnet18_until(backbone: nn.Module, x: torch.Tensor, stage: str) -> torch.Tensor:
    """Run ``backbone`` forward, stopping as soon as ``stage``'s output is produced.

    Later stages are never invoked, so pooling at an early stage (e.g. ``"l2"``)
    doesn't pay for ``layer3``/``layer4``'s compute.
    """
    for name, module in get_resnet18_stage_modules(backbone):
        x = module(x)
        if name == stage:
            return x
    raise ValueError(f"Unsupported stage: {stage!r}")


def probe_resnet18_stage_shapes(
    backbone: nn.Module,
    *,
    input_res: int,
    stage: str,
    input_channels: int = 3,
) -> ResNetStageShapes:
    param = next(backbone.parameters())
    probe = torch.zeros(
        1,
        int(input_channels),
        int(input_res),
        int(input_res),
        device=param.device,
        dtype=param.dtype,
    )
    with torch.no_grad():
        probe = run_resnet18_until(backbone, probe, stage)

    return ResNetStageShapes(
        stage_channels=probe.shape[1],
        stage_grid_h=probe.shape[2],
        stage_grid_w=probe.shape[3],
    )
