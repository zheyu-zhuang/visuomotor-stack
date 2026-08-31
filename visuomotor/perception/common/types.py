"""Small cross-domain contracts with no implementation dependencies."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class EncoderOutput:
    """Output shared by all observation encoders."""

    features: torch.Tensor
    streams: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention: Optional[torch.Tensor] = None
    geometry: Optional[Any] = None
    prepared_inputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_geometry: Optional[Any] = None
    voxel_crop_geometry: Optional[Any] = None
    voxel_crop_transform: Optional[Any] = None
    focus_records: Tuple[Any, ...] = ()
    auxiliary_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
