"""DP3 PointNet backbone for fixed-size XYZ/RGB point clouds."""

from __future__ import annotations

from torch import nn


class PointNetBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = 64,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
    ) -> None:
        super().__init__()
        hidden_dims = (64, 128, 256, 512) if input_dim > 3 else (64, 128, 256)
        layers = []
        width = int(input_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(width, hidden))
            if hidden != hidden_dims[-1]:
                if use_layernorm:
                    layers.append(nn.LayerNorm(hidden))
                layers.append(nn.ReLU())
            width = hidden
        self.mlp = nn.Sequential(*layers)
        if final_norm == "layernorm":
            self.projection = nn.Sequential(
                nn.Linear(width, int(output_dim)), nn.LayerNorm(int(output_dim))
            )
        elif final_norm == "none":
            self.projection = nn.Linear(width, int(output_dim))
        else:
            raise ValueError(f"unknown PointNet final norm {final_norm!r}")

    def forward(self, points):
        return self.projection(self.mlp(points).amax(dim=1))
