"""Proprioception-conditioned FiLM query builder for focus refinement.

Ported from ``seeker-dev``'s ``focus-pool`` branch
(``seeker/model/stage_pooled_resnet.py::QueryComposer``). Refiners explicitly
select any combination of normalized EEF position and gripper opening.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch
from torch import nn

QUERY_COND_DIMS = {"eef": 3, "gripper": 1}
QUERY_COND_KEYS = {"eef": "eef_pos", "gripper": "gripper_opening"}


def _select_query_proprio(
    composer_in: Mapping[str, torch.Tensor], query_cond: Tuple[str, ...]
) -> Optional[torch.Tensor]:
    if not query_cond:
        return None
    parts = [composer_in[QUERY_COND_KEYS[name]] for name in query_cond]
    return torch.cat(parts, dim=-1)


class QueryComposer(nn.Module):
    """A learned per-head base query, FiLM-modulated by proprioception."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        query_cond: Tuple[str, ...] = ("gripper",),
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        unknown = set(query_cond) - set(QUERY_COND_DIMS)
        if unknown:
            raise ValueError(f"unknown query_cond entries: {sorted(unknown)}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.query_cond = tuple(query_cond)
        proprio_dim = sum(QUERY_COND_DIMS[name] for name in self.query_cond)

        self.query_token = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        nn.init.normal_(self.query_token, std=0.02)
        self.proprio_to_film: Optional[nn.Module] = None
        if proprio_dim > 0:
            hidden_dim = 2 * self.head_dim
            self.proprio_to_film = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * self.head_dim),
                nn.Tanh(),
            )

    def forward(
        self,
        composer_in: Mapping[str, torch.Tensor],
        batch_size: int,
        noise: float = 0.0,
    ) -> torch.Tensor:
        proprio = _select_query_proprio(composer_in, self.query_cond)
        if proprio is None:
            token = self.query_token.unsqueeze(0).expand(batch_size, -1, -1)
            return token.reshape(batch_size, self.dim)

        proprio = proprio.unsqueeze(1).expand(-1, self.num_heads, -1)
        if noise > 0.0 and self.training:
            proprio = proprio + torch.randn_like(proprio) * noise
        token = self.query_token.unsqueeze(0).expand(proprio.shape[0], -1, -1)
        gamma, beta = self.proprio_to_film(proprio).chunk(2, dim=-1)
        query = (1.0 + gamma) * token + beta
        return query.reshape(proprio.shape[0], self.dim)
