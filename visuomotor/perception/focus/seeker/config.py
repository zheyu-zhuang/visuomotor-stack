"""Typed Seeker model settings and pretty-print helpers.

Defaults live here rather than in YAML: Hydra only needs to override them when
an experiment genuinely varies them.
"""

from dataclasses import dataclass, fields, replace
from typing import Any, Iterable, Mapping, Optional

from visuomotor.paths import WEIGHTS_DIR

DINOV3_CHECKPOINT_NAME = "dinov3.vits16plus.pth"


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    ckpt_path: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backbone.name must be set")
        if not self.ckpt_path:
            raise ValueError("backbone.ckpt_path must be set")


@dataclass(frozen=True)
class QueryComposerBaseConfig:
    use_rotation: bool
    task_emb_dim: int
    hidden_mult: int
    proprio_dim: int
    disable_proprio: bool

    def __post_init__(self) -> None:
        if self.task_emb_dim <= 0:
            raise ValueError("query_composer.task_emb_dim must be > 0")
        if self.hidden_mult <= 0:
            raise ValueError("query_composer.hidden_mult must be > 0")
        if self.proprio_dim <= 0:
            raise ValueError("query_composer.proprio_dim must be > 0")


@dataclass(frozen=True)
class QueryComposerConfig(QueryComposerBaseConfig):
    emb_dim: int
    num_robots: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.emb_dim <= 0:
            raise ValueError("query_composer.emb_dim must be > 0")
        if self.num_robots <= 0:
            raise ValueError("query_composer.num_robots must be > 0")


@dataclass(frozen=True)
class IntentRefinerBaseConfig:
    num_refinement_iters: int
    entmax_alpha: float
    hidden_multiplier: int
    disable_head_gating: bool

    def __post_init__(self) -> None:
        if self.num_refinement_iters < 1:
            raise ValueError("intent_refiner.num_refinement_iters must be >= 1")
        if self.entmax_alpha <= 1.0:
            raise ValueError("intent_refiner.entmax_alpha must be > 1.0")
        if self.hidden_multiplier <= 0:
            raise ValueError("intent_refiner.hidden_multiplier must be > 0")


@dataclass(frozen=True)
class IntentRefinerConfig(IntentRefinerBaseConfig):
    emb_dim: int
    num_heads: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.emb_dim <= 0:
            raise ValueError("intent_refiner.emb_dim must be > 0")
        if self.num_heads <= 0:
            raise ValueError("intent_refiner.num_heads must be > 0")


@dataclass(frozen=True)
class SeekerRuntimeConfig:
    top_p: float
    select_n_heads: int

    def __post_init__(self) -> None:
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("seeker.top_p must be in (0, 1]")
        if self.select_n_heads < 1:
            raise ValueError("seeker.select_n_heads must be >= 1")


@dataclass(frozen=True)
class SeekerModelConfig:
    backbone: BackboneConfig
    query_composer: QueryComposerConfig
    intent_refiner: IntentRefinerConfig
    seeker: SeekerRuntimeConfig


@dataclass(frozen=True)
class SeekerBaseConfig:
    backbone: BackboneConfig
    query_composer: QueryComposerBaseConfig
    intent_refiner: IntentRefinerBaseConfig
    seeker: SeekerRuntimeConfig


def default_seeker_base_config() -> SeekerBaseConfig:
    return SeekerBaseConfig(
        backbone=BackboneConfig(
            name="DINOv3",
            ckpt_path=str((WEIGHTS_DIR / DINOV3_CHECKPOINT_NAME).resolve()),
        ),
        query_composer=QueryComposerBaseConfig(
            use_rotation=False,
            task_emb_dim=512,
            hidden_mult=2,
            proprio_dim=4,
            disable_proprio=False,
        ),
        intent_refiner=IntentRefinerBaseConfig(
            num_refinement_iters=3,
            entmax_alpha=1.3,
            hidden_multiplier=4,
            disable_head_gating=False,
        ),
        seeker=SeekerRuntimeConfig(top_p=0.8, select_n_heads=4),
    )


def _plain_mapping(value):
    if hasattr(value, "items"):
        return {str(key): _plain_mapping(item) for key, item in value.items()}
    return value


def _override_section(section, overrides: Optional[Mapping[str, Any]], name: str):
    if not overrides:
        return section
    allowed = {entry.name for entry in fields(section)}
    unknown = sorted(set(overrides).difference(allowed))
    if unknown:
        raise ValueError(f"unknown {name} overrides: {unknown}")
    return replace(section, **dict(overrides))


def load_seeker_base_config(
    overrides: Optional[Mapping[str, Any]] = None,
) -> SeekerBaseConfig:
    """Return the Seeker defaults with any per-section overrides applied."""
    base = default_seeker_base_config()
    if not overrides:
        return base
    overrides = _plain_mapping(overrides)
    unknown = sorted(set(overrides).difference({entry.name for entry in fields(base)}))
    if unknown:
        raise ValueError(f"unknown Seeker config sections: {unknown}")
    return SeekerBaseConfig(
        backbone=_override_section(base.backbone, overrides.get("backbone"), "backbone"),
        query_composer=_override_section(
            base.query_composer, overrides.get("query_composer"), "query_composer"
        ),
        intent_refiner=_override_section(
            base.intent_refiner, overrides.get("intent_refiner"), "intent_refiner"
        ),
        seeker=_override_section(base.seeker, overrides.get("seeker"), "seeker"),
    )


def resolve_seeker_model_config(
    base_cfg: SeekerBaseConfig,
    *,
    emb_dim: int,
    num_heads: int,
    num_robots: int,
) -> SeekerModelConfig:
    """Resolve runtime-derived dimensions into the final Seeker component configs."""
    emb_dim = int(emb_dim)
    num_heads = int(num_heads)
    num_robots = int(num_robots)

    return SeekerModelConfig(
        backbone=base_cfg.backbone,
        query_composer=QueryComposerConfig(
            use_rotation=base_cfg.query_composer.use_rotation,
            task_emb_dim=base_cfg.query_composer.task_emb_dim,
            hidden_mult=base_cfg.query_composer.hidden_mult,
            proprio_dim=base_cfg.query_composer.proprio_dim,
            disable_proprio=base_cfg.query_composer.disable_proprio,
            emb_dim=emb_dim,
            num_robots=num_robots,
        ),
        intent_refiner=IntentRefinerConfig(
            num_refinement_iters=base_cfg.intent_refiner.num_refinement_iters,
            entmax_alpha=base_cfg.intent_refiner.entmax_alpha,
            hidden_multiplier=base_cfg.intent_refiner.hidden_multiplier,
            disable_head_gating=base_cfg.intent_refiner.disable_head_gating,
            emb_dim=emb_dim,
            num_heads=num_heads,
        ),
        seeker=base_cfg.seeker,
    )


def build_seeker_pretty_config(
    *,
    view_names: Iterable[str],
    ckpt_path: Optional[str],
    cfg: SeekerModelConfig,
    out_dim: int,
) -> dict[str, Any]:
    def pretty_view_name(name: str) -> str:
        if name == "external":
            return "External"
        if name == "wrist":
            return "Wrist"
        return name.replace("_", " ").title()

    stages_str = "coarse -> fine"
    names = list(view_names)
    branch_lines = []
    for i, view in enumerate(names):
        prefix = "└─" if i == len(names) - 1 else "├─"
        branch_lines.append(f"{prefix} {pretty_view_name(view):<12s}: {stages_str}")

    return {
        "Initialization": {
            "Source": "Checkpoint" if ckpt_path else "Random Initialization",
            "Checkpoint Path": ckpt_path,
        },
        "Seeker Tower": {
            "Backbone": f"{cfg.backbone.name} (frozen)",
            "Branches": branch_lines,
        },
        "Query Composer": {
            "Use rotation": cfg.query_composer.use_rotation,
            "Task emb dim": cfg.query_composer.task_emb_dim,
            "Disable proprio": cfg.query_composer.disable_proprio,
        },
        "Intent Refiner": {
            "Refinement iters": cfg.intent_refiner.num_refinement_iters,
            "Entmax alpha": cfg.intent_refiner.entmax_alpha,
            "Head gating": (
                "disabled (uniform)"
                if cfg.intent_refiner.disable_head_gating
                else "intent_alignment (fixed)"
            ),
            "Output dim": out_dim,
        },
        "Masking": {
            "Top-p": cfg.seeker.top_p,
            "Select heads": cfg.seeker.select_n_heads,
        },
    }
