"""Resolve Visuomotor Stack resource paths shared across domains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
PATHS_CONFIG = CONFIG_DIR / "paths.yaml"


def _resolve_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class ResourcePaths:
    """Resolved filesystem resources used by the stack."""

    package_root: Path
    dataset_root: Path
    weights_dir: Path
    textures_dir: Path
    backgrounds_dir: Path
    task_embedding_cache: Path


def load_resource_paths(
    overrides: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> ResourcePaths:
    """Load path defaults and apply supported environment overrides."""

    cfg_path = config_path or PATHS_CONFIG
    cfg_omega = OmegaConf.load(cfg_path)
    if overrides:
        cfg_omega = OmegaConf.merge(cfg_omega, OmegaConf.create(dict(overrides)))
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    paths = cfg.get("paths") if isinstance(cfg, Mapping) else None
    if not isinstance(paths, Mapping):
        raise ValueError(f"Missing or invalid 'paths' section in {cfg_path}")

    required = ("dataset_root", "weights_dir", "textures_dir", "backgrounds_dir", "task_emb_cache")
    missing = [key for key in required if key not in paths]
    if missing:
        raise ValueError(f"Missing path config keys: {missing}")
    resolved = {key: _resolve_path(paths[key]) for key in required}
    return ResourcePaths(
        package_root=PACKAGE_ROOT,
        dataset_root=resolved["dataset_root"],
        weights_dir=resolved["weights_dir"],
        textures_dir=resolved["textures_dir"],
        backgrounds_dir=resolved["backgrounds_dir"],
        task_embedding_cache=resolved["task_emb_cache"],
    )


RESOURCE_PATHS = load_resource_paths()
DATASETS_DIR = RESOURCE_PATHS.dataset_root.parent
MIMICGEN_DATASETS_DIR = RESOURCE_PATHS.dataset_root
WEIGHTS_DIR = RESOURCE_PATHS.weights_dir
TEXTURES_DIR = RESOURCE_PATHS.textures_dir
BACKGROUNDS_DIR = RESOURCE_PATHS.backgrounds_dir
