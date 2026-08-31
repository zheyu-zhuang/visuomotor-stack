"""Shared contracts and utilities for dataset rerendering."""

from __future__ import annotations

import json
import os
import queue
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np

from visuomotor.data.core import images as CoreImages

SPATIAL_OBS_KEYS = frozenset({"voxel", "point_cloud"})
COMMIT_EVERY_DEFAULT = 5000
LMDB_MAP_SIZE_GB_DEFAULT = 32
JPEG_QUALITY_DEFAULT = CoreImages.JPEG_QUALITY_DEFAULT
PROGRESS_EVENT_DEMO_DONE = "demo_done"
PROGRESS_POLL_INTERVAL_SEC = 0.2


@dataclass
class RenderedEpisode:
    absolute_action: np.ndarray
    lowdim: Dict[str, np.ndarray]
    rgb_jpeg: Dict[str, List[bytes]]
    voxel_frames: Dict[str, List[tuple]]
    point_cloud_frames: List[bytes]
    oracle: Dict[str, np.ndarray]
    success: bool


def _remove_existing_output_dir(path: Union[str, Path], *, overwrite: bool) -> None:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return
    if not overwrite:
        resp = input(f"Output dir exists: {path}\nOverwrite? (y/n): ").strip().lower()
        if resp != "y":
            raise SystemExit("Canceled.")
    shutil.rmtree(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _camera_obs_keys(camera_names: List[str]) -> List[str]:
    # Normalize to robosuite obs key convention: "<camera>_image"
    out = []
    for name in camera_names:
        if name.endswith("_image"):
            out.append(name)
        else:
            out.append(f"{name}_image")
    return out


def _sorted_demo_keys(h5_file: h5py.File) -> List[str]:
    return sorted(h5_file["data"].keys(), key=lambda x: int(x[5:]))


def _source_demo_indices(
    dataset_path: str,
    start_index: int,
    n_demo: Optional[int],
) -> List[int]:
    """Return source demo indices selected for rerender."""
    with h5py.File(str(Path(dataset_path).expanduser().resolve()), "r") as h5_file:
        demos = _sorted_demo_keys(h5_file)
    start_index = int(start_index)
    if start_index < 0 or start_index >= len(demos):
        raise ValueError(
            f"start episode {start_index} out of range (episodes={len(demos)})"
        )
    selected = [int(ep[5:]) for ep in demos[start_index:]]
    if n_demo is not None:
        selected = selected[: int(n_demo)]
    if not selected:
        raise ValueError("No demos selected for rerender")
    return selected


def _sibling_cache_dir_with_suffix(cache_dir: Union[str, Path], suffix: str) -> str:
    cache_path = Path(cache_dir).expanduser().resolve()
    if not suffix:
        return str(cache_path)
    return str(cache_path.parent / f"{cache_path.name}{suffix}")


def _discover_task_datasets(
    datasets_root: str,
    tasks: Optional[List[str]] = None,
) -> List[tuple[str, str]]:
    """Discover raw task HDF5 files under a MimicGen-style datasets root."""
    root = Path(datasets_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"datasets_root not found: {root}")

    if tasks:
        task_names = [str(task) for task in tasks]
    else:
        task_names = sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir() and (child / f"{child.name}.hdf5").is_file()
        )

    if not task_names:
        raise ValueError(f"No task datasets found under {root}")

    datasets = []
    missing = []
    for task_name in task_names:
        hdf5_path = root / task_name / f"{task_name}.hdf5"
        if hdf5_path.is_file():
            datasets.append((task_name, str(hdf5_path)))
        else:
            missing.append(str(hdf5_path))
    if missing:
        raise FileNotFoundError(
            "Missing task HDF5 files:\n" + "\n".join(f"  {path}" for path in missing)
        )
    return datasets


def _split_contiguous(values: List[int], num_chunks: int) -> List[List[int]]:
    """Split values into contiguous non-empty chunks."""
    num_chunks = max(1, min(int(num_chunks), len(values)))
    base = len(values) // num_chunks
    rem = len(values) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < rem else 0)
        chunks.append(values[start : start + size])
        start += size
    return [chunk for chunk in chunks if chunk]


def _drain_progress_events(
    progress_queue,
    pbar,
    samples: int,
    demos: int,
) -> tuple[int, int, bool]:
    drained = False
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break
        drained = True
        if event.get("event") != PROGRESS_EVENT_DEMO_DONE:
            continue
        samples += int(event["n_samples"])
        demos += 1
        pbar.update(1)
        pbar.set_postfix_str(f"samples={samples}")
    return samples, demos, drained
