"""Registry of stable per-task benchmark facts.

These are properties of the MimicGen benchmark, not experiment choices, so they
live in Python instead of being interpolated across YAML files.
"""

from __future__ import annotations

import json
import os
import re

from visuomotor.config.schema import TaskSpec
from visuomotor.data.mimicgen import tasks as MimicgenTasks

DEFAULT_MAX_STEPS = 800

# The released MimicGen voxel datasets use a fixed world-Z floor. This is
# independent of the physical table height used to remove table points.
VOXEL_Z_MIN = 0.7

MAX_STEPS = {
    "square_d0": 400,
    "square_d2": 400,
    "stack_d1": 400,
    "stack_three_d1": 400,
    "threading_d2": 400,
    "coffee_d2": 400,
    "three_piece_assembly_d2": 500,
    "hammer_cleanup_d1": 500,
    "mug_cleanup_d1": 500,
    "nut_assembly_d0": 500,
    "kitchen_d1": 800,
    "pick_place_d0": 1000,
}

# Each MimicGen task's robosuite ``table_offset``. Point-cloud production uses
# its Z value to remove the tabletop; voxel production keeps ``VOXEL_Z_MIN``.
# DEFAULT_TABLE_OFFSET covers tasks whose environment defines no table offset.
DEFAULT_TABLE_OFFSET = (0.0, 0.0, 0.7)

TABLE_OFFSET = {
    "square_d0": (0.0, 0.0, 0.82),
    "square_d2": (0.0, 0.0, 0.82),
    "stack_d1": (0.0, 0.0, 0.8),
    "stack_three_d1": (0.0, 0.0, 0.8),
    "threading_d2": (0.0, 0.0, 0.8),
    "coffee_d2": (0.0, 0.0, 0.8),
    "three_piece_assembly_d2": (0.0, 0.0, 0.8),
    "hammer_cleanup_d1": (-0.2, 0.0, 0.9),
    "mug_cleanup_d1": (0.0, 0.0, 0.8),
    "nut_assembly_d0": (0.0, 0.0, 0.82),
    "kitchen_d1": (-0.2, 0.0, 0.9),
    # pick_place_d0's robosuite env class defines no table_offset attribute.
    "pick_place_d0": DEFAULT_TABLE_OFFSET,
}


def canonical_name(name: str) -> str:
    """Normalize either a task name (``square_d0``) or an env name (``Square_D0``)."""
    canonical = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    return re.sub(r"_+", "_", canonical.replace(" ", "_")).lower()


def table_offset(name: str):
    return TABLE_OFFSET.get(canonical_name(name), DEFAULT_TABLE_OFFSET)


def get_task_spec(name: str) -> TaskSpec:
    canonical = canonical_name(name)
    return TaskSpec(
        name=canonical,
        max_steps=MAX_STEPS.get(canonical, DEFAULT_MAX_STEPS),
        spatial_cameras=MimicgenTasks.spatial_cameras(canonical),
        table_offset=table_offset(canonical),
    )


def voxel_bounds(task: TaskSpec, ws_size: float):
    """Return Equidiff-compatible task bounds for voxel reconstruction."""
    half = float(ws_size) / 2.0
    center_x, center_y, _ = task.table_offset
    return (
        (center_x - half, center_y - half, VOXEL_Z_MIN),
        (center_x + half, center_y + half, VOXEL_Z_MIN + float(ws_size)),
    )


def point_cloud_bounds(task: TaskSpec, ws_size: float, table_margin: float):
    """Return task-table bounds for point-cloud reconstruction."""
    half = float(ws_size) / 2.0
    center_x, center_y, center_z = task.table_offset
    return (
        (
            center_x - half,
            center_y - half,
            center_z + float(table_margin),
        ),
        (
            center_x + half,
            center_y + half,
            center_z + float(ws_size),
        ),
    )


def dataset_robot_ids(dataset_path: str, task_name: str):
    """Robot ids present in a rendered observation cache.

    A single-task cache holds one robot; a cache merged from several tasks spans
    every robot those tasks use, which is what makes per-robot normalization
    necessary. Falls back to the task name when the cache has not been built yet.
    """
    from visuomotor.data.mimicgen.tasks import env_name_to_robot_id

    names = [str(task_name)]
    meta_path = os.path.join(str(dataset_path), "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as handle:
            source_names = json.load(handle).get("source_task_names") or ()
        names = [str(name) for name in source_names] or names
    return tuple(sorted({env_name_to_robot_id(name) for name in names}))
