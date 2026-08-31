#!/usr/bin/env python3
"""Merge current rerendered task LMDB caches into one multitask cache."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

import lmdb
import numpy as np
from tqdm import tqdm

from visuomotor.data.core import images as CoreImages
from visuomotor.data.core import sparse_voxels as SparseVoxels
from visuomotor.data.mimicgen import cache as MgCache
from visuomotor.data.mimicgen.action import absolute_posmat_to_delta_chunks
from visuomotor.paths import MIMICGEN_DATASETS_DIR

DATASETS_ROOT = MIMICGEN_DATASETS_DIR
LMDB_MAP_SIZE_GB = 64
LMDB_COMMIT_EVERY_PUTS = 5000
REQUIRED_LOWDIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_rot",
    "robot0_gripper_qpos",
)


def _voxel_metadata_by_key(meta: dict, voxel_keys: list[str]) -> dict:
    metadata = meta.get("voxel_specs")
    if metadata is not None:
        return {str(key): value for key, value in dict(metadata).items()}
    if voxel_keys and meta.get("voxel_spec") is not None:
        return {voxel_keys[0]: meta["voxel_spec"]}
    return {}


def _voxel_field_by_key(meta: dict, field: str, voxel_keys: list[str]) -> dict:
    value = meta.get(field)
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {key: value for key in voxel_keys}


def _open_lmdb_read(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=False,
        max_readers=2048,
    )


def _open_lmdb_write(path: Path, map_size_gb: int) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        map_size=int(map_size_gb * (1024**3)),
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
        max_dbs=1,
    )


def _concat_oracle_chunks(key: str, chunks: list[np.ndarray]) -> np.ndarray:
    if key != "target_points":
        return np.concatenate(chunks, axis=0)
    max_points = max(int(chunk.shape[1]) for chunk in chunks)
    padded = []
    for chunk in chunks:
        if int(chunk.shape[1]) < max_points:
            pad = np.full(
                (
                    chunk.shape[0],
                    max_points - int(chunk.shape[1]),
                    chunk.shape[2],
                ),
                np.nan,
                dtype=chunk.dtype,
            )
            chunk = np.concatenate([chunk, pad], axis=1)
        padded.append(chunk)
    return np.concatenate(padded, axis=0)


def _mergeable_lowdim_keys(
    in_dirs: list[Path],
    metas: list[dict],
) -> tuple[list[str], list[str]]:
    """Keep only lowdim arrays that are structurally compatible across tasks."""

    ref_keys = [str(key) for key in metas[0]["lowdim_keys"]]
    key_sets = [set(map(str, meta["lowdim_keys"])) for meta in metas]
    kept = []
    skipped = []
    for key in ref_keys:
        if not all(key in keys for keys in key_sets):
            skipped.append(f"{key} (missing)")
            continue
        shapes = [
            tuple(
                MgCache.load_numpy_array(str(cache_dir), f"lowdim/{key}.npy").shape[1:]
            )
            for cache_dir in in_dirs
        ]
        if len(set(shapes)) != 1:
            skipped.append(f"{key} (shapes={shapes})")
            continue
        kept.append(key)

    missing_required = [key for key in REQUIRED_LOWDIM_KEYS if key not in kept]
    if missing_required:
        raise ValueError(
            "Required lowdim keys are not mergeable: "
            + ", ".join(missing_required)
            + ". Check rerendered cache lowdim arrays."
        )
    return kept, skipped


def merge_caches(
    in_dirs: list[Path],
    out_dir: Path,
    n_demo_per_input: Optional[int],
    source_task_names: Optional[List[str]] = None,
    episode_indices_per_input: Optional[List[List[int]]] = None,
    overwrite: bool = False,
    lmdb_map_size_gb: int = LMDB_MAP_SIZE_GB,
    commit_every: int = LMDB_COMMIT_EVERY_PUTS,
    delta_horizons: Optional[List[int]] = None,
) -> None:
    """Take the first N demos from each input cache and concatenate them."""
    if not in_dirs:
        raise ValueError("in_dirs is empty")

    in_dirs = [path.expanduser().resolve() for path in in_dirs]
    out_dir = out_dir.expanduser().resolve()
    source_task_names = list(
        source_task_names or [path.parent.name for path in in_dirs]
    )
    if len(source_task_names) != len(in_dirs):
        raise ValueError("source_task_names must match in_dirs length")
    if episode_indices_per_input is not None and len(episode_indices_per_input) != len(
        in_dirs
    ):
        raise ValueError("episode_indices_per_input must match in_dirs length")

    if out_dir.exists():
        if not overwrite:
            resp = input(f"Output dir exists: {out_dir}\nOverwrite? (y/n): ").strip()
            if resp.lower() != "y":
                raise SystemExit("Canceled.")
        shutil.rmtree(out_dir)

    metas = []
    for cache_dir in in_dirs:
        with (cache_dir / "meta.json").open("r") as f:
            metas.append(json.load(f))

    ref = metas[0]
    rgb_keys = [str(key) for key in ref["rgb_keys"]]
    oracle_keys = [str(key) for key in ref.get("oracle_keys", [])]
    voxel_keys = [str(key) for key in ref.get("voxel_keys", [])]
    point_cloud_keys = [str(key) for key in ref.get("point_cloud_keys", [])]

    if any([str(key) for key in meta["rgb_keys"]] != rgb_keys for meta in metas):
        raise ValueError("All input caches must use the same rgb_keys")
    if any(
        [str(key) for key in meta.get("oracle_keys", [])] != oracle_keys
        for meta in metas
    ):
        raise ValueError("All input caches must use the same oracle_keys")
    if any(
        [str(key) for key in meta.get("voxel_keys", [])] != voxel_keys for meta in metas
    ):
        raise ValueError("All input caches must use the same voxel_keys")
    if voxel_keys and any(
        any(
            storage != "sparse"
            for storage in _voxel_field_by_key(
                meta, "voxel_storage", voxel_keys
            ).values()
        )
        for meta in metas
    ):
        raise ValueError(
            "All input caches must use the sparse occupied-cell voxel storage "
            "format; re-render older caches with --enable-voxel"
        )
    ref_voxel_metadata = _voxel_metadata_by_key(ref, voxel_keys)
    if voxel_keys and any(
        _voxel_metadata_by_key(meta, voxel_keys) != ref_voxel_metadata
        for meta in metas
    ):
        raise ValueError("All input voxel caches must use the same per-key specs")
    if any(
        [str(key) for key in meta.get("point_cloud_keys", [])] != point_cloud_keys
        for meta in metas
    ):
        raise ValueError("All input caches must use the same point_cloud_keys")
    if point_cloud_keys and any(
        meta.get("point_cloud_spec") != ref.get("point_cloud_spec") for meta in metas
    ):
        raise ValueError(
            "All input point-cloud caches must use the same point_cloud_spec"
        )
    lowdim_keys, skipped_lowdim_keys = _mergeable_lowdim_keys(in_dirs, metas)
    if skipped_lowdim_keys:
        print("[merge] skipped incompatible lowdim keys:")
        for key in skipped_lowdim_keys:
            print(f"  - {key}")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_env = _open_lmdb_write(out_dir / "images.lmdb", map_size_gb=lmdb_map_size_gb)
    out_txn = out_env.begin(write=True)
    out_puts = 0

    # Selected frames' occupied cells, gathered in output order.
    out_voxel_frames = {key: [] for key in voxel_keys}

    episode_lengths = []
    source_episode_counts_by_task = {}
    task_name_to_id = {}
    unique_task_names = []
    task_embeddings = []
    task_language_tokens = []
    task_ids = []
    robot_ids = []
    task_instructions = []
    source_demo_indices = []
    table_texture_episode_files = []
    table_texture_meta = None
    action_abs = []
    lowdim = {key: [] for key in lowdim_keys}
    oracle = {key: [] for key in oracle_keys}
    global_step = 0

    for input_idx, (cache_dir, meta) in enumerate(zip(in_dirs, metas)):
        ep_lens = list(map(int, meta["episode_lengths"]))
        if episode_indices_per_input is None:
            n_take = len(ep_lens) if n_demo_per_input is None else int(n_demo_per_input)
            if n_take < 1 or n_take > len(ep_lens):
                raise ValueError(
                    f"{cache_dir}: requested {n_take} demos, found {len(ep_lens)}"
                )
            selected_episodes = np.arange(n_take, dtype=np.int64)
        else:
            selected_episodes = np.asarray(
                episode_indices_per_input[input_idx],
                dtype=np.int64,
            )
            if selected_episodes.size == 0:
                continue
            if selected_episodes.min() < 0 or selected_episodes.max() >= len(ep_lens):
                raise ValueError(
                    f"{cache_dir}: selected episode indices out of range "
                    f"for {len(ep_lens)} demos"
                )

        task_name = str(source_task_names[input_idx])
        if task_name not in task_name_to_id:
            task_name_to_id[task_name] = len(unique_task_names)
            unique_task_names.append(task_name)
            source_episode_counts_by_task[task_name] = 0
        task_id = task_name_to_id[task_name]

        cum_lens = np.cumsum([0, *ep_lens], dtype=np.int64)
        step_indices = np.concatenate(
            [
                np.arange(cum_lens[ep_idx], cum_lens[ep_idx + 1], dtype=np.int64)
                for ep_idx in selected_episodes.tolist()
            ],
            axis=0,
        )
        selected_lengths = [
            ep_lens[int(ep_idx)] for ep_idx in selected_episodes.tolist()
        ]

        episode_lengths.extend(selected_lengths)
        source_episode_counts_by_task[task_name] += int(selected_episodes.size)
        task_embeddings.append(
            np.asarray(
                MgCache.load_numpy_array(str(cache_dir), "lowdim/task_embedding.npy")[
                    selected_episodes
                ],
                dtype=np.float32,
            )
        )
        task_language_tokens.append(
            np.asarray(
                MgCache.load_numpy_array(
                    str(cache_dir), "lowdim/task_language_tokens.npy"
                )[selected_episodes],
                dtype=np.float32,
            )
        )
        task_ids.append(np.full((selected_episodes.size,), task_id, dtype=np.int64))
        robot_ids.append(
            np.asarray(
                MgCache.load_numpy_array(str(cache_dir), "lowdim/robot_id.npy")[
                    selected_episodes
                ],
                dtype=np.int64,
            ).reshape(-1)
        )
        task_instructions.append(
            MgCache.load_numpy_array(str(cache_dir), "task_instructions.npy")[
                selected_episodes
            ]
        )
        source_demo_indices.append(
            np.asarray(
                meta.get("source_demo_indices", np.arange(len(ep_lens))),
                dtype=np.int64,
            )[selected_episodes]
        )
        if "table_texture" in meta:
            texture_meta = dict(meta["table_texture"])
            episode_files = texture_meta.get("episode_files")
            if episode_files is not None:
                if table_texture_meta is None:
                    table_texture_meta = {
                        key: value
                        for key, value in texture_meta.items()
                        if key != "episode_files"
                    }
                    table_texture_meta["episode_files"] = table_texture_episode_files
                table_texture_episode_files.extend(
                    [
                        episode_files[int(ep_idx)]
                        for ep_idx in selected_episodes.tolist()
                    ]
                )
        selected_action = np.asarray(
            MgCache.load_numpy_array(str(cache_dir), "action/absolute_action.npy")[
                step_indices
            ],
            dtype=np.float32,
        )
        if selected_action.ndim != 2 or selected_action.shape[1] != 13:
            raise ValueError(
                f"{cache_dir}: absolute actions must have shape [T,13], "
                f"got {selected_action.shape}"
            )
        action_abs.append(selected_action)

        for key in lowdim_keys:
            lowdim[key].append(
                np.asarray(
                    MgCache.load_numpy_array(str(cache_dir), f"lowdim/{key}.npy")[
                        step_indices
                    ],
                    dtype=np.float32,
                )
            )
        for key in oracle_keys:
            oracle[key].append(
                MgCache.load_numpy_array(str(cache_dir), f"oracle/{key}.npy")[
                    step_indices
                ]
            )

        src_voxels = {}
        for key in voxel_keys:
            index_name, colour_name, offsets_name = SparseVoxels.array_names(key)
            src_voxels[key] = (
                MgCache.load_numpy_array(str(cache_dir), offsets_name),
                MgCache.load_numpy_array(str(cache_dir), index_name),
                MgCache.load_numpy_array(str(cache_dir), colour_name),
            )

        src_env = _open_lmdb_read(cache_dir / "images.lmdb")
        with src_env.begin(write=False) as src_txn:
            for source_step in tqdm(
                step_indices.tolist(),
                desc=f"Copy images {cache_dir.name}",
                unit="step",
                leave=False,
            ):
                for cam in rgb_keys:
                    value = src_txn.get(f"{cam}/{source_step:08d}".encode("ascii"))
                    if value is None:
                        raise KeyError(f"{cache_dir}: missing {cam}/{source_step:08d}")
                    out_txn.put(f"{cam}/{global_step:08d}".encode("ascii"), value)
                    out_puts += 1

                    if out_puts % commit_every == 0:
                        out_txn.commit()
                        out_txn = out_env.begin(write=True)

                for key in point_cloud_keys:
                    value = src_txn.get(f"{key}/{source_step:08d}".encode("ascii"))
                    if value is None:
                        raise KeyError(f"{cache_dir}: missing {key}/{source_step:08d}")
                    out_txn.put(f"{key}/{global_step:08d}".encode("ascii"), value)
                    out_puts += 1

                    if out_puts % commit_every == 0:
                        out_txn.commit()
                        out_txn = out_env.begin(write=True)

                for key, (
                    src_voxel_offsets,
                    src_voxel_index,
                    src_voxel_colour,
                ) in src_voxels.items():
                    start, end = (
                        int(src_voxel_offsets[source_step]),
                        int(src_voxel_offsets[source_step + 1]),
                    )
                    out_voxel_frames[key].append(
                        (
                            np.array(src_voxel_index[start:end]),
                            np.array(src_voxel_colour[start:end]),
                        )
                    )

                global_step += 1
        src_env.close()
        del src_voxels

    out_txn.put(b"__len__", str(global_step).encode("ascii"))
    out_txn.commit()
    out_env.sync()
    out_env.close()

    absolute_action = np.concatenate(action_abs, axis=0).astype(np.float32)
    arrays = {
        MgCache.archive_key("action/absolute_action.npy"): absolute_action,
    }
    for key, chunks in lowdim.items():
        arrays[MgCache.archive_key(f"lowdim/{key}.npy")] = np.concatenate(
            chunks,
            axis=0,
        ).astype(np.float32)

    arrays[MgCache.archive_key("lowdim/task_embedding.npy")] = np.concatenate(
        task_embeddings,
        axis=0,
    ).astype(np.float32)
    arrays[MgCache.archive_key("lowdim/task_language_tokens.npy")] = np.concatenate(
        task_language_tokens,
        axis=0,
    ).astype(np.float32)
    arrays[MgCache.archive_key("lowdim/task_id.npy")] = np.concatenate(
        task_ids, axis=0
    ).astype(np.int64)
    arrays[MgCache.archive_key("lowdim/robot_id.npy")] = np.concatenate(
        robot_ids,
        axis=0,
    ).astype(np.int64)
    arrays[MgCache.archive_key("task_instructions.npy")] = np.asarray(
        np.concatenate(task_instructions, axis=0),
        dtype=object,
    )
    delta_horizons = [16] if delta_horizons is None else list(delta_horizons)
    for delta_horizon in sorted({int(h) for h in delta_horizons}):
        if delta_horizon < 1:
            raise ValueError(f"delta horizon must be >= 1, got {delta_horizon}")
        arrays[MgCache.archive_key(f"action/delta_action_h{delta_horizon}.npy")] = (
            absolute_posmat_to_delta_chunks(
                eef_pos=arrays[MgCache.archive_key("lowdim/robot0_eef_pos.npy")],
                eef_rot=arrays[MgCache.archive_key("lowdim/robot0_eef_rot.npy")],
                action_posmat=absolute_action,
                horizon=delta_horizon,
                episode_lengths=episode_lengths,
            )
        )
    for key, chunks in oracle.items():
        if chunks:
            arrays[MgCache.archive_key(f"oracle/{key}.npy")] = _concat_oracle_chunks(
                key, chunks
            )
    voxel_packed = {
        key: SparseVoxels.pack(out_voxel_frames[key]) for key in voxel_keys
    }
    for key, packed in voxel_packed.items():
        for target, source in zip(
            SparseVoxels.array_names(key),
            (
                SparseVoxels.INDEX_ARRAY,
                SparseVoxels.COLOUR_ARRAY,
                SparseVoxels.OFFSETS_ARRAY,
            ),
        ):
            arrays[MgCache.archive_key(target)] = packed[source]
    np.savez(MgCache.archive_path(out_dir), **arrays)

    meta_out = {
        "cache_format": "lmdb_npz_v1",
        "merged_from": [str(path) for path in in_dirs],
        "source_task_names": unique_task_names,
        "source_env_metas": [meta["env_meta"] for meta in metas if "env_meta" in meta],
        "source_episode_counts": [
            int(source_episode_counts_by_task[name]) for name in unique_task_names
        ],
        "rgb_keys": rgb_keys,
        "lowdim_keys": lowdim_keys,
        "episode_lengths": list(map(int, episode_lengths)),
        "n_demo": len(episode_lengths),
        "n_samples": int(sum(episode_lengths)),
        "image_size": int(ref["image_size"]),
        "jpeg_quality": int(
            ref.get("jpeg_quality", CoreImages.JPEG_QUALITY_DEFAULT)
        ),
        "oracle_keys": oracle_keys,
        "source_demo_indices": list(
            map(int, np.concatenate(source_demo_indices, axis=0))
        ),
        "delta_action_horizons": sorted({int(h) for h in delta_horizons}),
    }
    if voxel_keys:
        meta_out["voxel_keys"] = voxel_keys
        if voxel_keys == ["voxel"]:
            meta_out["voxel_spec"] = ref_voxel_metadata["voxel"]
            meta_out["voxel_storage"] = "sparse"
            meta_out["voxel_max_points"] = int(
                voxel_packed["voxel"]["max_points"]
            )
        else:
            meta_out["voxel_specs"] = ref_voxel_metadata
            meta_out["voxel_storage"] = {key: "sparse" for key in voxel_keys}
            meta_out["voxel_max_points"] = {
                key: int(voxel_packed[key]["max_points"]) for key in voxel_keys
            }
    if point_cloud_keys:
        meta_out["point_cloud_keys"] = point_cloud_keys
        meta_out["point_cloud_spec"] = ref["point_cloud_spec"]
    source_env_names = [
        str(env_meta.get("env_name", ""))
        for env_meta in meta_out["source_env_metas"]
        if isinstance(env_meta, dict)
    ]
    if len(set(source_env_names)) == 1 and meta_out["source_env_metas"]:
        meta_out["env_meta"] = meta_out["source_env_metas"][0]
        meta_out["env_name"] = source_env_names[0]
    if table_texture_meta is not None:
        meta_out["table_texture"] = table_texture_meta
    for key in (
        "oracle_camera",
        "oracle_patch_size",
        "oracle_min_patch_area_fraction",
        "oracle_min_patch_pixels",
    ):
        if key in ref:
            meta_out[key] = ref[key]
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta_out, f, indent=2)
    with (out_dir / "build_done.flag").open("w") as f:
        f.write("build completed\n")

    print(f"[merged] out_dir={out_dir}")
    print(f"  demos   : {meta_out['n_demo']}")
    print(f"  samples : {meta_out['n_samples']}")
    print(f"  rgb_keys: {len(rgb_keys)} cams")
    print(f"  lowdim  : {len(lowdim_keys)} keys")
    print(f"  oracle  : {len(oracle_keys)} arrays")


def merge_task_caches(
    tasks,
    output_task: str,
    n_demo_per_task: int,
    *,
    datasets_root=DATASETS_ROOT,
    overwrite: bool = False,
) -> None:
    """Merge named task caches beneath one dataset root."""
    datasets_root = Path(datasets_root).expanduser().resolve()
    tasks = list(tasks)
    merge_caches(
        in_dirs=[
            MgCache.default_cache_dir_for_dataset(datasets_root / task / f"{task}.hdf5")
            for task in tasks
        ],
        out_dir=MgCache.default_cache_dir_for_dataset(
            datasets_root / output_task / f"{output_task}.hdf5"
        ),
        n_demo_per_input=n_demo_per_task,
        source_task_names=tasks,
        overwrite=overwrite,
        delta_horizons=[16],
    )
