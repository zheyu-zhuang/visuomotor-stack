"""LMDB and NumPy cache serialization for rendered episodes."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import lmdb
import numpy as np
from tqdm import tqdm

from visuomotor.data.core import sparse_voxels as SparseVoxels
from visuomotor.data.mimicgen import action as MgAction
from visuomotor.data.mimicgen import cache as MgCache
from visuomotor.data.mimicgen.oracle import oracle_cache as OracleCache
from visuomotor.data.mimicgen.tasks import env_name_to_meta, setup_task_embedding_cache
from visuomotor.environment._dataset_rendering import common as RenderingCommon
from visuomotor.environment._dataset_rendering import renderer as DatasetRenderer
from visuomotor.paths import TEXTURES_DIR


def _selected_demos(h5_file, demo_indices, start_index):
    all_demos = RenderingCommon._sorted_demo_keys(h5_file)
    if demo_indices is None:
        start_index = int(start_index)
        if start_index < 0 or start_index >= len(all_demos):
            raise ValueError(
                f"start episode {start_index} out of range "
                f"(episodes={len(all_demos)})"
            )
        return all_demos[start_index:]
    demos = [f"demo_{int(index)}" for index in demo_indices]
    missing = [episode for episode in demos if episode not in h5_file["data"]]
    if missing:
        raise KeyError(f"Requested demos not found in HDF5: {missing[:10]}")
    return demos


def _validate_rendered_buffers(rendered, *, length, voxel_keys, point_cloud):
    frame_lengths = {
        **{f"rgb:{key}": len(frames) for key, frames in rendered.rgb_jpeg.items()},
        **{
            f"voxel:{key}": len(rendered.voxel_frames[key])
            for key in voxel_keys
        },
        "point_cloud": len(rendered.point_cloud_frames) if point_cloud else length,
    }
    mismatched = {
        key: value for key, value in frame_lengths.items() if value != length
    }
    if mismatched:
        raise ValueError(
            f"rendered frame-buffer length mismatch for T={length}: {mismatched}"
        )
    for key, array in rendered.lowdim.items():
        if array.shape[0] != length:
            raise ValueError(
                f"lowdim '{key}' length mismatch: {array.shape[0]} vs {length}"
            )


def _archive_arrays(
    *,
    lowdim_chunks,
    abs_chunks,
    delta_horizons,
    oracle_chunks,
    episode_lengths,
    task_embeddings,
    task_language_token_embeddings,
    robot_ids,
    task_instructions,
    voxel_frames,
):
    arrays = {}
    lowdim_keys = sorted(lowdim_chunks)
    for key in lowdim_keys:
        arrays[MgCache.archive_key(os.path.join("lowdim", f"{key}.npy"))] = (
            np.concatenate(lowdim_chunks[key], axis=0).astype(np.float32)
        )
    abs_all = np.concatenate(abs_chunks, axis=0).astype(np.float32)
    arrays[MgCache.archive_key(os.path.join("action", "absolute_action.npy"))] = (
        abs_all
    )
    for horizon in sorted({int(value) for value in delta_horizons}):
        if horizon < 1:
            raise ValueError(f"delta horizon must be >= 1, got {horizon}")
        arrays[
            MgCache.archive_key(
                os.path.join("action", f"delta_action_h{horizon}.npy")
            )
        ] = MgAction.absolute_posmat_to_delta_chunks(
            eef_pos=arrays[
                MgCache.archive_key(os.path.join("lowdim", "robot0_eef_pos.npy"))
            ],
            eef_rot=arrays[
                MgCache.archive_key(os.path.join("lowdim", "robot0_eef_rot.npy"))
            ],
            action_posmat=abs_all,
            horizon=horizon,
            episode_lengths=episode_lengths,
        )
    oracle_arrays = OracleCache.build_oracle_arrays(
        oracle_chunks=oracle_chunks,
        expected_steps=int(sum(episode_lengths)),
    )
    for key, array in oracle_arrays.items():
        arrays[MgCache.archive_key(os.path.join("oracle", f"{key}.npy"))] = array
    arrays[MgCache.archive_key(os.path.join("lowdim", "task_embedding.npy"))] = (
        np.stack(task_embeddings, axis=0).astype(np.float32)
    )
    arrays[
        MgCache.archive_key(os.path.join("lowdim", "task_language_tokens.npy"))
    ] = np.stack(task_language_token_embeddings, axis=0).astype(np.float32)
    arrays[MgCache.archive_key(os.path.join("lowdim", "task_id.npy"))] = np.zeros(
        (len(episode_lengths),), dtype=np.int64
    )
    arrays[MgCache.archive_key(os.path.join("lowdim", "robot_id.npy"))] = (
        np.asarray(robot_ids, dtype=np.int64)
    )
    arrays[MgCache.archive_key("task_instructions.npy")] = np.asarray(
        task_instructions, dtype=object
    )
    voxel_packed = {
        key: SparseVoxels.pack(frames) for key, frames in voxel_frames.items()
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
    return arrays, lowdim_keys, list(oracle_arrays), voxel_packed


def _cache_metadata(
    renderer,
    *,
    lowdim_keys,
    episode_lengths,
    robot_names,
    source_demo_indices,
    delta_horizons,
    voxel_packed,
    table_texture_episode_files,
    oracle_keys,
):
    meta = {
        "cache_format": "lmdb_npz_v1",
        "env_meta": RenderingCommon._json_safe(renderer.source_env_meta),
        "env_name": str(renderer.env_meta.get("env_name", "")),
        "rgb_keys": list(renderer.rgb_keys),
        "lowdim_keys": lowdim_keys,
        "episode_lengths": list(map(int, episode_lengths)),
        "n_demo": int(len(episode_lengths)),
        "n_samples": int(sum(episode_lengths)),
        "image_size": int(renderer.res),
        "jpeg_quality": int(renderer.jpeg_quality),
        "robot_names": robot_names,
        "source_demo_indices": list(map(int, source_demo_indices)),
        "delta_action_horizons": sorted({int(value) for value in delta_horizons}),
    }
    if renderer.voxel_specs:
        keys = list(renderer.voxel_specs)
        meta["voxel_keys"] = keys
        if keys == ["voxel"]:
            meta["voxel_spec"] = renderer.voxel_specs["voxel"].metadata()
            meta["voxel_storage"] = "sparse"
            meta["voxel_max_points"] = int(voxel_packed["voxel"]["max_points"])
        else:
            meta["voxel_specs"] = {
                key: spec.declaration()
                for key, spec in renderer.voxel_specs.items()
            }
            meta["voxel_storage"] = {key: "sparse" for key in keys}
            meta["voxel_max_points"] = {
                key: int(voxel_packed[key]["max_points"]) for key in keys
            }
    if renderer.point_cloud_spec is not None:
        meta["point_cloud_keys"] = ["point_cloud"]
        meta["point_cloud_spec"] = renderer.point_cloud_spec.metadata()
    if renderer.table_texture_every is not None:
        meta["table_texture"] = {
            "mode": "ranked_table_texture",
            "demos_per_texture": int(renderer.table_texture_every),
            "texture_dir": str((Path(TEXTURES_DIR) / "train").resolve()),
            "files": [Path(path).name for path in renderer.table_texture_files],
            "episode_files": table_texture_episode_files,
        }
    if oracle_keys:
        meta["oracle_keys"] = oracle_keys
        meta["oracle_camera"] = renderer.oracle_camera
        meta["oracle_patch_size"] = int(renderer.oracle_patch_size)
        meta["oracle_min_patch_area_fraction"] = float(
            renderer.oracle_min_patch_area_fraction
        )
        meta["oracle_min_mask_pixels"] = int(renderer.oracle_min_mask_pixels)
    return meta


class DatasetRerenderToCache(DatasetRenderer.DatasetRenderer):
    def render_to_cache(
        self,
        n_demo: Optional[int],
        start_index: int,
        jpeg_quality: int,
        lmdb_map_size_gb: int,
        commit_every: int,
        delta_horizons: List[int],
        demo_indices: Optional[List[int]] = None,
        show_progress: bool = True,
        progress_queue=None,
    ) -> None:
        """Process selected demos and write full cache outputs."""
        setup_task_embedding_cache()

        env_name = self.env_meta.get("env_name", None)
        if env_name is None:
            raise ValueError("env_name missing in env_meta")

        # Task metadata derived from env_name.
        task_meta = env_name_to_meta(env_name)
        task_instruction = str(task_meta["instruction"])
        robot_name = str(task_meta["robot"])
        robot_id_scalar = int(task_meta["robot_id"])
        task_emb = task_meta["task_embedding"].cpu().numpy().astype(np.float32)
        task_language_tokens = (
            task_meta["task_language_tokens"].cpu().numpy().astype(np.float32)
        )

        # LMDB init
        lmdb_path = os.path.join(self.out_dir, "images.lmdb")
        if os.path.exists(lmdb_path):
            os.remove(lmdb_path)

        env_lmdb = lmdb.open(
            lmdb_path,
            map_size=int(lmdb_map_size_gb * (1024**3)),
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True,
            max_dbs=1,
        )
        txn = env_lmdb.begin(write=True)
        put_count = 0

        # Store rendered voxels as occupied cells.
        voxel_frames: Dict[str, List[tuple]] = {
            key: [] for key in self.voxel_specs
        }

        lowdim_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
        abs_chunks: List[np.ndarray] = []
        episode_lengths: List[int] = []
        oracle_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Episode-level outputs
        task_instructions: List[str] = []
        robot_names: List[str] = []
        robot_ids: List[int] = []
        task_embeddings: List[np.ndarray] = []
        task_language_token_embeddings: List[np.ndarray] = []
        source_demo_indices: List[int] = []
        table_texture_episode_files: List[str] = []

        global_step = 0
        kept_demos = 0

        with h5py.File(self.input_path, "r") as h5_file:
            demos = _selected_demos(h5_file, demo_indices, start_index)

            target_n_demo = None if n_demo is None else int(n_demo)
            if n_demo is None:
                n_demo = len(demos)

            pbar = (
                tqdm(total=n_demo, desc="Rerender -> LMDB cache", unit="ok")
                if show_progress
                else None
            )
            scanned = 0

            for ep in demos:
                scanned += 1
                if kept_demos >= int(n_demo):
                    break

                source_demo_idx = int(ep.split("_")[-1])
                texture_demo_rank = None
                texture_file = None
                if self.texture_rank_by_source_demo is not None:
                    texture_demo_rank = self.texture_rank_by_source_demo[
                        source_demo_idx
                    ]
                if self.table_texture_every is not None:
                    demo_rank = (
                        kept_demos if texture_demo_rank is None else texture_demo_rank
                    )
                    texture_file = self._table_texture_for_rank(demo_rank)

                rendered = self.rerender_demonstration(
                    h5_file,
                    ep,
                    texture_demo_rank=texture_demo_rank,
                    jpeg_quality=jpeg_quality,
                )
                if not rendered.success:
                    if pbar is not None:
                        pbar.set_postfix_str(
                            f"kept={kept_demos}/{n_demo or '-'} scanned={scanned} skip_failed"
                        )
                    continue

                T = int(rendered.absolute_action.shape[0])
                episode_lengths.append(T)
                for key, arr in rendered.oracle.items():
                    oracle_chunks[key].append(arr)

                # Episode-level arrays (one per written demo).
                task_instructions.append(task_instruction)
                robot_names.append(robot_name)
                robot_ids.append(robot_id_scalar)
                task_embeddings.append(task_emb)
                task_language_token_embeddings.append(task_language_tokens)
                source_demo_indices.append(source_demo_idx)
                if texture_file is not None:
                    table_texture_episode_files.append(Path(texture_file).name)

                if rendered.absolute_action.ndim != 2 or (
                    rendered.absolute_action.shape[1] != 13
                ):
                    raise ValueError(
                        "cached absolute actions must have shape [T,13], "
                        f"got {rendered.absolute_action.shape}"
                    )
                abs_chunks.append(rendered.absolute_action)

                _validate_rendered_buffers(
                    rendered,
                    length=T,
                    voxel_keys=self.voxel_specs,
                    point_cloud=self.enable_point_cloud,
                )
                for key, arr in rendered.lowdim.items():
                    lowdim_chunks[key].append(arr)

                for t in range(T):
                    for camera_key in self.rgb_keys:
                        key = f"{camera_key}/{global_step:08d}".encode("ascii")
                        txn.put(key, rendered.rgb_jpeg[camera_key][t])
                        put_count += 1

                        if put_count % int(commit_every) == 0:
                            txn.commit()
                            txn = env_lmdb.begin(write=True)

                    for key in self.voxel_specs:
                        voxel_frames[key].append(rendered.voxel_frames[key][t])

                    if self.enable_point_cloud:
                        key = f"point_cloud/{global_step:08d}".encode("ascii")
                        txn.put(key, rendered.point_cloud_frames[t])
                        put_count += 1
                        if put_count % int(commit_every) == 0:
                            txn.commit()
                            txn = env_lmdb.begin(write=True)

                    global_step += 1

                kept_demos += 1
                self.rerender_counter = kept_demos
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "event": RenderingCommon.PROGRESS_EVENT_DEMO_DONE,
                            "n_samples": T,
                        }
                    )
                if pbar is not None:
                    pbar.update(1)  # only on kept/success
                failed = scanned - kept_demos
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"kept={kept_demos}/{n_demo or '-'} scanned={scanned} failed={failed}"
                    )
        if pbar is not None:
            pbar.close()
        if target_n_demo is not None and kept_demos < target_n_demo:
            txn.abort()
            env_lmdb.close()
            raise RuntimeError(
                f"Requested {target_n_demo} successful demos, but only "
                f"{kept_demos} succeeded before source demos were exhausted."
            )
        # Finalize LMDB.
        txn.put(b"__len__", str(global_step).encode("ascii"))
        txn.commit()
        env_lmdb.sync()
        env_lmdb.close()

        if kept_demos == 0:
            raise RuntimeError("No demos were written (all failed or none selected).")

        arrays, lowdim_keys, oracle_keys, voxel_packed = _archive_arrays(
            lowdim_chunks=lowdim_chunks,
            abs_chunks=abs_chunks,
            delta_horizons=delta_horizons,
            oracle_chunks=oracle_chunks,
            episode_lengths=episode_lengths,
            task_embeddings=task_embeddings,
            task_language_token_embeddings=task_language_token_embeddings,
            robot_ids=robot_ids,
            task_instructions=task_instructions,
            voxel_frames=voxel_frames,
        )
        np.savez(MgCache.archive_path(self.out_dir), **arrays)

        meta = _cache_metadata(
            self,
            lowdim_keys=lowdim_keys,
            episode_lengths=episode_lengths,
            robot_names=robot_names,
            source_demo_indices=source_demo_indices,
            delta_horizons=delta_horizons,
            voxel_packed=voxel_packed,
            table_texture_episode_files=table_texture_episode_files,
            oracle_keys=oracle_keys,
        )
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        with open(os.path.join(self.out_dir, "build_done.flag"), "w") as f:
            f.write("build completed\n")

        print(
            f"[built] out_dir={self.out_dir} demos={meta['n_demo']} samples={meta['n_samples']}"
        )
