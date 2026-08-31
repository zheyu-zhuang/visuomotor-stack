"""Single- and multi-process dataset rerender orchestration."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import cv2
from tqdm import tqdm

from visuomotor.data.core import spatial as Spatial
from visuomotor.data.mimicgen import cache as MgCache
from visuomotor.environment._dataset_rendering import cache as RenderingCache
from visuomotor.environment._dataset_rendering import common as RenderingCommon

_NATIVE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def _single_threaded_native_libraries():
    """Keep spawned workers from multiplying native thread pools."""
    previous = {name: os.environ.get(name) for name in _NATIVE_THREAD_ENV}
    os.environ.update({name: "1" for name in _NATIVE_THREAD_ENV})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_cpu_sets(num_workers: int) -> List[tuple[int, ...]]:
    """Assign workers bounded CPU sets while leaving host capacity responsive."""
    num_workers = int(num_workers)
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    available = sorted(os.sched_getaffinity(0))
    reserve = min(4, max(0, len(available) - num_workers))
    usable = available[: len(available) - reserve]
    return [tuple(usable[index::num_workers]) for index in range(num_workers)]


def _configure_rerender_worker(*, worker_id: int, cpu_ids: tuple[int, ...]) -> None:
    """Lower one worker's scheduling pressure and cap its native parallelism."""
    if cpu_ids:
        os.sched_setaffinity(0, cpu_ids)
    os.nice(5)
    cv2.setNumThreads(1)
    # Avoid simultaneous simulator / EGL initialization spikes.
    time.sleep(0.25 * int(worker_id))


def _rerender_worker(
    *,
    worker_id: int,
    dataset_path: str,
    out_cache_dir: str,
    camera_resolution: int,
    table_texture_every: Optional[int],
    texture_rank_by_source_demo: Optional[Dict[int, int]],
    oracle_camera: Optional[str],
    oracle_patch_size: int,
    oracle_min_patch_area_fraction: float,
    oracle_min_mask_pixels: int,
    demo_indices: List[int],
    cpu_ids: tuple[int, ...],
    overwrite: bool,
    voxel_spec: Optional[Spatial.VoxelProducerSpec] = None,
    voxel_specs: Optional[Mapping[str, Spatial.VoxelProducerSpec]] = None,
    point_cloud_spec: Optional[Spatial.PointCloudProducerSpec] = None,
    progress_queue=None,
) -> dict:
    """Worker entrypoint for shard-based parallel rerender."""
    _configure_rerender_worker(worker_id=worker_id, cpu_ids=cpu_ids)
    builder = None
    try:
        builder = RenderingCache.DatasetRerenderToCache(
            dataset_path=dataset_path,
            out_cache_dir=out_cache_dir,
            camera_resolution=camera_resolution,
            table_texture_every=table_texture_every,
            texture_rank_by_source_demo=texture_rank_by_source_demo,
            oracle_camera=oracle_camera,
            oracle_patch_size=oracle_patch_size,
            oracle_min_patch_area_fraction=oracle_min_patch_area_fraction,
            oracle_min_mask_pixels=oracle_min_mask_pixels,
            voxel_spec=voxel_spec,
            voxel_specs=voxel_specs,
            point_cloud_spec=point_cloud_spec,
            overwrite=overwrite,
        )
        try:
            builder.render_to_cache(
                n_demo=None,
                start_index=0,
                jpeg_quality=RenderingCommon.JPEG_QUALITY_DEFAULT,
                lmdb_map_size_gb=RenderingCommon.LMDB_MAP_SIZE_GB_DEFAULT,
                commit_every=RenderingCommon.COMMIT_EVERY_DEFAULT,
                delta_horizons=[],
                demo_indices=demo_indices,
                show_progress=False,
                progress_queue=progress_queue,
            )
        except RuntimeError as exc:
            if "No demos were written" not in str(exc):
                raise
            return {
                "worker_id": int(worker_id),
                "out_dir": str(out_cache_dir),
                "n_demo": 0,
                "n_samples": 0,
            }
        with open(os.path.join(out_cache_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        return {
            "worker_id": int(worker_id),
            "out_dir": str(out_cache_dir),
            "n_demo": int(meta["n_demo"]),
            "n_samples": int(meta["n_samples"]),
        }
    finally:
        if builder is not None:
            builder.close()


def _run_parallel_rerender(
    *,
    dataset_path: str,
    output_dir: str,
    camera_resolution: int,
    table_texture_every: Optional[int],
    oracle_camera: Optional[str],
    oracle_patch_size: int,
    oracle_min_patch_area_fraction: float,
    oracle_min_mask_pixels: int,
    start_index: int,
    n_demo: Optional[int],
    num_workers: int,
    delta_horizons: List[int],
    overwrite: bool,
    voxel_spec: Optional[Spatial.VoxelProducerSpec] = None,
    voxel_specs: Optional[Mapping[str, Spatial.VoxelProducerSpec]] = None,
    point_cloud_spec: Optional[Spatial.PointCloudProducerSpec] = None,
) -> None:
    """Rerender source demos in parallel shard caches and merge them."""
    from visuomotor.data.cache_merge import merge_caches

    output_path = Path(output_dir).expanduser().resolve()
    RenderingCommon._remove_existing_output_dir(output_path, overwrite=overwrite)

    shard_root = output_path.parent / f".{output_path.name}_worker_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    selected = RenderingCommon._source_demo_indices(dataset_path, start_index, None)
    texture_rank_by_source_demo = None
    if table_texture_every is not None:
        texture_rank_by_source_demo = {
            int(source_idx): rank for rank, source_idx in enumerate(selected)
        }
    target_successes = len(selected) if n_demo is None else int(n_demo)
    if target_successes < 1:
        raise ValueError(f"n_demo must be >= 1, got {n_demo}")
    if n_demo is not None and target_successes > len(selected):
        raise ValueError(
            f"n_demo={target_successes} exceeds remaining source demos={len(selected)}"
        )
    print(
        f"[parallel] rerendering {target_successes} successful demos "
        f"with up to {int(num_workers)} workers",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    thread_limits = _single_threaded_native_libraries()
    thread_limits.__enter__()
    manager = None
    try:
        manager = ctx.Manager()
        progress_queue = manager.Queue()
        planned_shards = []
        built_shard_paths = set()
        cursor = 0
        round_idx = 0
        success_count = 0
        samples = 0
        with tqdm(total=target_successes, desc="parallel rerender", unit="ok") as pbar:
            while cursor < len(selected) and success_count < target_successes:
                remaining = target_successes - success_count
                candidates = selected[cursor : cursor + remaining]
                cursor += len(candidates)
                chunks = RenderingCommon._split_contiguous(candidates, num_workers)
                worker_cpu_sets = _worker_cpu_sets(len(chunks))
                round_root = shard_root / f"round_{round_idx:03d}"
                shard_dirs = [
                    round_root / f"worker_{i:03d}" for i in range(len(chunks))
                ]
                planned_shards.extend(shard_dirs)
                futures = []
                with ProcessPoolExecutor(
                    max_workers=len(chunks), mp_context=ctx
                ) as pool:
                    for worker_id, (indices, shard_dir) in enumerate(
                        zip(chunks, shard_dirs)
                    ):
                        futures.append(
                            pool.submit(
                                _rerender_worker,
                                worker_id=worker_id,
                                dataset_path=dataset_path,
                                out_cache_dir=str(shard_dir),
                                camera_resolution=camera_resolution,
                                table_texture_every=table_texture_every,
                                texture_rank_by_source_demo=texture_rank_by_source_demo,
                                oracle_camera=oracle_camera,
                                oracle_patch_size=oracle_patch_size,
                                oracle_min_patch_area_fraction=(
                                    oracle_min_patch_area_fraction
                                ),
                                oracle_min_mask_pixels=oracle_min_mask_pixels,
                                demo_indices=indices,
                                cpu_ids=worker_cpu_sets[worker_id],
                                overwrite=True,
                                voxel_spec=voxel_spec,
                                voxel_specs=voxel_specs,
                                point_cloud_spec=point_cloud_spec,
                                progress_queue=progress_queue,
                            )
                        )

                    done = set()
                    while len(done) < len(futures):
                        for future in futures:
                            if future in done or not future.done():
                                continue
                            result = future.result()
                            done.add(future)
                            if int(result["n_demo"]) > 0:
                                built_shard_paths.add(Path(result["out_dir"]).resolve())
                            print(
                                f"[parallel] worker {result['worker_id']:03d} built "
                                f"{result['n_demo']} demos / {result['n_samples']} samples",
                                flush=True,
                            )

                        samples, success_count, drained = (
                            RenderingCommon._drain_progress_events(
                                progress_queue,
                                pbar,
                                samples,
                                success_count,
                            )
                        )
                        if not drained and len(done) < len(futures):
                            time.sleep(RenderingCommon.PROGRESS_POLL_INTERVAL_SEC)

                    samples, success_count, _ = RenderingCommon._drain_progress_events(
                        progress_queue,
                        pbar,
                        samples,
                        success_count,
                    )
                round_idx += 1
    finally:
        if manager is not None:
            manager.shutdown()
        thread_limits.__exit__(None, None, None)

    built_shards = [
        shard_dir
        for shard_dir in planned_shards
        if shard_dir.resolve() in built_shard_paths
    ]
    if not built_shards:
        raise RuntimeError("Parallel rerender produced no shard caches")
    if n_demo is not None and success_count < target_successes:
        raise RuntimeError(
            f"[parallel] requested {target_successes} successful demos, "
            f"built {success_count}"
        )

    episode_indices_per_input = None
    if n_demo is not None:
        successful = []
        for input_idx, shard_dir in enumerate(built_shards):
            with (shard_dir / "meta.json").open("r") as f:
                meta = json.load(f)
            for episode_idx, source_idx in enumerate(meta["source_demo_indices"]):
                successful.append((int(source_idx), input_idx, int(episode_idx)))
        successful.sort()
        episode_indices_per_input = [[] for _ in built_shards]
        for _, input_idx, episode_idx in successful[:target_successes]:
            episode_indices_per_input[input_idx].append(episode_idx)

    merge_caches(
        in_dirs=built_shards,
        out_dir=output_path,
        n_demo_per_input=None,
        source_task_names=[Path(dataset_path).parent.name] * len(built_shards),
        episode_indices_per_input=episode_indices_per_input,
        overwrite=True,
        lmdb_map_size_gb=RenderingCommon.LMDB_MAP_SIZE_GB_DEFAULT,
        commit_every=RenderingCommon.COMMIT_EVERY_DEFAULT,
        delta_horizons=delta_horizons,
    )

    shutil.rmtree(shard_root)


def _run_single_rerender(
    *,
    dataset_path: str,
    output_dir: str,
    camera_resolution: int,
    table_texture_every: Optional[int],
    oracle_camera: Optional[str],
    oracle_patch_size: int,
    oracle_min_patch_area_fraction: float,
    oracle_min_mask_pixels: int,
    start_index: int,
    n_demo: Optional[int],
    num_workers: int,
    delta_horizons: List[int],
    overwrite: bool,
    voxel_spec: Optional[Spatial.VoxelProducerSpec] = None,
    voxel_specs: Optional[Mapping[str, Spatial.VoxelProducerSpec]] = None,
    point_cloud_spec: Optional[Spatial.PointCloudProducerSpec] = None,
) -> None:
    """Rerender one raw HDF5 dataset to one LMDB cache."""
    if int(num_workers) > 1:
        _run_parallel_rerender(
            dataset_path=dataset_path,
            output_dir=output_dir,
            camera_resolution=camera_resolution,
            table_texture_every=table_texture_every,
            oracle_camera=oracle_camera,
            oracle_patch_size=oracle_patch_size,
            oracle_min_patch_area_fraction=oracle_min_patch_area_fraction,
            oracle_min_mask_pixels=oracle_min_mask_pixels,
            start_index=start_index,
            n_demo=n_demo,
            num_workers=num_workers,
            delta_horizons=delta_horizons,
            overwrite=overwrite,
            voxel_spec=voxel_spec,
            voxel_specs=voxel_specs,
            point_cloud_spec=point_cloud_spec,
        )
        return

    builder = RenderingCache.DatasetRerenderToCache(
        dataset_path=dataset_path,
        out_cache_dir=output_dir,
        camera_resolution=camera_resolution,
        table_texture_every=table_texture_every,
        oracle_camera=oracle_camera,
        oracle_patch_size=oracle_patch_size,
        oracle_min_patch_area_fraction=oracle_min_patch_area_fraction,
        oracle_min_mask_pixels=oracle_min_mask_pixels,
        voxel_spec=voxel_spec,
        voxel_specs=voxel_specs,
        point_cloud_spec=point_cloud_spec,
        overwrite=overwrite,
    )

    try:
        builder.render_to_cache(
            n_demo=n_demo,
            start_index=start_index,
            jpeg_quality=RenderingCommon.JPEG_QUALITY_DEFAULT,
            lmdb_map_size_gb=RenderingCommon.LMDB_MAP_SIZE_GB_DEFAULT,
            commit_every=RenderingCommon.COMMIT_EVERY_DEFAULT,
            delta_horizons=delta_horizons,
        )
    finally:
        builder.close()


def render_datasets(
    *,
    dataset=None,
    datasets_root=None,
    tasks=None,
    n_demo=None,
    output_dir=None,
    output_suffix="",
    table_texture_every=None,
    oracle_camera="agentview",
    oracle_patch_size=16,
    oracle_min_patch_area_fraction=0.05,
    oracle_min_mask_pixels=16,
    num_workers=4,
    voxel_specs: Optional[Mapping[str, Spatial.VoxelProducerSpec]] = None,
    point_cloud_specs: Optional[Mapping[str, Spatial.PointCloudProducerSpec]] = None,
    overwrite=False,
) -> None:
    """Rerender one dataset or a collection of task datasets."""
    if table_texture_every is not None and int(table_texture_every) < 1:
        raise ValueError("table_texture_every must be at least one")
    oracle_camera = str(oracle_camera).strip() or None
    kwargs = dict(
        camera_resolution=256,
        table_texture_every=table_texture_every,
        oracle_camera=oracle_camera,
        oracle_patch_size=oracle_patch_size,
        oracle_min_patch_area_fraction=oracle_min_patch_area_fraction,
        oracle_min_mask_pixels=oracle_min_mask_pixels,
        start_index=0,
        n_demo=n_demo,
        num_workers=num_workers,
        delta_horizons=[16],
        overwrite=overwrite,
    )
    voxel_specs = dict(voxel_specs or {})
    point_cloud_specs = dict(point_cloud_specs or {})

    def spatial_specs(dataset_path: str) -> dict:
        key = str(Path(dataset_path).expanduser().resolve())
        selected_voxels = voxel_specs.get(key)
        if (
            isinstance(selected_voxels, Spatial.VoxelProducerSpec)
            or selected_voxels is None
        ):
            voxel_kwargs = {"voxel_spec": selected_voxels}
        else:
            voxel_kwargs = {"voxel_specs": dict(selected_voxels)}
        return {
            **voxel_kwargs,
            "point_cloud_spec": point_cloud_specs.get(key),
        }

    if dataset is not None:
        dataset_path = str(Path(dataset).expanduser().resolve())
        base = MgCache.default_cache_dir_for_dataset(dataset_path)
        target = (
            str(Path(output_dir).expanduser().resolve())
            if output_dir is not None
            else RenderingCommon._sibling_cache_dir_with_suffix(base, output_suffix)
        )
        _run_single_rerender(
            dataset_path=dataset_path,
            output_dir=target,
            **spatial_specs(dataset_path),
            **kwargs,
        )
        return
    if datasets_root is None:
        raise ValueError("dataset or datasets_root is required")
    if output_dir is not None:
        raise ValueError("output_dir is valid only for a single dataset")
    discovered = RenderingCommon._discover_task_datasets(datasets_root, tasks=tasks)
    for task_name, dataset_path in discovered:
        base = MgCache.default_cache_dir_for_dataset(
            dataset_path, output_root=datasets_root
        )
        target = RenderingCommon._sibling_cache_dir_with_suffix(base, output_suffix)
        print(f"[bulk] {task_name}: {dataset_path} -> {target}", flush=True)
        _run_single_rerender(
            dataset_path=dataset_path,
            output_dir=target,
            **spatial_specs(dataset_path),
            **kwargs,
        )
