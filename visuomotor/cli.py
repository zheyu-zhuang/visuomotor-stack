"""The sole command-line parsing boundary for Visuomotor Stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run_train(args, hydra_args) -> int:
    import hydra
    from omegaconf import OmegaConf

    from visuomotor.config.tasks import get_task_spec
    from visuomotor.workspace.base import BaseWorkspace

    resolvers = {
        "eval": eval,
        "add_int": lambda a, b: int(a) + int(b),
        "multiply": lambda a, b: int(a) * int(b),
        "divide": lambda a, b: float(a) / float(b),
        "get_max_steps": lambda name: get_task_spec(name).max_steps,
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)

    @hydra.main(version_base=None, config_path="config")
    def hydra_main(cfg: OmegaConf) -> None:
        OmegaConf.resolve(cfg)
        workspace: BaseWorkspace = hydra.utils.get_class(cfg._target_)(cfg)
        workspace.run()

    sys.argv = ["vmstack train"]
    if args.config_name:
        sys.argv.append(f"--config-name={args.config_name}")
    sys.argv.extend(hydra_args)
    hydra_main()
    return 0


def _load_setup_dependencies():
    """Load the repo-root setup_dependencies.py by path.

    It lives outside the visuomotor package (like .dep/, which it reads
    from) since it only makes sense against a source checkout, so it can't be
    imported as `visuomotor.setup_dependencies`.
    """
    import importlib.util
    import sys

    from visuomotor.paths import REPO_ROOT

    path = REPO_ROOT / "setup_dependencies.py"
    spec = importlib.util.spec_from_file_location("setup_dependencies", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses needs cls.__module__ registered in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_setup(args) -> int:
    setup_dependencies = _load_setup_dependencies()
    if not args.assets_only:
        setup_dependencies.install_suite_dependencies(
            deps_root=args.suite_deps_root, force=args.force
        )
    return setup_dependencies.install_assets(
        repo=args.repo,
        release_tag=args.release_tag,
        force=args.force,
        build_task_cache=not args.skip_task_cache,
    )


def _run_rollout(args) -> int:
    import torch

    from visuomotor.config.build import (
        build_runner,
        load_rollout_checkpoint,
    )
    from visuomotor.visualization.artifacts import allocate_rollout_output

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_rollout_checkpoint(
        args.checkpoint, map_location=device, weights=args.weights
    )
    policy = checkpoint.policy
    policy.to(device).eval()
    progress = ""
    if checkpoint.epoch is not None:
        progress += f", epoch={checkpoint.epoch}"
    if checkpoint.global_step is not None:
        progress += f", global_step={checkpoint.global_step}"
    print(
        f"[rollout] checkpoint: {checkpoint.checkpoint_format}, "
        f"weights={checkpoint.weights}{progress}"
    )
    if args.inspect:
        print(policy.model_spec.describe())
        return 0
    output_dir = allocate_rollout_output(args.checkpoint, args.output_dir)
    metrics, artifacts = build_runner(
        checkpoint.runner_spec,
        output_dir=str(output_dir),
    ).run(policy)
    print(f"[rollout] output: {output_dir}")
    print(f"[rollout] artifacts: {len(artifacts)}, metrics: {len(metrics)}")
    return 0


def _run_checkpoint(args) -> int:
    import torch

    from visuomotor.policy.checkpoint import strip_backbone

    if args.checkpoint_command == "strip-backbone":
        payload = torch.load(args.input, map_location="cpu")
        stripped = strip_backbone(payload)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stripped, output)
        print(f"[checkpoint strip-backbone] Saved: {output}")
    return 0


def _run_data(args) -> int:
    if args.data_command == "convert-actions":
        from visuomotor.environment.action_conversion import convert_dataset_file

        convert_dataset_file(args.dataset, args.output, overwrite=args.overwrite)
    elif args.data_command == "merge":
        from visuomotor.data.cache_merge import merge_task_caches

        merge_task_caches(
            args.tasks,
            args.output_task,
            args.n_demo_per_task,
            datasets_root=args.datasets_root,
            overwrite=args.overwrite,
        )
    elif args.data_command == "playback":
        from visuomotor.environment.dataset_playback import playback_dataset

        playback_dataset(
            args.dataset_path,
            use_obs=args.use_obs,
            use_actions=args.use_actions,
            cache_dir=args.cache_dir,
            cameras=args.cameras,
            wait_ms=args.wait_ms,
            save_first_frame=args.save_first_frame,
            show_window=not args.no_window,
            max_frames=args.max_frames,
            start_from=args.start_from,
        )
    elif args.data_command == "prepare":
        from visuomotor.config import tasks as ConfigTasks
        from visuomotor.data.core import spatial as Spatial
        from visuomotor.environment.dataset_rendering import render_datasets

        dataset_paths = []
        if args.dataset is not None:
            dataset_paths.append(Path(args.dataset).expanduser().resolve())
        elif args.datasets_root is not None:
            root = Path(args.datasets_root).expanduser().resolve()
            task_names = args.tasks
            if task_names is None:
                task_names = sorted(
                    child.name
                    for child in root.iterdir()
                    if child.is_dir() and (child / f"{child.name}.hdf5").is_file()
                )
            dataset_paths.extend(
                root / str(task_name) / f"{task_name}.hdf5" for task_name in task_names
            )

        voxel_specs = {}
        point_cloud_specs = {}
        for dataset_path in dataset_paths:
            task = ConfigTasks.get_task_spec(dataset_path.parent.name)
            key = str(dataset_path)
            if args.enable_voxel:
                bounds_min, bounds_max = ConfigTasks.voxel_bounds(task, 0.6)
                voxel_specs[key] = Spatial.VoxelProducerSpec(
                    cameras=task.spatial_cameras,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                )
            if args.enable_point_cloud:
                bounds_min, bounds_max = ConfigTasks.point_cloud_bounds(
                    task, 0.6, 0.005
                )
                point_cloud_specs[key] = Spatial.PointCloudProducerSpec(
                    cameras=task.spatial_cameras,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                )

        render_datasets(
            dataset=args.dataset,
            datasets_root=args.datasets_root,
            tasks=args.tasks,
            n_demo=args.n_demo,
            output_dir=args.output_dir,
            output_suffix=args.output_suffix,
            table_texture_every=args.table_texture_every,
            oracle_camera=args.oracle_camera,
            oracle_patch_size=args.oracle_patch_size,
            oracle_min_patch_area_fraction=args.oracle_min_patch_area_fraction,
            oracle_min_mask_pixels=args.oracle_min_mask_pixels,
            num_workers=args.num_workers,
            voxel_specs=voxel_specs,
            point_cloud_specs=point_cloud_specs,
            overwrite=args.overwrite,
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmstack", description="Visuomotor Stack tools"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="install dependencies and assets")
    setup.add_argument("--assets-only", action="store_true")
    setup.add_argument("--suite-deps-root")
    setup.add_argument("--repo", default="zheyu-zhuang/visuomotor-stack")
    setup.add_argument("--release-tag", default="assets")
    setup.add_argument(
        "--force",
        action="store_true",
        help="restore locked dependency checkouts and re-download assets",
    )
    setup.add_argument("--skip-task-cache", action="store_true")

    train = commands.add_parser("train", help="run a Hydra training workspace")
    train.add_argument("--config-name", default="train_rgb_diffusion")

    rollout = commands.add_parser("rollout", help="run a checkpoint")
    rollout.add_argument("checkpoint")
    rollout.add_argument("--device")
    rollout.add_argument("--output-dir")
    rollout.add_argument(
        "--weights",
        choices=("auto", "ema", "model"),
        default="auto",
        help="training-checkpoint weights (default: EMA when available)",
    )
    rollout.add_argument("--inspect", action="store_true")
    checkpoint = commands.add_parser("checkpoint", help="checkpoint utilities")
    checkpoint_ops = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    strip = checkpoint_ops.add_parser(
        "strip-backbone",
        help="remove frozen DINOv3 backbone weights from a checkpoint",
    )
    strip.add_argument("--input", required=True)
    strip.add_argument("--output", required=True)

    data = commands.add_parser("data", help="dataset operations")
    operations = data.add_subparsers(dest="data_command", required=True)
    convert = operations.add_parser("convert-actions")
    convert.add_argument("--dataset", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--overwrite", action="store_true")

    merge = operations.add_parser("merge")
    merge.add_argument("--datasets-root", default="datasets/mimicgen")
    merge.add_argument("--tasks", nargs="+", required=True)
    merge.add_argument("--output-task", required=True)
    merge.add_argument("--n-demo-per-task", type=int, required=True)
    merge.add_argument("--overwrite", action="store_true")

    playback = operations.add_parser("playback")
    playback.add_argument("--dataset-path", required=True)
    mode = playback.add_mutually_exclusive_group(required=True)
    mode.add_argument("--use-obs", action="store_true")
    mode.add_argument("--use-actions", action="store_true")
    playback.add_argument("--cache-dir")
    playback.add_argument("--cameras", nargs="+")
    playback.add_argument("--wait-ms", type=int, default=1)
    playback.add_argument("--save-first-frame")
    playback.add_argument("--no-window", action="store_true")
    playback.add_argument("--max-frames", type=int)
    playback.add_argument("--start-from", type=int, default=0)

    prepare = operations.add_parser("prepare")
    prepare.add_argument("--dataset")
    prepare.add_argument("--datasets-root")
    prepare.add_argument("--tasks", nargs="+")
    prepare.add_argument("--n-demo", type=int)
    prepare.add_argument("--output-dir")
    prepare.add_argument("--output-suffix", default="")
    prepare.add_argument("--table-texture-every", type=int)
    prepare.add_argument("--oracle-camera", default="agentview")
    prepare.add_argument("--oracle-patch-size", type=int, default=16)
    prepare.add_argument("--oracle-min-patch-area-fraction", type=float, default=0.05)
    prepare.add_argument("--oracle-min-mask-pixels", type=int, default=16)
    prepare.add_argument("--num-workers", type=int, default=4)
    prepare.add_argument("--enable-voxel", action="store_true")
    prepare.add_argument("--enable-point-cloud", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    try:
        args, extra = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as error:
        return int(error.code)
    if extra and args.command != "train":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "train":
        return _run_train(args, extra)
    if args.command == "setup":
        return _run_setup(args)
    if args.command == "rollout":
        return _run_rollout(args)
    if args.command == "checkpoint":
        return _run_checkpoint(args)
    return _run_data(args)


if __name__ == "__main__":
    raise SystemExit(main())
