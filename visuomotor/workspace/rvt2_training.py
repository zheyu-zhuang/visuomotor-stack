"""Train a MVT heatmap predictor on RVT2Heatmap heuristic labels."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from visuomotor.config import build as Build
from visuomotor.config import schema as Schema
from visuomotor.config.resolve import resolve_rvt2_pretraining
from visuomotor.data.core import images as CoreImages
from visuomotor.data.core import normalization as CoreNormalization
from visuomotor.data.mimicgen.rvt2 import rvt2_dataset as Rvt2Dataset
from visuomotor.perception.common.augmentation import BackgroundRandomizer
from visuomotor.perception.focus.rvt2 import model as Rvt2Heatmap
from visuomotor.visualization.artifacts import ArtifactStore, publish_artifacts
from visuomotor.visualization.diagnostics import isolated_evaluation
from visuomotor.visualization.rendering import draw_patch_diagnostic
from visuomotor.workspace import training_utils as TrainingUtils
from visuomotor.workspace.base import BaseWorkspace


def _gaussian_patch_targets(
    target: torch.Tensor,
    *,
    grid_size: int,
    sigma: float,
) -> torch.Tensor:
    """Build soft patch heatmap labels centered on the target patch."""
    if sigma <= 0:
        raise ValueError("sigma must be positive for Gaussian targets")
    device = target.device
    coords = torch.arange(grid_size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    patch_y = (target // grid_size).to(torch.float32)[:, None, None]
    patch_x = (target % grid_size).to(torch.float32)[:, None, None]
    dist2 = (yy[None] - patch_y).square() + (xx[None] - patch_x).square()
    heatmap = torch.exp(-0.5 * dist2 / float(sigma * sigma))
    heatmap = heatmap.flatten(1)
    return heatmap / heatmap.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _patch_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    grid_size: int,
    sigma: float,
) -> torch.Tensor:
    if sigma <= 0:
        return F.cross_entropy(logits, target)
    soft_target = _gaussian_patch_targets(
        target,
        grid_size=grid_size,
        sigma=sigma,
    )
    return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _prepare_image(image: torch.Tensor, image_size: int) -> torch.Tensor:
    return CoreImages.resize_image(
        CoreNormalization.Normalizer.normalize_rgb(
            CoreImages.image_to_float01(image, source="raw")
        ),
        image_size,
    )


def _save_visualization_grid(
    *,
    patch_backbone: Rvt2Heatmap.PatchFeatureBackbone,
    head: Rvt2Heatmap.PatchActivationHead,
    background_randomizer: Optional[BackgroundRandomizer],
    overlay_alpha: Optional[float],
    loader: DataLoader,
    device: torch.device,
    dino_image_size: int,
    patch_size: int,
    num_samples: int,
    alpha: float,
) -> Optional[Image.Image]:
    if num_samples <= 0:
        return None
    try:
        batch = next(iter(loader))
    except StopIteration:
        return None

    with isolated_evaluation(head, seed=0), isolated_evaluation(
        patch_backbone, seed=0
    ):
        image = batch["image"].to(device, non_blocking=True)
        image_norm = _prepare_image(image, dino_image_size)
        image_norm = _random_overlay_like_seeker(
            image_norm,
            background_randomizer=background_randomizer,
            overlay_alpha=overlay_alpha,
        )
        image_vis = CoreImages.image_to_float01(image_norm, source="imagenet")
        target = batch["target_patch"].to(device, non_blocking=True)
        logits = _head_logits(
            head=head,
            patches=patch_backbone(image_norm),
            batch=batch,
            device=device,
        )
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

    grid_size = int(dino_image_size) // int(patch_size)
    n = min(int(num_samples), int(image.shape[0]))
    panels = []
    for i in range(n):
        prob_grid = probs[i].detach().cpu().reshape(grid_size, grid_size)
        panels.append(
            draw_patch_diagnostic(
                image=image_vis[i].detach().cpu(),
                prob_grid=prob_grid,
                target_patch=int(target[i].detach().cpu()),
                pred_patch=int(pred[i].detach().cpu()),
                source_frame=int(batch["source_frame"][i]),
                target_frame=int(batch["target_frame"][i]),
                alpha=alpha,
            )
        )
    if not panels:
        return None

    cols = min(4, len(panels))
    rows = int(math.ceil(len(panels) / cols))
    canvas = Image.new(
        "RGB",
        (cols * dino_image_size, rows * dino_image_size),
        color=(255, 255, 255),
    )
    for i, panel in enumerate(panels):
        canvas.paste(
            panel, ((i % cols) * dino_image_size, (i // cols) * dino_image_size)
        )
    return canvas


def _make_visualization_loader(
    dataset: Dataset,
    *,
    num_samples: int,
    mode: str,
) -> Optional[DataLoader]:
    n = len(dataset)
    if n == 0 or num_samples <= 0:
        return None
    k = min(int(num_samples), n)
    mode = str(mode).strip().lower()
    if mode not in ("even", "uniform"):
        raise ValueError(f"Unknown visualization sampling mode: {mode!r}")
    indices = np.linspace(0, n - 1, num=k, dtype=np.int64).tolist()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=k,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _head_logits(
    *,
    head: Rvt2Heatmap.PatchActivationHead,
    patches: torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    eef_pos = batch.get("eef_pos")
    task_context = batch["task_context"]
    task_language_tokens = task_context.get("task_language_tokens")
    return head(
        patches,
        batch["gripper"].to(device, non_blocking=True),
        task_context["task_embedding"].to(device, non_blocking=True),
        task_context["robot_id"].to(device, non_blocking=True),
        eef_pos=None if eef_pos is None else eef_pos.to(device, non_blocking=True),
        task_language_tokens=(
            None
            if task_language_tokens is None
            else task_language_tokens.to(device, non_blocking=True)
        ),
    )


def _run_epoch(
    *,
    patch_backbone: Rvt2Heatmap.PatchFeatureBackbone,
    head: Rvt2Heatmap.PatchActivationHead,
    background_randomizer: Optional[BackgroundRandomizer],
    overlay_alpha: Optional[float],
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    dino_image_size: int,
    patch_size: int,
    target_sigma_patches: float,
    max_steps: Optional[int],
    desc: str,
    show_progress: bool,
) -> dict:
    train = optimizer is not None
    patch_backbone.train(train and patch_backbone.backbone_type != "dino")
    head.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_top5 = 0.0
    total_count = 0
    grid_size = int(dino_image_size) // int(patch_size)

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc=desc,
            leave=True,
            file=sys.stdout,
            dynamic_ncols=True,
        )
    for step, batch in enumerate(iterator, start=1):
        image = batch["image"].to(device, non_blocking=True)
        image = _prepare_image(image, dino_image_size)
        if train:
            image = _random_overlay_like_seeker(
                image,
                background_randomizer=background_randomizer,
                overlay_alpha=overlay_alpha,
            )
        target = batch["target_patch"].to(device, non_blocking=True)

        patches = patch_backbone(image)
        logits = _head_logits(head=head, patches=patches, batch=batch, device=device)
        loss = _patch_heatmap_loss(
            logits,
            target,
            grid_size=grid_size,
            sigma=float(target_sigma_patches),
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            count = int(target.shape[0])
            total_loss += float(loss.item()) * count
            total_acc += float((pred == target).float().sum().item())
            total_top5 += float(
                (topk == target[:, None]).any(dim=1).float().sum().item()
            )
            total_count += count
            if show_progress:
                iterator.set_postfix(
                    loss=total_loss / max(total_count, 1),
                    acc=total_acc / max(total_count, 1),
                )

        if max_steps is not None and step >= max_steps:
            break

    if total_count == 0:
        return {"loss": None, "acc": None, "top5": None, "n": 0}
    return {
        "loss": total_loss / total_count,
        "acc": total_acc / total_count,
        "top5": total_top5 / total_count,
        "n": total_count,
    }


def _save_checkpoint(
    path: Path,
    *,
    head: Rvt2Heatmap.PatchActivationHead,
    optimizer: torch.optim.Optimizer,
    patch_backbone: Rvt2Heatmap.PatchFeatureBackbone,
    training_config: dict,
    rvt2_heatmap_config: dict,
    head_config: dict,
    gripper_mean: np.ndarray,
    gripper_std: np.ndarray,
    label_stats: dict,
    epoch: int,
    metrics: dict,
) -> None:
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "patch_backbone_state_dict": patch_backbone.state_dict(),
            "patch_backbone": patch_backbone.backbone_type,
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": training_config,
            "rvt2_heatmap_config": rvt2_heatmap_config,
            "head_config": head_config,
            "gripper_mean": gripper_mean,
            "gripper_std": gripper_std,
            "label_stats": label_stats,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    *,
    head: Rvt2Heatmap.PatchActivationHead,
    patch_backbone: Rvt2Heatmap.PatchFeatureBackbone,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    load_optimizer: bool,
) -> int:
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"RVT2Heatmap checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict) or "head_state_dict" not in payload:
        raise ValueError(f"Invalid RVT2Heatmap checkpoint: {ckpt_path}")

    head.load_state_dict(payload["head_state_dict"], strict=True)
    if "patch_backbone_state_dict" not in payload:
        raise ValueError("RVT2Heatmap checkpoint is missing patch_backbone_state_dict.")
    patch_backbone.load_state_dict(payload["patch_backbone_state_dict"], strict=True)
    if load_optimizer and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if "epoch" not in payload:
        raise ValueError("RVT2Heatmap checkpoint is missing epoch.")
    return int(payload["epoch"]) + 1


def _seeker_overlay_alpha(epoch: int, stage_stride: float) -> float:
    return 0.5 if int(epoch) < 3 * float(stage_stride) else 0.6


def _random_overlay_like_seeker(
    image: torch.Tensor,
    *,
    background_randomizer: Optional[BackgroundRandomizer],
    overlay_alpha: Optional[float],
) -> torch.Tensor:
    if background_randomizer is None or overlay_alpha is None:
        return image
    B = int(image.shape[0])
    count = int(B * 0.5)
    if count <= 0:
        return image
    bg = background_randomizer(B).to(device=image.device, dtype=image.dtype)
    if bg.shape[-2:] != image.shape[-2:]:
        bg = F.interpolate(
            bg, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
    idx = torch.randperm(B, device=image.device)[:count]
    out = image.clone()
    alpha = float(overlay_alpha)
    out[idx] = image[idx] * alpha + bg[idx] * (1.0 - alpha)
    return out


def _prepare_runtime(spec: Schema.Rvt2PretrainingSpec, output_dir: str) -> dict:
    training_config = Schema.to_dict(spec)
    training = spec.training
    model = spec.model
    rvt2_heatmap_cfg = dict(model.heatmap)
    rvt2_heatmap_cfg_dict = dict(rvt2_heatmap_cfg)
    if training.resume and training.load_weights:
        raise ValueError("Use only one of --resume or --load-weights.")

    np.random.seed(training.seed)
    torch.manual_seed(training.seed)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    background_randomizer = Build.build_rvt2_background_randomizer(
        model.background_overlay,
        image_size=int(rvt2_heatmap_cfg["dino_image_size"]),
    )
    return {
        "training_config": training_config,
        "spec": spec,
        "rvt2_heatmap_cfg": rvt2_heatmap_cfg,
        "rvt2_heatmap_cfg_dict": rvt2_heatmap_cfg_dict,
        "background_randomizer": background_randomizer,
        "output_dir": output_path,
    }


def _prepare_training_data(runtime: dict) -> dict:
    spec = runtime["spec"]
    dataset_spec = spec.dataset
    training = spec.training
    workspace = spec.workspace
    rvt2_heatmap_cfg = runtime["rvt2_heatmap_cfg"]
    dataset_path, dataset = Build.build_rvt2_pretraining_dataset(spec)
    runtime["dataset_path"] = dataset_path
    start_episode = min(
        int(dataset_spec.skip_first_episodes), dataset.n_demo_active
    )
    episode_indices = list(range(start_episode, dataset.n_demo_active))
    if not episode_indices:
        raise ValueError("No episodes selected after --skip-first-episodes.")

    records, label_stats = Rvt2Dataset.build_rvt2_heatmap_patch_records(
        dataset,
        episode_indices=episode_indices,
        camera=rvt2_heatmap_cfg["camera"],
        dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
        patch_size=rvt2_heatmap_cfg["patch_size"],
        joint_vel_atol=rvt2_heatmap_cfg["joint_vel_atol"],
        stopped_buffer_len=rvt2_heatmap_cfg["stopped_buffer_len"],
        include_final=rvt2_heatmap_cfg["include_final"],
        mute_initial_gripper_open=rvt2_heatmap_cfg["mute_initial_gripper_open"],
        show_progress=workspace.show_label_progress,
    )
    if not records:
        raise ValueError(f"No trainable labels were built. Label stats: {label_stats}")

    train_records, val_records = Rvt2Dataset.split_patch_records(
        records,
        val_ratio=dataset_spec.val_ratio,
        seed=training.seed,
    )
    gripper_mean, gripper_std = Rvt2Dataset.gripper_stats(train_records)

    train_set = Rvt2Dataset.RVT2HeatmapPatchDataset(
        dataset, train_records, gripper_mean, gripper_std
    )
    val_set = Rvt2Dataset.RVT2HeatmapPatchDataset(
        dataset, val_records, gripper_mean, gripper_std
    )
    train_sampler = Build.build_sampler(
        train_set,
        workspace.sampler,
        seed=training.seed,
    )
    train_loader = Build.build_dataloader(
        train_set, workspace.train_loader, sampler=train_sampler
    )
    val_loader = Build.build_dataloader(val_set, workspace.val_loader)
    return {
        "dataset": dataset,
        "train_set": train_set,
        "val_set": val_set,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_sampler": train_sampler,
        "records": records,
        "train_records": train_records,
        "val_records": val_records,
        "label_stats": label_stats,
        "gripper_mean": gripper_mean,
        "gripper_std": gripper_std,
    }


def _prepare_model_state(runtime: dict) -> Build.Rvt2PretrainingModel:
    return Build.build_rvt2_pretraining_model(runtime["spec"])


def _restore_training_state(
    runtime: dict, model_state: Build.Rvt2PretrainingModel
) -> int:
    training = runtime["spec"].training
    if training.resume:
        return _load_checkpoint(
            training.resume,
            head=model_state.head,
            patch_backbone=model_state.patch_backbone,
            optimizer=model_state.optimizer,
            device=model_state.device,
            load_optimizer=training.resume_optimizer,
        )
    if training.load_weights:
        _load_checkpoint(
            training.load_weights,
            head=model_state.head,
            patch_backbone=model_state.patch_backbone,
            optimizer=model_state.optimizer,
            device=model_state.device,
            load_optimizer=False,
        )
    return 1


def _write_run_summary(
    runtime: dict,
    data: dict,
    model_state: Build.Rvt2PretrainingModel,
    start_epoch: int,
) -> None:
    training = runtime["spec"].training
    workspace = runtime["spec"].workspace
    run_summary = {
        "dataset_path": str(runtime["dataset_path"]),
        "cache_dir": data["dataset"].cache_dir,
        "dino_ckpt": model_state.dino_checkpoint,
        "patch_backbone": runtime["rvt2_heatmap_cfg"]["patch_backbone"],
        "output_dir": str(runtime["output_dir"]),
        "records": len(data["records"]),
        "train_records": len(data["train_records"]),
        "val_records": len(data["val_records"]),
        "label_stats": data["label_stats"],
        "resume": training.resume,
        "load_weights": training.load_weights,
        "start_epoch": start_epoch,
        "rvt2_heatmap_config": runtime["rvt2_heatmap_cfg_dict"],
        "head_config": model_state.head_config,
    }
    with (runtime["output_dir"] / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_summary, f, indent=2)
        f.write("\n")
    if workspace.print_run_summary:
        print(json.dumps(run_summary, indent=2), flush=True)


def _run_training_loop(
    *,
    runtime: dict,
    data: dict,
    model_state: Build.Rvt2PretrainingModel,
    start_epoch: int,
    json_logger: TrainingUtils.JsonLogger,
    wandb_run=None,
) -> None:
    spec = runtime["spec"]
    model = spec.model
    training = spec.training
    workspace = spec.workspace
    visualization = workspace.visualization
    rvt2_heatmap_cfg = runtime["rvt2_heatmap_cfg"]
    artifact_store = ArtifactStore(
        runtime["output_dir"],
        save_images=visualization.enabled and visualization.save.images,
        save_videos=visualization.enabled and visualization.save.videos,
    )
    if visualization.enabled and visualization.augmentation_preview:
        preview_loader = _make_visualization_loader(
            data["train_set"],
            num_samples=visualization.num_samples,
            mode=workspace.visualization_sampling,
        )
        if preview_loader is not None:
            preview = _save_visualization_grid(
                patch_backbone=model_state.patch_backbone,
                head=model_state.head,
                background_randomizer=runtime["background_randomizer"],
                overlay_alpha=_seeker_overlay_alpha(start_epoch, model.stage_stride),
                loader=preview_loader,
                device=model_state.device,
                dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
                patch_size=rvt2_heatmap_cfg["patch_size"],
                num_samples=visualization.num_samples,
                alpha=workspace.visualization_alpha,
            )
            if preview is not None:
                artifact_store.save_image(
                    preview,
                    artifact_store.training_image(),
                    key="media/training/augmentation_preview",
                )
    checkpoint_every = int(training.checkpoint_every or 0)
    for epoch in range(start_epoch, int(training.epochs) + 1):
        overlay_alpha = _seeker_overlay_alpha(epoch, model.stage_stride)
        if data["train_sampler"] is not None:
            data["train_sampler"].set_epoch(epoch)
        train_metrics = _run_epoch(
            patch_backbone=model_state.patch_backbone,
            head=model_state.head,
            background_randomizer=runtime["background_randomizer"],
            overlay_alpha=overlay_alpha,
            loader=data["train_loader"],
            optimizer=model_state.optimizer,
            device=model_state.device,
            dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
            patch_size=rvt2_heatmap_cfg["patch_size"],
            target_sigma_patches=rvt2_heatmap_cfg["target_sigma_patches"],
            max_steps=training.max_train_steps,
            desc=f"epoch {epoch}/{training.epochs}",
            show_progress=True,
        )
        if len(data["val_set"]) > 0:
            val_metrics = _run_epoch(
                patch_backbone=model_state.patch_backbone,
                head=model_state.head,
                background_randomizer=runtime["background_randomizer"],
                overlay_alpha=None,
                loader=data["val_loader"],
                optimizer=None,
                device=model_state.device,
                dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
                patch_size=rvt2_heatmap_cfg["patch_size"],
                target_sigma_patches=rvt2_heatmap_cfg["target_sigma_patches"],
                max_steps=training.max_val_steps,
                desc=f"epoch {epoch} val",
                show_progress=False,
            )
        else:
            val_metrics = {"loss": None, "acc": None, "top5": None, "n": 0}

        payload = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if visualization.enabled:
            split_name = "val" if len(data["val_set"]) > 0 else "train"
            vis_set = data["val_set"] if len(data["val_set"]) > 0 else data["train_set"]
            vis_loader = _make_visualization_loader(
                vis_set,
                num_samples=visualization.num_samples,
                mode=workspace.visualization_sampling,
            )
            if vis_loader is not None:
                panel = _save_visualization_grid(
                    patch_backbone=model_state.patch_backbone,
                    head=model_state.head,
                    background_randomizer=runtime["background_randomizer"],
                    overlay_alpha=None,
                    loader=vis_loader,
                    device=model_state.device,
                    dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
                    patch_size=rvt2_heatmap_cfg["patch_size"],
                    num_samples=visualization.num_samples,
                    alpha=workspace.visualization_alpha,
                )
                record = artifact_store.save_image(
                    panel,
                    artifact_store.eval_image("rvt2", epoch=epoch, step=epoch),
                    key="media/eval/rvt2",
                    caption=f"RVT2 {split_name} epoch {epoch}",
                )
                if record is not None:
                    payload["visualization"] = str(record.path)
                    if wandb_run is not None:
                        publish_artifacts(
                            wandb_run,
                            [record],
                            upload_images=visualization.upload.images,
                            upload_videos=False,
                            step=epoch,
                        )
        json_logger.log(payload)
        if workspace.print_epoch_metrics:
            print(json.dumps(payload), flush=True)
        if wandb_run is not None:
            step_log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_top5": val_metrics["top5"],
            }
            wandb_run.log(step_log, step=epoch)

        should_checkpoint = checkpoint_every > 0 and (
            (epoch % checkpoint_every) == 0 or epoch == int(training.epochs)
        )
        if should_checkpoint:
            _save_checkpoint(
                runtime["output_dir"] / "latest.pt",
                head=model_state.head,
                patch_backbone=model_state.patch_backbone,
                optimizer=model_state.optimizer,
                training_config=runtime["training_config"],
                rvt2_heatmap_config=runtime["rvt2_heatmap_cfg_dict"],
                head_config=model_state.head_config,
                gripper_mean=data["gripper_mean"],
                gripper_std=data["gripper_std"],
                label_stats=data["label_stats"],
                epoch=epoch,
                metrics=payload,
            )


class TrainRVT2HeatmapWorkspace(BaseWorkspace):
    """Hydra workspace for RVT2Heatmap training."""

    def __init__(self, cfg, output_dir=None):
        spec = resolve_rvt2_pretraining(cfg)
        super().__init__(Schema.to_dict(spec), output_dir=output_dir)
        self.spec = spec
        self.runtime = _prepare_runtime(self.spec, self.output_dir)
        self.model_state = _prepare_model_state(self.runtime)
        self.start_epoch = 1
        self.data = None

    def run(self):
        wandb_run = TrainingUtils.init_wandb(
            self.spec.workspace.logging,
            output_dir=self.runtime["output_dir"],
            config=Schema.to_dict(self.spec),
        )

        self.data = _prepare_training_data(self.runtime)
        self.start_epoch = _restore_training_state(self.runtime, self.model_state)
        _write_run_summary(
            self.runtime,
            self.data,
            self.model_state,
            self.start_epoch,
        )
        log_path = self.runtime["output_dir"] / "train_log.jsonl"
        with TrainingUtils.JsonLogger(
            log_path, filter_fn=lambda _key, _value: True
        ) as logger:
            _run_training_loop(
                runtime=self.runtime,
                data=self.data,
                model_state=self.model_state,
                start_epoch=self.start_epoch,
                json_logger=logger,
                wandb_run=wandb_run,
            )
