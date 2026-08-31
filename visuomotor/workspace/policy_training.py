import copy
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Subset

from visuomotor.config import build as Build
from visuomotor.config import schema as Schema
from visuomotor.config.resolve import resolve_policy_run
from visuomotor.data.core import sparse_voxels as SparseVoxels
from visuomotor.data.core.mirror import MirrorObsActionAugmentor
from visuomotor.data.core.scene_augmentation import SceneYawObsActionAugmentor
from visuomotor.data.core.tensors import optimizer_to
from visuomotor.policy.base import BaseImagePolicy
from visuomotor.visualization import diagnostics as VisualizationDiagnostics
from visuomotor.visualization import rendering as VisualizationRendering
from visuomotor.visualization.artifacts import ArtifactStore, publish_artifacts
from visuomotor.workspace import training_utils as TrainingUtils
from visuomotor.workspace.base import BaseWorkspace
from visuomotor.workspace.display import pretty_print_nested

_CROP_RUNTIME_KEYS = ("Voxel Crop", "RGB Crop", "Random Crop", "Focus Box Crop")


def _group_augmentation_runtime_config(
    sample_augmentations: dict,
    model_cfg: dict,
) -> dict:
    augmentations = dict(sample_augmentations)
    for section in model_cfg.values():
        if not isinstance(section, dict):
            continue
        for key in _CROP_RUNTIME_KEYS:
            if key in section:
                augmentations[key] = section.pop(key)
    return augmentations


def _format_metric_line(log: dict, phase: str) -> str:
    sections = [
        f"epoch {int(log['epoch']):03d}",
        f"step {int(log['global_step']):,}",
    ]
    loss = log.get(f"{phase}_loss")
    if loss is not None:
        sections.append(f"loss: {loss:.4f}")
    if phase == "train" and log.get("lr") is not None:
        sections.append(f"lr: {log['lr']:.3e}")
    return " | ".join(sections)


def _score_label(key: str) -> str:
    prefix = key.rsplit("mean_score", 1)[0].rstrip("/_")
    return prefix.replace("_", " ").replace("/", " ") or "rollout"


def _format_rollout_line(log: dict) -> str:
    scores = []
    for key, value in log.items():
        if not key.endswith("mean_score"):
            continue
        best_key = key[: -len("mean_score")] + "max_score"
        scores.append(
            f"{_score_label(key)} {value:.1%} (best {log[best_key]:.1%})"
        )
    return " | ".join(
        (
            f"epoch {int(log['epoch']):03d}",
            f"step {int(log['global_step']):,}",
            f"success: {' · '.join(scores)}",
        )
    )


def _load_rollout_bests(path: Path) -> dict[str, float]:
    bests = {}
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return bests
    for line in lines:
        success = line.partition("| success: ")[2]
        for score in success.split(" · "):
            match = re.fullmatch(r"(.+?) [\d.]+% \(best ([\d.]+)%\)", score.strip())
            if match:
                label, best = match.groups()
                bests[label] = max(bests.get(label, -math.inf), float(best) / 100)
    return bests


def _rollout_outputs(
    runner_log: dict,
    *,
    global_step: int,
    epoch: int,
    best_scores: dict[str, float],
) -> dict:
    outputs = {"global_step": global_step, "epoch": epoch}
    for key, value in runner_log.items():
        if key.startswith("performance/"):
            outputs[key] = value
            continue
        if not key.endswith("mean_score"):
            continue
        best_key = key[: -len("mean_score")] + "max_score"
        label = _score_label(key)
        best = max(
            best_scores.get(label, -math.inf),
            runner_log.get(best_key, -math.inf),
            value,
        )
        best_scores[label] = best
        outputs[key] = value
        outputs[best_key] = best
    return outputs


class TrainPolicyWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg, output_dir=None):
        run_spec = resolve_policy_run(cfg)
        super().__init__(Schema.to_dict(run_spec), output_dir=output_dir)
        self.run_spec = run_spec
        training = self.run_spec.training

        seed = training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BaseImagePolicy = Build.build_policy(self.run_spec.model)

        self.ema_model: BaseImagePolicy = None
        if training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.ema: TrainingUtils.EMAModel = None
        if self.ema_model is not None:
            ema_spec = self.run_spec.workspace.ema
            self.ema = TrainingUtils.EMAModel(
                model=self.ema_model,
                update_after_step=ema_spec.update_after_step,
                inv_gamma=ema_spec.inv_gamma,
                power=ema_spec.power,
                min_value=ema_spec.min_value,
                max_value=ema_spec.max_value,
            )

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=training.lr,
            betas=training.betas,
            eps=training.eps,
            weight_decay=training.weight_decay,
        )

        self.global_step = 0
        self.epoch = 0

    def _build_sample_augmentors(self, shape_meta: dict):
        """Sample-level augmentation: multi-field, geometrically coupled, applied
        to whole batches before any encoder runs. Mirror and scene-yaw are
        mutually exclusive (rejected in config.schema.validate)."""
        dataset = self.run_spec.dataset
        canonical_keys = {
            "eef_pos": "eef_pos",
            "eef_rot": "eef_rot6d",
            "gripper_qpos": "gripper_qpos",
        }
        mirror = MirrorObsActionAugmentor(
            shape_meta=shape_meta,
            action_rep=dataset.trajectory.action_rep,
            config=dataset.mirror_augmentation,
            source_keys=canonical_keys,
        )
        scene_yaw = SceneYawObsActionAugmentor(
            shape_meta=shape_meta,
            action_rep=dataset.trajectory.action_rep,
            config=dataset.scene_yaw_augmentation,
            source_keys=canonical_keys,
            fixed_camera_rgb_keys=[
                key
                for key, field in shape_meta["obs"].items()
                if field.get("type") == "rgb" and key != "rgb_wrist"
            ],
        )
        return mirror, scene_yaw

    def _augmentation_runtime_config(self) -> dict:
        dataset = self.run_spec.dataset
        yaw = dataset.scene_yaw_augmentation
        mirror = dataset.mirror_augmentation
        return {
            "Scene Yaw": (
                f"Enabled ([{yaw.min_deg}, {yaw.max_deg}] deg)"
                if yaw is not None and yaw.enable
                else "Disabled"
            ),
            "Mirror": (
                f"Enabled (p={mirror.prob})"
                if mirror is not None and mirror.enable
                else "Disabled"
            ),
        }

    @staticmethod
    def _build_voxel_materializer(shape_meta: dict):
        """Build the device-side sparse-to-dense voxel materializer."""
        return SparseVoxels.VoxelMaterializer(
            {
                key: field["shape"]
                for key, field in shape_meta["obs"].items()
                if field.get("type") == "voxel"
            }
        )

    def _train_epoch(
        self,
        train_iter,
        *,
        batches_per_epoch,
        num_epochs,
        max_train_steps,
        log_freq,
        device,
        lr_scheduler,
        voxel_materializer,
        mirror_augmentor,
        scene_yaw_augmentor,
    ):
        training = self.run_spec.training
        step_log = {}
        train_losses = []
        with tqdm.tqdm(
            range(batches_per_epoch),
            desc=f"Training epoch {self.epoch}",
            leave=False,
            mininterval=training.tqdm_interval_sec,
        ) as progress:
            for batch_idx in progress:
                batch = next(train_iter)
                batch = tree_map(
                    lambda value: value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value,
                    batch,
                )
                voxel_materializer(batch)
                mirror_augmentor(batch)
                scene_yaw_augmentor(batch)
                loss_kwargs = {"global_step": self.global_step}
                if getattr(self.model, "wants_training_progress", False):
                    loss_kwargs["training_progress"] = self.epoch / max(
                        1, num_epochs - 1
                    )
                loss = self.model.loss(batch, **loss_kwargs)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if self.ema is not None:
                    self.ema.step(self.model)

                if self.global_step % log_freq == 0:
                    loss_value = loss.item()
                    progress.set_postfix(loss=loss_value, refresh=False)
                    train_losses.append(loss_value)
                step_log = {
                    "train_loss": train_losses[-1] if train_losses else None,
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                }

                if batch_idx != batches_per_epoch - 1:
                    self.global_step += 1
                if max_train_steps is not None and batch_idx >= max_train_steps - 1:
                    break

        step_log["train_loss"] = (
            float(np.mean(train_losses)) if train_losses else loss.item()
        )
        return step_log

    def _validate_epoch(
        self,
        policy,
        val_dataloader,
        step_log,
        *,
        max_val_steps,
        device,
        voxel_materializer,
        mirror_augmentor,
        prepare_diagnostic_batch,
        artifact_store,
        wandb_run,
    ):
        training = self.run_spec.training
        visualization = self.run_spec.workspace.visualization
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = tree_map(
                    lambda value: value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value,
                    batch,
                )
                voxel_materializer(batch)
                mirror_augmentor.center_batch(batch)
                loss = policy.loss(batch)
                val_losses.append(loss.item())
                if max_val_steps is not None and batch_idx >= max_val_steps - 1:
                    break
        if val_losses:
            step_log["val_loss"] = float(np.mean(val_losses))

        fixed_batch = prepare_diagnostic_batch()
        if fixed_batch is None or not visualization.enabled:
            return
        with VisualizationDiagnostics.isolated_evaluation(policy, seed=training.seed):
            diagnostics = policy.collect_diagnostics(
                fixed_batch["obs"],
                task_context=fixed_batch.get("task_context"),
            )
        panels = {
            "actions": VisualizationRendering.render_action_comparison(
                diagnostics.predicted_actions,
                fixed_batch["action"],
                num_samples=visualization.num_samples,
                action_rep=self.run_spec.model.trajectory.action_rep,
                observations=fixed_batch["obs"],
                voxel_geometry=diagnostics.encoder.voxel_crop_geometry,
            ),
        }
        focus = VisualizationRendering.render_focus_diagnostics(
            diagnostics.encoder,
            num_samples=visualization.num_samples,
            observations=fixed_batch["obs"],
            targets=fixed_batch.get("targets"),
        )
        if focus is not None:
            panels["focus"] = focus

        records = []
        for category, panel in panels.items():
            record = artifact_store.save_image(
                panel,
                artifact_store.eval_image(
                    category, epoch=self.epoch, step=self.global_step
                ),
                key=f"media/eval/{category}",
                caption=f"epoch {self.epoch} step {self.global_step}",
            )
            if record is not None:
                records.append(record)
        publish_artifacts(
            wandb_run,
            records,
            upload_images=visualization.upload.images,
            upload_videos=False,
            step=self.global_step,
        )

    def run(self):
        spec = self.run_spec
        training = spec.training
        workspace = spec.workspace
        trajectory = spec.model.trajectory
        resume_status = "From Scratch"

        if training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                resume_status = "Loaded From Saved"
                TrainingUtils.restore_checkpoint_with_optional_ema(self, lastest_ckpt_path)
                self.epoch += 1
                self.global_step += 1

        model_cfg = dict(self.model.get_runtime_config(num_epochs=training.num_epochs))
        augmentations = _group_augmentation_runtime_config(
            self._augmentation_runtime_config(), model_cfg
        )
        policy_block = model_cfg.pop("Training", {})
        normalizer = model_cfg.pop("Normalizer", None)
        flow_summary = model_cfg.pop("Flow", None)
        inputs_block = model_cfg.pop("Voxel Encoder", None)

        runtime_cfg = {
            "Experiment": spec.exp_name,
            "Seed": training.seed,
            "Task / Data": {
                "Task": f"{spec.task.name} · {spec.regime.name}",
                "Dataset": f"{spec.dataset.n_demo} demos",
                "Actions": f"{trajectory.action_rep} actions",
                "Horizon": (
                    f"{trajectory.observation_horizon} obs → "
                    f"{trajectory.prediction_horizon} pred → "
                    f"{trajectory.execution_horizon} exec"
                ),
                **({"Normalizer": normalizer} if normalizer is not None else {}),
            },
            "Policy": {
                "Stack": (
                    f"{spec.model.input.name} → "
                    f"{spec.model.encoder.name} → "
                    f"{spec.model.policy.name}"
                ),
                **policy_block,
                **({"Flow": flow_summary} if flow_summary is not None else {}),
            },
        }
        if inputs_block is not None:
            runtime_cfg["Inputs"] = inputs_block
        runtime_cfg.update(model_cfg)
        runtime_cfg["Training"] = {
            "Batch Size": workspace.train_loader.batch_size,
            "Checkpoint": resume_status,
            "Episodes": f"{spec.runner.n_test} episodes · {spec.runner.n_envs} envs",
            "Max Steps": spec.task.max_steps,
        }
        runtime_cfg["Augmentations"] = augmentations
        pretty_print_nested(runtime_cfg, title="Runtime Configuration", pad_before=True, pad_after=True)

        model_meta = spec.dataset.observation.model_meta(trajectory.action_dim)
        mirror_augmentor, scene_yaw_augmentor = self._build_sample_augmentors(model_meta)
        voxel_materializer = self._build_voxel_materializer(model_meta)
        train_dataset = Build.build_dataset(spec.dataset, mode="train")
        train_dataloader = Build.build_infinite_dataloader(train_dataset, workspace.train_loader)
        val_dataset = Build.build_dataset(spec.dataset, mode="eval")
        val_dataloader = None
        if len(val_dataset) > 0:
            val_dataloader = Build.build_dataloader(val_dataset, workspace.val_loader)
        diagnostic_dataset = val_dataset if len(val_dataset) > 0 else train_dataset
        diagnostic_indices = VisualizationDiagnostics.evenly_spaced_indices(
            len(diagnostic_dataset), workspace.visualization.num_samples
        )
        diagnostic_loader = DataLoader(
            Subset(diagnostic_dataset, diagnostic_indices),
            batch_size=max(1, len(diagnostic_indices)),
            shuffle=False,
            num_workers=0,
        )
        preview_indices = VisualizationDiagnostics.evenly_spaced_indices(
            len(train_dataset), workspace.visualization.num_samples
        )
        preview_loader = DataLoader(
            Subset(train_dataset, preview_indices),
            batch_size=max(1, len(preview_indices)),
            shuffle=False,
            num_workers=0,
        )
        normalizer = train_dataset.get_normalizer(spec.model.normalizer)

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = TrainingUtils.get_scheduler(
            training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * training.num_epochs,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        env_runner = Build.build_runner(spec.runner, output_dir=str(self.output_dir))

        wandb_run = TrainingUtils.init_wandb(
            workspace.logging,
            output_dir=self.output_dir,
            config=Schema.to_dict(spec),
            updates={"exp_name": spec.exp_name},
        )
        artifact_store = ArtifactStore(
            self.output_dir,
            save_images=workspace.visualization.enabled
            and workspace.visualization.save.images,
            save_videos=workspace.visualization.enabled
            and workspace.visualization.save.videos,
        )

        topk_manager = TrainingUtils.TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            monitor_key=workspace.checkpoint.topk.monitor_key,
            mode=workspace.checkpoint.topk.mode,
            k=workspace.checkpoint.topk.k,
            format_str=workspace.checkpoint.topk.format_str,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("training requires a CUDA GPU")
        device = torch.device(training.device)
        torch.backends.cudnn.benchmark = True
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        for model in filter(None, (self.model, self.ema_model)):
            initialize = getattr(model.encoder, "initialize_internal", None)
            if initialize is not None:
                initialize(dataset_size=train_dataset.n_frames_active, device=device)

        def prepare_diagnostic_batch(loader=diagnostic_loader, *, center=True):
            try:
                batch = next(iter(loader))
            except StopIteration:
                return None
            batch = tree_map(
                lambda value: value.to(device) if torch.is_tensor(value) else value,
                batch,
            )
            voxel_materializer(batch)
            if center:
                mirror_augmentor.center_batch(batch)
            return batch

        if workspace.visualization.enabled and workspace.visualization.augmentation_preview:
            with VisualizationDiagnostics.isolated_evaluation(
                self.model, seed=training.seed
            ):
                preview_batch = prepare_diagnostic_batch(preview_loader, center=False)
                if preview_batch is not None:
                    canonical_preview = tree_map(
                        lambda value: value.clone() if torch.is_tensor(value) else copy.deepcopy(value),
                        preview_batch["obs"],
                    )
                    mirror_augmentor(preview_batch)
                    scene_yaw_augmentor(preview_batch)
                    diagnostics = self.model.collect_diagnostics(
                        preview_batch["obs"], task_context=preview_batch.get("task_context")
                    )
            if preview_batch is not None:
                panels = [
                    VisualizationRendering.render_observation_strip(
                        canonical_preview, num_samples=workspace.visualization.num_samples
                    ),
                    VisualizationRendering.render_observation_strip(
                        preview_batch["obs"], num_samples=workspace.visualization.num_samples
                    ),
                ]
                if diagnostics.encoder and diagnostics.encoder.prepared_inputs:
                    panels.append(
                        VisualizationRendering.render_observation_strip(
                            diagnostics.encoder.prepared_inputs,
                            num_samples=workspace.visualization.num_samples,
                            rgb_source="imagenet",
                        )
                    )
                preview = VisualizationRendering.sample_row_report(
                    list(
                        zip(
                            [
                                "Canonical",
                                "Augmented",
                                "Encoder",
                            ][: len(panels)],
                            panels,
                        )
                    )
                )
                artifact_store.save_image(
                    preview,
                    artifact_store.training_image(),
                    key="media/training/augmentation_preview",
                )

        num_epochs = training.num_epochs
        max_train_steps = training.max_train_steps
        max_val_steps = training.max_val_steps
        rollout_every = training.rollout_every
        checkpoint_every = training.checkpoint_every
        val_every = training.val_every
        log_freq = training.log_freq
        if training.debug:
            num_epochs = 2
            max_train_steps = 3
            max_val_steps = 3
            rollout_every = 1
            checkpoint_every = 1
            val_every = 1
            log_freq = 1

        # Preserve worker prefetching across fixed-size bookkeeping epochs.
        batches_per_epoch = len(train_dataloader)
        train_iter = iter(train_dataloader)

        train_path = Path(self.output_dir) / "train.log"
        metrics_path = Path(self.output_dir) / "metrics.log"
        success_path = Path(self.output_dir) / "rollout_success.log"
        rollout_bests = _load_rollout_bests(success_path)
        with (
            train_path.open("a", buffering=1) as train_log,
            metrics_path.open("a", buffering=1) as metrics_log,
            success_path.open("a", buffering=1) as success_log,
        ):
            while self.epoch < num_epochs:
                step_log = self._train_epoch(
                    train_iter,
                    batches_per_epoch=batches_per_epoch,
                    num_epochs=num_epochs,
                    max_train_steps=max_train_steps,
                    log_freq=log_freq,
                    device=device,
                    lr_scheduler=lr_scheduler,
                    voxel_materializer=voxel_materializer,
                    mirror_augmentor=mirror_augmentor,
                    scene_yaw_augmentor=scene_yaw_augmentor,
                )

                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()

                should_validate = (
                    val_dataloader is not None
                    and val_every > 0
                    and (self.epoch % val_every) == 0
                )
                if should_validate:
                    self._validate_epoch(
                        policy,
                        val_dataloader,
                        step_log,
                        max_val_steps=max_val_steps,
                        device=device,
                        voxel_materializer=voxel_materializer,
                        mirror_augmentor=mirror_augmentor,
                        prepare_diagnostic_batch=prepare_diagnostic_batch,
                        artifact_store=artifact_store,
                        wandb_run=wandb_run,
                    )

                rollout_log = {}
                if (self.epoch % rollout_every) == 0 and self.epoch != 0:
                    rollout_metrics, rollout_artifacts = env_runner.run(
                        policy, epoch=self.epoch
                    )
                    rollout_log = _rollout_outputs(
                        rollout_metrics,
                        global_step=self.global_step,
                        epoch=self.epoch,
                        best_scores=rollout_bests,
                    )
                    publish_artifacts(
                        wandb_run,
                        rollout_artifacts,
                        upload_images=workspace.visualization.upload.images,
                        upload_videos=workspace.visualization.upload.videos,
                        step=self.global_step,
                    )

                # Log as soon as epoch metrics are ready. Checkpointing can be slow,
                # so logging after it makes rollout media appear one cycle late.
                wandb_run.log({**step_log, **rollout_log}, step=self.global_step)
                train_log.write(_format_metric_line(step_log, "train") + "\n")
                if "val_loss" in step_log:
                    metrics_log.write(_format_metric_line(step_log, "val") + "\n")
                if rollout_log:
                    success_log.write(_format_rollout_line(rollout_log) + "\n")

                is_last_epoch = self.epoch == (num_epochs - 1)
                should_checkpoint = checkpoint_every > 0 and (
                    ((self.epoch % checkpoint_every) == 0 and self.epoch != 0) or is_last_epoch
                )
                if should_checkpoint:
                    if workspace.checkpoint.save_last:
                        self.save_checkpoint()

                    metric_dict = {
                        key.replace("/", "_"): value
                        for key, value in {**step_log, **rollout_log}.items()
                    }

                    if topk_manager.monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            if workspace.checkpoint.save_topk_full:
                                self.save_checkpoint(path=topk_ckpt_path)
                            else:
                                policy_key = "ema_model" if self.ema_model is not None else "model"
                                self.save_state_dict_checkpoint(
                                    path=topk_ckpt_path,
                                    state_dicts={policy_key: policy},
                                    metadata={
                                        "lightweight": True,
                                        "checkpoint_type": "topk_policy",
                                        "policy_key": policy_key,
                                        "epoch": int(self.epoch),
                                        "global_step": int(self.global_step),
                                    },
                                )
                            policy.save_checkpoint(
                                Path(topk_ckpt_path).with_suffix(".release.pth"),
                                runner_spec=spec.runner,
                            )
                self.model.train()

                self.global_step += 1
                self.epoch += 1
