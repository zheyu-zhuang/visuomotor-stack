if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Subset

from visuomotor.config import build as Build
from visuomotor.config import schema as Schema
from visuomotor.config.resolve import resolve_seeker_pretraining
from visuomotor.data.core.tensors import dict_apply, optimizer_to
from visuomotor.policy.staged_seeker import SeekerTrainingPolicy
from visuomotor.visualization import diagnostics as VisualizationDiagnostics
from visuomotor.visualization import rendering as VisualizationRendering
from visuomotor.visualization.artifacts import ArtifactStore, publish_artifacts
from visuomotor.workspace import training_utils as TrainingUtils
from visuomotor.workspace.base import BaseWorkspace
from visuomotor.workspace.display import pretty_print_nested


class TrainSeekerWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg, output_dir=None):
        spec = resolve_seeker_pretraining(cfg)
        super().__init__(Schema.to_dict(spec), output_dir=output_dir)
        self.spec = spec
        training = self.spec.training

        seed = training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: SeekerTrainingPolicy = Build.build_seeker_pretraining_policy(self.spec)

        self.ema_model: SeekerTrainingPolicy = None
        if training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.ema: TrainingUtils.EMAModel = None
        if self.ema_model is not None:
            ema_spec = self.spec.workspace.ema
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

    def _restore(self) -> str:
        training = self.spec.training
        resume_status = "From Scratch"
        if training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                resume_status = "Loaded From Saved"
                TrainingUtils.restore_checkpoint_with_optional_ema(self, lastest_ckpt_path)
                self.epoch += 1
                self.global_step += 1
        return resume_status

    def _report_runtime(self, resume_status: str) -> None:
        runtime_cfg = copy.deepcopy(self.model.get_runtime_config())
        runtime_cfg["Training"]["Checkpoint"] = resume_status
        runtime_cfg["Training"]["Batch Size"] = self.spec.workspace.train_loader.batch_size
        pretty_print_nested(
            runtime_cfg,
            title="Runtime Configuration",
            pad_before=True,
            pad_after=True,
        )

    def _build_training_data(self):
        spec = self.spec
        workspace = spec.workspace
        train_dataset = Build.build_seeker_pretraining_dataset(spec.dataset)
        train_dataset.set_mode("train")
        train_sampler = Build.build_sampler(
            train_dataset,
            workspace.sampler,
            seed=spec.training.seed,
        )
        train_dataloader = Build.build_dataloader(
            train_dataset, workspace.train_loader, sampler=train_sampler
        )
        normalizer = train_dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        return train_dataset, train_dataloader

    def _build_training_services(self, train_dataloader):
        spec = self.spec
        training = spec.training
        workspace = spec.workspace
        lr_scheduler = TrainingUtils.get_scheduler(
            training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * training.num_epochs,
            last_epoch=self.global_step - 1,
        )
        wandb_run = TrainingUtils.init_wandb(
            workspace.logging,
            output_dir=self.output_dir,
            config=Schema.to_dict(spec),
        )
        return lr_scheduler, self.ema, wandb_run

    def _prepare_device(self, train_dataset) -> torch.device:
        device = torch.device(self.spec.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self.model.obs_encoder.initialize_internal(
            dataset_size=train_dataset.n_frames_active,
            device=device,
        )
        return device

    def _train_epoch(
        self,
        train_dataloader,
        *,
        device: torch.device,
        lr_scheduler,
        ema,
        wandb_run,
        json_logger,
        max_train_steps,
    ) -> dict:
        training = self.spec.training
        step_log = {}
        train_losses = []
        with tqdm.tqdm(
            train_dataloader,
            desc=f"Training epoch {self.epoch}",
            leave=False,
            mininterval=training.tqdm_interval_sec,
        ) as progress:
            for batch_idx, batch in enumerate(progress):
                task_instruction = batch.pop("task_instruction", None)
                if task_instruction is not None:
                    task_instruction = task_instruction.copy()
                batch = dict_apply(
                    batch, lambda value: value.to(device, non_blocking=True)
                )
                raw_loss = self.model.loss(
                    batch,
                    task_instruction=task_instruction,
                    epoch_idx=self.epoch,
                )
                raw_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                lr_scheduler.step()
                if ema is not None:
                    ema.step(self.model)

                raw_loss_cpu = raw_loss.item()
                progress.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    "train_loss": raw_loss_cpu,
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if batch_idx != len(train_dataloader) - 1:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1
                if max_train_steps is not None and batch_idx >= max_train_steps - 1:
                    break
        if not train_losses:
            raise ValueError("Seeker pretraining dataloader produced no batches")
        step_log["train_loss"] = float(np.mean(train_losses))
        return step_log

    def _save_epoch_checkpoint(self, checkpoint_every: int) -> None:
        if self.epoch % checkpoint_every != 0:
            return
        checkpoint = self.spec.workspace.checkpoint
        if checkpoint.save_last:
            self.save_checkpoint()
        if checkpoint.save_snapshot:
            self.save_snapshot()
        if checkpoint.seeker_light_path is None:
            return
        seeker_weights = self.model.obs_encoder.seeker.state_dict()
        prefix = "obs_encoder.seeker."
        seeker_weights = {
            key.removeprefix(prefix): value for key, value in seeker_weights.items()
        }
        seeker_light_dir = os.path.dirname(checkpoint.seeker_light_path)
        if seeker_light_dir:
            os.makedirs(seeker_light_dir, exist_ok=True)
        torch.save(seeker_weights, checkpoint.seeker_light_path)

    def run(self):
        training = self.spec.training
        self._report_runtime(self._restore())
        train_dataset, train_dataloader = self._build_training_data()
        lr_scheduler, ema, wandb_run = self._build_training_services(train_dataloader)
        device = self._prepare_device(train_dataset)
        num_epochs = 2 if training.debug else training.num_epochs
        max_train_steps = 3 if training.debug else training.max_train_steps
        checkpoint_every = 1 if training.debug else training.checkpoint_every
        visualization = self.spec.workspace.visualization
        artifact_store = ArtifactStore(
            self.output_dir,
            save_images=visualization.enabled and visualization.save.images,
            save_videos=visualization.enabled and visualization.save.videos,
        )
        indices = VisualizationDiagnostics.evenly_spaced_indices(
            len(train_dataset), visualization.num_samples
        )
        diagnostic_loader = DataLoader(
            Subset(train_dataset, indices),
            batch_size=max(1, len(indices)),
            shuffle=False,
            num_workers=0,
        )

        def render_diagnostics(epoch):
            try:
                batch = next(iter(diagnostic_loader))
            except StopIteration:
                return None
            instruction = batch.pop("task_instruction", None)
            batch = dict_apply(batch, lambda value: value.to(device))
            policy = self.ema_model if self.ema_model is not None else self.model
            with VisualizationDiagnostics.isolated_evaluation(
                policy, seed=training.seed
            ):
                stages = policy.collect_diagnostics(
                    batch, epoch_idx=epoch, task_instruction=instruction
                )
            return VisualizationRendering.render_seeker_stages(
                stages, num_samples=visualization.num_samples
            )

        if visualization.enabled and visualization.augmentation_preview:
            preview = render_diagnostics(self.epoch)
            if preview is not None:
                artifact_store.save_image(
                    preview,
                    artifact_store.training_image(),
                    key="media/training/augmentation_preview",
                )
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with TrainingUtils.JsonLogger(log_path) as json_logger:
            while self.epoch < num_epochs:
                step_log = self._train_epoch(
                    train_dataloader,
                    device=device,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    wandb_run=wandb_run,
                    json_logger=json_logger,
                    max_train_steps=max_train_steps,
                )
                self._save_epoch_checkpoint(checkpoint_every)
                is_last_epoch = self.epoch == num_epochs - 1
                if visualization.enabled and (
                    self.epoch % checkpoint_every == 0 or is_last_epoch
                ):
                    panel = render_diagnostics(self.epoch)
                    if panel is not None:
                        record = artifact_store.save_image(
                            panel,
                            artifact_store.eval_image(
                                "seeker", epoch=self.epoch, step=self.global_step
                            ),
                            key="media/eval/seeker",
                            caption=f"Seeker epoch {self.epoch}",
                        )
                        if record is not None:
                            publish_artifacts(
                                wandb_run,
                                [record],
                                upload_images=visualization.upload.images,
                                upload_videos=False,
                                step=self.global_step,
                            )
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
