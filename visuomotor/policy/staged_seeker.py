"""Training-only Seeker policy for diffusion visual-focus pretraining."""

from copy import deepcopy
from typing import Optional

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from visuomotor.action_generation.diffusion import DiffusionActionGenerator
from visuomotor.data.core import actions as CoreActions
from visuomotor.data.core.normalization import Normalizer, normalize_obs
from visuomotor.perception.focus.seeker.training_encoder import SeekerTrainingEncoder
from visuomotor.policy.base import ModuleAttrMixin


class SeekerTrainingPolicy(ModuleAttrMixin):
    """Train Seeker-based visual-focus policies with the diffusion objective."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        image_size,
        num_robots,
        weights=None,
        stage_stride=30,
        visual_mode="external",
        obs_dropout=0.0,
        action_rep="absolute",
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        unet_channels=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        background_path=None,
        seeker_overrides=None,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1

        enc_n_hidden = 128
        action_dim = action_shape[0]

        obs_feature_dim = enc_n_hidden * int(n_obs_steps)
        generator = DiffusionActionGenerator(
            action_dim=action_dim,
            condition_dim=obs_feature_dim,
            prediction_horizon=horizon,
            noise_scheduler=noise_scheduler,
            unet_channels=unet_channels,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            num_inference_steps=num_inference_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
        )

        self.coarse_generator = generator
        self.fine_generator = deepcopy(generator)

        self.obs_encoder = SeekerTrainingEncoder(
            n_hidden=enc_n_hidden,
            num_robots=num_robots,
            obs_dropout=obs_dropout,
            image_size=image_size,
            visual_mode=visual_mode,
            background_path=background_path,
            weights=weights,
            seeker_config=seeker_overrides,
        )

        self.normalizer = Normalizer()

        self.stage_stride = stage_stride

        model_num_params = sum(p.numel() for p in generator.parameters()) / 1e6
        enc_num_params = sum(p.numel() for p in self.obs_encoder.parameters()) / 1e6
        action_rep = CoreActions.validate_action_rep(action_rep)

        self.runtime_config = {
            "Training": {
                "Objective": "Diffusion",
                "Action Chunk Representation": action_rep,
                "Visual Mode": visual_mode.replace("_", " ").title(),
                "Horizon": horizon,
                "Action Steps": n_action_steps,
                "Observation Steps": n_obs_steps,
                "Stage Stride": f"{int(self.stage_stride)} epochs",
                "Model Params": f"{model_num_params:.2f} M",
                "Encoder Params": f"{enc_num_params:.2f} M",
            },
            "Attention Seeker": self.obs_encoder.seeker.get_runtime_config(),
            "Background Randomizer": (
                self.obs_encoder.background_randomizer.get_runtime_config()
                if self.obs_encoder.background_randomizer is not None
                else {"Status": "Disabled"}
            ),
        }

    def get_runtime_config(self, num_epochs: Optional[int] = None) -> dict:
        """Return the runtime configuration block shown at startup."""
        return self.runtime_config

    def set_normalizer(self, normalizer: Normalizer):
        """Set action/observation normalizer for this policy and encoder."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

    def lambda_scheduler(self, epoch, duration, max_lambda=1.0):
        """Linear warmup scheduler in [0, max_lambda]."""
        if epoch < 0:
            return 0.0
        return min((epoch + 1) / (duration + 1e-6), max_lambda)

    def _forward_batch(self, batch, epoch_idx, task_instruction=None):
        """Shared preprocessing and stage selection for one training batch."""
        assert "valid_mask" not in batch

        obs = batch["obs"]
        task_context = batch["task_context"]
        obs_index = batch.get("obs_index", None)
        robot_id = task_context.get("robot_id", None)
        task_embedding = task_context.get("task_embedding", None)
        assert robot_id is not None, "robot_id must be provided"
        assert task_embedding is not None, "task_embedding must be provided"

        nactions = self.normalizer.normalize_action(batch["action"], robot_id)
        model_obs = normalize_obs(
            obs,
            self.normalizer,
            observation_kinds={
                key: "rgb" for key in ("rgb_external", "rgb_wrist") if key in obs
            },
            robot_id=robot_id,
        )

        s = self.stage_stride
        stage = "coarse" if epoch_idx < s else "fine"
        alpha = 0.5 if epoch_idx < 3 * s else 0.6

        if epoch_idx >= 3 * s:
            self.obs_encoder.disable_random_crop = True

        feat_dict, consistency_loss = self.obs_encoder(
            obs=model_obs,
            canonical_obs=obs,
            task_context=task_context,
            stage=stage,
            overlay_alpha=alpha,
            task_instruction=task_instruction,
            obs_index=obs_index,
        )

        return nactions, feat_dict, consistency_loss, epoch_idx

    def loss(
        self,
        batch,
        epoch_idx,
        task_instruction=None,
    ):
        """Compute diffusion loss plus stage-consistency regularization."""
        nactions, feat_dict, consistency_loss, epoch_idx = self._forward_batch(
            batch=batch,
            epoch_idx=epoch_idx,
            task_instruction=task_instruction,
        )

        batch_size = nactions.shape[0]
        trajectory = nactions

        loss = 0.0
        if feat_dict["coarse"] is not None:
            loss += self._train_step_diffusion(
                trajectory=trajectory,
                generator=self.coarse_generator,
                global_cond=feat_dict["coarse"].reshape(batch_size, -1),
            )
        if feat_dict["fine"] is not None:
            loss += self._train_step_diffusion(
                trajectory=trajectory,
                generator=self.fine_generator,
                global_cond=feat_dict["fine"].reshape(batch_size, -1),
            )
            loss /= 2.0

        s = self.stage_stride
        trimming_lambda = self.lambda_scheduler(epoch_idx - 2 * s, s // 2)
        return loss + trimming_lambda * consistency_loss

    @torch.no_grad()
    def collect_diagnostics(self, batch, *, epoch_idx, task_instruction=None):
        obs = batch["obs"]
        task_context = batch["task_context"]
        robot_id = task_context["robot_id"]
        model_obs = normalize_obs(
            obs,
            self.normalizer,
            observation_kinds={
                key: "rgb" for key in ("rgb_external", "rgb_wrist") if key in obs
            },
            robot_id=robot_id,
        )
        stage = "coarse" if epoch_idx < self.stage_stride else "fine"
        alpha = 0.5 if epoch_idx < 3 * self.stage_stride else 0.6
        return self.obs_encoder.collect_diagnostics(
            model_obs,
            canonical_obs=obs,
            task_context=task_context,
            stage=stage,
            overlay_alpha=alpha,
        )

    def _train_step_diffusion(self, trajectory, generator, global_cond):
        """Single diffusion training step."""
        return generator.loss(trajectory, global_cond)
