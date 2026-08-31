"""Training encoder that wraps Seeker for policy learning."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from visuomotor.data.core.images import resize_image
from visuomotor.geometry.roi import box_px_to_grid_mask, grid_mask_to_pixel_box
from visuomotor.perception.common.augmentation import (
    BackgroundRandomizer,
    CropRandomizer,
)
from visuomotor.perception.common.inputs import EncoderInputs, ObsInputProcessor
from visuomotor.perception.focus.seeker.model import Seeker, SeekerStageOut


@dataclass
class EncoderStageOut:
    """Encoder wrapper around one Seeker stage output."""

    stage: str
    image: torch.Tensor
    seeker: SeekerStageOut
    tight_box: torch.Tensor  # grid_mask_to_pixel_box output (used for norm + loss)
    input_box: Optional[torch.Tensor]  # box used to create this stage input
    feat: torch.Tensor

    @property
    def ctx(self) -> torch.Tensor:
        return self.seeker.ctx

    @property
    def attn(self) -> torch.Tensor:
        return self.seeker.attn_map

    @property
    def head_score(self) -> torch.Tensor:
        return self.seeker.head_score

    @property
    def mask(self) -> torch.Tensor:
        return self.seeker.mask


class SeekerTrainingEncoder(ObsInputProcessor):
    """Seeker-backed visual encoder used during training.

    Supports three modes via `visual_mode`:
    - `external`: external camera only
    - `external_wrist`: external + wrist
    - `finetune_wrist`: train the wrist view with cached external coarse features
    """

    def __init__(
        self,
        image_size: int,
        num_robots: int,
        input_res: int = 224,
        n_hidden: int = 128,
        obs_dropout: float = 0.0,
        *,
        visual_mode: str = "external",  # | "external_wrist" | "finetune_wrist"
        background_path: Optional[str] = None,
        weights: Optional[str] = None,
        strict_weights: bool = True,
        seeker_config: Optional[dict] = None,
    ) -> None:
        """Initialize SeekerTrainingEncoder.

        Args:
            image_size: Input image resolution before crop/resize.
            input_res: Resolution passed into Seeker.
            n_hidden: Hidden dimension for projection layers
            obs_dropout: Dropout probability for observation features
            visual_mode: One of "external", "external_wrist", "finetune_wrist".
            background_path: Optional texture/background directory for overlay augmentation.
            weights: Optional Seeker checkpoint path (used in finetune_wrist).
            seeker_config: Optional overrides merged onto model/seeker.yaml
                (e.g. intent_refiner.num_refinement_iters, query_composer.disable_proprio).
        """
        super().__init__(num_robots=num_robots)
        assert visual_mode in (
            "external",
            "external_wrist",
            "finetune_wrist",
        ), f"Unknown visual_mode: {visual_mode}"

        # Core modules
        obs_shape = (3, image_size, image_size)
        self.crop_randomizer = CropRandomizer(obs_shape, crop_size=input_res)

        self.visual_mode = visual_mode

        self.enable_wrist_view = visual_mode != "external"
        self.use_cached_external = visual_mode == "finetune_wrist"

        views = ["external"]
        if self.enable_wrist_view:
            views.append("wrist")

        self.seeker = Seeker(
            num_robots=num_robots,
            views=views,
            weights=weights,
            verbose=False,
            config=seeker_config,
            strict_weights=strict_weights,
        )

        # Feature projections
        feat_dim = self.seeker.out_dim
        self.task_emb_proj = nn.Linear(self.seeker.task_emb_dim, 16)
        in_dim = feat_dim + 11 + 16 + self.num_robots  # proprio + task_emb + robot_id
        in_dim += feat_dim if self.enable_wrist_view else 0

        self.fine_proj = nn.Linear(in_dim, n_hidden)
        self.coarse_proj = nn.Linear(in_dim, n_hidden)

        # Training parameters
        self.obs_dropout = obs_dropout
        self.proprio_noise = 0.005

        # Processing parameters
        self.input_res = input_res
        self.image_size = image_size
        self.margin = 8
        self.box_jitter = 0.1

        # External-feature caching
        self.buffer = None
        self.disable_random_crop = False

        # background randomizer
        if background_path is not None:
            self.background_randomizer = BackgroundRandomizer(
                input_shape=(self.input_res, self.input_res),
                background_path=background_path,
            )
        else:
            self.background_randomizer = None

    def set_normalizer(self, normalizer):
        self.seeker.set_normalizer(normalizer)

    def initialize_internal(self, dataset_size, device):
        if not self.use_cached_external:
            return

        feat_dim = self.seeker.out_dim
        self.buffer = torch.empty(
            dataset_size, feat_dim, dtype=torch.float32, device=device
        )
        self.buffer_valid = torch.zeros(dataset_size, dtype=torch.bool, device=device)

    def forward(
        self,
        obs,
        *,
        canonical_obs=None,
        task_context=None,
        stage: str = "coarse",
        overlay_alpha=None,
        task_instruction=None,
        obs_index=None,
    ):
        shared_args = dict(
            obs=obs,
            canonical_obs=canonical_obs,
            task_context=task_context,
            overlay_alpha=overlay_alpha,
            task_instruction=task_instruction,
            stage=stage,
        )
        if self.use_cached_external:
            return self._forward_finetune_wrist(obs_index=obs_index, **shared_args)
        else:
            return self._forward(**shared_args)

    def _forward_finetune_wrist(
        self,
        obs,
        *,
        canonical_obs,
        task_context,
        stage: str,
        obs_index,
        overlay_alpha=None,
        task_instruction=None,
    ):
        """Forward path for `finetune_wrist` mode using cached external coarse features."""
        assert self.use_cached_external
        assert self.enable_wrist_view
        assert self.buffer is not None, "Call initialize_internal() before finetuning."
        assert obs_index is not None, "obs_index required for caching."
        if obs_index.shape[-1] == 1:
            obs_index = obs_index.squeeze(-1)
        assert obs_index.dim() == 1, "obs_index must be shape [B]."

        # Keep deterministic external preprocessing so cache lookup is stable.
        enc_in = self.process_obs(obs, canonical_obs, task_context)
        assert enc_in.T == 1, (
            "finetune_wrist assumes T==1 (frame-level indexing). "
            "If T>1, pass per-frame indices or reshape indices accordingly."
        )

        if self.enable_wrist_view:
            assert enc_in.wrist is not None, "wrist image is missing"

        # 1) Retrieve cached external coarse ctx (or populate cache).
        idx = obs_index.to(self.buffer.device, non_blocking=True)
        need_cache = ~self.buffer_valid[idx]  # [B] bool

        if need_cache.any():
            # Compute external coarse ctx only for missing entries.
            with torch.no_grad():
                external = enc_in.external[need_cache]
                composer_in = {k: v[need_cache] for k, v in enc_in.composer_in.items()}
                external_outs_missing = self.run_stages(
                    external,
                    stage="coarse",
                    view="external",
                    composer_in=composer_in,
                    overlay_alpha=None,
                )
                assert (
                    len(external_outs_missing) > 0
                ), "No stage output while populating cache."
                external_ctx_missing = (
                    external_outs_missing[0].feat.detach().float()
                )  # [Bm, D]

            self.buffer[idx[need_cache]] = external_ctx_missing
            self.buffer_valid[idx[need_cache]] = True

        external_ctx = self.buffer[idx]  # [B, D]

        # 2) Compute EIH coarse ctx (trainable path).
        wrist_outs = self.run_stages(
            enc_in.wrist,
            stage=stage,
            view="wrist",
            composer_in=enc_in.composer_in,
            overlay_alpha=overlay_alpha,
        )
        assert len(wrist_outs) > 0, "wrist stages empty."

        # Randomly drop external features during training.
        mask_prob = 0.2
        if self.training and mask_prob > 0.0:
            mask = torch.rand(external_ctx.size(0), 1, device=external_ctx.device) > mask_prob
            external_ctx = external_ctx * mask.float()

        external_feats = [external_ctx for _ in range(len(wrist_outs))]

        feat_dict = self.aggregate_features(
            enc_in=enc_in,
            external_feats=external_feats,
            wrist_feats=[out.feat for out in wrist_outs],
        )

        return feat_dict, self.stage_consistency_loss(wrist_outs)

    def _forward(
        self,
        obs,
        *,
        canonical_obs,
        task_context,
        stage: str,
        overlay_alpha=None,
        task_instruction=None,
    ):
        assert stage in ["fine", "coarse"], f"Unknown Stage: {stage}"

        enc_in = self.process_obs(obs, canonical_obs, task_context)

        shared = dict(composer_in=enc_in.composer_in, overlay_alpha=overlay_alpha)

        external_outs = self.run_stages(
            enc_in.external, stage=stage, view="external", **shared
        )
        wrist_outs = self.run_stages(
            enc_in.wrist, stage=stage, view="wrist", **shared
        )

        if self.enable_wrist_view:
            assert len(wrist_outs) > 0, "wrist image is missing"

        wrist_feats = [out.feat for out in wrist_outs] if self.enable_wrist_view else None
        feat_dict = self.aggregate_features(
            enc_in=enc_in,
            external_feats=[out.feat for out in external_outs],
            wrist_feats=wrist_feats,
        )

        consistency_loss = self.stage_consistency_loss(external_outs)
        consistency_loss += self.stage_consistency_loss(wrist_outs)

        return feat_dict, consistency_loss

    def collect_diagnostics(
        self,
        obs,
        *,
        canonical_obs,
        task_context,
        stage: str,
        overlay_alpha=None,
    ):
        """Return coarse/fine stage geometry without retaining visualization state."""
        enc_in = self.process_obs(obs, canonical_obs, task_context)
        shared = dict(composer_in=enc_in.composer_in, overlay_alpha=overlay_alpha)
        return {
            "external": self.run_stages(
                enc_in.external, stage=stage, view="external", **shared
            ),
            "wrist": self.run_stages(
                enc_in.wrist, stage=stage, view="wrist", **shared
            ),
        }

    def run_stages(
        self,
        image: torch.Tensor,
        view: str,
        *,
        composer_in,
        stage,
        overlay_alpha=None,
    ) -> list[EncoderStageOut]:
        """Run Seeker stages on one view and wrap outputs for encoder use."""

        if image is None:
            return []

        assert stage in ("coarse", "fine")
        assert image.shape[-1] in (self.input_res, self.input_res // 2)

        image_aug = self._random_overlay(image, overlay_alpha)

        B, _, H, W = image_aug.shape
        device = image_aug.device
        full_box = torch.tensor([[0, 0, W - 1, H - 1]], device=device).expand(B, -1)

        out = self.seeker(
            image=image_aug,
            view=view,
            composer_in=composer_in,
            proprio_noise=self.proprio_noise if self.training else 0.0,
            stage=stage,
        )

        stage_outs = [("coarse", out.coarse)]
        if out.fine is not None:
            stage_outs.append(("fine", out.fine))

        outs: list[EncoderStageOut] = []
        for stage_name, stage_out in stage_outs:
            ctx = stage_out.ctx
            mask = stage_out.mask
            tight_box = grid_mask_to_pixel_box(mask.squeeze(1), full_box)

            # Store raw ctx as (B, D), assuming Nq == 1.
            ctx_vec = ctx.squeeze(1)

            outs.append(
                EncoderStageOut(
                    stage=stage_name,
                    image=image_aug,
                    seeker=stage_out,
                    tight_box=tight_box,
                    input_box=full_box,
                    feat=ctx_vec,
                )
            )
        return outs

    def aggregate_features(
        self,
        *,
        external_feats: list[torch.Tensor],
        wrist_feats: Optional[list[torch.Tensor]],
        enc_in,
    ) -> dict:
        assert len(external_feats) > 0, "external_feats cannot be empty."

        if self.enable_wrist_view:
            assert wrist_feats is not None and len(wrist_feats) == len(
                external_feats
            ), "Stage mismatch"

        proprio = enc_in.proprio
        task_embedding = enc_in.task_embedding
        robot_id = enc_in.composer_in["robot_id"]

        task_ind = self.task_emb_proj(task_embedding)
        aux = torch.cat([proprio, task_ind, robot_id], dim=-1)

        # -------- stage 0 (coarse) --------
        coarse_in = torch.cat([external_feats[0], aux], dim=-1)
        if self.enable_wrist_view:
            coarse_in = torch.cat([wrist_feats[0], coarse_in], dim=-1)
        coarse_feat = self.coarse_proj(coarse_in)

        # -------- stage 1 (fine, optional) --------
        fine_feat = None
        if len(external_feats) > 1:
            fine_in = torch.cat([external_feats[1], aux], dim=-1)
            if self.enable_wrist_view:
                fine_in = torch.cat([wrist_feats[1], fine_in], dim=-1)
            fine_feat = self.fine_proj(fine_in)

        feat_dict = {"coarse": coarse_feat, "fine": fine_feat}

        # global feature dropout
        if self.obs_dropout > 0.0 and self.training:
            for k, v in feat_dict.items():
                if v is not None:
                    feat_dict[k] = F.dropout(v, p=self.obs_dropout, training=True)

        return feat_dict

    def stage_consistency_loss(
        self, stages: list[EncoderStageOut], pad_box=True
    ) -> torch.Tensor:
        """KL consistency between coarse attention and final-stage tight box."""
        if stages is None or len(stages) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        coarse_attn = stages[0].attn  # [(B*T), H, Nq, Nk] or [B, H, Nq, Nk]
        tight_box = stages[-1].tight_box  # [B,4] or [(B*T),4]
        # pad tight_box to avoid cutting off attention at borders
        if pad_box:
            tight_box = tight_box.clone()
            tight_box[:, 0] = torch.clamp(tight_box[:, 0] - self.margin, min=0)
            tight_box[:, 1] = torch.clamp(tight_box[:, 1] - self.margin, min=0)
            tight_box[:, 2] = torch.clamp(
                tight_box[:, 2] + self.margin, max=self.input_res - 1
            )
            tight_box[:, 3] = torch.clamp(
                tight_box[:, 3] + self.margin, max=self.input_res - 1
            )
        head_score = stages[0].head_score  # [B, H, Nq, 1] or [B, H, Nq]

        # Select top-k heads per (b, nq)
        head_score = rearrange(head_score.squeeze(-1), "b h nq -> (b nq) h")
        k = int(self.seeker.select_n_heads)
        sel_heads = torch.topk(head_score, k=k, dim=1, largest=True).indices

        # Gather selected heads
        sel_idx = (
            sel_heads.unsqueeze(-1).expand(-1, -1, coarse_attn.size(-1)).unsqueeze(2)
        )
        coarse_attn_sel = torch.gather(coarse_attn, 1, sel_idx)

        # Trim attention with tight box
        Nk = coarse_attn_sel.shape[-1]
        S = int(Nk**0.5)
        assert Nk == S * S, f"Expected square grid, got {Nk} != {S}^2"

        box_grid = box_px_to_grid_mask(tight_box, image_size=self.input_res, grid_size=S)
        trimmed = coarse_attn_sel * box_grid.unsqueeze(1)

        # Flatten (time may already be folded into batch)
        coarse_attn_sel = coarse_attn_sel.reshape(-1, Nk)
        trimmed = trimmed.reshape(-1, Nk)

        # KL divergence
        eps = 1e-8
        q = trimmed / (trimmed.sum(dim=-1, keepdim=True) + eps)  # teacher
        p = coarse_attn_sel / (
            coarse_attn_sel.sum(dim=-1, keepdim=True) + eps
        )  # student
        p = p.clamp(min=1e-4)

        return F.kl_div(p.log(), q.detach(), reduction="none").mean()

    def _random_overlay(self, image, overlay_alpha):
        if self.background_randomizer is None or overlay_alpha is None:
            return image
        B, C, H, W = image.shape
        bg = self.background_randomizer(B)
        if bg.shape[-2:] != (H, W):
            bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)
        a = overlay_alpha if overlay_alpha is not None else 0.5
        rand_indices = torch.randperm(B)[: int(B * 0.5)]
        img_aug = image.clone()
        img_aug[rand_indices] = image[rand_indices] * a + bg[rand_indices] * (1 - a)
        return img_aug

    def process_obs(
        self,
        obs: Dict[str, torch.Tensor],
        canonical_obs: Optional[Dict[str, torch.Tensor]] = None,
        task_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> EncoderInputs:
        """Preprocess normalized observations for Seeker forward."""
        enc_in = self.obs_to_input(
            obs,
            canonical_obs,
            task_context,
            self.seeker.normalizer,
            resize=False,
        )

        def _proc_img(image, resize_only: bool = False):
            if image is None:
                return None
            if resize_only or self.disable_random_crop:
                return resize_image(image, self.input_res)
            return self.crop_randomizer(image)

        # Disable random crop on cached external path for consistency.
        enc_in.external = _proc_img(
            enc_in.external, resize_only=self.use_cached_external
        )
        enc_in.wrist = _proc_img(enc_in.wrist)
        return enc_in
