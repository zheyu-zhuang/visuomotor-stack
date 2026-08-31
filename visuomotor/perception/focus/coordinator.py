"""Per-view focus transforms for the released Seeker model."""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from visuomotor.config.schema import FocusConditionedEncoderSpec, OverlaySpec
from visuomotor.data.core.normalization import Normalizer, build_normalizer_module
from visuomotor.geometry.roi import crop_with_box, grid_mask_to_pixel_box
from visuomotor.perception.common import oracle as CommonOracle
from visuomotor.perception.common.augmentation import (
    BackgroundOverlay,
    ResizeCropRandomizer,
)
from visuomotor.perception.common.prediction import VisualFocusPrediction
from visuomotor.perception.focus.rvt2.model import RVT2Heatmap
from visuomotor.perception.focus.seeker.model import Seeker

MODE_OPERATION = {
    "pass_through": "pass_through",
    "focus_condition": "focus_condition",
    "focus_crop": "focus_crop",
    "lowres_crop_only": "focus_crop",
    "focus_mask": "focus_mask",
    "focus_mask_crop": "focus_mask_crop",
    "random_overlay": "random_overlay",
    "disabled": "disabled",
}
VALID_FOCUS_MODES = set(MODE_OPERATION)
FOCUS_BACKENDS = {"seeker", "rvt2_heatmap", "oracle"}
NO_FOCUS_MODES = {"pass_through", "random_overlay"}
FOCUS_OPERATION_MODES = {
    mode
    for mode, operation in MODE_OPERATION.items()
    if operation in {"focus_condition", "focus_crop", "focus_mask", "focus_mask_crop"}
}
FOCUS_CROP_MODES = {
    mode for mode, operation in MODE_OPERATION.items() if operation == "focus_crop"
}
FOCUS_MASK_CROP_MODES = {
    mode for mode, operation in MODE_OPERATION.items() if operation == "focus_mask_crop"
}
MASK_CACHE_MODES = {"focus_mask", "focus_mask_crop"}
GUIDED_OVERLAY_MODES = {"focus_mask", "focus_mask_crop"}
BOX_FILM_MODES = {
    mode
    for mode, operation in MODE_OPERATION.items()
    if operation in {"focus_condition", "focus_crop", "focus_mask_crop"}
}
VISUAL_FOCUS_RECORD_MODES = FOCUS_OPERATION_MODES


class FocusViewTransform(nn.Module):
    """Apply released Seeker crop/mask pipelines for each enabled camera view."""

    def __init__(
        self,
        spec: FocusConditionedEncoderSpec,
        *,
        normalizer_kind: str = "multi_robot_linear",
        verbose: bool = False,
    ):
        super().__init__()

        self.spec = spec
        self.verbose = bool(verbose)

        source = spec.source
        self.focus_source = str(source.name).strip().lower()
        if self.focus_source in {"", "null", "none"}:
            self.focus_source = "none"
        self.seeker_weights = source.weights
        self.seeker_strict_weights = bool(source.strict_weights)
        self.rvt2_checkpoint = source.checkpoint
        self.vit_in = int(spec.vit_in)
        self.random_crop = spec.random_crop

        guided = spec.guided_overlay or OverlaySpec()
        random_overlay = spec.random_overlay or OverlaySpec()

        self.guided_overlay_prob = float(guided.prob)
        self.guided_overlay_noise_std = float(guided.noise_std)
        self.guided_overlay_alpha_min = float(guided.alpha_min)
        self.guided_overlay_alpha_max = float(guided.alpha_max)
        self.guided_overlay_warmup_steps = int(guided.warmup_steps)
        self.guided_overlay_background_path = guided.background_path

        self.random_overlay_prob = float(random_overlay.prob)
        self.random_overlay_alpha_min = float(random_overlay.alpha_min)
        self.random_overlay_alpha_max = float(random_overlay.alpha_max)
        self.random_overlay_warmup_steps = int(random_overlay.warmup_steps)
        self.random_overlay_background_path = random_overlay.background_path

        self.box_jitter = 0.05
        self.box_margin_px = 8

        self.views = []
        self.view_modes = {}
        for view, mode in spec.view_modes:
            if mode not in VALID_FOCUS_MODES or mode == "disabled":
                raise ValueError(
                    f"FocusViewTransform: invalid mode {mode!r} for {view!r}"
                )
            self.views.append(str(view))
            self.view_modes[str(view)] = str(mode)

        if not self.views:
            raise ValueError("FocusViewTransform: no enabled views")
        if "external" not in self.views:
            raise ValueError("FocusViewTransform requires 'external'")

        self.enable_wrist_view = "wrist" in self.views
        self.focus_views = [
            view
            for view, mode in self.view_modes.items()
            if mode in FOCUS_OPERATION_MODES
        ]
        if self.focus_views and self.focus_source not in FOCUS_BACKENDS:
            valid = ", ".join(sorted(FOCUS_BACKENDS))
            raise ValueError(
                f"focus_view_transform.source={self.focus_source!r} cannot provide "
                f"focus boxes; expected one of {valid}"
            )

        self.uses_seeker = bool(self.focus_views) and self.focus_source == "seeker"
        self.uses_rvt2_heatmap = (
            bool(self.focus_views) and self.focus_source == "rvt2_heatmap"
        )
        self.uses_oracle = bool(self.focus_views) and self.focus_source == "oracle"

        if self.uses_rvt2_heatmap:
            for view in self.focus_views:
                if view != "external":
                    raise ValueError(
                        "RVT2Heatmap focus currently supports the external view only"
                    )
        if self.uses_oracle:
            for view in self.focus_views:
                if view != "external":
                    raise ValueError("Oracle focus currently supports the external view only")
        self.uses_random_overlay = any(
            mode == "random_overlay" for mode in self.view_modes.values()
        )
        self.uses_masked_overlay = any(
            mode in GUIDED_OVERLAY_MODES for mode in self.view_modes.values()
        )
        self.uses_guided_overlay = (
            self.guided_overlay_prob > 0.0 and self.uses_masked_overlay
        )
        self.uses_random_bg_overlay = (
            self.random_overlay_prob > 0.0 and self.uses_random_overlay
        )
        self.uses_overlay = self.uses_guided_overlay or self.uses_random_bg_overlay

        self.seeker: Optional[Seeker] = None
        if self.uses_seeker:
            if not self.seeker_weights:
                raise ValueError(
                    "FocusViewTransform requires released Seeker weights. "
                    "Set focus_view_transform.source.weights."
                )
            if not os.path.isfile(self.seeker_weights):
                raise FileNotFoundError(
                    f"Seeker checkpoint not found: {self.seeker_weights}"
                )
            seeker_views = list(source.checkpoint_views) or list(self.views)
            self.seeker = Seeker(
                num_robots=spec.num_robots,
                weights=self.seeker_weights,
                views=seeker_views,
                verbose=False,
                strict_weights=self.seeker_strict_weights,
            )
            self.seeker.eval()

        # Seeker keeps the per-robot normalization it was trained with; without a
        # Seeker the composer input is normalized like the rest of the policy.
        if self.seeker is not None:
            self.normalizer = Normalizer()
            self.normalizer.load_state_dict(self.seeker.normalizer.state_dict())
        else:
            self.normalizer = build_normalizer_module(normalizer_kind)

        self.patch_size = int(getattr(self.seeker, "patch_size", 16))
        self.grid_res = self.vit_in // self.patch_size

        self.rvt2_heatmap: Optional[RVT2Heatmap] = None
        if self.uses_rvt2_heatmap:
            self.rvt2_heatmap = RVT2Heatmap(
                checkpoint=self.rvt2_checkpoint,
                vit_in=self.vit_in,
            )
            self.patch_size = int(self.rvt2_heatmap.patch_size)
            self.grid_res = int(self.rvt2_heatmap.grid_res)

        self.cached_views = [
            view for view in self.views if self.view_modes[view] not in NO_FOCUS_MODES
        ]
        self.mask_cached_views = [
            view
            for view in self.cached_views
            if self.view_modes[view] in MASK_CACHE_MODES
        ]
        self.view_to_box_idx = {view: i for i, view in enumerate(self.cached_views)}
        self.view_to_mask_idx = {
            view: i for i, view in enumerate(self.mask_cached_views)
        }

        self.guided_background_overlay = None
        if self.uses_guided_overlay:
            if not self.guided_overlay_background_path:
                raise ValueError(
                    "focus_view_transform.overlay.guided.background_path is required "
                    "when guided overlay probability is positive."
                )
            alpha_mid = (
                self.guided_overlay_alpha_min + self.guided_overlay_alpha_max
            ) / 2.0
            self.guided_background_overlay = BackgroundOverlay(
                {
                    "prob": self.guided_overlay_prob,
                    "alpha": [alpha_mid, alpha_mid],
                    "background_path": self.guided_overlay_background_path,
                },
                cache_res=self.vit_in,
            )

        self.random_background_overlay = None
        if self.uses_random_bg_overlay:
            if not self.random_overlay_background_path:
                raise ValueError(
                    "focus_view_transform.overlay.random.background_path is required "
                    "when random overlay probability is positive."
                )
            self.random_background_overlay = BackgroundOverlay(
                {
                    "prob": self.random_overlay_prob,
                    "alpha": [
                        self.random_overlay_alpha_min,
                        self.random_overlay_alpha_max,
                    ],
                    "background_path": self.random_overlay_background_path,
                },
                cache_res=self.vit_in,
            )

        self.lowres_crop = ResizeCropRandomizer(
            self.random_crop,
            channels=3,
            resize_mode="bilinear" if self.uses_seeker else "area",
        )
        self.buffer_valid: Optional[torch.Tensor] = None
        self.box_buffer: Optional[torch.Tensor] = None
        self.mask_buffer: Optional[torch.Tensor] = None
        self.counter = 0

    def initialize_buffer(self, buffer_size: int, device: torch.device):
        self.buffer_valid = torch.zeros(
            (buffer_size,), dtype=torch.uint8, device=device
        )
        self.box_buffer = torch.zeros(
            (buffer_size, len(self.cached_views), 4), dtype=torch.uint8, device=device
        )
        if self.mask_cached_views:
            self.mask_buffer = torch.zeros(
                (
                    buffer_size,
                    len(self.mask_cached_views),
                    self.grid_res * self.grid_res,
                ),
                dtype=torch.uint8,
                device=device,
            )
        else:
            self.mask_buffer = None

    def set_normalizer(self, normalizer: nn.Module) -> None:
        if self.seeker is None:
            self.normalizer.load_state_dict(normalizer.state_dict())

    def retrieve_from_buffer(
        self, obs_index: torch.Tensor
    ) -> Optional[Dict[str, VisualFocusPrediction]]:
        if self.buffer_valid is None or self.box_buffer is None:
            return None
        obs_index = obs_index.to(torch.int64)
        if (self.buffer_valid[obs_index] == 0).any():
            return None

        out = {}
        for view in self.cached_views:
            box_u8 = self.box_buffer[obs_index, self.view_to_box_idx[view]]
            box_px = (box_u8.float() / 255.0) * float(self.vit_in - 1)
            mask_grid = None
            if view in self.view_to_mask_idx:
                if self.mask_buffer is None:
                    return None
                mask_u8 = self.mask_buffer[obs_index, self.view_to_mask_idx[view]]
                mask_grid = (mask_u8.float() / 255.0).view(
                    -1, 1, self.grid_res, self.grid_res
                )
            out[view] = VisualFocusPrediction(
                box_px=box_px,
                mask_grid=mask_grid,
                source=self._source_for_view(view),
            )
        return out

    def fill_buffer(
        self, *, obs_index: torch.Tensor, payloads: Dict[str, VisualFocusPrediction]
    ):
        if self.buffer_valid is None or self.box_buffer is None:
            return
        obs_index = obs_index.to(torch.int64)
        for view, value in payloads.items():
            box01 = (value.box_px / float(self.vit_in - 1)).clamp(0.0, 1.0)
            self.box_buffer[obs_index, self.view_to_box_idx[view]] = (
                (box01 * 255.0).round().to(torch.uint8)
            )
            if view in self.view_to_mask_idx:
                assert value.mask_grid is not None
                assert self.mask_buffer is not None
                mask_u8 = (value.mask_grid.clamp(0.0, 1.0) * 255.0).round()
                self.mask_buffer[obs_index, self.view_to_mask_idx[view]] = (
                    mask_u8.to(torch.uint8).squeeze(1).flatten(1)
                )
        self.buffer_valid[obs_index] = 1

    @torch.no_grad()
    def infer_all_visual_focus(
        self,
        *,
        images_vit_by_view: Dict[str, torch.Tensor],
        composer_in: dict,
        obs_index: Optional[torch.Tensor],
        oracle_info: Optional[dict] = None,
    ) -> Dict[str, Optional[VisualFocusPrediction]]:
        if self.training and obs_index is not None:
            cached = self.retrieve_from_buffer(obs_index)
            if cached is not None:
                return {
                    view: None
                    if self.view_modes[view] in NO_FOCUS_MODES
                    else cached[view]
                    for view in self.views
                }

        payloads = {}
        out = {}
        for view in self.views:
            mode = self.view_modes[view]
            if mode in NO_FOCUS_MODES:
                out[view] = None
                continue
            if mode in FOCUS_OPERATION_MODES:
                prediction = self.predict_visual_focus(
                    view=view,
                    image=images_vit_by_view[view],
                    composer_in=composer_in,
                    oracle_info=oracle_info,
                )
                payloads[view] = prediction
                out[view] = prediction
                continue

        if self.training and obs_index is not None and self.buffer_valid is not None:
            self.fill_buffer(obs_index=obs_index, payloads=payloads)
        return out

    def predict_visual_focus(
        self,
        *,
        view: str,
        image: torch.Tensor,
        composer_in: dict,
        oracle_info: Optional[dict],
    ) -> VisualFocusPrediction:
        if self.focus_source == "seeker":
            if self.seeker is None:
                raise RuntimeError("Seeker backend is not initialized")
            default_box = torch.tensor(
                [[0.0, 0.0, float(self.vit_in - 1), float(self.vit_in - 1)]],
                device=image.device,
            ).expand(image.shape[0], -1)
            mask_grid = self.seeker(
                image=image,
                view=view,
                composer_in=composer_in,
            ).final.mask
            box_px = grid_mask_to_pixel_box(mask_grid.squeeze(1), default_box)
            mask_grid = mask_grid / (mask_grid.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            return VisualFocusPrediction(
                box_px=box_px,
                mask_grid=mask_grid.clamp(0.0, 1.0),
                source="seeker",
            )

        if self.focus_source == "rvt2_heatmap":
            if self.rvt2_heatmap is None:
                raise RuntimeError("RVT2Heatmap backend is not initialized")
            return self.rvt2_heatmap.predict_visual_focus(
                image=image,
                composer_in=composer_in,
                view_name=view,
            )

        if self.focus_source == "oracle":
            return self.oracle_visual_focus(
                view=view,
                oracle_info=oracle_info,
                batch_size=image.shape[0],
                device=image.device,
            )

        raise RuntimeError(f"Unhandled focus source: {self.focus_source}")

    def process_view(
        self,
        *,
        view: str,
        image_vit: torch.Tensor,
        visual_focus: Optional[VisualFocusPrediction],
    ) -> Dict[str, Optional[torch.Tensor]]:
        mode = self.view_modes[view]
        # The full image, in the canonical continuous [0, 0, W, H] box convention
        # (this box is only ever reported as metadata here, never fed into the
        # inclusive-pixel-box math Seeker's iterative refinement uses).
        default_box_px = torch.tensor(
            [[0.0, 0.0, float(self.vit_in), float(self.vit_in)]],
            device=image_vit.device,
        ).expand(image_vit.shape[0], -1)

        if mode == "pass_through":
            return {
                "image": self.lowres_crop(image_vit),
                "box_px": default_box_px,
                "visual_focus": None,
            }
        if mode == "random_overlay":
            image_aug = self.overlay(image_vit, None)
            return {
                "image": self.lowres_crop(image_aug),
                "box_px": default_box_px,
                "visual_focus": None,
            }

        assert visual_focus is not None
        operation = MODE_OPERATION[mode]
        if operation == "focus_condition":
            return self.focus_condition(image_vit, visual_focus)
        if operation == "focus_crop":
            return self.focus_crop(image_vit, visual_focus)
        if operation == "focus_mask_crop":
            return self.focus_mask_crop(image_vit, visual_focus)
        if operation == "focus_mask":
            return self.focus_mask(image_vit, visual_focus, default_box_px)

        raise RuntimeError(f"Unhandled focus operation {operation!r} for mode {mode!r}")

    def focus_condition(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "image": self.lowres_crop(image_vit),
            "box_px": visual_focus.box_px,
            "visual_focus": visual_focus,
        }

    def focus_crop(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
    ) -> Dict[str, Optional[torch.Tensor]]:
        crop, _, box_px = self.box_crop(image_vit, None, visual_focus.box_px)
        return {"image": crop, "box_px": box_px, "visual_focus": visual_focus}

    def focus_mask(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
        default_box_px: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        mask_px = self.upsample_mask(visual_focus.mask_grid)
        image_aug = self.overlay(image_vit, mask_px)
        return {
            "image": self.lowres_crop(image_aug),
            "box_px": default_box_px,
            "visual_focus": visual_focus,
        }

    def focus_mask_crop(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
    ) -> Dict[str, Optional[torch.Tensor]]:
        mask_px = self.upsample_mask(visual_focus.mask_grid)
        crop, mask_px, box_px = self.box_crop(image_vit, mask_px, visual_focus.box_px)
        crop = self.overlay(crop, mask_px)
        return {"image": crop, "box_px": box_px, "visual_focus": visual_focus}

    def upsample_mask(self, mask_grid: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            mask_grid,
            size=(self.vit_in, self.vit_in),
            mode="bilinear",
            align_corners=False,
        )

    def box_crop(
        self,
        image: torch.Tensor,
        mask_px: Optional[torch.Tensor],
        box_px: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        jitter = self.box_jitter if self.training else 0.0
        return crop_with_box(
            image=image,
            box=box_px,
            mask=mask_px,
            output_size=(self.random_crop.output_res, self.random_crop.output_res),
            box_jitter=jitter,
            margin=self.box_margin_px,
        )

    def _source_for_view(self, view: str) -> str:
        _ = view
        return self.focus_source

    @staticmethod
    def _oracle_array(oracle_info: dict, names: list[str]) -> Optional[torch.Tensor]:
        for name in names:
            value = oracle_info.get(name)
            if value is not None:
                return value if torch.is_tensor(value) else torch.as_tensor(value)
        return None

    def oracle_visual_focus(
        self,
        *,
        view: str,
        oracle_info: Optional[dict],
        batch_size: int,
        device: torch.device,
    ) -> VisualFocusPrediction:
        if oracle_info is None:
            raise RuntimeError(
                f"{self.view_modes[view]} requires oracle_info: training reads it from "
                "the dataset's oracle cache, rollouts from the simulator."
            )

        mask = self._oracle_array(
            oracle_info,
            [
                f"target_patch_mask_{view}",
                f"oracle_target_patch_mask_{view}",
            ],
        )
        if mask is None:
            available = ", ".join(sorted(str(k) for k in oracle_info.keys()))
            raise RuntimeError(
                f"Missing oracle target patch mask for view {view!r}. "
                f"Available oracle keys: {available}"
            )

        mask = mask.to(device=device)
        if mask.dim() == 4:
            mask = mask.reshape(-1, *mask.shape[-2:])
        if mask.dim() != 3:
            raise RuntimeError(
                "Expected oracle target patch mask [N,S,S] or [B,T,S,S], "
                f"got {tuple(mask.shape)}"
            )
        if int(mask.shape[0]) != int(batch_size):
            raise RuntimeError(
                f"Oracle target patch mask frame count {mask.shape[0]} does not "
                f"match flattened image batch {batch_size}"
            )

        box_px = CommonOracle.target_patch_tight_boxes(mask, image_size=self.vit_in)
        box_px = CommonOracle.replace_invalid_boxes(box_px, image_size=self.vit_in)
        return VisualFocusPrediction(
            box_px=box_px.to(device=device),
            mask_grid=mask.float().unsqueeze(1),
            source=self._source_for_view(view),
            metadata={
                "view": view,
                "mode": self.view_modes[view],
            },
        )

    def overlay(
        self, image: torch.Tensor, mask_px: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not self.training:
            return image
        if mask_px is None:
            prob = self.random_overlay_prob
            if self.random_overlay_warmup_steps > 0:
                prob *= min(
                    1.0,
                    float(self.counter) / float(self.random_overlay_warmup_steps),
                )
            if self.random_background_overlay is None or prob <= 0.0:
                return image
            return self.random_background_overlay(image, prob=prob)

        prob = self.guided_overlay_prob
        if self.guided_overlay_warmup_steps > 0:
            prob *= min(
                1.0,
                float(self.counter) / float(self.guided_overlay_warmup_steps),
            )
        if self.guided_background_overlay is None or prob <= 0.0:
            return image

        mask = mask_px
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear")
        if self.guided_overlay_noise_std > 0:
            mask = (
                mask + torch.randn_like(mask) * self.guided_overlay_noise_std
            ).clamp(0.0, 1.0)
        mask = mask.clamp(self.guided_overlay_alpha_min, self.guided_overlay_alpha_max)
        return self.guided_background_overlay(
            image, alpha=mask, prob=prob, replacement=True
        )

    def forward(
        self,
        view_imgs: Dict[str, torch.Tensor],
        composer_in: dict,
        obs_index: Optional[torch.Tensor] = None,
        oracle_info: Optional[dict] = None,
    ) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        assert set(view_imgs.keys()) == set(self.views), (
            f"Expected views {self.views}, got {list(view_imgs.keys())}"
        )
        # Focus runs over frames flattened across time, so the cache index does too.
        if obs_index is not None:
            obs_index = obs_index.reshape(-1)
        prepared_images = {}
        for view in self.views:
            image = view_imgs[view]
            prepare_at_vit_resolution = (
                self.uses_seeker or self.view_modes[view] in FOCUS_OPERATION_MODES
            )
            if prepare_at_vit_resolution and image.shape[-2:] != (
                self.vit_in,
                self.vit_in,
            ):
                image = F.interpolate(
                    image,
                    size=(self.vit_in, self.vit_in),
                    mode="bilinear",
                    align_corners=False,
                )
            prepared_images[view] = image

        focus_by_view = self.infer_all_visual_focus(
            images_vit_by_view=prepared_images,
            composer_in=composer_in,
            obs_index=obs_index,
            oracle_info=oracle_info,
        )
        out = {
            view: self.process_view(
                view=view,
                image_vit=prepared_images[view],
                visual_focus=focus_by_view[view],
            )
            for view in self.views
        }
        if self.training:
            self.counter += 1
        return out

    def get_runtime_config(self) -> dict:
        view_names = {"external": "External", "wrist": "Wrist"}
        mode_names = {
            "pass_through": "Pass Through",
            "focus_condition": "Focus Condition",
            "focus_crop": "Focus Crop",
            "lowres_crop_only": "Low-Res Crop",
            "focus_mask": "Focus Mask",
            "focus_mask_crop": "Focus Mask Crop",
            "random_overlay": "Random Overlay",
        }
        crop_modes = FOCUS_CROP_MODES | FOCUS_MASK_CROP_MODES
        focus_cfg = {}
        for view in self.views:
            mode = self.view_modes[view]
            label = view_names.get(view, view.replace("_", " ").title())
            is_box_crop = mode in crop_modes
            if is_box_crop:
                # A jittered box (cropped from the vit_in image) is resized
                # directly to output_res; the random-crop resize/crop pipeline
                # below never runs for this view, so there is no fixed input_res.
                focus_cfg[label] = {
                    "Mode": mode_names.get(mode, mode.replace("_", " ").title()),
                    "Input Res": "crop",
                    "Output Res": self.random_crop.output_res,
                }
            else:
                random_crop_status = (
                    "train: random, eval: center" if self.random_crop.enabled else "Disabled (no margin)"
                )
                focus_cfg[label] = {
                    "Mode": mode_names.get(mode, mode.replace("_", " ").title()),
                    "Input Res": self.random_crop.input_res,
                    "Output Res": self.random_crop.output_res,
                    "Random Crop": random_crop_status,
                }

        overlay_lines = []
        if self.uses_guided_overlay:
            overlay_lines.append(
                f"guided {self.guided_overlay_prob:.2f} prob, "
                f"warmup {self.guided_overlay_warmup_steps}, "
                f"alpha "
                f"{self.guided_overlay_alpha_min:.2f}-"
                f"{self.guided_overlay_alpha_max:.2f}"
            )
        if self.uses_random_bg_overlay:
            overlay_lines.append(
                f"random {self.random_overlay_prob:.2f} prob, "
                f"warmup {self.random_overlay_warmup_steps}, "
                f"alpha "
                f"{self.random_overlay_alpha_min:.2f}-"
                f"{self.random_overlay_alpha_max:.2f}"
            )
        focus_cfg["Overlay"] = "; ".join(overlay_lines) if overlay_lines else "Disabled"

        if any(mode in crop_modes for mode in self.view_modes.values()):
            crop = f"box jitter {100.0 * self.box_jitter:.1f}%"
            if self.box_margin_px:
                crop += f", margin {self.box_margin_px}px"
            focus_cfg["Focus Box Crop"] = crop
        else:
            focus_cfg["Focus Box Crop"] = "Disabled"

        focus_cfg["Source"] = self.focus_source.replace("_", " ").title()
        if self.uses_seeker:
            # A runtime report is requested only after Seeker weights are loaded.
            # loaded without error in __init__; no need for a separate block.
            focus_cfg["Seeker Weights"] = f"loaded ({self.seeker_weights})"
        config = {"Focus Transform": focus_cfg}

        if self.uses_rvt2_heatmap:
            config["RVT2 Heatmap"] = {
                "Status": "Enabled",
                "Checkpoint": self.rvt2_checkpoint,
                "Zoom": (
                    None
                    if self.rvt2_heatmap is None
                    else f"{self.rvt2_heatmap.zoom:.1f}x"
                ),
            }
        elif self.uses_oracle:
            config["Oracle Focus"] = {
                "Status": "Enabled",
            }
        elif not self.uses_seeker:
            config["Visual Focus"] = "Disabled"
        return config
