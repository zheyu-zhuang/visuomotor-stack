"""Policy observation encoder using an external visual-focus model."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from visuomotor.config.schema import FocusConditionedEncoderSpec
from visuomotor.geometry.roi import normalize_box
from visuomotor.perception.backbone.resnet.build import (
    build_resnet18_stages,
    get_resnet18_stage_modules,
)
from visuomotor.perception.common import prediction as Prediction
from visuomotor.perception.common.inputs import EncoderInputs, ObsInputProcessor
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.perception.focus import coordinator as FocusView


class FocusConditionedObsEncoder(ObsInputProcessor):
    """Encode observations with focus-guided crops and ResNet backbones."""

    def __init__(
        self,
        spec: FocusConditionedEncoderSpec,
        *,
        normalizer_kind: str = "multi_robot_linear",
    ):
        super().__init__(num_robots=spec.num_robots)

        self.spec = spec
        self.focus_view_transform = FocusView.FocusViewTransform(
            spec, normalizer_kind=normalizer_kind, verbose=False
        )
        self.enable_wrist_view = self.focus_view_transform.enable_wrist_view

        # Per-view visual encoders
        feat_dim = int(spec.feature_dim)
        out_res = self.focus_view_transform.random_crop.output_res
        cnn_input_shape = (3, out_res, out_res)
        self.external_cnn = ResNet18Encoder(
            input_shape=cnn_input_shape,
            out_dim=feat_dim,
            pretrained_imagenet=spec.resnet_pretrained_imagenet,
            norm=spec.norm,
        )
        out_feature_dim = feat_dim + 11
        self.wrist_cnn: Optional[nn.Module] = None
        if self.enable_wrist_view:
            self.wrist_cnn = ResNet18Encoder(
                input_shape=cnn_input_shape,
                out_dim=feat_dim,
                pretrained_imagenet=spec.resnet_pretrained_imagenet,
                norm=spec.norm,
            )
            out_feature_dim += feat_dim

        self.output_dim = int(out_feature_dim)

        # Box FiLM conditioning using normalized (cx, cy, scale)
        self.box_film = nn.Linear(3, feat_dim * 2)
        # Keep preprocessing aligned with the focus transform/Seeker Input Resn.
        self.input_res = self.focus_view_transform.vit_in

    def initialize_internal(self, dataset_size: int, device: torch.device):
        """Initialize the focus cache."""
        self.focus_view_transform.initialize_buffer(dataset_size, device)

    def forward(
        self,
        obs,
        canonical_obs=None,
        task_context=None,
        obs_index=None,
        oracle_info=None,
    ):
        """Run focus-conditioned observation encoding and ResNet encoding."""
        enc_in = self.process_obs(obs, canonical_obs, task_context)

        views = {"external": enc_in.external}
        if self.enable_wrist_view:
            views["wrist"] = enc_in.wrist

        processed_views = self.focus_view_transform(
            views,
            composer_in=enc_in.composer_in,
            obs_index=obs_index,
            oracle_info=oracle_info,
        )
        focus_records = self._focus_records(processed_views, enc_in.T)

        external_ret = processed_views["external"]
        external_image = external_ret["image"]
        external_box_px = external_ret["box_px"]

        wrist_image, wrist_box_px = None, None
        if self.enable_wrist_view:
            wrist_ret = processed_views["wrist"]
            wrist_image, wrist_box_px = wrist_ret["image"], wrist_ret["box_px"]

        feats = [enc_in.proprio]
        external_feature = self.encode_external(
            external_image,
            external_box_px,
        )
        feats.append(external_feature)
        streams = {"rgb_external": external_feature}
        if self.enable_wrist_view:
            wrist_feature = self.encode_wrist(wrist_image, wrist_box_px)
            feats.append(wrist_feature)
            streams["rgb_wrist"] = wrist_feature
        feat = torch.cat(feats, dim=-1)

        # Views are encoded flattened over time; the policy conditions per sample.
        prepared_inputs = {
            "rgb_external": external_image.reshape(-1, enc_in.T, *external_image.shape[1:])
        }
        if wrist_image is not None:
            prepared_inputs["rgb_wrist"] = wrist_image.reshape(
                -1, enc_in.T, *wrist_image.shape[1:]
            )
        return EncoderOutput(
            features=feat.reshape(-1, enc_in.T, feat.shape[-1]),
            streams={
                key: value.reshape(-1, enc_in.T, value.shape[-1])
                for key, value in streams.items()
            },
            prepared_inputs=prepared_inputs,
            focus_records=tuple(focus_records),
            metadata={"encoder": self.spec.name},
        )

    def _focus_records(self, processed_views, T: int):
        """Return latest-step focus records without retaining model side state."""
        B = next(iter(processed_views.values()))["box_px"].shape[0] // int(T)
        records: list[Prediction.VisualFocusRecord] = []
        for view, ret in processed_views.items():
            box_px = ret.get("box_px")
            if box_px is None:
                continue

            mode = self.focus_view_transform.view_modes.get(view, "disabled")
            if mode in ("pass_through", "random_overlay"):
                continue
            if mode in FocusView.VISUAL_FOCUS_RECORD_MODES:
                source = (
                    ret["visual_focus"].source if ret.get("visual_focus") else "seeker"
                )
            else:
                source = mode

            last_box = box_px.view(B, int(T), 4)[:, -1].detach()
            prediction = Prediction.VisualFocusPrediction(
                box_px=last_box, source=source
            )
            records.append(
                Prediction.VisualFocusRecord(
                    source=source,
                    view=view,
                    timestep=int(T) - 1,
                    prediction=prediction,
                    image_size=(
                        int(self.focus_view_transform.vit_in),
                        int(self.focus_view_transform.vit_in),
                    ),
                )
            )
        return records

    def set_normalizer(self, normalizer) -> None:
        """Offer the run normalizer to the focus transform, which keeps a Seeker's own."""
        self.focus_view_transform.set_normalizer(normalizer)

    def encode_external(self, image, box_px):
        """Encode the external image and optionally FiLM-condition on the Seeker box."""
        mode = self.focus_view_transform.view_modes["external"]
        feat = self.external_cnn(image)
        if mode in FocusView.BOX_FILM_MODES:
            feat = self.apply_box_film(feat, box_px)
        return feat

    def apply_box_film(self, feat, box_px):
        """Condition visual features on normalized focus box geometry."""
        normed_box = normalize_box(box_px, self.focus_view_transform.vit_in)
        gamma, beta = self.box_film(normed_box).chunk(2, dim=-1)
        return feat * (1 + gamma) + beta

    def encode_wrist(self, image, box_px):
        """Encode eye-in-hand image when enabled."""
        if image is None:
            return None
        feat = self.wrist_cnn(image)
        mode = self.focus_view_transform.view_modes["wrist"]
        if mode in FocusView.BOX_FILM_MODES:
            feat = self.apply_box_film(feat, box_px)
        return feat

    def process_obs(
        self,
        obs: Dict[str, torch.Tensor],
        canonical_obs: Optional[Dict[str, torch.Tensor]] = None,
        task_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> EncoderInputs:
        """Format normalized observations for the focus view transform."""
        enc_in = self.obs_to_input(
            obs,
            canonical_obs,
            task_context,
            self.focus_view_transform.normalizer,
            resize=False,
        )
        if self.focus_view_transform.view_modes["external"] == "lowres_crop_only":
            enc_in.external = self.degrade_obs_resolution(enc_in.external)
        if (
            self.enable_wrist_view
            and self.focus_view_transform.view_modes["wrist"]
            == "lowres_crop_only"
        ):
            enc_in.wrist = self.degrade_obs_resolution(enc_in.wrist)
        return enc_in

    def degrade_obs_resolution(self, image: torch.Tensor) -> torch.Tensor:
        """Remove image detail by downsampling to the crop's input_res, then restoring vit_in."""
        low_res = self.focus_view_transform.random_crop.input_res
        vit_in = self.focus_view_transform.vit_in
        image = F.interpolate(
            image,
            size=(low_res, low_res),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(
            image,
            size=(vit_in, vit_in),
            mode="bilinear",
            align_corners=False,
        )

    def get_runtime_config(self) -> dict:
        """Expose focus model config for logging/debugging."""
        return self.focus_view_transform.get_runtime_config()


class ResNet18Encoder(nn.Module):
    """Lightweight ResNet-18 feature encoder."""

    def __init__(
        self,
        *,
        input_shape,
        out_dim: int,
        pretrained_imagenet: bool = True,
        norm: str = "groupnorm",
    ):
        super().__init__()

        backbone = build_resnet18_stages(pretrained_imagenet=pretrained_imagenet, norm=norm)

        # Keep convolutional trunk only; pooling/head are defined below.
        self.trunk = nn.Sequential(*[module for _, module in get_resnet18_stage_modules(backbone)])

        with torch.no_grad():
            feat = self.trunk(torch.zeros(1, *input_shape))
            c = feat.shape[1]
            assert c == 512

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c, out_dim)

    def forward(self, x):
        x = self.trunk(x)
        x = self.pool(x)
        return self.head(x.view(x.shape[0], -1))
