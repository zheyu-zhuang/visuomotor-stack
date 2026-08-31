import types

import pytest
import torch

from visuomotor.config.schema import (
    AttentionPriorSpec,
    FocusConditionedEncoderSpec,
    FocusSourceSpec,
    RandomCropSpec,
)
from visuomotor.data.core.normalization import Normalizer
from visuomotor.data.core.observations import (
    canonicalize_rgb_from_uint8,
    canonicalize_voxel_from_uint8,
)
from visuomotor.geometry.grid import FeatureGridGeometry
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.perception.encoder.focus_pool import FocusRefineEncoder
from visuomotor.perception.focus.coordinator import FocusViewTransform
from visuomotor.perception.focus.refine.attention_prior import (
    FocusAttentionPrior,
    gaussian_feature_target,
)
from visuomotor.perception.focus.refine.planar import FocusRefine2d
from visuomotor.perception.focus.refine.position_encoding import (
    get_freq_position_embedding_3d,
    get_normalized_grid_coordinates_3d,
)
from visuomotor.perception.focus.refine.stage_pooled_resnet import StagePooledResNet2d
from visuomotor.perception.focus.refine.volumetric import FocusRefine3d


def _model_rgb(image):
    return Normalizer.normalize_canonical_rgb(canonicalize_rgb_from_uint8(image))


def _model_voxel(voxel):
    return Normalizer.normalize_voxel(canonicalize_voxel_from_uint8(voxel))


def test_gaussian_feature_target_is_rank_agnostic():
    geometry_2d = FeatureGridGeometry.from_stride((16, 16), (4, 4), stride=4)
    assert geometry_2d.shape == (4, 4)
    target = gaussian_feature_target(torch.zeros(3, 2), geometry_2d, sigma_cells=1.0)
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(3))

    geometry_3d = FeatureGridGeometry.from_stride((16, 16, 16), (4, 4, 4), stride=4)
    assert geometry_3d.shape == (4, 4, 4)
    target_3d = gaussian_feature_target(torch.zeros(3, 3), geometry_3d, sigma_cells=1.0)
    torch.testing.assert_close(target_3d.sum(dim=-1), torch.ones(3))


def test_focus_attention_prior_is_bootstrapping_defaults_to_always_active():
    prior = FocusAttentionPrior({"enabled": True})
    assert prior.is_bootstrapping(None) is True
    assert prior.is_bootstrapping(0) is True
    assert prior.is_bootstrapping(10_000) is True


def test_focus_attention_prior_is_bootstrapping_decays_after_bootstrap_steps():
    prior = FocusAttentionPrior({"enabled": True, "bootstrap_steps": 100})
    assert prior.is_bootstrapping(0) is True
    assert prior.is_bootstrapping(99) is True
    assert prior.is_bootstrapping(100) is False
    assert prior.is_bootstrapping(None) is False


def test_attention_prior_spec_round_trips_through_focus_refine_encoder_spec():
    from visuomotor.config.schema import FocusRefineEncoderSpec, from_dict, to_dict

    spec = FocusRefineEncoderSpec(
        name="rgb_focus_pool2d",
        architecture="focus_pool2d",
        rgb_keys=("rgb_external",),
        proprio_fields=("gripper_qpos",),
        gripper_key="gripper_qpos",
        input_res=16,
        attention_prior=AttentionPriorSpec(enabled=True, weight=0.1, sigma_cells=1.5),
    )
    restored = from_dict(to_dict(spec))
    assert restored == spec


def test_focus_refine_2d_ctx_dim_is_head_dim_not_heads_times_head_dim():
    torch.manual_seed(0)
    module = FocusRefine2d(in_channels=6, grid_h=4, grid_w=5, iters=2, heads=3, head_dim=8)
    feat = torch.randn(2, 6, 4, 5, requires_grad=True)
    out = module(feat, {"gripper_opening": torch.randn(2, 1)})

    assert out.ctx.shape == (2, 8)
    assert out.pool_map.shape == (2, 3, 4, 5)
    assert out.keypoints.shape == (2, 3, 2)

    out.ctx.sum().backward()
    assert feat.grad is not None


def test_focus_refine_3d_matches_planar_query_position_and_context_contract():
    torch.manual_seed(0)
    module = FocusRefine3d(
        in_channels=6,
        grid_d=2,
        grid_h=3,
        grid_w=4,
        iters=2,
        heads=3,
        head_dim=8,
        query_cond=("eef", "gripper"),
    )
    feat = torch.randn(2, 6, 2, 3, 4)
    composer_in = {
        "eef_pos": torch.randn(2, 3),
        "gripper_opening": torch.randn(2, 1),
    }

    context, attention = module(feat, composer_in, return_attn=True)
    assert context.shape == (2, 8)
    assert attention.shape == (2, 3, 2, 3, 4)
    assert module.query_builder.query_cond == ("eef", "gripper")
    assert module.pos_enc.shape == (24, 8)
    assert module.to_key.in_features == 14
    feature_tokens = feat.flatten(2).transpose(1, 2)
    position_tokens = module.pos_enc.unsqueeze(0).expand(feat.shape[0], -1, -1)
    value_tokens = torch.cat((feature_tokens, position_tokens), dim=-1)
    values = module.to_value(value_tokens).view(2, 24, 3, 8).permute(0, 2, 1, 3)
    expected = torch.einsum("bhn,bhnd->bhd", attention.flatten(2), values).mean(1)
    torch.testing.assert_close(context, expected)


def test_3d_frequency_positions_follow_planar_coordinate_ordering():
    coordinates = get_normalized_grid_coordinates_3d(2, 3, 4)
    embedding = get_freq_position_embedding_3d(2, 3, 4, 8)

    assert coordinates.shape == (24, 3)
    assert coordinates[0].tolist() == pytest.approx([-1.0, -1.0, -1.0])
    assert coordinates[-1].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert embedding.shape == (24, 8)


def test_stage_pooled_resnet_2d_truncates_before_layer3():
    torch.manual_seed(0)
    module = StagePooledResNet2d(
        input_res=32,
        feat_dim=16,
        pretrained_imagenet=False,
        pooling_stage="l2",
        iters=1,
        heads=2,
        head_dim=8,
    )
    # layer2 output channels are 128 on a standard resnet18.
    assert module.pool.query_builder.dim == 16
    out, pool_map, keypoints = module(
        torch.randn(2, 3, 32, 32),
        composer_in={"gripper_opening": torch.randn(2, 1)},
        return_pool_map=True,
    )
    assert out.shape == (2, 16)
    assert pool_map.shape[0] == 2
    assert keypoints.shape == (2, 2, 2)


def test_focus_refine_encoder_accepts_observation_contract_fields_only():
    torch.manual_seed(1)
    encoder = FocusRefineEncoder(
        spatial_rank=2,
        feature_dim=8,
        input_res=8,
        rgb_keys=("camera",),
        gripper_key="gripper",
        num_heads=2,
        num_iterations=2,
        pretrained_imagenet=False,
    ).eval()
    observations = {
        "camera": _model_rgb(torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)),
        "gripper": torch.randn(2, 2, 1),
    }

    output = encoder(observations)

    assert isinstance(output, EncoderOutput)
    assert output.features.shape == (2, 2, encoder.output_dim)
    assert output.metadata["uses_task_embedding"] is False
    assert output.metadata["uses_robot_id"] is False


def test_focus_refine_encoder_rgb_random_crop_resizes_crops_and_settles_in_eval():
    torch.manual_seed(2)
    encoder = FocusRefineEncoder(
        spatial_rank=2,
        feature_dim=8,
        input_res=32,
        rgb_keys=("camera",),
        gripper_key="gripper",
        num_heads=2,
        num_iterations=2,
        pretrained_imagenet=False,
        random_crop=RandomCropSpec(input_res=16, output_res=12),
    )
    observations = {
        "camera": _model_rgb(torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8)),
        "gripper": torch.randn(2, 2, 1),
    }

    output = encoder.train()(observations)
    assert output.features.shape == (2, 2, encoder.output_dim)

    encoder.eval()
    torch.testing.assert_close(
        encoder(observations).features, encoder(observations).features
    )


def test_focus_refine_encoder_rejects_random_crop_for_voxel():
    with pytest.raises(ValueError, match="planar"):
        FocusRefineEncoder(
            spatial_rank=3,
            feature_dim=8,
            input_res=16,
            voxel_key="voxel",
            gripper_key="gripper",
            random_crop=RandomCropSpec(input_res=16, output_res=12),
        )


def test_focus_refine_encoder_voxel_forward_shape():
    torch.manual_seed(4)
    encoder = FocusRefineEncoder(
        spatial_rank=3,
        feature_dim=32,
        input_res=16,
        input_channels=4,
        voxel_key="voxel",
        gripper_key="gripper",
        num_heads=2,
        num_iterations=2,
        pool_stage=2,
    ).eval()
    occupancy = torch.randint(0, 2, (2, 2, 1, 16, 16, 16), dtype=torch.uint8)
    colour = torch.randint(0, 256, (2, 2, 3, 16, 16, 16), dtype=torch.uint8)
    observations = {
        "voxel": _model_voxel(torch.cat((occupancy, colour), dim=2)),
        "gripper": torch.randn(2, 2, 2),
    }
    output = encoder(observations)
    assert output.features.shape == (2, 2, encoder.output_dim)


def test_focus_refine_encoder_reports_random_crop_status():
    cropped = FocusRefineEncoder(
        spatial_rank=2,
        input_res=84,
        rgb_keys=("camera",),
        gripper_key="gripper",
        pretrained_imagenet=False,
        random_crop=RandomCropSpec(84, 76),
    )
    assert "76" in cropped.get_runtime_config()["FocusRefine Encoder"]["Random Crop"]

    uncropped = FocusRefineEncoder(
        spatial_rank=2,
        input_res=84,
        rgb_keys=("camera",),
        gripper_key="gripper",
        pretrained_imagenet=False,
    )
    assert uncropped.get_runtime_config()["FocusRefine Encoder"]["Random Crop"] == "Disabled"


def _center_projecting_camera_matrix(batch: int, raw_res: int) -> torch.Tensor:
    """A synthetic world->pixel matrix mapping world (0,0,z) to the image center."""
    matrix = torch.zeros(batch, 4, 4)
    matrix[:, 0, 0] = 1.0
    matrix[:, 0, 3] = raw_res / 2
    matrix[:, 1, 1] = 1.0
    matrix[:, 1, 3] = raw_res / 2
    matrix[:, 2, 3] = 1.0
    return matrix


def _rgb_attention_prior_encoder(**overrides) -> FocusRefineEncoder:
    kwargs = dict(
        spatial_rank=2,
        feature_dim=8,
        input_res=32,
        rgb_keys=("camera",),
        gripper_key="gripper",
        num_heads=2,
        num_iterations=2,
        pretrained_imagenet=False,
    )
    kwargs.update(overrides)
    return FocusRefineEncoder(**kwargs)


def test_focus_refine_encoder_rgb_attention_prior_end_to_end():
    torch.manual_seed(5)
    raw_res = 16
    encoder = _rgb_attention_prior_encoder(
        attention_prior=AttentionPriorSpec(enabled=True, weight=0.5, sigma_cells=1.0, bootstrap_steps=2),
        attention_prior_view="camera",
        attention_prior_raw_res=raw_res,
    ).eval()

    batch = 3
    observations = {
        "camera": _model_rgb(torch.randint(0, 256, (batch, 3, 32, 32), dtype=torch.uint8)),
        "gripper": torch.randn(batch, 1),
    }
    oracle_info = {"camera_matrix_camera": _center_projecting_camera_matrix(batch, raw_res)}
    focus_target = {"pos": torch.zeros(batch, 3), "valid": torch.ones(batch, dtype=torch.bool)}

    out = encoder(observations, oracle_info=oracle_info, focus_target=focus_target, global_step=0)
    assert torch.isfinite(out.auxiliary_losses["attention_prior"])
    assert out.auxiliary_losses["attention_prior"].item() > 0.0

    off = encoder(observations, oracle_info=oracle_info, focus_target=focus_target, global_step=5)
    assert off.auxiliary_losses["attention_prior"].item() == 0.0

    no_target = encoder(observations)
    assert not no_target.auxiliary_losses


def test_focus_refine_encoder_rgb_attention_prior_disabled_by_default():
    encoder = _rgb_attention_prior_encoder().eval()
    observations = {
        "camera": _model_rgb(torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)),
        "gripper": torch.randn(2, 1),
    }
    assert not encoder(observations).auxiliary_losses


def test_focus_refine_encoder_rgb_attention_prior_requires_oracle_info():
    encoder = _rgb_attention_prior_encoder(
        attention_prior=AttentionPriorSpec(enabled=True, weight=0.5, sigma_cells=1.0),
        attention_prior_view="camera",
        attention_prior_raw_res=16,
    ).eval()
    observations = {
        "camera": _model_rgb(torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)),
        "gripper": torch.randn(2, 1),
    }
    focus_target = {"pos": torch.zeros(2, 3), "valid": torch.ones(2, dtype=torch.bool)}
    with pytest.raises(ValueError, match="oracle_info"):
        encoder(observations, focus_target=focus_target)


def test_focus_refine_encoder_rgb_attention_prior_requires_view_metadata():
    with pytest.raises(ValueError, match="attention_prior_view"):
        _rgb_attention_prior_encoder(
            attention_prior=AttentionPriorSpec(enabled=True, weight=0.5, sigma_cells=1.0),
        )


def _voxel_attention_prior_encoder(**overrides) -> FocusRefineEncoder:
    kwargs = dict(
        spatial_rank=3,
        feature_dim=32,
        input_res=16,
        input_channels=4,
        voxel_key="voxel",
        gripper_key="gripper",
        num_heads=2,
        num_iterations=2,
        pool_stage=2,
        workspace_min=(-0.3, -0.3, 0.8),
        ws_size=0.6,
    )
    kwargs.update(overrides)
    return FocusRefineEncoder(**kwargs)


def test_focus_refine_encoder_voxel_attention_prior_end_to_end():
    torch.manual_seed(6)
    encoder = _voxel_attention_prior_encoder(
        attention_prior=AttentionPriorSpec(enabled=True, weight=0.5, sigma_cells=1.0, bootstrap_steps=2),
    ).eval()

    batch = 3
    occupancy = torch.randint(0, 2, (batch, 1, 16, 16, 16), dtype=torch.uint8)
    colour = torch.randint(0, 256, (batch, 3, 16, 16, 16), dtype=torch.uint8)
    observations = {
        "voxel": _model_voxel(torch.cat((occupancy, colour), dim=1)),
        "gripper": torch.randn(batch, 2),
    }
    focus_target = {"pos": torch.full((batch, 3), 0.0), "valid": torch.ones(batch, dtype=torch.bool)}
    focus_target["pos"][:, 2] = 1.1  # inside the workspace's z range [0.8, 1.4]

    out = encoder(observations, focus_target=focus_target, global_step=0)
    assert torch.isfinite(out.auxiliary_losses["attention_prior"])
    assert out.auxiliary_losses["attention_prior"].item() > 0.0

    off = encoder(observations, focus_target=focus_target, global_step=5)
    assert off.auxiliary_losses["attention_prior"].item() == 0.0

    no_target = encoder(observations)
    assert not no_target.auxiliary_losses


def test_focus_refine_encoder_voxel_attention_prior_disabled_by_default():
    encoder = _voxel_attention_prior_encoder().eval()
    occupancy = torch.randint(0, 2, (2, 1, 16, 16, 16), dtype=torch.uint8)
    colour = torch.randint(0, 256, (2, 3, 16, 16, 16), dtype=torch.uint8)
    observations = {
        "voxel": _model_voxel(torch.cat((occupancy, colour), dim=1)),
        "gripper": torch.randn(2, 2),
    }
    assert not encoder(observations).auxiliary_losses


def test_focus_refine_encoder_voxel_attention_prior_requires_workspace_geometry():
    with pytest.raises(ValueError, match="workspace_min"):
        FocusRefineEncoder(
            spatial_rank=3,
            feature_dim=8,
            input_res=16,
            voxel_key="voxel",
            gripper_key="gripper",
            attention_prior=AttentionPriorSpec(enabled=True, weight=0.5, sigma_cells=1.0),
        )


def test_random_crop_spec_validates_resolutions():
    with pytest.raises(ValueError):
        RandomCropSpec(input_res=76, output_res=84)
    assert RandomCropSpec(84, 76).enabled
    assert not RandomCropSpec(84, 84).enabled


def _pass_through_focus_transform() -> FocusViewTransform:
    spec = FocusConditionedEncoderSpec(
        name="focus",
        feature_architecture="resnet18",
        source=FocusSourceSpec(),
        view_modes=(("external", "pass_through"), ("wrist", "pass_through")),
        view_augmentations=(("external", "none"), ("wrist", "none")),
        view_keys=(("external", "rgb_external"), ("wrist", "rgb_wrist")),
        proprio_fields=("eef_pos", "eef_rot6d", "gripper_qpos"),
        num_robots=2,
        vit_in=224,
        random_crop=RandomCropSpec(84, 76),
    )
    return FocusViewTransform(spec).eval()


def test_focus_transform_does_not_interpolate_matching_pass_through_input(monkeypatch):
    transform = _pass_through_focus_transform()

    def fail_interpolate(*_args, **_kwargs):
        raise AssertionError("matching 84-resolution pass-through input was interpolated")

    monkeypatch.setattr(
        "visuomotor.perception.common.augmentation.F.interpolate", fail_interpolate
    )
    monkeypatch.setattr(
        "visuomotor.perception.focus.coordinator.F.interpolate",
        fail_interpolate,
    )
    result = transform(
        {
            "external": torch.randn(2, 3, 84, 84),
            "wrist": torch.randn(2, 3, 84, 84),
        },
        composer_in={},
    )
    assert result["external"]["image"].shape == (2, 3, 76, 76)
    assert result["wrist"]["image"].shape == (2, 3, 76, 76)


def test_focus_transform_prepares_only_focus_operated_view_at_224(monkeypatch):
    transform = _pass_through_focus_transform()
    transform.view_modes["external"] = "focus_crop"
    captured = {}

    def capture(self, *, images_vit_by_view, **_kwargs):
        captured.update(
            {view: tuple(image.shape[-2:]) for view, image in images_vit_by_view.items()}
        )
        raise RuntimeError("captured")

    transform.infer_all_visual_focus = types.MethodType(capture, transform)
    with pytest.raises(RuntimeError, match="captured"):
        transform(
            {
                "external": torch.randn(2, 3, 256, 256),
                "wrist": torch.randn(2, 3, 84, 84),
            },
            composer_in={},
        )
    assert captured == {"external": (224, 224), "wrist": (84, 84)}


def test_seeker_prepares_all_views_through_released_bilinear_path():
    transform = _pass_through_focus_transform()
    transform.uses_seeker = True
    transform.lowres_crop.resize_mode = "bilinear"
    images = {
        "external": torch.randn(2, 3, 256, 256),
        "wrist": torch.randn(2, 3, 256, 256),
    }

    result = transform(images, composer_in={})

    for view, image in images.items():
        expected = torch.nn.functional.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        expected = torch.nn.functional.interpolate(
            expected,
            size=(84, 84),
            mode="bilinear",
            align_corners=False,
        )
        expected = expected[..., 4:80, 4:80]
        torch.testing.assert_close(result[view]["image"], expected)
