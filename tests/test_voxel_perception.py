import pytest
import torch

from visuomotor.config.schema import AttentionPriorSpec
from visuomotor.data.core.normalization import Normalizer
from visuomotor.data.core.observations import (
    canonicalize_rgb_from_uint8,
    canonicalize_voxel_from_uint8,
)
from visuomotor.perception.backbone.resnet.rgb import ResNet18Backbone
from visuomotor.perception.backbone.resnet.voxel import VoxelResNetBackbone
from visuomotor.perception.backbone.voxel_simple import VoxelSimpleBackbone
from visuomotor.perception.encoder.voxel import VoxelObservationEncoder
from visuomotor.perception.focus.refine.attention_prior import (
    FocusAttentionPrior,
    gaussian_feature_target,
)
from visuomotor.perception.focus.refine.volumetric import (
    FocusRefine3d,
    FocusVoxelBackbone,
)


def _random_voxel(batch, channels, size, dtype=torch.uint8):
    occupancy = torch.randint(0, 2, (batch, 1, size, size, size), dtype=dtype)
    colour = torch.randint(0, 256, (batch, channels - 1, size, size, size), dtype=dtype)
    return torch.cat((occupancy, colour), dim=1)


def test_voxel_resnet_backbone_output_shape():
    backbone = VoxelResNetBackbone(obs_channel=4, n_out=64)
    out = backbone(torch.rand(2, 4, 64, 64, 64))
    assert out.shape == (2, 64)


def test_voxel_simple_backbone_accepts_configured_voxel_resolution():
    backbone = VoxelSimpleBackbone(obs_channel=4, n_out=64)
    assert backbone(torch.rand(2, 4, 58, 58, 58)).shape == (2, 64, 1, 1, 1)
    assert backbone(torch.rand(2, 4, 64, 64, 64)).shape == (2, 64, 1, 1, 1)


def test_voxel_simple_backbone_matches_source_channel_and_conv_contract():
    backbone = VoxelSimpleBackbone(obs_channel=4)
    convolutions = [
        module for module in backbone.modules() if isinstance(module, torch.nn.Conv3d)
    ]

    assert [(layer.in_channels, layer.out_channels) for layer in convolutions] == [
        (4, 16),
        (16, 32),
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (128, 256),
    ]
    assert all(layer.bias is not None for layer in convolutions)
    assert not any(isinstance(module, torch.nn.GroupNorm) for module in backbone.modules())


def test_focus_refine_3d_output_shape_and_attention_grid():
    refine = FocusRefine3d(
        in_channels=16,
        grid_d=4,
        grid_h=4,
        grid_w=4,
        iters=3,
        heads=4,
        head_dim=8,
    )
    features = torch.rand(2, 16, 4, 4, 4)
    composer_in = {"gripper_opening": torch.rand(2, 1)}
    context = refine(features, composer_in)
    assert context.shape == (2, 8)
    context_again, attention = refine(features, composer_in, return_attn=True)
    assert attention.shape == (2, 4, 4, 4, 4)
    torch.testing.assert_close(context, context_again)


def test_focus_attention_prior_gating_and_single_cell_loss():
    prior = FocusAttentionPrior({"enabled": True, "weight": 0.5, "sigma_cells": 0.05})
    assert prior.active_weight(bootstrapping=True) == 0.5
    assert prior.active_weight(bootstrapping=False) == 0.0

    from visuomotor.geometry.grid import FeatureGridGeometry

    geometry = FeatureGridGeometry.from_stride((8, 8, 8), (4, 4, 4), stride=2)
    # A near-delta attention map centered on the same cell the target sits on.
    logits = torch.full((1, 2, 4, 4, 4), -10.0)
    logits[0, :, 0, 0, 0] = 10.0
    attention = logits.flatten(2).softmax(dim=-1).reshape(1, 2, 4, 4, 4)
    target = {"pos": geometry.centers[0, 0, 0].unsqueeze(0), "valid": torch.ones(1, dtype=torch.bool)}
    loss = prior.loss(attention, target, geometry)
    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_gaussian_feature_target_sums_to_one():
    from visuomotor.geometry.grid import FeatureGridGeometry

    geometry = FeatureGridGeometry.from_stride((8, 8, 8), (4, 4, 4), stride=2)
    target = gaussian_feature_target(torch.zeros(3, 3), geometry, sigma_cells=1.0)
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(3))


def test_focus_voxel_backbone_attention_prior_end_to_end():
    backbone = FocusVoxelBackbone(
        obs_channel=4,
        n_out=32,
        in_size=58,
        pool_stage=3,
        focus_pool={"iters": 2, "heads": 4, "attention_prior": {"enabled": True, "weight": 0.1}},
    )
    voxels = _random_voxel(2, 4, 58, dtype=torch.float32)
    gripper = torch.rand(2, 2)
    context, attention = backbone(voxels, gripper, return_attn=True)
    assert context.shape == (2, 32)
    assert not hasattr(backbone, "proj")
    backbone.feature_geometry.validate_attention(attention)


def test_resnet_backbone_output_shape():
    image = torch.rand(2, 3, 84, 84)
    assert ResNet18Backbone(out_size=32, weights=None)(image).shape == (2, 32)


def _observations(batch, steps, size, channels=4):
    voxel = _random_voxel(batch * steps, channels, size).reshape(batch, steps, channels, size, size, size)
    return {
        "voxel": Normalizer.normalize_voxel(canonicalize_voxel_from_uint8(voxel)),
        "rgb_wrist": Normalizer.normalize_canonical_rgb(
            canonicalize_rgb_from_uint8(
                torch.randint(0, 256, (batch, steps, 3, 84, 84), dtype=torch.uint8)
            )
        ),
        "eef_pos": torch.randn(batch, steps, 3),
        "eef_rot6d": torch.eye(3).reshape(1, 1, 9)[..., :6].expand(batch, steps, 6).clone(),
        "gripper_qpos": torch.rand(batch, steps, 2),
    }


@pytest.mark.parametrize("architecture", ["voxel_simple", "voxel_resnet3d"])
def test_voxel_observation_encoder_fuses_modalities(architecture):
    size = 64
    encoder = VoxelObservationEncoder(
        source_shape=(4, size, size, size),
        crop_size=size,
        voxel_architecture=architecture,
        rgb_keys=("rgb_wrist",),
        proprio_fields=("eef_pos", "eef_rot6d", "gripper_qpos"),
        proprio_dims=(3, 6, 2),
        feature_dim=32,
        rgb_feature_dim=32,
    )
    out = encoder(_observations(2, 3, size))
    assert encoder.output_dim == 32 + 32 + 11
    assert out.features.shape == (2, 3, encoder.output_dim)
    assert out.streams["voxel"].shape == (2, 3, 32)
    assert out.streams["rgb_wrist"].shape == (2, 3, 32)
    assert out.attention is None


def test_voxel_observation_encoder_ignores_attention_collection_when_unsupported():
    size = 64
    encoder = VoxelObservationEncoder(
        source_shape=(4, size, size, size),
        voxel_architecture="voxel_simple",
        feature_dim=32,
    )

    out = encoder(_observations(2, 3, size), collect_attention=True)

    assert encoder.output_dim == 32
    assert out.features.shape == (2, 3, encoder.output_dim)
    assert out.attention is None


def test_voxel_observation_encoder_focus_pool_attention_prior():
    size = 58
    encoder = VoxelObservationEncoder(
        source_shape=(4, size, size, size),
        crop_size=size,
        voxel_architecture="voxel_focus_pool3d",
        rgb_keys=("rgb_wrist",),
        proprio_fields=("eef_pos", "eef_rot6d", "gripper_qpos"),
        proprio_dims=(3, 6, 2),
        feature_dim=32,
        coord_conv=True,
        num_iterations=2,
        num_heads=4,
        attention_prior=AttentionPriorSpec(
            enabled=True, weight=0.02, bootstrap_steps=2
        ),
        source_workspace_min=(0.0, 0.0, 0.0),
        source_workspace_size=0.6,
    )
    batch, steps = 2, 3
    observations = _observations(batch, steps, size)
    flat = batch * steps
    target_world = {"pos": torch.zeros(flat, 3), "valid": torch.ones(flat, dtype=torch.bool)}
    out = encoder(
        observations, attention_target_world=target_world, global_step=0
    )
    assert out.attention.shape[:2] == (batch, steps)
    assert out.voxel_crop_transform.starts.shape == (batch * steps, 3)
    assert tuple(out.auxiliary_losses) == ("attention_prior",)
    assert torch.isfinite(out.auxiliary_losses["attention_prior"])
    off = encoder(
        observations, attention_target_world=target_world, global_step=5
    )
    assert off.auxiliary_losses["attention_prior"].item() == 0.0


def test_voxel_observation_encoder_rejects_non_cubic_or_unsupported_shapes():
    with pytest.raises(ValueError):
        VoxelObservationEncoder(source_shape=(4, 64, 32, 64))


def test_coord_conv_coordinates_are_source_grid_not_crop_window():
    """The crop must slice coordinates with content.

    A training crop lands at a random offset. If the coordinate grid were built
    at the cropped size it would be re-laid over each window, so one source cell
    would carry a different coordinate on every sample and the channels would
    encode position-within-window rather than position-in-scene.
    """
    source_size, crop_size = 16, 12
    encoder = VoxelObservationEncoder(
        source_shape=(4, source_size, source_size, source_size),
        crop_size=crop_size,
        voxel_architecture="voxel_simple",
        rgb_keys=(),
        proprio_fields=(),
        proprio_dims=(),
        feature_dim=16,
        coord_conv=True,
        source_workspace_min=(0.0, 0.0, 0.0),
        source_workspace_size=0.6,
    )
    assert tuple(encoder.coord_grid.shape) == (3, source_size, source_size, source_size)

    voxels = torch.zeros(1, 4, source_size, source_size, source_size)
    axis = torch.linspace(-1, 1, source_size)

    encoder.train()
    starts = set()
    for _ in range(16):
        prepared, transform = encoder._prepare_voxel(voxels)
        assert prepared.shape[1] == 7
        start = int(transform.starts[0, 0])
        starts.add(start)
        torch.testing.assert_close(
            prepared[0, 4, :, 0, 0], axis[start : start + crop_size]
        )
    assert len(starts) > 1, "the training crop should land at varying offsets"

    encoder.eval()
    _, centered = encoder._prepare_voxel(voxels)
    assert int(centered.starts[0, 0]) == (source_size - crop_size) // 2


@pytest.mark.parametrize("coord_conv", [False, True])
def test_voxel_encoder_reports_coord_conv_from_the_built_backbone(coord_conv):
    """The runtime block reads the backbone, not the flag.

    A block that echoed `self.coord_conv` could print Enabled while the stack
    was built for four channels. Reporting the conv's own input width cannot
    disagree with what the model actually runs.
    """
    size = 16
    encoder = VoxelObservationEncoder(
        source_shape=(4, size, size, size),
        crop_size=12,
        voxel_architecture="voxel_simple",
        rgb_keys=(),
        proprio_fields=("eef_pos", "eef_delta_pos"),
        proprio_dims=(3, 3),
        feature_dim=16,
        coord_conv=coord_conv,
        source_workspace_min=(0.0, 0.0, 0.0),
        source_workspace_size=0.6,
    )
    block = encoder.get_runtime_config()["Voxel Encoder"]
    assert block["Architecture"] == "voxel_simple"
    assert block["Proprio"] == "eef_pos, eef_delta_pos"
    assert "16×16×16→12³" in block["Voxel Crop"]
    if coord_conv:
        assert block["Coord Conv"] == "Enabled (7 input channels)"
        assert encoder.voxel_backbone_in_channels == 7
    else:
        assert block["Coord Conv"] == "Disabled"
        assert encoder.voxel_backbone_in_channels == 4


def test_encoder_features_are_the_streams_followed_by_proprioception():
    """The fused layout keeps visual streams before proprioception."""
    size = 64
    encoder = VoxelObservationEncoder(
        source_shape=(4, size, size, size),
        crop_size=size,
        voxel_architecture="voxel_simple",
        rgb_keys=("rgb_wrist",),
        proprio_fields=("eef_pos", "eef_rot6d", "gripper_qpos"),
        proprio_dims=(3, 6, 2),
        feature_dim=32,
        rgb_feature_dim=32,
    )
    encoder.eval()
    observations = _observations(2, 3, size)
    out = encoder(observations)

    rebuilt = torch.cat(
        tuple(out.streams.values())
        + tuple(observations[key] for key in encoder.proprio_fields),
        dim=-1,
    )
    torch.testing.assert_close(rebuilt, out.features, rtol=0, atol=0)
    assert tuple(out.streams) == ("voxel", "rgb_wrist")
    assert rebuilt.shape[-1] == encoder.output_dim
