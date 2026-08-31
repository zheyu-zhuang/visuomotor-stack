import torch

from visuomotor.config.build import build_generator
from visuomotor.config.schema import GeneratorSpec, TrajectoryContract
from visuomotor.data.core.normalization import Normalizer, normalize_obs
from visuomotor.perception.encoder.point_cloud import PointCloudObservationEncoder


def test_dp3_encoder_fuses_temporal_point_cloud_and_proprioception():
    encoder = PointCloudObservationEncoder(
        source_shape=(32, 6),
        proprio_fields=("eef_pos", "eef_rot6d", "gripper_qpos"),
        proprio_dims=(3, 6, 2),
        feature_dim=16,
        state_mlp_dims=(16, 8),
    )
    observations = {
        "point_cloud": torch.randn(2, 3, 32, 6),
        "eef_pos": torch.randn(2, 3, 3),
        "eef_rot6d": torch.randn(2, 3, 6),
        "gripper_qpos": torch.randn(2, 3, 2),
    }
    output = encoder(observations)
    assert output.features.shape == (2, 3, 24)
    assert output.streams["point_cloud"].shape == (2, 3, 16)
    assert output.streams["proprio"].shape == (2, 3, 8)


def test_point_cloud_crosses_the_fitted_normalization_boundary():
    normalizer = Normalizer()
    normalizer.update_bounds(
        "point_cloud",
        torch.tensor([-1.0, -2.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
    )
    normalizer.finalize()
    points = torch.tensor([[[0.0, 0.0, 1.0, 0.5, 0.5, 0.5]]])
    normalized = normalize_obs(
        {"point_cloud": points},
        normalizer,
        observation_kinds={"point_cloud": "point_cloud"},
    )
    torch.testing.assert_close(normalized["point_cloud"], torch.zeros_like(points))


def test_dp3_ddim_generator_trains_and_samples_through_shared_action_path():
    generator = build_generator(
        GeneratorSpec(
            kind="diffusion",
            scheduler="ddim",
            unet_channels=(16, 32),
            num_train_timesteps=4,
            num_inference_steps=2,
            prediction_type="sample",
        ),
        trajectory=TrajectoryContract(10, 8, 2, 4, "absolute"),
        condition_dim=24,
    )
    actions = torch.randn(2, 8, 10)
    condition = torch.randn(2, 24)
    assert torch.isfinite(generator.loss(actions, condition))
    assert generator.sample(condition).shape == actions.shape
