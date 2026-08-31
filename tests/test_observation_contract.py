import pytest
import torch

from visuomotor.data.core import normalization as CoreNormalization
from visuomotor.data.core.normalization import Normalizer
from visuomotor.data.core.observations import (
    canonicalize_rgb_from_uint8,
    canonicalize_voxel_from_uint8,
)


def test_rgb_canonicalization_then_imagenet_normalization_and_input_immutability():
    image = torch.tensor([[[0]], [[128]], [[255]]], dtype=torch.uint8)
    original = image.clone()
    canonical = canonicalize_rgb_from_uint8(image)
    torch.testing.assert_close(canonical, image)
    assert torch.equal(image, original)

    result = CoreNormalization.Normalizer().normalize("rgb", canonical)
    expected = (image.float().flatten() / 255 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(result.flatten(), expected)


def test_voxel_canonicalization_then_colour_masked_and_scaled_to_unit_range():
    voxel = torch.zeros((4, 2, 1, 1), dtype=torch.uint8)
    voxel[:, 0] = torch.tensor([0, 255, 255, 255], dtype=torch.uint8).view(4, 1, 1)
    voxel[:, 1] = torch.tensor([1, 0, 128, 255], dtype=torch.uint8).view(4, 1, 1)
    canonical = canonicalize_voxel_from_uint8(voxel)
    # Canonicalization never masks/normalizes -- the empty cell's RGB channels
    # still carry the raw source values at this stage.
    torch.testing.assert_close(
        canonical[1:, 0].flatten(), torch.tensor([255, 255, 255], dtype=torch.uint8)
    )

    result = CoreNormalization.Normalizer().normalize("voxel", canonical)
    assert torch.equal(result[1:, 0], torch.zeros_like(result[1:, 0]))
    expected_rgb = torch.tensor([0.0, 128 / 255, 1.0])
    torch.testing.assert_close(result[:, 1, 0, 0], torch.cat((torch.tensor([1.0]), expected_rgb)))


def test_voxel_rejects_nonbinary_occupancy():
    voxel = torch.zeros((4, 2, 2, 2), dtype=torch.uint8)
    voxel[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        canonicalize_voxel_from_uint8(voxel)


def test_normalizer_round_trip_and_checkpoint_round_trip():
    normalizer = Normalizer()
    values = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    normalizer.update_samples("eef_pos", values)
    normalizer.finalize()
    normalized = normalizer.normalize("eef_pos", values)
    torch.testing.assert_close(normalized, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
    torch.testing.assert_close(normalizer.denormalize("eef_pos", normalized), values)
    restored = Normalizer()
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored.normalize("eef_pos", values), normalized)


def test_per_robot_normalization_round_trip():
    normalizer = Normalizer()
    normalizer.update_samples("eef_pos", torch.tensor([[0.0], [2.0]]), robot_id=0)
    normalizer.update_samples("eef_pos", torch.tensor([[10.0], [14.0]]), robot_id=1)
    normalizer.finalize()
    actions = torch.tensor([[[1.0]], [[12.0]]])
    robot_id = torch.tensor([0, 1])
    normalized = normalizer.normalize("eef_pos", actions, robot_id=robot_id)
    torch.testing.assert_close(normalized, torch.zeros_like(actions))
    torch.testing.assert_close(
        normalizer.denormalize("eef_pos", normalized, robot_id=robot_id), actions
    )


def test_normalize_action_round_trip_with_batched_robot_id():
    normalizer = Normalizer()
    for robot_id, low, high in ((0, 0.0, 2.0), (1, 10.0, 14.0)):
        pos = torch.tensor([[low, low, low], [high, high, high]])
        gripper = torch.tensor([[low], [high]])
        normalizer.update_samples("action_pos", pos, robot_id=robot_id)
        normalizer.update_samples("action_gripper", gripper, robot_id=robot_id)
    normalizer.finalize()

    rotation = torch.zeros(2, 6)
    actions = torch.cat(
        [torch.tensor([[1.0, 1.0, 1.0], [12.0, 12.0, 12.0]]), rotation, torch.tensor([[1.0], [12.0]])],
        dim=-1,
    )
    robot_id = torch.tensor([0, 1])
    normalized = normalizer.normalize_action(actions, robot_id=robot_id)
    torch.testing.assert_close(normalized[..., :3], torch.zeros(2, 3))
    torch.testing.assert_close(normalized[..., 3:9], rotation)
    torch.testing.assert_close(normalized[..., 9:], torch.zeros(2, 1))
    torch.testing.assert_close(normalizer.denormalize_action(normalized, robot_id=robot_id), actions)


def test_normalize_routed_field_raises_for_unfitted_robot():
    normalizer = Normalizer()
    normalizer.update_samples("eef_pos", torch.tensor([[0.0], [2.0]]), robot_id=0)
    normalizer.finalize()
    with pytest.raises(KeyError):
        normalizer.normalize("eef_pos", torch.tensor([[1.0]]), robot_id=torch.tensor([7]))
