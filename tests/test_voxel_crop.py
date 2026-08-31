import torch

from visuomotor.perception.common.voxel_crop import VoxelCropper


def test_train_time_crop_shape_and_bounds():
    cropper = VoxelCropper(crop_shape=(8, 8, 8)).train()
    voxels = torch.rand(4, 2, 12, 12, 12)
    result = cropper(voxels)
    assert result.voxels.shape == (4, 2, 8, 8, 8)
    assert (result.transform.starts >= 0).all()
    assert (result.transform.starts <= 4).all()


def test_eval_time_crop_is_centered_and_deterministic():
    cropper = VoxelCropper(crop_shape=(8, 8, 8)).eval()
    voxels = torch.rand(3, 2, 12, 12, 12)
    result = cropper(voxels)
    torch.testing.assert_close(result.transform.starts, torch.full((3, 3), 2.0))


def test_crop_center_maps_to_grid_coordinate_zero():
    cropper = VoxelCropper(crop_shape=(8, 8, 8)).eval()
    voxels = torch.rand(1, 1, 12, 12, 12)
    result = cropper(voxels)
    center_index = result.transform.starts + (torch.tensor([8.0, 8.0, 8.0]) - 1) / 2
    center_source_norm = center_index / 11.0 * 2 - 1
    crop_norm = result.transform.source_to_crop(center_source_norm)
    torch.testing.assert_close(crop_norm, torch.zeros(1, 3), atol=1e-5, rtol=1e-5)


def test_no_crop_shape_is_the_identity():
    cropper = VoxelCropper(crop_shape=None).train()
    voxels = torch.rand(2, 1, 6, 6, 6)
    result = cropper(voxels)
    torch.testing.assert_close(result.voxels, voxels)
    torch.testing.assert_close(result.transform.starts, torch.zeros(2, 3))


def test_no_crop_does_not_advance_random_state():
    cropper = VoxelCropper(crop_shape=None).train()
    voxels = torch.rand(2, 1, 6, 6, 6)
    state = torch.random.get_rng_state()
    cropper(voxels)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
