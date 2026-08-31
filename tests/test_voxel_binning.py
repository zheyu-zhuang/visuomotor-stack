"""The patched robomimic voxelizer must match the Open3D path it replaced.

``_bin_points_to_voxel_grid`` bins points in numpy rather than building an
Open3D ``VoxelGrid`` and reading one Python ``Voxel`` object per occupied cell.
The grids it produces have to stay byte-identical to that retired path, since
spatial caches and trained checkpoints were produced against it.
"""

import numpy as np
import open3d as o3d
import pytest
from robomimic.envs.env_robosuite import EnvRobosuite, _accumulate_channels

BOUNDS_MIN = (-0.3, -0.3, 0.75)
BOUNDS_MAX = (0.3, 0.3, 1.35)
RESOLUTION = (64, 64, 64)


def _open3d_reference(points, colors, bounds_min, bounds_max, resolution):
    """The retired path: Open3D bins the cloud and averages colours per voxel."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    res = int(resolution[0])
    out = np.zeros((4, res, res, res), dtype=np.uint8)
    if len(pcd.points) == 0:
        return out
    bmin = np.asarray(bounds_min, dtype=np.float64)
    bmax = np.asarray(bounds_max, dtype=np.float64)
    voxel_size = (bmax[0] - bmin[0]) / res + 1e-4
    grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, voxel_size=voxel_size, min_bound=bmin, max_bound=bmax
    )
    voxels = grid.get_voxels()
    if not voxels:
        return out
    indices = np.stack([voxel.grid_index for voxel in voxels])
    vcolors = np.stack([voxel.color for voxel in voxels])
    keep = np.all((indices >= 0) & (indices < res), axis=-1)
    indices = indices[keep]
    vcolors = vcolors[keep]
    out[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    out[1:, indices[:, 0], indices[:, 1], indices[:, 2]] = np.clip(
        vcolors.T * 255, 0, 255
    ).astype(np.uint8)
    return out


def _cloud(seed, num_points=40000, pad=0.1):
    """Points spanning the workspace and spilling outside it on every axis."""
    rng = np.random.default_rng(seed)
    lower = np.asarray(BOUNDS_MIN) - pad
    upper = np.asarray(BOUNDS_MAX) + pad
    points = rng.uniform(lower, upper, size=(num_points, 3))
    colors = rng.integers(0, 256, size=(num_points, 3)).astype(np.uint8)
    return points, colors


def _binner():
    env = object.__new__(EnvRobosuite)
    return lambda points, colors: EnvRobosuite._bin_points_to_voxel_grid(
        env, points, colors, BOUNDS_MIN, BOUNDS_MAX, RESOLUTION
    )


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_numpy_binning_reproduces_the_open3d_grid_exactly(seed):
    points, colors = _cloud(seed)

    produced = _binner()(points, colors)
    expected = _open3d_reference(
        points, colors, BOUNDS_MIN, BOUNDS_MAX, RESOLUTION
    )

    # A wrong voxel_size or index offset would still occupy plausible cells.
    assert int(produced[0].sum()) > 1000
    np.testing.assert_array_equal(produced[0], expected[0])
    np.testing.assert_array_equal(produced[1:], expected[1:])


def test_points_outside_the_bounds_occupy_no_cell():
    points = np.asarray(BOUNDS_MAX, dtype=np.float64) + np.array(
        [[1.0, 1.0, 1.0], [0.5, 0.0, 0.0]]
    )
    colors = np.full((2, 3), 255, dtype=np.uint8)

    produced = _binner()(points, colors)

    assert int(produced[0].sum()) == 0


def test_an_empty_cloud_yields_an_empty_grid():
    produced = _binner()(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))

    assert produced.shape == (4, *RESOLUTION)
    assert not produced.any()


def test_a_cell_takes_the_mean_colour_of_its_points():
    centre = (np.asarray(BOUNDS_MIN) + np.asarray(BOUNDS_MAX)) / 2.0
    points = np.repeat(centre.reshape(1, 3), 4, axis=0)
    colors = np.array(
        [[0, 0, 0], [100, 100, 100], [200, 200, 200], [255, 255, 255]],
        dtype=np.uint8,
    )

    produced = _binner()(points, colors)

    assert int(produced[0].sum()) == 1
    occupied = produced[1:, produced[0] > 0].reshape(3)
    np.testing.assert_array_equal(occupied, np.full(3, 138, dtype=np.uint8))


def test_channel_accumulation_matches_a_scatter_add():
    rng = np.random.default_rng(3)
    cells = 512
    flat = rng.integers(0, cells, size=20000)
    values = rng.uniform(0.0, 255.0, size=(20000, 3))
    expected = np.zeros((cells, 3), dtype=np.float64)
    np.add.at(expected, flat, values)

    np.testing.assert_array_equal(_accumulate_channels(flat, values, cells), expected)
