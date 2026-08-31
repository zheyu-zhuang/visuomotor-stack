import math

import pytorch3d.transforms as p3t
import torch

from visuomotor.geometry import bounds as Bounds
from visuomotor.geometry import grid as Grid
from visuomotor.geometry import projection as Projection
from visuomotor.geometry import representation as Representation
from visuomotor.geometry import rigid as Rigid
from visuomotor.geometry import roi as Roi


def _random_rotation(batch: int) -> torch.Tensor:
    axis_angle = torch.randn(batch, 3)
    return torch.linalg.matrix_exp(_skew(axis_angle))


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    row0 = torch.stack((zero, -z, y), dim=-1)
    row1 = torch.stack((z, zero, -x), dim=-1)
    row2 = torch.stack((-y, x, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


# ---------------------------------------------------------------- projection


def test_projection_rejects_points_behind_the_camera():
    projected = Projection.world_xyz_to_pixel_row_col(
        torch.tensor([[1.0, 2.0, -1.0]]), torch.eye(4), image_size=8
    )
    assert torch.isnan(projected).all()


def test_projection_supports_rectangular_images_and_unclamped_coordinates():
    points = torch.tensor([[30.0, 20.0, 2.0]])
    matrix = torch.eye(4)
    unclamped = Projection.world_xyz_to_pixel_row_col(
        points, matrix, image_size=(6, 10), clamp=False
    )
    torch.testing.assert_close(unclamped, torch.tensor([[10.0, 15.0]]))
    clamped = Projection.world_xyz_to_pixel_row_col(
        points, matrix, image_size=(6, 10)
    )
    torch.testing.assert_close(clamped, torch.tensor([[5.0, 9.0]]))


# --------------------------------------------------------------------- rigid


def test_identity_transform_is_a_no_op():
    identity_R = torch.eye(3).unsqueeze(0)
    zero_t = torch.zeros(1, 3)
    points = torch.randn(1, 5, 3)
    torch.testing.assert_close(Rigid.transform(identity_R.expand(1, 5, 3, 3), zero_t.expand(1, 5, 3), points), points)


def test_inverse_round_trip():
    rotation = _random_rotation(3)
    translation = torch.randn(3, 3)
    R_inv, t_inv = Rigid.inv(rotation, translation)
    R_id, t_id = Rigid.compose(rotation, translation, R_inv, t_inv)
    torch.testing.assert_close(R_id, torch.eye(3).unsqueeze(0).expand(3, 3, 3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(t_id, torch.zeros(3, 3), atol=1e-5, rtol=1e-5)


def test_composition_equivalence():
    R_AB, t_AB = _random_rotation(2), torch.randn(2, 3)
    R_BC, t_BC = _random_rotation(2), torch.randn(2, 3)
    R_AC, t_AC = Rigid.compose(R_AB, t_AB, R_BC, t_BC)
    p_C = torch.randn(2, 3)
    p_A_direct = Rigid.transform(R_AC, t_AC, p_C)
    p_B = Rigid.transform(R_BC, t_BC, p_C)
    p_A_chained = Rigid.transform(R_AB, t_AB, p_B)
    torch.testing.assert_close(p_A_direct, p_A_chained, atol=1e-5, rtol=1e-5)


def test_relative_transform_handles_translated_and_rotated_source_frame():
    R_WE = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    t_WE = torch.tensor([[10.0, 20.0, 30.0]])
    R_EA = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]]
    )
    t_EA = torch.tensor([[1.0, 2.0, 3.0]])
    R_WA, t_WA = Rigid.compose(R_WE, t_WE, R_EA, t_EA)

    relative_rotation, relative_translation = Rigid.relative(
        R_WE, t_WE, R_WA, t_WA
    )

    torch.testing.assert_close(relative_rotation, R_EA)
    torch.testing.assert_close(relative_translation, t_EA)


def test_transform_pose_xyz_equals_transform():
    rotation, translation = _random_rotation(2), torch.randn(2, 3)
    pose = torch.cat((torch.randn(2, 3), Representation.mat_to_rot6d(_random_rotation(2))), dim=-1)
    transformed_pose = Rigid.transform_pose(rotation, translation, pose)
    torch.testing.assert_close(transformed_pose[..., :3], Rigid.transform(rotation, translation, pose[..., :3]))


def test_orientation_composition_correctness():
    R_AB, R_BC = _random_rotation(2), _random_rotation(2)
    R_AC = Rigid.transform_rotation(R_AB, R_BC)
    torch.testing.assert_close(R_AC, R_AB @ R_BC)


def test_transform_pose_preserves_trailing_extras():
    rotation, translation = _random_rotation(2), torch.randn(2, 3)
    extras = torch.randn(2, 2)
    pose = torch.cat((torch.randn(2, 3), Representation.mat_to_rot6d(_random_rotation(2)), extras), dim=-1)
    transformed = Rigid.transform_pose(rotation, translation, pose)
    torch.testing.assert_close(transformed[..., 9:], extras)


def test_transform_pose_broadcasts_one_frame_over_a_horizon():
    rotation, translation = _random_rotation(2), torch.randn(2, 3)
    poses = torch.cat((torch.randn(2, 5, 3), Representation.mat_to_rot6d(_random_rotation(2 * 5)).reshape(2, 5, 6)), dim=-1)
    transformed = Rigid.transform_pose(rotation, translation, poses)
    assert transformed.shape == poses.shape
    for step in range(5):
        torch.testing.assert_close(transformed[:, step], Rigid.transform_pose(rotation, translation, poses[:, step]))


def test_transform_pose_inverse_round_trip():
    rotation, translation = _random_rotation(2), torch.randn(2, 3)
    pose = torch.cat((torch.randn(2, 3), Representation.mat_to_rot6d(_random_rotation(2))), dim=-1)
    R_inv, t_inv = Rigid.inv(rotation, translation)
    local = Rigid.transform_pose(R_inv, t_inv, pose)
    torch.testing.assert_close(Rigid.transform_pose(rotation, translation, local), pose, atol=1e-5, rtol=1e-5)


def test_inv_rotation_matches_the_rotation_half_of_inv():
    rotation = _random_rotation(3)
    R_BA, _ = Rigid.inv(rotation, torch.randn(3, 3))
    torch.testing.assert_close(Rigid.inv_rotation(rotation), R_BA)


def test_frame_with_mismatched_batch_dims_is_rejected():
    rotation = _random_rotation(4)
    for call in (
        lambda: Rigid.inv(rotation, torch.randn(3, 3)),
        lambda: Rigid.transform(rotation, torch.randn(3, 3), torch.randn(4, 3)),
        lambda: Rigid.to_homogeneous(rotation, torch.randn(3, 3)),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for mismatched frame batch dims")


def test_non_rotation_shapes_are_rejected():
    for call in (
        lambda: Rigid.inv_rotation(torch.randn(4, 3, 2)),
        lambda: Rigid.transform_rotation(_random_rotation(2), torch.randn(2, 3)),
        lambda: Rigid.transform(_random_rotation(2), torch.randn(2, 3), torch.randn(2, 4)),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for a non-rotation shape")


def test_homogeneous_round_trip():
    rotation, translation = _random_rotation(2), torch.randn(2, 3)
    X = Rigid.to_homogeneous(rotation, translation)
    round_R, round_t = Rigid.from_homogeneous(X)
    torch.testing.assert_close(round_R, rotation)
    torch.testing.assert_close(round_t, translation)


def test_geodesic_angle_of_identical_rotations_is_zero():
    rotation = _random_rotation(4)
    torch.testing.assert_close(Rigid.geodesic_angle(rotation, rotation), torch.zeros(4), atol=1e-2, rtol=1e-2)


def test_geometric_median_recovers_a_tight_cluster_against_one_outlier():
    cluster = torch.tensor([1.0, 1.0, 1.0]) + 0.01 * torch.randn(9, 3)
    outlier = torch.tensor([[50.0, -50.0, 50.0]])
    points = torch.cat((cluster, outlier), dim=0).unsqueeze(0)
    median = Rigid.geometric_median(points, dim=1)
    torch.testing.assert_close(median, torch.tensor([[1.0, 1.0, 1.0]]), atol=0.1, rtol=0.1)


def test_rotation_geometric_median_converges_on_near_identical_rotations():
    base = _random_rotation(1)
    noise = torch.eye(3).unsqueeze(0)
    rotations = base.unsqueeze(1).expand(1, 8, 3, 3) @ noise.unsqueeze(1).expand(1, 8, 3, 3)
    median = Rigid.rotation_geometric_median(rotations, dim=1)
    torch.testing.assert_close(median, base, atol=1e-4, rtol=1e-4)


def test_rotation_geometric_median_projects_numerically_invalid_particles():
    base = _random_rotation(1).unsqueeze(1).expand(1, 8, 3, 3).clone()
    base[:, ::2] *= 1.01

    median = Rigid.rotation_geometric_median(base, dim=1)

    identity = torch.eye(3).expand(1, 3, 3)
    torch.testing.assert_close(
        median.transpose(-1, -2) @ median, identity, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(torch.linalg.det(median), torch.ones(1), atol=1e-5, rtol=1e-5)


def test_rotation_geometric_median_does_not_use_strict_trace_log_map(monkeypatch):
    rotations = _random_rotation(8).reshape(1, 8, 3, 3)

    def reject_trace_validation(*args, **kwargs):
        raise AssertionError("strict trace log map must not be used")

    monkeypatch.setattr(Rigid.p3t, "so3_log_map", reject_trace_validation)

    median = Rigid.rotation_geometric_median(rotations, dim=1)

    assert torch.isfinite(median).all()


# --------------------------------------------------------------- representation


def test_rotation_representation_round_trips():
    rotation = _random_rotation(4)
    torch.testing.assert_close(
        Representation.rot6d_to_mat(Representation.mat_to_rot6d(rotation)), rotation, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        Representation.rotvec_to_mat(Representation.mat_to_rotvec(rotation)), rotation, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        Representation.quat_to_mat(Representation.mat_to_quat(rotation)), rotation, atol=1e-5, rtol=1e-5
    )


def test_mat_to_rotvec_uses_the_shortest_arc():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    long_way = axis * (2 * math.pi - 0.5)  # 5.78 rad about +z == 0.5 rad about -z
    rotvec = Representation.mat_to_rotvec(Representation.rotvec_to_mat(long_way))
    assert float(torch.linalg.vector_norm(rotvec, dim=-1)) <= math.pi + 1e-6
    torch.testing.assert_close(rotvec, -axis * 0.5, atol=1e-5, rtol=1e-5)


def test_rotation_transformer_decodes_axis_angle_on_the_shortest_arc():
    """Rollout inverts rot6d back to the convention the caches were written in.

    ``convert_actions`` records cached actions through ``mat_to_rotvec``. With
    pytorch3d's raw decode a rollout emits the 2*pi complement for some
    rotations -- a second convention for one quantity, and one the robosuite
    controllers are not written against.
    """
    generator = torch.Generator().manual_seed(0)
    rotvec = torch.randn(512, 3, generator=generator) * 2.5
    rot6d = Representation.mat_to_rot6d(Representation.rotvec_to_mat(rotvec))

    transformer = Representation.RotationTransformer("axis_angle", "rotation_6d")
    decoded = transformer.inverse(rot6d)

    # The raw decode does exceed pi here, so this batch actually exercises it.
    raw = p3t.matrix_to_axis_angle(Representation.rot6d_to_mat(rot6d))
    assert float(torch.linalg.vector_norm(raw, dim=-1).max()) > math.pi

    assert float(torch.linalg.vector_norm(decoded, dim=-1).max()) <= math.pi + 1e-6
    torch.testing.assert_close(
        decoded, Representation.mat_to_rotvec(Representation.rot6d_to_mat(rot6d))
    )
    torch.testing.assert_close(
        Representation.rotvec_to_mat(decoded),
        Representation.rotvec_to_mat(rotvec),
        atol=1e-5,
        rtol=1e-5,
    )


def test_rotation_transformer_axis_angle_round_trip_is_stable():
    """Forward then inverse must land on the canonical form, not oscillate."""
    transformer = Representation.RotationTransformer("axis_angle", "rotation_6d")
    generator = torch.Generator().manual_seed(1)
    rotvec = torch.randn(256, 3, generator=generator) * 2.5

    once = transformer.inverse(transformer.forward(rotvec))
    twice = transformer.inverse(transformer.forward(once))

    assert float(torch.linalg.vector_norm(once, dim=-1).max()) <= math.pi + 1e-6
    torch.testing.assert_close(once, twice, atol=1e-6, rtol=1e-6)


def test_canonicalize_rotvec_preserves_short_arcs_and_the_rotation():
    short = torch.tensor([[0.1, -0.2, 0.3]])
    torch.testing.assert_close(Representation.canonicalize_rotvec(short), short)
    long_way = torch.tensor([[0.0, 3.0, -2.5]])  # norm 3.9 > pi
    canonical = Representation.canonicalize_rotvec(long_way)
    assert float(torch.linalg.vector_norm(canonical, dim=-1)) <= math.pi
    torch.testing.assert_close(
        Representation.rotvec_to_mat(canonical),
        Representation.rotvec_to_mat(long_way),
        atol=1e-5,
        rtol=1e-5,
    )


def test_quat_xyzw_to_mat_matches_identity_quaternion():
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(Representation.quat_xyzw_to_mat(quaternion), torch.eye(3).unsqueeze(0))


# ------------------------------------------------------------------------ grid


def test_voxelize_points_shape_and_nonempty():
    points = torch.tensor([[[0.3, 0.3, 0.3], [-0.5, 0.5, 0.2]]])
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    grid = Grid.voxelize_points(points, features, bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], resolution=4)
    assert grid.shape == (1, 2, 4, 4, 4)
    assert grid.sum() > 0


def test_source_voxel_geometry_round_trip():
    geometry = Grid.SourceVoxelGeometry.optional((0.0, 0.0, 0.0), 0.6, (64, 64, 64))
    points = torch.tensor([[0.3, 0.3, 0.3], [0.1, 0.5, 0.2]])
    grid = geometry.world_to_grid(points)
    torch.testing.assert_close(geometry.grid_to_world(grid), points, atol=1e-4, rtol=1e-4)
    assert Grid.SourceVoxelGeometry.optional(None, 0.6, (64, 64, 64)) is None


def test_voxel_crop_transform_center_point_maps_to_zero():
    starts = torch.tensor([[3.0, 3.0, 3.0]])
    transform = Grid.VoxelCropTransform(starts=starts, source_shape=(64, 64, 64), crop_shape=(58, 58, 58))
    center_index = starts + (torch.tensor([58.0, 58.0, 58.0]) - 1) / 2
    center_source_norm = (center_index / 63.0) * 2 - 1
    crop_norm = transform.source_to_crop(center_source_norm)
    torch.testing.assert_close(crop_norm, torch.zeros(1, 3), atol=1e-4, rtol=1e-4)
    assert transform.contains_crop(crop_norm).all()


def test_voxel_crop_transform_rejects_out_of_range_starts():
    try:
        Grid.VoxelCropTransform(starts=torch.tensor([[10.0, 0.0, 0.0]]), source_shape=(10, 10, 10), crop_shape=(8, 8, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for an out-of-range start")


def test_feature_grid_geometry_shape_and_validation():
    geometry = Grid.FeatureGridGeometry.from_stride((64, 64, 64), (8, 8, 8), stride=8)
    assert geometry.shape == (8, 8, 8)
    geometry.validate_attention(torch.zeros(2, 4, 8, 8, 8))
    try:
        geometry.validate_attention(torch.zeros(2, 4, 7, 8, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a mismatched attention shape")


# ---------------------------------------------------------------------- bounds


def test_yaw_envelope_bounds_covers_every_point_on_the_circle():
    center = torch.tensor([0.5, -0.25, 0.0])
    point = torch.tensor([1.5, -0.25, 0.3])  # radius 1 about the center, in the xy-plane
    lo, hi = Bounds.yaw_envelope_bounds(point, center)
    angles = torch.linspace(0, 2 * math.pi, 37)[:-1]
    rotated_xy = center[:2] + torch.stack((angles.cos(), angles.sin()), dim=-1)
    assert bool((rotated_xy >= lo[:2] - 1e-5).all())
    assert bool((rotated_xy <= hi[:2] + 1e-5).all())
    torch.testing.assert_close(lo[2], point[2])
    torch.testing.assert_close(hi[2], point[2])


def test_planar_workspace_contains_xy():
    workspace = Bounds.PlanarWorkspace(center_xy=(1.0, 1.0), size=2.0)
    inside = torch.tensor([1.5, 0.5, 3.0])
    outside = torch.tensor([3.0, 3.0, 3.0])
    assert bool(workspace.contains_xy(inside))
    assert not bool(workspace.contains_xy(outside))


def test_planar_workspace_position_bounds_use_workspace_xy_and_data_z():
    workspace = Bounds.PlanarWorkspace(center_xy=(1.0, -2.0), size=0.6)
    positions = torch.tensor([[1.1, -1.9, 0.4], [0.9, -2.1, 0.8]])

    lo, hi = workspace.position_bounds(positions)

    torch.testing.assert_close(lo[:, :2], torch.tensor([[0.7, -2.3], [0.7, -2.3]]))
    torch.testing.assert_close(hi[:, :2], torch.tensor([[1.3, -1.7], [1.3, -1.7]]))
    torch.testing.assert_close(lo[:, 2], positions[:, 2])
    torch.testing.assert_close(hi[:, 2], positions[:, 2])


# ------------------------------------------------------------------------ roi


def test_full_image_roi_is_exactly_0_0_w_h():
    box = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    mask = Roi.box_px_to_grid_mask(box, image_size=8, grid_size=4, guard_cells=0)
    torch.testing.assert_close(mask, torch.ones_like(mask))


def test_expand_box_px_is_symmetric_and_clamps():
    box = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
    expanded = Roi.expand_box_px(box, margin_px=1.0)
    torch.testing.assert_close(expanded, torch.tensor([[1.0, 1.0, 7.0, 7.0]]))
    clamped = Roi.expand_box_px(box, margin_px=10.0, image_size=8.0)
    torch.testing.assert_close(clamped, torch.tensor([[0.0, 0.0, 8.0, 8.0]]))


def test_box_px_to_grid_mask_guard_cells_grows_the_mask():
    box = torch.tensor([[3.0, 3.0, 5.0, 5.0]])
    tight = Roi.box_px_to_grid_mask(box, image_size=8, grid_size=4, guard_cells=0)
    guarded = Roi.box_px_to_grid_mask(box, image_size=8, grid_size=4, guard_cells=1)
    assert bool((guarded.bool() | ~tight.bool()).all())  # guarded is a superset of tight
    assert guarded.sum() >= tight.sum()
