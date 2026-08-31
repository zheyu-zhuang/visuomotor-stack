import pytest

from visuomotor.config import schema as Schema
from visuomotor.config.build import sensor_specs
from visuomotor.data.core.spatial import PointCloudProducerSpec, VoxelProducerSpec


def test_voxel_metadata_exact_validation_and_override():
    spec = VoxelProducerSpec(resolution=(32, 48, 64), cameras=("front",))
    assert spec.metadata()["reconstruction_resolution"] == 84
    assert tuple(spec.metadata()) == (
        "resolution",
        "bounds_min",
        "bounds_max",
        "ws_size",
        "cameras",
        "channels",
        "reconstruction_resolution",
    )
    assert tuple(spec.declaration()) == (
        "output_key",
        "frame",
        "resolution",
        "bounds_min",
        "bounds_max",
        "ws_size",
        "cameras",
        "channels",
        "reconstruction_resolution",
    )
    spec.validate_metadata(spec.metadata())
    assert spec.observation_shape == (4, 32, 48, 64)
    bad = spec.metadata()
    bad["resolution"] = [64, 64, 64]
    with pytest.raises(ValueError, match="mismatch"):
        spec.validate_metadata(bad)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"output_key": ""}, "output_key"),
        ({"channels": ("occupancy",)}, "channels"),
        ({"ws_size": 0.0}, "ws_size"),
        ({"bounds_min": (0.0, 0.0, 0.0)}, "both be set"),
        (
            {"bounds_min": (1.0, 0.0, 0.0), "bounds_max": (0.0, 1.0, 1.0)},
            "lower bound",
        ),
        (
            {
                "frame": "eef_centered",
                "ws_size": 0.2,
                "bounds_min": (-0.2, -0.1, -0.1),
                "bounds_max": (0.1, 0.1, 0.1),
            },
            "symmetric",
        ),
    ),
)
def test_voxel_producer_rejects_invalid_declarations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VoxelProducerSpec(**kwargs)


def test_spatial_producers_round_trip_and_are_not_copied_by_builders():
    voxel = VoxelProducerSpec(cameras=("front",))
    point_cloud = PointCloudProducerSpec(cameras=("front",))
    source = Schema.SourceObservationSpec(
        fields=(), producers=(voxel, point_cloud)
    )

    restored = Schema.from_dict(Schema.to_dict(source))
    assert restored == source
    assert Schema.to_dict(voxel)["__spec__"] == "VoxelProducerSpec"
    assert Schema.to_dict(point_cloud)["__spec__"] == "PointCloudProducerSpec"

    voxels, resolved_point_cloud = sensor_specs(source)
    assert voxels["voxel"] is voxel
    assert resolved_point_cloud is point_cloud


def test_sensor_specs_reject_duplicate_spatial_outputs():
    duplicate_voxels = Schema.SourceObservationSpec(
        fields=(),
        producers=(VoxelProducerSpec(), VoxelProducerSpec()),
    )
    with pytest.raises(ValueError, match="duplicate voxel"):
        sensor_specs(duplicate_voxels)

    duplicate_points = Schema.SourceObservationSpec(
        fields=(),
        producers=(PointCloudProducerSpec(), PointCloudProducerSpec()),
    )
    with pytest.raises(ValueError, match="duplicate point-cloud"):
        sensor_specs(duplicate_points)
