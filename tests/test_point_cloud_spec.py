import pytest

from visuomotor.data.core.spatial import PointCloudProducerSpec


def test_point_cloud_metadata_exact_validation_and_override():
    spec = PointCloudProducerSpec(num_points=512, cameras=("front",))
    assert spec.table_margin == 0.005
    assert spec.observation_shape == (512, 6)
    assert "output_key" not in spec.metadata()
    assert tuple(spec.metadata()) == (
        "num_points",
        "bounds_min",
        "bounds_max",
        "ws_size",
        "table_margin",
        "cameras",
        "channels",
        "reconstruction_resolution",
    )
    assert spec.metadata()["table_margin"] == 0.005
    assert spec.metadata()["reconstruction_resolution"] == 84
    spec.validate_metadata(spec.metadata())
    bad = spec.metadata()
    bad["num_points"] = 1024
    with pytest.raises(ValueError, match="mismatch"):
        spec.validate_metadata(bad)


def test_point_cloud_table_margin_must_remove_a_nonempty_slab():
    with pytest.raises(ValueError, match="table_margin"):
        PointCloudProducerSpec(table_margin=0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"output_key": "cloud"}, "output_key"),
        ({"channels": ("x", "y", "z")}, "XYZRGB"),
        ({"ws_size": 0.0}, "ws_size"),
        ({"bounds_max": (1.0, 1.0, 1.0)}, "both be set"),
        (
            {"bounds_min": (0.0, 0.0, 1.0), "bounds_max": (1.0, 1.0, 0.0)},
            "lower bound",
        ),
    ),
)
def test_point_cloud_producer_rejects_invalid_declarations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PointCloudProducerSpec(**kwargs)
