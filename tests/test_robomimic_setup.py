from visuomotor.environment.robomimic import robomimic_setup as RobomimicSetup


def test_create_env_registers_hidden_spatial_rgbd_streams(monkeypatch):
    registered = {}
    marker = object()

    monkeypatch.setattr(
        RobomimicSetup.ObsUtils,
        "initialize_obs_modality_mapping_from_dict",
        lambda mapping: registered.update(mapping),
    )
    monkeypatch.setattr(
        RobomimicSetup.EnvUtils,
        "create_env_from_metadata",
        lambda **_kwargs: marker,
    )

    env_meta = {
        "env_kwargs": {
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "use_voxel_obs": True,
            "voxel_cameras": [
                "birdview",
                "agentview",
                "sideview",
                "robot0_eye_in_hand",
            ],
        }
    }
    shape_meta = {
        "obs": {
            "agentview_image": {"type": "rgb"},
            "voxel": {"type": "voxel"},
            "robot0_eef_pos": {"type": "low_dim"},
        }
    }

    result = RobomimicSetup.create_env(env_meta, shape_meta)

    assert result is marker
    assert registered["low_dim"] == ["robot0_eef_pos"]
    assert registered["rgb"] == [
        "agentview_image",
        "birdview_image",
        "sideview_image",
        "robot0_eye_in_hand_image",
    ]
    assert registered["depth"] == [
        "birdview_depth",
        "agentview_depth",
        "sideview_depth",
        "robot0_eye_in_hand_depth",
    ]
    assert "voxel" not in registered
