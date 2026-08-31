from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visuomotor.config import schema as Schema
from visuomotor.environment.gym_wrappers.video_recording_wrapper import VideoRecorder
from visuomotor.geometry.grid import (
    FeatureGridGeometry,
    SourceVoxelGeometry,
    VoxelCropTransform,
)
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.visualization import diagnostics as Diagnostics
from visuomotor.visualization import rendering as Rendering
from visuomotor.visualization import rollout as Rollout
from visuomotor.visualization import rollout_media as RolloutMedia
from visuomotor.visualization.artifacts import (
    ArtifactRecord,
    ArtifactStore,
    allocate_rollout_output,
    publish_artifacts,
)


def test_visualization_config_is_local_first_and_rejects_unsaved_uploads():
    spec = Schema.VisualizationSpec()
    assert spec.save.images and spec.save.videos
    assert not spec.upload.images and not spec.upload.videos
    with pytest.raises(ValueError, match="upload images"):
        Schema.VisualizationSpec(
            save=Schema.MediaToggleSpec(images=False, videos=True),
            upload=Schema.MediaToggleSpec(images=True, videos=False),
        )
    with pytest.raises(ValueError, match="upload videos"):
        Schema.VisualizationSpec(
            save=Schema.MediaToggleSpec(images=True, videos=False),
            upload=Schema.MediaToggleSpec(images=False, videos=True),
        )


def test_rollout_start_frames_accept_compact_worker_rgb():
    images = np.arange(2 * 3 * 2 * 2, dtype=np.uint8).reshape(1, 2, 3, 2, 2)
    frames = RolloutMedia.extract_start_frames({"camera": images}, "camera")
    np.testing.assert_array_equal(frames, np.moveaxis(images[:, 0], 1, -1))
    assert not np.shares_memory(frames, images)


def test_artifact_store_uses_stable_webp_and_rollout_paths(tmp_path):
    store = ArtifactStore(tmp_path)
    image_path = store.eval_image("focus", epoch=3, step=17)
    record = store.save_image(
        np.zeros((31, 45, 3), dtype=np.uint8),
        image_path,
        key="media/eval/focus",
    )
    assert image_path == tmp_path / "media/eval/focus/epoch_0003_step_00000017.webp"
    assert record.path == image_path
    assert Image.open(image_path).format == "WEBP"
    assert not image_path.with_suffix(".png").exists()
    assert store.rollout_video(epoch=4, split="test", seed=8).name == "test_seed_8.mp4"
    assert store.rollout_summary(epoch=4) == tmp_path / "media/rollout/epoch_0004/summary.webp"


def test_disabled_local_media_creates_nothing(tmp_path):
    store = ArtifactStore(tmp_path, save_images=False, save_videos=False)
    assert store.save_image(
        np.zeros((8, 8, 3), dtype=np.uint8),
        store.training_image(),
        key="media/training/preview",
    ) is None
    assert not (tmp_path / "media").exists()


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step=None):
        self.calls.append((payload, step))


def test_upload_toggles_are_independent_and_choose_one_video(tmp_path):
    image_path = tmp_path / "image.webp"
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    image_path.touch()
    video_a.touch()
    video_b.touch()
    records = [
        ArtifactRecord("media/eval/focus", "image", image_path),
        ArtifactRecord("media/rollout/video", "video", video_b),
        ArtifactRecord("media/rollout/video", "video", video_a),
    ]
    run = _FakeRun()
    publish_artifacts(
        run,
        records,
        upload_images=True,
        upload_videos=False,
        image_factory=lambda path, caption="": (path, caption),
        video_factory=lambda *args, **kwargs: pytest.fail("video factory called"),
    )
    assert tuple(run.calls[0][0]) == ("media/eval/focus",)

    run = _FakeRun()
    publish_artifacts(
        run,
        records,
        upload_images=False,
        upload_videos=True,
        image_factory=lambda *args, **kwargs: pytest.fail("image factory called"),
        video_factory=lambda path, **kwargs: (Path(path).name, kwargs),
    )
    assert run.calls[0][0]["media/rollout/video"][0] == "a.mp4"


def test_fixed_cohort_and_diagnostics_restore_rng_and_mode():
    assert Diagnostics.evenly_spaced_indices(10, 4) == (0, 3, 6, 9)
    model = torch.nn.Linear(2, 2)
    model.train()
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    with Diagnostics.isolated_evaluation(model, seed=99):
        assert not model.training
        _ = torch.rand(20)
    assert model.training
    assert torch.equal(torch.rand(3), expected)


def test_rgb_voxel_point_cloud_and_action_renderers_are_deterministic():
    rgb = {"rgb_external": torch.zeros(2, 1, 3, 16, 16, dtype=torch.uint8)}
    first = np.asarray(Rendering.render_observations(rgb, num_samples=2))
    second = np.asarray(Rendering.render_observations(rgb, num_samples=2))
    assert np.array_equal(first, second)

    voxel = torch.zeros(1, 1, 4, 8, 8, 8, dtype=torch.uint8)
    voxel[:, :, 0, 3, 3, 3] = 1
    voxel[:, :, 1:, 3, 3, 3] = 255
    assert Rendering.render_observations({"voxel": voxel}).size[0] > 0

    points = torch.tensor([[[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]]]])
    assert Rendering.render_observations({"point_cloud": points}).size[0] > 0
    actions = torch.zeros(1, 4, 10)
    assert Rendering.render_action_comparison(actions, actions).size[0] > 0


def test_rgb_observation_grid_uses_one_unlabelled_row_per_view():
    observations = {
        "rgb_external": torch.zeros(3, 1, 3, 16, 20, dtype=torch.uint8),
        "rgb_wrist": torch.full((3, 1, 3, 16, 20), 255, dtype=torch.uint8),
    }

    grid = Rendering.render_rgb_observations(observations, num_samples=3)

    assert grid.height == 2 * 16 + 3 * 6
    assert grid.width > 3 * 20


def test_diagnostic_sections_keep_their_natural_heights():
    first = Image.new("RGB", (80, 20))
    second = Image.new("RGB", (40, 100))

    report = Rendering.stack_sections([("Input", first), ("Encoder", second)])

    assert report.height == 8 + (18 + 20 + 8) + (18 + 100 + 8)
    assert report.width == 80 + 16


def test_generic_rollout_payload_hud_and_outcome_filename():
    payload = RolloutMedia.extract_rollout_diagnostics(
        {"diagnostics": {"focus": ()}},
        action_positions=np.zeros((1, 4, 3), dtype=np.float32),
        eef_positions=np.zeros((1, 3), dtype=np.float32),
        n_envs=1,
    )[0]
    payload["seed"] = 12
    state = Rollout.RolloutDiagnosticState()
    state.update(payload)
    state.observe(np.array([0.01, 0.0, 0.0]))
    current = state.current()
    frame = np.zeros((101, 99, 3), dtype=np.uint8)
    hud = Rollout.draw_hud(frame, current)
    assert hud.shape[0] > frame.shape[0]
    assert hud.shape[1:] == frame.shape[1:]
    assert np.array_equal(hud[: frame.shape[0]], frame)
    assert np.any(hud[frame.shape[0] :] != (16, 20, 24))
    assert RolloutMedia.rollout_video_filename(
        prefix="test/", seed=12, outcome=True
    ) == "test_seed_12_success.mp4"


def test_rollout_grid_reserves_footer_for_long_labels():
    frame = np.full((48, 96, 3), 120, dtype=np.uint8)
    grid = RolloutMedia.make_image_grid(
        [frame],
        success_flags=[True],
        labels=["START an excessively long split name seed 10000 SUCCESS"],
    )

    assert grid.shape[0] > frame.shape[0]
    assert grid.shape[1] == frame.shape[1]
    assert np.array_equal(
        grid[24 : frame.shape[0] - 4, 4:-4], frame[24:-4, 4:-4]
    )
    assert np.any(grid[frame.shape[0] :] != (26, 29, 34))


def test_action_diagnostics_overlay_trajectories_on_voxel_triview():
    batch, steps, size = 2, 8, 16
    identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    target = torch.zeros(batch, steps, 10)
    target[..., :3] = torch.linspace(-0.2, 0.2, steps)[None, :, None]
    target[..., 3:9] = identity_rot6d
    predicted = target.clone()
    predicted[..., 1] += 0.04
    predicted[..., -1] = torch.linspace(-1, 1, steps)
    voxel = torch.zeros(batch, 1, 4, size, size, size, dtype=torch.uint8)
    voxel[:, :, 0, 4:12, 4:12, 4:12] = 1
    voxel[:, :, 1:, 4:12, 4:12, 4:12] = 160

    image = Rendering.render_action_comparison(
        predicted,
        target,
        num_samples=batch,
        action_rep="absolute",
        observations={"voxel": voxel},
        voxel_geometry=SourceVoxelGeometry(
            (-0.5, -0.5, -0.5), 1.0, (size, size, size)
        ),
    )

    assert image.size == (48 + batch * 180 + 5, 43 + 3 * 180 + 3 * 5 + 96 + 32)
    colors = np.asarray(image).reshape(-1, 3)
    assert np.any(np.all(colors == (80, 220, 100), axis=1))
    assert np.any(np.all(colors == (255, 105, 90), axis=1))


def test_focus_pool_3d_diagnostics_overlay_attention_on_voxel_triview():
    batch, heads, source_size, crop_size = 2, 4, 10, 8
    voxel = torch.zeros(batch, 1, 4, source_size, source_size, source_size, dtype=torch.uint8)
    voxel[:, :, 0, 2:8, 2:8, 2:8] = 1
    voxel[:, :, 1:, 2:8, 2:8, 2:8] = 120
    attention = torch.zeros(batch, 1, heads, 2, 2, 2)
    attention[:, :, :, 1, 1, 1] = 1
    crop_transform = VoxelCropTransform(
        starts=torch.ones(batch, 3),
        source_shape=(source_size,) * 3,
        crop_shape=(crop_size,) * 3,
    )
    output = EncoderOutput(
        features=torch.zeros(batch, 1),
        attention=attention,
        attention_geometry=FeatureGridGeometry.from_stride(
            (crop_size,) * 3, (2,) * 3, stride=4
        ),
        voxel_crop_geometry=SourceVoxelGeometry(
            (-0.5, -0.5, -0.5), 1.0, (source_size,) * 3
        ),
        voxel_crop_transform=crop_transform,
    )

    image = Rendering.render_focus_diagnostics(
        output,
        num_samples=batch,
        observations={
            "voxel": voxel,
            "eef_pos": torch.zeros(batch, 1, 3),
        },
        targets={
            "focus_target_pos": torch.zeros(batch, 3),
            "focus_target_valid": torch.ones(batch, dtype=torch.bool),
        },
    )

    assert image.size == (
        24 + batch * crop_size * 3 + (batch - 1) * 4,
        16 + 3 * crop_size * 3 + 2 * 4 + 18,
    )
    colors = np.asarray(image).reshape(-1, 3)
    assert np.any(np.all(colors == (255, 48, 220), axis=1))
    assert np.any(np.all(colors == (255, 255, 255), axis=1))


def test_standalone_rollout_output_is_numbered(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "policy.pth"
    checkpoint.parent.mkdir(parents=True)
    first = allocate_rollout_output(checkpoint)
    second = allocate_rollout_output(checkpoint)
    assert first.name == "run_rollout_0001"
    assert second.name == "run_rollout_0002"


def test_h264_video_is_crf28_yuv420p_even_and_has_no_frame_dump(tmp_path):
    import av

    path = tmp_path / "rollout.mp4"
    recorder = VideoRecorder.create_h264(fps=10, crf=28)
    assert recorder.kwargs["pix_fmt"] == "yuv420p"
    assert recorder.kwargs["options"]["crf"] == "28"
    recorder.start(str(path))
    recorder.write_frame(np.zeros((101, 99, 3), dtype=np.uint8))
    recorder.stop()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (98, 100)
        assert float(stream.average_rate) == 10
    assert list(tmp_path.glob("*.png")) == []
