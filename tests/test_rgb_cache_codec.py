import json

import lmdb
import numpy as np
import pytest

from visuomotor.data import cache_merge as CacheMerge
from visuomotor.data.core import images as Images
from visuomotor.data.mimicgen import observations as Observations
from visuomotor.environment import dataset_playback as Playback
from visuomotor.environment.robomimic import robomimic_setup as Setup


def _write_cache(path, codec, *, resolution=84, seed=0):
    path.mkdir()
    frames = np.random.default_rng(seed).integers(
        0, 256, (2, resolution, resolution, 3), dtype=np.uint8
    )
    lowdim = {
        "robot0_eef_pos": np.zeros((2, 3), dtype=np.float32),
        "robot0_eef_rot": np.tile(np.eye(3).reshape(1, 9), (2, 1)).astype(np.float32),
        "robot0_gripper_qpos": np.zeros((2, 2), dtype=np.float32),
    }
    meta = {
        "cache_format": "lmdb_npz_v1", "image_size": resolution,
        "rgb_keys": ["agentview_image"], "lowdim_keys": list(lowdim),
        "episode_lengths": [2], "n_samples": 2, "n_demo": 1,
        "source_demo_indices": [seed], **codec.metadata(),
    }
    (path / "meta.json").write_text(json.dumps(meta))
    arrays = {f"lowdim/{key}": value for key, value in lowdim.items()}
    arrays.update({
        "action/absolute_action": np.concatenate([
            lowdim["robot0_eef_pos"], lowdim["robot0_eef_rot"],
            np.zeros((2, 1), dtype=np.float32),
        ], axis=1),
        "lowdim/task_embedding": np.zeros((1, 2), dtype=np.float32),
        "lowdim/task_language_tokens": np.zeros((1, 2, 3), dtype=np.float32),
        "lowdim/robot_id": np.zeros(1, dtype=np.int64),
        "task_instructions": np.array(["assemble"]),
    })
    np.savez(path / "arrays.npz", **arrays)
    with lmdb.open(str(path / "images.lmdb"), subdir=False, map_size=2**22) as env:
        with env.begin(write=True) as txn:
            for index, frame in enumerate(frames):
                txn.put(f"agentview_image/{index:08d}".encode(), codec.encode(frame))
            txn.put(b"__len__", b"2")
    (path / "build_done.flag").touch()
    return frames


@pytest.mark.parametrize("resolution", [84, 256])
def test_lossless_codec_preserves_all_pixels_and_accepts_noncontiguous_input(resolution):
    frame = np.random.default_rng(1).integers(
        0, 256, (resolution, resolution, 3), dtype=np.uint8
    )[::-1]
    codec = Images.RGBCodec()
    decoded = codec.decode(memoryview(codec.encode(frame)), render_resolution=resolution)
    np.testing.assert_array_equal(decoded, np.moveaxis(frame, -1, 0))
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous


def test_lossless_rollout_does_not_call_any_codec(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("lossless rollout must not compress or decompress")

    monkeypatch.setattr(Images.imagecodecs, "blosc_encode", forbidden)
    monkeypatch.setattr(Images.imagecodecs, "blosc_decode", forbidden)
    frame = np.full((84, 84, 3), [255, 31, 0], dtype=np.uint8)
    actual = Images.canonical_rgb_from_source(frame, load_resolution=None)
    np.testing.assert_array_equal(actual, np.moveaxis(frame, -1, 0))


def test_codec_rejects_unknown_metadata_and_wrong_frame_size():
    with pytest.raises(ValueError, match="re-render"):
        Images.RGBCodec.from_metadata({"rgb_codec": "unknown"})
    codec = Images.RGBCodec()
    encoded = codec.encode(np.zeros((84, 84, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="byte count"):
        codec.decode(encoded, render_resolution=256)
    with pytest.raises(ValueError, match="HWC uint8"):
        codec.encode(np.zeros((84, 84, 3), dtype=np.float32))


def test_lmdb_read_playback_and_rollout_setup_require_lossless_codec(tmp_path):
    codec = Images.RGBCodec()
    cache = tmp_path / "cache"
    frames = _write_cache(cache, codec)
    resolution = Setup._load_cache_rgb_resolution(
        dataset_path=str(cache), cache_dir=str(cache)
    )
    assert resolution == 84
    adapter = Observations.MimicGenObservationAdapter(
        shape_meta={"obs": {"agentview_image": {"type": "rgb", "shape": [3, 84, 84]}}},
        cache_dir=str(cache), image_size=None, lmdb_readahead=False,
    )
    try:
        actual = adapter.read([1, 0])["rgb_external"]
        expected = np.stack([
            Images.canonical_rgb_from_source(frame, load_resolution=None)
            for frame in frames[::-1]
        ])
        np.testing.assert_array_equal(actual, expected)
        playback = object.__new__(Playback.DatasetPlayback)
        playback.meta = adapter.meta
        playback.rgb_codec = adapter.rgb_codec
        bgr = playback._decode_cached_frame(adapter._lmdb_txn, "agentview_image", 1)
        np.testing.assert_array_equal(bgr, np.moveaxis(expected[0], 0, -1)[..., ::-1])
    finally:
        adapter._lmdb_txn.abort()
        adapter._lmdb_env.close()


def test_merge_preserves_codec_and_compressed_frame_bytes(tmp_path):
    codec = Images.RGBCodec()
    inputs = [tmp_path / "first", tmp_path / "second"]
    for seed, path in enumerate(inputs):
        _write_cache(path, codec, seed=seed)
    output = tmp_path / "merged"
    CacheMerge.merge_caches(inputs, output, n_demo_per_input=None, delta_horizons=[])
    meta = json.loads((output / "meta.json").read_text())
    assert Images.RGBCodec.from_metadata(meta) == codec
    assert meta["episode_lengths"] == [2, 2]
    with lmdb.open(str(output / "images.lmdb"), subdir=False, readonly=True) as merged:
        with merged.begin() as dst:
            for episode, path in enumerate(inputs):
                with lmdb.open(str(path / "images.lmdb"), subdir=False, readonly=True) as src:
                    with src.begin() as txn:
                        for index in range(2):
                            assert dst.get(f"agentview_image/{2 * episode + index:08d}".encode()) == txn.get(
                                f"agentview_image/{index:08d}".encode()
                            )


def test_merge_rejects_incompatible_rgb_sizes(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    codec = Images.RGBCodec()
    _write_cache(first, codec)
    _write_cache(second, codec, resolution=256)
    with pytest.raises(ValueError, match="same RGB"):
        CacheMerge.merge_caches([first, second], tmp_path / "merged", n_demo_per_input=None)


@pytest.mark.parametrize("codec_name", [None, "jpeg", "unknown"])
@pytest.mark.parametrize("consumer", ["dataset", "rollout", "playback", "merge"])
def test_consumers_reject_missing_or_nonlossless_codec_metadata(tmp_path, codec_name, consumer):
    cache = tmp_path / "cache"
    _write_cache(cache, Images.RGBCodec())
    meta_path = cache / "meta.json"
    meta = json.loads(meta_path.read_text())
    if codec_name is None:
        meta.pop("rgb_codec")
    else:
        meta["rgb_codec"] = codec_name
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="re-render"):
        if consumer == "dataset":
            Observations.MimicGenObservationAdapter(
                shape_meta={"obs": {"agentview_image": {"type": "rgb"}}},
                cache_dir=str(cache), image_size=None, lmdb_readahead=False,
            )
        elif consumer == "rollout":
            Setup._load_cache_rgb_resolution(dataset_path=str(cache), cache_dir=str(cache))
        elif consumer == "playback":
            Playback.DatasetPlayback(str(cache), use_obs=True, use_actions=False, show_window=False)
        else:
            valid = tmp_path / "valid"
            _write_cache(valid, Images.RGBCodec())
            CacheMerge.merge_caches([valid, cache], tmp_path / "merged", n_demo_per_input=None)
