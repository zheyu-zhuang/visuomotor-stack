# Implementation State

Only `Current Focus` defines active structural work. Stable architecture belongs
in `docs/architecture.md`, contributor rules in `AGENTS.md`, and execution
history in git commits.

## Current Focus

No active implementation task.

## Completed Summary

### Lossless RGB cache encoding and rollout render skipping

- New LMDB RGB caches declare `rgb_codec: blosc_zstd` and store each HWC uint8 frame losslessly with Blosc Zstd level 3, typesize 1, no shuffle, and one codec thread. `imagecodecs==2022.9.26` is a runtime dependency. Generation, merge, dataset reads, playback, and rollout setup enforce the declaration; existing JPEG caches require re-rendering. Merge also requires equal RGB sizes.
- Rollout consumes direct RGB through the same resize and CHW conversion as cache reads. The Robosuite patch caches each camera's RGB-D pair while skipped control steps keep observable sampling clocks advancing. The wrapper renders fresh RGB on video-due steps and skips discarded visual processing. The multistep wrapper renders the retained observation window, including extra history and truncated chunks. The installed Robosuite dependency must carry the patch.
- Verification: `OMP_NUM_THREADS=2 NUMBA_CACHE_DIR=/tmp/vmstack-numba-cache /home/zheyu/Anaconda/envs/vmstack/bin/python -m pytest -q tests/test_rgb_cache_codec.py tests/test_canonical_parity.py tests/test_mimicgen_dataset_boundary.py tests/test_rollout_observation_gating.py tests/test_rollout_performance.py tests/test_robomimic_setup.py tests/test_dataset_rendering.py` passed 84 with one CUDA skip; `OMP_NUM_THREADS=2 NUMBA_CACHE_DIR=/tmp/vmstack-numba-cache /home/zheyu/Anaconda/envs/vmstack/bin/python -m pytest -q tests/test_architecture.py` passed 15. Scoped `ruff check`, `git diff --check`, and `git -C /home/zheyu/Codebase/visuomotor-deps/mimic/robosuite apply --reverse --check /tmp/vmstack-port/.dep/robosuite.patch` passed.

### Architecture and boundaries

- Existing top-level package domains remain stable and are enforced by
  architecture tests. Hydra/OmegaConf stop at configuration resolution;
  runtime code consumes typed specs or explicit values.
- `ObservationContract` remains the source of truth for model-facing tensors.
  Source conversion, canonical representation, preparation, and normalization
  remain separate layers.
- Spatial and SE(3) math belongs to `visuomotor.geometry`; reflection remains
  separate. Perception follows `backbone -> focus -> encoder`, with
  `EncoderOutput` as the policy-facing boundary.

### Public CLI

- Observation-cache creation is exposed as `vmstack data prepare`. The former
  `generate-observations` subcommand is retired without a compatibility alias;
  its arguments and cache-generation behavior are unchanged.
- CLI help, cache recovery guidance, RVT-2 guidance, and operational
  documentation use the new command name.

### Public policy surface

- Public and private development now use separate repositories. The public
  repository starts from a clean root commit and contains no private
  experimental policy implementation, configuration, runtime support, tests,
  documentation, or prior history.
- Public recipes cover RGB, Seeker, voxel, global diffusion, and global flow.
  Launchers compose defaults directly and do not compose other launchers.
- Policy generators use `unet_channels` for public UNet widths. Global flow
  uses explicit integration steps; the low-level UNet retains its internal
  `down_dims` parameter.
- W&B projects are generator-owned: `vm-global-flow` and
  `vm-global-diffusion`. Tasks are groups and encoder names identify runs.

### Data and runtime

- Source, cached-absolute, and model action contracts are 7D, 13D, and 10D.
- Cached observations remain canonical uint8 or sparse data; model voxels
  contain binary occupancy and occupied RGB in `[0, 1]`.
- Dataset generation and rollout share spatial producer specs. World voxels use
  a fixed 0.7 m floor; point clouds remove a 5 mm tabletop slab before
  deterministic sampling.
- Rollout accepts release artifacts and workspace checkpoints containing typed
  run specs, prefers EMA weights, and supports explicit raw weights.
- Checkpoints, logs, metrics, and media inherit the configured run root.

### Verification

- Sanitized public release: 380 tests passed and 1 CUDA-only test skipped with
  `NUMBA_DISABLE_JIT=1`; 152 focused tests passed. Ruff, compileall,
  `git diff --check`, and private-surface searches passed.
- Data preparation CLI rename: `conda run -n vmstack pytest -q
  tests/test_architecture.py` passed 15 tests; the focused CLI and dataset
  boundary command passed 16 tests. Direct parser smoke checks, retired-name
  searches, and `git diff --check` passed.
