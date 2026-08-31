# Visuomotor Stack Architecture

This document defines stable system structure. Active engineering work belongs in `IMPLEMENTATION.md`; operational usage belongs in `docs/workflows.md`.

## System structure

Visuomotor Stack is organised into nine domains:

```text
action_generation/  action prediction mechanisms
config/             experiment selection and construction
data/               datasets and observation boundaries
environment/        simulator interaction and rollout
geometry/           spatial and SE(3) operations
perception/         representation and feature extraction
policy/             policy composition and runtime contracts
visualization/      pure rendering and compressed experiment artifacts
workspace/          training lifecycle orchestration
```

The ownership rule is:

```text
config/workspace
        -> runtime construction

environment
        -> policy
             -> perception

environment/workspace
        -> visualization

runtime domains
        -> data.core / geometry
```

Runtime modules consume typed declarations. They do not discover configuration or depend on benchmark-specific implementation details.

---

## Configuration boundary

The configuration flow is:

```text
Hydra YAML
    -> config/resolve.py
        -> typed specifications
            -> config/build.py
                -> runtime objects
```

Configuration selects experiments. Builders construct runtime objects. Workspaces orchestrate lifecycle only.

Policy generator configuration uses `unet_channels` for UNet stage widths.
Flow solver counts use `integration_steps`.

Runtime code must not:

- parse Hydra/OmegaConf trees
- call configuration builders
- use configuration as a service locator

Derived state should be computed from typed specifications rather than duplicated as configuration.

---

## Experiment composition

Experiments are composed from independent roles:

```text
input + encoder + policy + regime
```

`input`
- selects observations
- defines canonical fields

`encoder`
- consumes canonical observations
- defines representation and feature extraction

`policy`
- defines action prediction and control family

`regime`
- defines experimental conditions such as augmentation

Compatibility is determined by declared requirements and capabilities, not preset-specific pairing rules.

Policy configs define reusable model families. Non-launch
`*_training_defaults.yaml` configs own reusable training selections. Executable
`train_*.yaml` configs compose those defaults and explicitly select their policy;
ablation launchers own experiment-local policy overrides, and one launcher never
composes another launcher.

Shared trajectory structure is owned by:

```text
TrajectoryContract
    -> ModelSpec
    -> DatasetSpec
    -> RunnerSpec
```

---

## Data boundary

The observation path is:

```text
benchmark source
    -> benchmark adapter
        -> canonical representation
            -> preparation / augmentation
                -> normalization
                    -> model
```

Ownership:

- Benchmark adapters own source names and simulator semantics.
- `data/core/observations.py` owns canonical semantic fields, source-to-canonical
  conversion, and canonical validation for both dataset and rollout backends.
- `data/core/images.py` owns the RGB cache codec, so a rendered frame becomes a
  canonical tensor by exactly one route.
- Normalization owns canonical-to-model conversion.
- Perception validates model-space inputs and never converts source layouts.
- Models consume contracts, not source layouts.

The canonical policy-facing contract includes:

```text
RGB:     uint8, CHW, RGB, [0,255]
point cloud: float32, [points, XYZRGB]
voxel:   uint8 world-frame volumetric representation
proprio: float32 physical state
action:  position + rotation_6d + gripper
```

A rendered frame reaches the policy with the same bytes whether it arrives
from the dataset cache or from a live rollout:

```text
simulator frame  (camera render resolution, from the cache's `image_size`)
    -> JPEG encode at the cache's quality
        -> JPEG decode at the encoder's RGB load resolution
            -> canonical uint8 CHW
```

Training splits that codec across the cache write and the cache read; rollout
runs both halves inline. Rollout therefore renders at the resolution the cache
was built from, and loads at the resolution the encoder was trained on --
neither is chosen independently. Voxel and point-cloud grids are fused from the
same RGB-D streams at the same reconstruction resolution, and sparse cache
transport is lossless, so they match without a codec of their own.

Released Seeker checkpoints transport native RGB through that codec, then apply
their trained bilinear resize to 224 inside the focus preparation boundary.

MimicGen action conversion is exact at each boundary:

```text
source / environment: xyz + axis-angle + gripper command       [7]
absolute cache:       xyz + rotation matrix + gripper command  [13]
model:                xyz + rotation_6d + gripper command      [10]
```

Delta cache actions use the same 10 model fields per horizon step, with pose
expressed in the observation frame's end-effector coordinates. Measured
`gripper_qpos` remains observation state and is never appended to an action.

For keypose targets, every opening or closing command transition is mandatory.
Measured finger motion relocates that event to the first stable window after
the response: motion onset is an opening change above `5e-4`, and stability is
four consecutive changes at or below `2e-4`. These defaults separate response
from the upper tail of stable motion measured on StackThree D1. If no response
window fits before the next command, the event falls back to the end of its own
command interval and is not dropped. Targets advance at the keypose. A
keypose-targeted action chunk retains demonstrated actions through its target,
then repeats the target-indexed endpoint for the rest of the fixed horizon. It
never mixes actions governed by the following segment.
Non-keypose action chunks retain the sampled dataset trajectory unchanged.

Canonical names describe semantic views:

```text
rgb_external
rgb_wrist
voxel
point_cloud
```

Simulator names such as `agentview` and `robot0_eye_in_hand` remain inside adapters.

Voxel and point-cloud production is selected by typed producer specs owned by
`data/core/spatial.py`. The shared declarations own intrinsic validation and
cache metadata; configuration resolves directly to them, while environment
adapters own their binding to simulator arguments. Voxel specs are keyed by
canonical output, so one observation can contain grids with different frames,
bounds, and resolutions. Dataset generation and rollout receive the same
ordered camera tuple, reconstruction resolution, workspace size, bounds, output
resolution, channel layout, and frame for every key. One fused RGB-D
reconstruction feeds all selected spatial producers. The world grid retains its
existing voxelization. The local grid subtracts the current EEF translation
from reconstructed world points and applies symmetric bounds; its axes stay
aligned with the world and do not depend on EEF rotation.

Voxelization bins the fused cloud in numpy. It reproduces Open3D's
`CreateFromPointCloudWithinBounds` indexing and its per-voxel mean colour
exactly -- the same inflated `voxel_size`, the same `[0, 1]` colour
accumulation -- because existing spatial caches and trained checkpoints were
produced against that path. Open3D returned one Python object per occupied
voxel, which dominated the per-control-step cost. The Open3D point cloud is now
built only for the point-cloud observation that needs it.

Single-grid caches retain the legacy `voxel_spec` and sparse array paths.
Multi-grid caches record `voxel_specs`, sparse storage, and maximum occupied-cell
counts per key; each key has independent sparse index, colour, and offset
arrays. Model-facing loaders and rollout setup validate exact per-key producer
metadata. Point-cloud padding is deterministic, so generation and rollout
cannot diverge merely because a crop has too few points.

Parallel rerender workers materialize only one simulator observation at a time.
They retain episode-local JPEG bytes, sparse voxel cells, point-cloud bytes, and
low-dimensional arrays until success is known; dense observations and unused
`next_obs` trajectories are never accumulated. Native thread pools and CPU
affinity are bounded per worker, host CPUs remain reserved, intermediate shards
omit derived delta actions, and simulator/EGL resources are closed explicitly
before shard merge.

Dataset rerendering keeps its public entrypoint in `dataset_rendering.py` and
places its implementation in the private `_dataset_rendering` package:
`renderer.py` owns simulator construction and frame capture, `cache.py` owns
LMDB and NumPy serialization, `orchestration.py` owns dataset discovery,
workers, and shard merging, and `common.py` owns their shared contracts and
utilities. Oracle metadata retains a frame-aligned world-to-pixel matrix for
every rendered RGB camera.

---

## Normalization boundary

Normalization is the only transition from canonical data to model space.

It owns:

- visual dequantization
- RGB/voxel scaling
- fitted low-dimensional normalization
- fitted action normalization

It does not repair:

- source naming
- channel layout
- invalid canonical values
- missing conversions

Observation state and action semantics remain separate (`gripper_qpos` versus gripper action).

Fitted low-dimensional and global action fields use per-axis affine min/max
maps. All fitted state is checkpoint state.

The composed `augmentation/defaults.yaml` owns RGB crop, voxel crop,
image-overlay, mirror, and scene-yaw defaults. Overlay method selection remains paired with each focus
encoder's per-view/per-regime transform because mask-guided and random overlays
require different image preparation.
Experiments select them under `input_augmentation` using only `enabled` or
`disabled`. A mapping
with `method` plus selected fields overrides pool values at the same boundary.
Resolution merges that declaration before constructing typed runtime specs.
Coupled scene yaw rotates the full canonical grid and matching world-space
observations and targets before normalization. A single-scale voxel encoder
applies its resolved random-train/center-eval crop to its model input.

Cached voxel fields contain only occupancy and RGB.

---

## Geometry boundary

All spatial and rigid-body operations belong in `visuomotor.geometry`.

Responsibilities:

- SE(3) composition and transforms
- rotation conversion
- projection
- voxel geometry
- workspace bounds
- reflection operations

Convention:

```text
A_R_B = rotation of B expressed in A
```

Reflection is an O(3) operation and remains separate from SE(3).

---

## Perception internal structure

`visuomotor.perception` is organised into four tiers:

```text
backbone/   raw per-view feature extraction, no focus mechanism
focus/      focus mechanisms: internal refinement primitives,
            external focus models, and per-view delegation
encoder/    observation -> EncoderOutput, the only policy-facing tier
common/     cross-domain contracts and utilities
```

`backbone/`, `focus/`, and `encoder/` are a strict layer stack, not three
coequal peers:

```text
backbone -> focus -> encoder
```

Each layer only depends on layers before it in that order — `backbone/` has
no perception-internal dependencies, `focus/` may depend on `backbone/`
(e.g. `focus/refine/stage_pooled_resnet.py` on `backbone/resnet/build.py`,
`focus/seeker/` and `focus/rvt2/` on `backbone/dinov3_core/`), and
`encoder/` may depend on both. They share tree depth under `perception/`,
not dependency rank. `common/` is the one zero-dependency cross-cutting
layer any of the three may use.

Within `focus/refine/`, files are named by mechanism spatial-rank
(`planar.py`, `volumetric.py`), not by view modality — a different axis
from `backbone/resnet/`'s `rgb.py`/`voxel.py`, which name by view modality.
The two axes correlate (planar pooling operates on RGB features, volumetric
pooling on voxel features) but are not the same concept, so `focus/refine/`
uses its own vocabulary rather than reusing `backbone/`'s.

---

## Policy and rollout boundary

`ModelSpec` combines:

```text
input contract
    + encoder
    + policy family
    + normalizer
```

Policies consume shared encoder contracts rather than benchmark-specific data.

Derived proprioception (`eef_delta_pos`, `eef_delta_rotvec`,
`gripper_qpos_delta`) is selected through `input.proprio` like any other field.
See [`data.md`](./data.md) for the contract.

Environment runners own:

- native observation retrieval
- action conversion
- rollout
- logging

`MultiStepWrapper` reads only the last `n_obs_steps` observations of an action
chunk, so the wrappers declare per-control-step whether the next observation is
read. On a step whose observation is discarded, the patched robomimic env
disables every camera observable, which skips both the offscreen renders and the
RGB-D fusion; the image wrapper hands back the last produced visual arrays so
the observation contract is unchanged. Lanes recording video keep the render
camera enabled on the steps the recorder encodes. Rewards, terminations, and
MimicGen oracle subtask signals advance on every control step regardless; only
oracle focus projection follows the observation cadence.

Dataset and rollout paths must produce equivalent canonical observations before shared normalization.

Diagnostics cross this boundary explicitly: `EncoderOutput` owns prepared
inputs, attention geometry, voxel crop geometry, and focus records;
`BaseImagePolicy.collect_diagnostics` adds predicted actions. Rollout
predictions carry rollout-safe diagnostics and runners do
not inspect mutable model attributes.

`EncoderOutput.auxiliary_losses` is the sole encoder-objective contract. Each
entry is a final weighted scalar whose encoder-owned schedule has already been
evaluated; policies add every entry exactly once.

Rollout construction crosses into the environment through an environment-owned
typed request. Configuration constructs that request from the resolved runner
spec; the runner and robomimic setup do not repeatedly flatten configuration
fields or import configuration types. Runner lifecycle helpers own chunk
initialization, policy-input preparation, diagnostic publication, artifact
finalization, and metric aggregation without changing control-step ordering.

---

## Visualization boundary

`visuomotor.visualization` owns pure observation, focus, action, and rollout
renderers plus compressed artifact storage. Workspaces and runners own cadence
and optional W&B publishing.

Local artifacts are the source of truth. Images use WebP and rollout videos use
native-resolution H.264 MP4; online publishing reuses those files. Image and
video saving and uploading are independently typed, and uploading a disabled
local media type is invalid configuration.

---

## Architectural invariants

- Runtime code does not parse Hydra/OmegaConf.
- Workspaces own lifecycle, not component construction.
- Config builders own runtime construction.
- Perception and policy do not depend on benchmark-specific packages.
- Source naming exists only in adapters/configuration.
- Canonicalization and normalization are separate stages.
- Encoders validate canonical contracts but do not convert data.
- Geometry operations use shared utilities.
- Model, dataset, and runner share one trajectory contract.
- Derived state is computed rather than duplicated.
