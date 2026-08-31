# Data Boundary

This document defines how observations move through Visuomotor Stack.

`docs/architecture.md` is authoritative for the observation and action
contracts; this document is the short orientation view.

Stable ownership:

```text
benchmark source
    -> benchmark adapter
        -> canonical representation
            -> preparation / augmentation
                -> normalization
                    -> model
```

## Ownership

Each layer has one responsibility.

Benchmark adapter owns:

- source observation names
- simulator/dataset-specific layouts
- source-to-canonical conversion

Canonical data owns:

- semantic observation names
- tensor contracts
- modality definitions

Normalization owns:

- conversion from canonical values to model inputs
- visual dequantization
- numeric scaling

Encoders and policies consume canonical/model representations only. They do not
know benchmark-specific keys or repair invalid inputs.

## Canonical observation contract

```text
RGB:     uint8, CHW, RGB, [0,255]
point cloud: float32, [points, XYZRGB]
voxel:   uint8 world-frame volumetric representation
proprio: float32 physical state
action:  position + rotation_6d + gripper
```

Canonical representation is stable across datasets and environments.

### Derived proprioception

`eef_delta_pos`, `eef_delta_rotvec`, and `gripper_qpos_delta` are differenced
from a pair of consecutive frames rather than read from a source field. They
give a policy the recent motion without widening the observation horizon, which
would re-encode the visual stream for every extra step.

The pose delta is expressed in the previous frame's own body frame, so it is
invariant to the world frame and scene-yaw augmentation leaves it alone. The
rotation delta is a rotation vector, not rot6d: over one control step it stays
far from the antipodal wrap that motivates rot6d for absolute orientation, and
its three components share one fitted scale where six do not. All three are
fitted low-dim fields.

`visuomotor.data.core.observations.proprio_deltas` is the only definition. The
dataset differences the whole cache once and indexes the result; a rollout
differences the extra frame the step wrapper already retains, which is a real
control step because only visual rendering is skipped between replans. Both
zero the delta at an episode's first frame -- the cache clamps it, and the
wrapper pads a reset with the first observation. `tests/test_proprio_deltas.py`
runs both schedulings over one trajectory and asserts they agree.

Mirror augmentation rejects these fields: a reflection is not a rotation, and
the augmenter reflects poses and actions only.

Point-cloud production crops to the task workspace and removes the 5 mm slab
above the tabletop before farthest-point sampling. The margin is recorded in
cache metadata so older caches cannot silently enter DP3 training.

## Observation naming

Canonical names describe semantic views:

```text
rgb_external
rgb_wrist
point_cloud
voxel
```

Simulator names remain inside adapters:

```text
agentview
robot0_eye_in_hand
```

Do not expose source names downstream.

## Normalization boundary

The transition is:

```text
canonical representation
        ->
model representation
```

Normalization owns:

- RGB/visual conversion
- fitted low-dimensional scaling
- fitted action scaling

It does not own:

- source conversion
- channel inference
- layout repair

## Adding a new modality

Define:

1. source representation
2. canonical representation
3. validation rules
4. normalization behaviour
5. model contract changes

Keep source-specific logic inside the adapter boundary.
