# Development Guide

This document explains where changes belong in Visuomotor Stack.

Architecture ownership is defined in `docs/architecture.md`.

## Package ownership

```text
action_generation/  action prediction mechanisms
config/             experiment selection and construction
data/               adapters, datasets, observation boundaries
environment/        simulator interaction and rollout
geometry/           spatial and SE(3) operations
perception/         visual feature extraction
policy/             policy composition and runtime contracts
visualization/      pure rendering and compressed experiment artifacts
workspace/          training lifecycle orchestration
```

## Where should my change go?

Add a benchmark:

```text
data/<benchmark>
environment/<benchmark>
```

Keep benchmark-specific names inside adapters.

Add an observation field:

```text
adapter
ObservationContract
normalizer
```

Add an encoder:

```text
perception/
config/encoder/
```

Add a policy family:

```text
policy/
action_generation/
config/policy/
```

Add experiment choices:

```text
config/
```

Add geometry functionality:

```text
geometry/
```

Add a visualization output:

```text
visualization/
```

## Development rules

- Configuration selects experiments; runtime receives typed specifications.
- Encoders consume canonical observations only.
- Geometry operations belong in the shared geometry package.
- Workspaces orchestrate lifecycle but do not own component construction.
- Avoid adding parallel abstractions when an existing ownership boundary exists.

## Verification

Run focused tests first.

For structural changes run architecture tests.

Keep active engineering state in `IMPLEMENTATION.md` rather than creating
separate trackers.
