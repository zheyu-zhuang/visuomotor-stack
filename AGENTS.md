# Agent Rules

Visuomotor Stack is the infrastructure for RGB, point-cloud, and voxel visuomotor manipulation policies.

## Reading order

Always:
1. Read `AGENTS.md`.
2. Read `IMPLEMENTATION.md` for structural changes.
3. Read `docs/architecture.md` only when architectural context is required.

## Working style

- Inspect before editing.
- Prefer the simplest implementation satisfying the task.
- Make surgical changes; do not absorb adjacent cleanup.
- Do not add speculative abstractions, compatibility layers, or configurability.
- Remove obsolete code created by your own changes.
- Record only structural follow-up work in `IMPLEMENTATION.md`.
- No note-to-self or narrative comments. Comments should be simple, explicit, and critical.
- Prefer compact module imports with qualified references over large symbol imports.

## Architecture rules

- Keep existing top-level package domains; do not introduce new domains casually.
- Configuration selects experiments; runtime code receives typed specs or explicit values.
- Hydra/OmegaConf stay at configuration boundaries.
- `ObservationContract` is the source of truth for model-facing tensors.
- Source conversion, canonical representation, and model normalization are separate layers.
- Spatial and SE(3) math belongs in `visuomotor.geometry`.
- Reflection is separate from SE(3).
- Do not reintroduce retired names or config routes rejected by architecture tests.

## Task progression

`IMPLEMENTATION.md` is the authoritative record for structural work only.
Structural changes include package or module layout, ownership boundaries,
cross-layer contracts, configuration/runtime boundaries, and persistent data
formats. Do not update it for localized fixes, cleanup, comments, display
changes, tests, or parameter tuning.

Before structural coding:

1. Define the goal.
2. Define primary search scope.
3. Define dependencies that may be followed if required.
4. Define targeted searches and stop conditions.
5. Define non-goals.
6. Define implementation steps and verification.

During structural implementation:

- Stay inside declared scope unless direct dependencies require expansion.
- Update the active task block in place.
- Do not create session logs, handoff documents, or parallel trackers.

When structural work is completed:

1. Move the task from `Current Focus` into `Completed Summary`.
2. Remove execution history, debugging traces, and temporary notes.
3. Preserve only:
   - what changed,
   - architectural impact,
   - introduced constraints,
   - verification status.
4. Create a new `Current Focus` only if another task is ready.

Completed items are constraints and context, not active work.

When finished:

- Run targeted tests first.
- Run architecture tests for structural changes.
- Record actual verification commands and outcomes.
