# Visuomotor Stack

Infrastructure for RGB, point-cloud, and voxel visuomotor manipulation policies.

This repository provides shared infrastructure for datasets, observation handling, perception, policy training, environments, configuration, and rollout. Input modalities, visual encoders, focus mechanisms, and policy families are interchangeable experiment choices rather than fixed commitments.

Seeker is one supported focus mechanism among several, alongside RVT-2 heatmaps, oracle focus, and refinement-based variants. See [`docs/workflows.md`](./docs/workflows.md) for the recipes and encoder overrides that select each one.

The installed distribution is `visuomotor-stack`, the Python namespace is `visuomotor`, and the command line entry point is `vmstack`.

---

## Overview

A Visuomotor Stack run follows two connected flows:

```
Experiment definition

Hydra config
    ↓
config/resolve.py
    ↓
typed specifications
    ↓
config/build.py
    ↓
runtime objects
```

```
Observations

benchmark source
    ↓
adapter
    ↓
canonical observations
    ↓
augmentation
    ↓
normalization
    ↓
perception
    ↓
policy
    ↓
action generation
```

The key design principle is:

> Configuration selects experiments. Runtime code operates on explicit typed specifications.

Runtime components do not read Hydra configuration directly.

---

## Repository structure

The codebase is organised around clear ownership boundaries:

```
visuomotor/

├── config/              Experiment definitions, resolution, construction
├── data/                Dataset loading and observation boundaries
├── environment/         Simulator interaction and rollout
├── geometry/            Spatial, SE(3), projection, and grid operations
├── perception/          Visual representation and feature extraction
├── policy/              Policy models and runtime interfaces
├── action_generation/   Action prediction mechanisms
├── visualization/       Rendering and compressed local experiment media
└── workspace/           Training lifecycle orchestration
```

When adding functionality, modify the package that owns the responsibility rather than creating cross-domain helpers.

---

## Experiment model

Experiments are composed from four independent choices:

```
input + encoder + policy + regime
```

`input`
: Defines available observations and canonical fields.

`encoder`
: Defines how observations are transformed into features.

`policy`
: Defines the policy family and action generation.

`regime`
: Defines experimental conditions such as augmentation or domain randomization.

A complete architecture description is available in:

- [`docs/architecture.md`](./docs/architecture.md)

---

## Quickstart

### Prerequisites

- Linux with an NVIDIA GPU and a CUDA-capable driver
- Git and Mamba (or Conda)
- Access to a MimicGen dataset

The environment pins Python 3.9, PyTorch 2.1, and CUDA 11.8. Training defaults to `cuda:0`.

### 1. Clone and create the environment

```bash
git clone https://github.com/zheyu-zhuang/visuomotor-stack.git
cd visuomotor-stack
mamba env create -f conda_environment.yaml
mamba activate vmstack
```

The conda environment is the authoritative dependency specification.

### 2. Install system libraries

MuJoCo/robosuite offscreen rendering needs a few system libraries.

Option A: system packages via `sudo` (preferred)

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Option B: conda fallback, if `sudo` is unavailable

```bash
mamba install -c conda-forge glew mesalib
mamba install -c menpo glfw3
```

### 3. Set up suite dependencies and assets

```bash
vmstack setup
```

`vmstack setup` must be run from a source checkout. It clones, pins, patches, and installs the MimicGen suite dependencies from `.dep/mimicgen.lock` (that lock file, not this document, is the source of truth for suite versions), then downloads and verifies model weights, textures, and backgrounds from this repository's `assets` release and builds the task-embedding cache. By default, dependency checkouts are placed in `../visuomotor-deps/mimic/`.

It refuses to run outside the `vmstack` conda environment unless `VISUOMOTOR_CONDA_ENV` is set to another env name, or cleared to skip the check.

Useful flags: `--assets-only` (skip the suite checkouts), `--force` (restore dependency checkouts to their locked revisions, reapply patches, and re-download assets), `--skip-task-cache`, and `--suite-deps-root <path>`.

### 4. Download a raw dataset

Raw MimicGen demonstrations are hosted at
<https://huggingface.co/datasets/amandlek/mimicgen_datasets/>. Download one task
into the expected layout:

```bash
TASK=three_piece_assembly_d2

mkdir -p datasets/mimicgen/${TASK}
wget -O "datasets/mimicgen/${TASK}/${TASK}.hdf5" \
  "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/${TASK}.hdf5?download=true"
```

The expected raw layout is:

```text
datasets/
  mimicgen/
    <task_name>/
      <task_name>.hdf5
```

Task metadata ([`visuomotor/data/mimicgen/tasks.py`](./visuomotor/data/mimicgen/tasks.py)) recognizes 8 MimicGen task families; any other task name fails at generation or train time:

`coffee_preparation`, `mug_cleanup`, `square`, `nut_assembly`, `stack_three`, `three_piece_assembly`, `pick_place`, `threading`

Repeat the download with a different `TASK` for another public task, using the difficulty suffix (`_d0`, `_d1`, `_d2`) published for that family.

### 5. Prepare data

Generate the model-ready observation cache for the downloaded task, or pass another raw dataset path with `--dataset`:

```bash
vmstack data generate-observations \
  --dataset datasets/mimicgen/${TASK}/${TASK}.hdf5 \
  --n-demo 100 \
  --num-workers 4
```

Generated data is written to `datasets/mimicgen/${TASK}/${TASK}_lmdb/`. Use `--output-dir` to select a different cache directory, or `--overwrite` to replace an existing cache.

### 6. Train

```bash
vmstack train \
  --config-name=train_seeker_diffusion \
  task=${TASK} \
  n_demo=100
```

This example uses RGB external and wrist observations with a Seeker ResNet-18 encoder and diffusion policy. The full set of recipes, including the RGB Diffusion Policy and DP3 baselines and the encoder overrides that reach the remaining focus mechanisms, is tabulated in [`docs/workflows.md`](./docs/workflows.md#recipes).

Runs are written to:

```text
experiments/<regime>/<task>/<input>/<encoder>/<policy>/<n_demo>d_<action_rep>_s<seed>/
```

The latest full checkpoint is stored at `checkpoints/latest.ckpt` inside the run directory. Training and rollout are GPU-oriented; reduce `batch_size`, `rollout.n_envs`, and data-loader worker counts when working with limited resources.

### 7. Roll out

```bash
vmstack rollout \
  experiments/in_domain/${TASK}/rgb_external_wrist/seeker_resnet18/global_diffusion/100d_absolute_s0/checkpoints/latest.ckpt \
  --output-dir experiments/${TASK}/manual_rollout
```

The checkpoint records the runner specification used to reconstruct its evaluation environment. Without `--output-dir`, rollout allocates a non-overwriting numbered directory beside the checkpoint run. Use `vmstack rollout <checkpoint> --inspect` to inspect the model without starting an environment.

Experiment images and videos are saved locally below each run's `media/`
directory by default. Images use WebP and rollout videos use annotated H.264
MP4. W&B media upload is opt-in through the independent
`visualization.upload.images` and `visualization.upload.videos` switches.

Additional operational recipes and conventions live in [`docs/workflows.md`](./docs/workflows.md).

---

## Documentation

| Document | Purpose |
|---|---|
| [`docs/architecture.md`](./docs/architecture.md) | Stable architecture, ownership, and invariants |
| [`docs/data.md`](./docs/data.md) | Observation contracts and data flow |
| [`docs/workflows.md`](./docs/workflows.md) | Training, evaluation, and experiment recipes |
| [`docs/development.md`](./docs/development.md) | How to modify and extend the codebase |
| [`AGENTS.md`](./AGENTS.md) | Coding and agent execution rules |
| [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) | Current engineering state and active tasks |

---

## Project status

Visuomotor Stack is an actively developed research codebase. Its APIs, experiment configurations, and checkpoint formats may evolve with ongoing work. The source is available under the [MIT License](./LICENSE).

---

## Development principles

Before changing the repository:

1. Identify the ownership boundary.
2. Make changes inside the responsible package.
3. Avoid speculative abstractions and compatibility layers.
4. Update architecture documentation only when stable design changes.
5. Record active engineering work in `IMPLEMENTATION.md`.

For non-trivial changes, follow the workflow defined in `AGENTS.md`:

- define scope and non-goals
- trace ownership and dependencies
- implement the smallest correct change
- add focused regression tests
- verify before completion
