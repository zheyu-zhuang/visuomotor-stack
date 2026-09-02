# Workflows

This document contains operational recipes.

Architecture decisions belong in `docs/architecture.md`.

## Installation and setup

Visuomotor Stack targets Linux with an NVIDIA GPU and CUDA-capable driver. The
authoritative dependency specification is `conda_environment.yaml`, which pins
Python 3.9, PyTorch 2.1, and CUDA 11.8.

```bash
git clone https://github.com/zheyu-zhuang/visuomotor-stack.git
cd visuomotor-stack
mamba env create -f conda_environment.yaml
mamba activate vmstack
```

MuJoCo and robosuite offscreen rendering require system libraries. Install them
through the operating system when possible:

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

If `sudo` is unavailable, install the rendering libraries through Conda:

```bash
mamba install -c conda-forge glew mesalib
mamba install -c menpo glfw3
```

From the source checkout, install the suite dependencies and assets:

```bash
vmstack setup
```

`vmstack setup` clones, pins, patches, and installs the MimicGen suite
dependencies recorded in `.dep/mimicgen.lock`. It also downloads and verifies
the model weights, textures, and backgrounds from the repository's `assets`
release and builds the task-embedding cache. Dependency checkouts default to
`../visuomotor-deps/mimic/`.

The command requires the `vmstack` Conda environment. Set
`VISUOMOTOR_CONDA_ENV` to a different environment name, or clear it to skip the
check. Use `--assets-only` to skip suite checkouts, `--suite-deps-root <path>`
to choose their location, `--skip-task-cache` to omit the embedding cache, or
`--force` to restore locked revisions, reapply patches, and redownload assets.
Run `vmstack setup --help` for the complete command interface.

## Typical policy workflow

```text
setup environment
        ->
prepare dataset
        ->
generate observation cache
        ->
train policy
        ->
rollout checkpoint
```

## Configuration model

A run is selected by:

```text
input + encoder + policy + regime
```

Where:

- `input` defines available observations and canonical fields.
- `encoder` defines the feature pipeline.
- `policy` defines action prediction.
- `regime` defines experimental conditions.

The resolver converts these choices into typed runtime specifications.

## Recipes

Each recipe below is a complete `input + encoder + policy` selection, run with
`vmstack train --config-name=<name>`.

| `--config-name` | input / encoder / policy | What it is |
| --- | --- | --- |
| `train_rgb_diffusion` | rgb_external_wrist / rgb_resnet18 / global_diffusion | RGB Diffusion Policy baseline (CLI default) |
| `train_seeker_diffusion` | rgb_external_wrist / seeker_resnet18 / global_diffusion | RGB diffusion with Seeker focus crops |
| `train_voxel_diffusion` | voxel_wrist / voxel_simple / global_diffusion | Source-width voxel DDPM baseline |
| `train_voxel_flow` | voxel_wrist / voxel_simple / global_flow | Voxel global flow-matching baseline |

Each recipe requires a matching observation cache:

| Recipe family | Required `prepare` flags |
| --- | --- |
| `train_rgb_*`, `train_seeker_*` | none beyond the defaults |
| `train_voxel_*` | `--enable-voxel` |

Training and rollout reject caches whose producer metadata or action widths do
not match the resolved contract, so a mismatched cache fails immediately rather
than training on the wrong observations.

## Recombining a recipe

Any recipe's `input`, `encoder`, `policy`, or `regime` can be replaced on the
command line. This is how encoders that have no dedicated recipe are run. The
available options are the file names in `visuomotor/config/input/`,
`visuomotor/config/encoder/`, `visuomotor/config/policy/`, and
`visuomotor/config/regime/`.

```bash
# 3D ResNet voxel encoder
vmstack train --config-name=train_voxel_flow \
  encoder=voxel_resnet3d task=${TASK} n_demo=100

# RVT-2 heatmap focus, requires .weights/rvt2_heatmap.mimicgen.pth
vmstack train --config-name=train_seeker_diffusion \
  encoder=rvt2_resnet18 task=${TASK} n_demo=100

# oracle focus upper bound
vmstack train --config-name=train_seeker_diffusion \
  encoder=oracle_resnet18 task=${TASK} n_demo=100

# 2D focus pooling
vmstack train --config-name=train_seeker_diffusion \
  encoder=rgb_focus_pool2d task=${TASK} n_demo=100
```

Policies recombine the same way, for example an RGB flow-matching run:

```bash
vmstack train --config-name=train_rgb_diffusion \
  policy=global_flow task=${TASK} n_demo=100
```

### Regimes

`regime=in_domain` is the default in every recipe. Two others are available:

- `regime=image_aug` adds background augmentation and table texture shuffling
  on the in-domain scene. It uses the standard cache.
- `regime=domain_rand` adds full domain randomization over 25 rendered
  backgrounds. It resolves its dataset to `<task>_lmdb_tex25`, so it requires a
  separately generated cache:

```bash
vmstack data prepare \
  --dataset datasets/mimicgen/${TASK}/${TASK}.hdf5 \
  --n-demo 100 \
  --table-texture-every 25 \
  --output-suffix _tex25
```

## Training

Example:

```bash
vmstack train \
  --config-name=train_seeker_diffusion \
  task=three_piece_assembly_d2 \
  n_demo=100
```

The RGB Diffusion Policy baseline uses the same form:

```bash
vmstack train \
  --config-name=train_rgb_diffusion \
  task=${TASK} \
  n_demo=100
```

Runs are written to:

```text
experiments/<wandb_project>/<task>/<encoder><input_suffix>/<n_demo>d_<action_rep>_s<seed>/
```

The latest full training checkpoint is stored at `checkpoints/latest.ckpt`
inside the run directory. Training and rollout are GPU-oriented; reduce
`batch_size`, `rollout.n_envs`, and data-loader worker counts on hosts with
limited resources.

The primary voxel baseline uses DDPM with the source U-Net widths:

```bash
vmstack train \
  --config-name=train_voxel_diffusion \
  task=${TASK} \
  n_demo=100 \
  input_augmentation.scene_yaw=enabled
```

Point-cloud caches created before tabletop-removal metadata was introduced must
be regenerated; pass `--overwrite` when replacing an existing cache.

Generate the combined spatial cache used by voxel and point-cloud experiments:

```bash
TASK=stack_three_d1

vmstack data prepare \
  --dataset datasets/mimicgen/${TASK}/${TASK}.hdf5 \
  --n-demo 100 \
  --num-workers 12 \
  --enable-voxel \
  --enable-point-cloud
```

The cache stores the complete spatial producer specifications and exact 13D
absolute / 10D-per-step delta action contracts. Training and rollout reject
producer metadata or action widths that do not match their resolved contracts.
Workers are CPU-bounded and keep only compressed/sparse episode buffers; choose
the worker count from a controlled smoke run for other hosts.

## Pretraining

The Seeker and RVT-2 focus providers are trained by their own workspaces,
launched through the same entry point with a config subpath:

```bash
vmstack train --config-name=pretrain/seeker task_name=square_d2 n_demo=250
vmstack train --config-name=pretrain/rvt2_heatmap n_demo=100
```

This step is not required for normal use. `vmstack setup` downloads
`seeker.mimicgen.pth` and `rvt2_heatmap.mimicgen.pth` into `.weights/`, and the
encoders load them with `strict_weights: true`. Run pretraining only to
regenerate those weights.

## Dataset preparation

Raw MimicGen demonstrations are hosted at
<https://huggingface.co/datasets/amandlek/mimicgen_datasets/>. Place the raw
dataset at `datasets/mimicgen/<task>/<task>.hdf5`, or pass another location
with `--dataset`:

```bash
TASK=three_piece_assembly_d2

mkdir -p datasets/mimicgen/${TASK}
wget -O "datasets/mimicgen/${TASK}/${TASK}.hdf5" \
  "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/${TASK}.hdf5?download=true"
```

Task metadata (`visuomotor/data/mimicgen/tasks.py`) recognizes 8 MimicGen task
families — `coffee_preparation`, `mug_cleanup`, `square`, `nut_assembly`,
`stack_three`, `three_piece_assembly`, `pick_place`, `threading` — each with the
difficulty suffix (`_d0`, `_d1`, `_d2`) published for that family. Any other
task name fails at generation or train time.

Generate the model-ready observation cache with:

```bash
vmstack data prepare \
  --dataset datasets/mimicgen/${TASK}/${TASK}.hdf5 \
  --n-demo 100 \
  --num-workers 4
```

The cache is written to `datasets/mimicgen/${TASK}/${TASK}_lmdb/`. Relevant
options:

- `--output-dir` selects a different cache directory.
- `--output-suffix` appends to the generated cache name, as required by
  `regime=domain_rand`.
- `--overwrite` replaces an existing cache.
- `--num-workers` sets rerendering parallelism; workers are CPU-bounded.
- `--enable-voxel` and `--enable-point-cloud` add the spatial
  producers required by the corresponding recipes.

Other dataset operations:

- `vmstack data convert-actions` rewrites an HDF5 action representation.
- `vmstack data merge` combines several tasks into one multitask dataset.
- `vmstack data playback` replays a dataset from stored observations or actions.

## Rollout

Rollout takes a checkpoint path and reconstructs its evaluation environment from
the runner specification recorded in the checkpoint. A training workspace
checkpoint is accepted directly and uses EMA weights when available:

```bash
vmstack rollout experiments/task/run/checkpoints/latest.ckpt --device cuda:0
```

Standalone rollouts allocate a new numbered directory beside the checkpoint run.
Pass `--weights model` to evaluate the raw training model instead of EMA, and
`--inspect` to inspect the model without starting an environment. Exported
release checkpoints remain supported. Select an explicit destination when
required:

```bash
vmstack rollout experiments/task/run/checkpoints/policy.release.pth \
  --output-dir experiments/task/manual_rollout
```

## Media and logging

Training, validation, and rollout media are written below `media/`. WebP grids
and annotated H.264 videos are retained locally. W&B image and video publishing
is off by default and can be enabled independently:

```bash
vmstack train --config-name=train_rgb_diffusion \
  visualization.upload.images=true \
  visualization.upload.videos=false
```

## Adding workflows

Add concise reproducible commands only.

Avoid duplicating architecture explanations here.
