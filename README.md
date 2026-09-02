# Visuomotor Stack

Visuomotor Stack is infrastructure for training and evaluating RGB,
point-cloud, and voxel visuomotor manipulation policies. It provides a shared
path from MimicGen demonstrations to prepared observations, policy training,
checkpoints, and simulator rollouts.

The installed package is `visuomotor-stack`, the Python namespace is
`visuomotor`, and the command-line interface is `vmstack`.

## Capabilities

| Input | Representative encoders and focus mechanisms | Policy families |
| --- | --- | --- |
| RGB, external and wrist views | ResNet-18, Seeker, RVT-2, oracle focus, 2D focus pooling | Diffusion, flow matching |
| Point cloud with proprioception | DP3 PointNet | Diffusion, flow matching |
| Voxel with optional wrist RGB | Voxel encoder, 3D ResNet, 3D focus pooling | Diffusion, flow matching |

Inputs, encoders, policies, and experimental regimes are selected independently
through configuration. The complete recipe matrix and compatible overrides are
documented in [Workflows](./docs/workflows.md#recipes).

## Interface

The main workflow is exposed through four commands:

| Command | Purpose |
| --- | --- |
| `vmstack setup` | Install the pinned MimicGen dependencies and project assets |
| `vmstack data prepare` | Convert demonstrations into model-ready observation caches |
| `vmstack train` | Resolve an experiment configuration and train its policy |
| `vmstack rollout` | Reconstruct an evaluation environment and run a checkpoint |

Experiments compose four choices:

```text
input + encoder + policy + regime
```

A complete recipe selects useful defaults. Individual choices can then be
overridden from the command line:

```bash
vmstack train \
  --config-name=train_rgb_diffusion \
  encoder=seeker_resnet18 \
  policy=global_flow \
  regime=image_aug \
  task=three_piece_assembly_d2 \
  n_demo=100
```

Hydra/OmegaConf is confined to configuration resolution. Runtime components
receive explicit typed specifications. The resulting artifact flow is:

```text
raw demonstrations -> observation cache -> checkpoint -> rollout
```

## Quickstart

### Prerequisites

- Linux with an NVIDIA GPU and CUDA-capable driver
- Git and Mamba or Conda
- Access to a MimicGen dataset

The environment pins Python 3.9, PyTorch 2.1, and CUDA 11.8. The commands below
run one RGB Diffusion Policy experiment from setup through rollout:

```bash
git clone https://github.com/zheyu-zhuang/visuomotor-stack.git
cd visuomotor-stack

mamba env create -f conda_environment.yaml
mamba activate vmstack

sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
vmstack setup

TASK=three_piece_assembly_d2
mkdir -p "datasets/mimicgen/${TASK}"
wget -O "datasets/mimicgen/${TASK}/${TASK}.hdf5" \
  "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/${TASK}.hdf5?download=true"

vmstack data prepare \
  --dataset "datasets/mimicgen/${TASK}/${TASK}.hdf5" \
  --n-demo 100 \
  --num-workers 4

vmstack train \
  --config-name=train_rgb_diffusion \
  task=${TASK} \
  n_demo=100

vmstack rollout \
  "experiments/vm-global-diffusion/${TASK}/rgb_resnet18/100d_absolute_s0/checkpoints/latest.ckpt" \
  --output-dir "experiments/${TASK}/manual_rollout"
```

`vmstack rollout <checkpoint> --inspect` inspects a model without starting an
environment. See [Workflows](./docs/workflows.md) for installation alternatives,
dataset conventions, voxel and point-cloud cache generation, recipes, training
options, output paths, checkpoints, and media settings.

## Documentation

| Document | Purpose |
| --- | --- |
| [Workflows](./docs/workflows.md) | Setup, datasets, training recipes, rollout, and artifacts |
| [Architecture](./docs/architecture.md) | Stable system structure, ownership, and invariants |
| [Data boundary](./docs/data.md) | Observation contracts, representations, and normalization |
| [Development guide](./docs/development.md) | Package ownership and contribution workflow |
| [Agent rules](./AGENTS.md) | Repository-specific coding and execution rules |
| [Implementation state](./IMPLEMENTATION.md) | Active structural work and completed architectural changes |

## Project status

Visuomotor Stack is an actively developed research codebase. APIs, experiment
configurations, and checkpoint formats may evolve with ongoing work. The source
is available under the [MIT License](./LICENSE).
