from dataclasses import replace
from pathlib import Path

import dill
import pytest
import torch

from visuomotor.config import schema as Schema
from visuomotor.config.build import (
    build_policy,
    load_policy_checkpoint,
    load_rollout_checkpoint,
)
from visuomotor.config.schema import (
    GeneratorSpec,
    GlobalPolicySpec,
    InputSpec,
    ModelSpec,
    ObservationContract,
    ObsFieldSpec,
    RandomCropSpec,
    RgbEncoderSpec,
    TrajectoryContract,
)
from visuomotor.data.core.normalization import Normalizer
from visuomotor.data.core.observations import canonicalize_rgb_from_uint8
from visuomotor.policy.checkpoint import strip_backbone


def _policy_spec(generator_kind="diffusion"):
    input_spec = InputSpec(name="rgb_external", rgb_views=("external",))
    encoder = RgbEncoderSpec(
        name="rgb_resnet18",
        architecture="resnet18",
        rgb_keys=("rgb_external",),
        proprio_fields=(),
        feature_dim=8,
        random_crop=RandomCropSpec(32, 32),
        pretrained_imagenet=False,
    )
    return ModelSpec(
        input=input_spec,
        observation=ObservationContract(
            (ObsFieldSpec("rgb_external", "rgb", (3, 32, 32)),)
        ),
        encoder=encoder,
        policy=GlobalPolicySpec(
            name=f"global_{generator_kind}",
            generator=GeneratorSpec(
                kind=generator_kind,
                num_inference_steps=2,
                diffusion_step_embed_dim=16,
                unet_channels=(8, 16),
                n_groups=4,
                num_train_timesteps=4,
                integration_steps=2,
                time_embedding_dim=16,
            ),
        ),
        normalizer="linear",
        trajectory=TrajectoryContract(
            action_dim=3,
            prediction_horizon=8,
            observation_horizon=2,
            execution_horizon=4,
            action_rep="absolute",
        ),
    )


def _batch():
    return {
        "obs": {
            # Already canonical: the training source->canonical boundary runs in
            # the training workspace, not inside GenerativePolicy.
            "rgb_external": canonicalize_rgb_from_uint8(
                torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8)
            ),
        },
        "action": torch.randn(2, 8, 3),
    }


def _runner_spec(model_spec):
    source_observation = Schema.SourceObservationSpec(
        (Schema.SourceObsFieldSpec("agentview_image", "rgb", (3, 32, 32)),)
    )
    return Schema.RunnerSpec(
        dataset_path="dataset.hdf5",
        observation=model_spec.observation,
        source_observation=source_observation,
        trajectory=model_spec.trajectory,
        rgb_load_resolutions=(("rgb_external", 32),),
    )


def _run_spec(model_spec, *, use_ema=True):
    runner = _runner_spec(model_spec)
    loader = Schema.DataLoaderSpec(
        batch_size=2,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        persistent_workers=False,
    )
    return Schema.RunSpec(
        task=Schema.TaskSpec("square_d0", 400, ("agentview",), (0.0, 0.0, 0.82)),
        regime=Schema.RegimeSpec("in_domain"),
        model=model_spec,
        dataset=Schema.DatasetSpec(
            path="dataset.hdf5",
            observation=model_spec.observation,
            source_observation=runner.source_observation,
            trajectory=model_spec.trajectory,
            rgb_load_resolutions=(("rgb_external", 32),),
            n_demo=100,
        ),
        runner=runner,
        training=replace(Schema.TrainingSpec(), use_ema=use_ema),
        workspace=Schema.PolicyWorkspaceSpec(
            train_loader=loader,
            val_loader=loader,
            ema=Schema.EmaSpec(),
            logging=Schema.LoggingSpec(
                project="test",
                resume=False,
                name="test",
                group="test",
                job_type="test",
                tags=(),
            ),
            checkpoint=Schema.CheckpointSpec(
                topk=Schema.TopKCheckpointSpec("score", "max", 1, "{score}"),
                save_last=True,
            ),
        ),
        exp_name="test",
    )


@pytest.mark.parametrize("generator_kind", ("diffusion", "flow"))
def test_optimizer_step_and_spec_only_checkpoint_restore(
    tmp_path: Path, generator_kind
):
    torch.manual_seed(2)
    spec = _policy_spec(generator_kind)
    policy = build_policy(spec)
    normalizer = Normalizer()
    normalizer.update_samples("action", torch.randn(4, 3))
    normalizer.finalize()
    policy.set_normalizer(normalizer)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = policy.loss(_batch(), generator=torch.Generator().manual_seed(3))
    loss.backward()
    optimizer.step()
    assert loss.detach() > 0

    batch = _batch()
    policy.eval()
    expected = policy.sample(batch["obs"], generator=torch.Generator().manual_seed(4))
    checkpoint = tmp_path / "policy.ckpt"
    policy.save_checkpoint(checkpoint)

    restored = load_policy_checkpoint(checkpoint).eval()
    actual = restored.sample(batch["obs"], generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.model_spec == spec


def test_rollout_checkpoint_loads_training_ema_and_runner(tmp_path: Path):
    spec = _policy_spec()
    policy = build_policy(spec)
    model_state = policy.state_dict()
    ema_state = {key: value.clone() for key, value in model_state.items()}
    changed_key = next(
        key for key, value in ema_state.items() if value.is_floating_point()
    )
    ema_state[changed_key].add_(1)
    checkpoint = tmp_path / "latest.ckpt"
    torch.save(
        {
            "cfg": Schema.to_dict(_run_spec(spec)),
            "state_dicts": {"model": model_state, "ema_model": ema_state},
            "pickles": {
                "epoch": dill.dumps(12),
                "global_step": dill.dumps(345),
            },
        },
        checkpoint,
        pickle_module=dill,
    )

    loaded = load_rollout_checkpoint(checkpoint)
    assert loaded.checkpoint_format == "training"
    assert loaded.weights == "ema_model"
    assert loaded.epoch == 12
    assert loaded.global_step == 345
    assert loaded.training_demo_count == 100
    assert loaded.runner_spec == _runner_spec(spec)
    torch.testing.assert_close(
        loaded.policy.state_dict()[changed_key], ema_state[changed_key]
    )

    raw = load_rollout_checkpoint(checkpoint, weights="model")
    assert raw.weights == "model"
    torch.testing.assert_close(
        raw.policy.state_dict()[changed_key], model_state[changed_key]
    )


def test_rollout_checkpoint_uses_raw_model_when_ema_is_disabled(tmp_path: Path):
    spec = _policy_spec()
    policy = build_policy(spec)
    checkpoint = tmp_path / "latest.ckpt"
    torch.save(
        {
            "cfg": Schema.to_dict(_run_spec(spec, use_ema=False)),
            "state_dicts": {"model": policy.state_dict()},
            "pickles": {},
        },
        checkpoint,
        pickle_module=dill,
    )

    loaded = load_rollout_checkpoint(checkpoint)
    assert loaded.weights == "model"
    with pytest.raises(ValueError, match="cannot provide 'ema'"):
        load_rollout_checkpoint(checkpoint, weights="ema")


def test_rollout_checkpoint_keeps_release_support(tmp_path: Path):
    spec = _policy_spec()
    policy = build_policy(spec)
    runner = _runner_spec(spec)
    checkpoint = tmp_path / "policy.release.pth"
    policy.save_checkpoint(checkpoint, runner_spec=runner)

    loaded = load_rollout_checkpoint(checkpoint)
    assert loaded.checkpoint_format == "release"
    assert loaded.weights == "release"
    assert loaded.runner_spec == runner


def test_strip_backbone_removes_vit_tensors_from_nested_state_dict():
    payload = {
        "schema_version": 3,
        "model_spec": {"__spec__": "ModelSpec"},
        "state_dict": {
            "vit.blocks.0.weight": torch.zeros(1),
            "vit.blocks.1.weight": torch.zeros(1),
            "head.weight": torch.zeros(1),
        },
    }
    stripped = strip_backbone(payload)
    assert set(stripped["state_dict"]) == {"head.weight"}
    assert stripped["model_spec"] == payload["model_spec"]


def test_strip_backbone_leaves_rvt2_payload_untouched_by_seeker_path():
    payload = {
        "head_state_dict": {"w": torch.zeros(1)},
        "patch_backbone_state_dict": {"vit.blocks.0.weight": torch.zeros(1)},
        "patch_backbone": "dino",
    }
    stripped = strip_backbone(payload)
    assert "patch_backbone_state_dict" not in stripped
    assert stripped["head_state_dict"] == payload["head_state_dict"]
