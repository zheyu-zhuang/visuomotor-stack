import pytest
import torch

from visuomotor.action_generation.base import ActionGenerator
from visuomotor.data.core.normalization import Normalizer
from visuomotor.perception.common.types import EncoderOutput
from visuomotor.policy.generative import GenerativePolicy


class Encoder(torch.nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, observations):
        return EncoderOutput(observations["features"])


class AuxiliaryEncoder(torch.nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, observations):
        features = observations["features"]
        return EncoderOutput(
            features,
            auxiliary_losses={"attention_prior": features.new_tensor(0.25)},
        )


class Generator(ActionGenerator):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def loss(self, actions, condition, *, generator=None):
        return {"trajectory": ((actions - condition * self.weight) ** 2).mean()}

    def sample(self, condition, *, generator=None):
        return condition * self.weight


class SequenceGenerator(ActionGenerator):
    prediction_horizon = 5

    def loss(self, actions, condition, *, generator=None):
        return actions.mean() * 0

    def sample(self, condition, *, generator=None):
        return condition[:, None].expand(-1, self.prediction_horizon, -1)


def test_runtime_config_keeps_parameter_counts_with_model_names():
    policy = GenerativePolicy(
        encoder=Encoder(),
        generator=Generator(),
        observation_kinds={},
        observation_feature_dim=2,
    )

    assert policy.get_runtime_config()["Training"] == {
        "Encoder": "Encoder (0.00 M)",
        "Generator": "Generator (0.00 M)",
    }


def test_generative_policy_optimizer_step_and_prediction():
    normalizer = Normalizer()
    normalizer.update_samples("action", torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
    normalizer.finalize()
    policy = GenerativePolicy(
        encoder=Encoder(),
        generator=Generator(),
        observation_kinds={},
        observation_feature_dim=2,
        action_normalizer=normalizer,
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    observations = {"features": torch.tensor([[0.5, -0.5]])}
    loss = policy.loss(observations, torch.tensor([[0.5, -0.5]]))["loss"]
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert policy.sample(observations).shape == (1, 2)


def test_generative_policy_sums_each_weighted_encoder_loss_once():
    policy = GenerativePolicy(
        encoder=AuxiliaryEncoder(),
        generator=Generator(),
        observation_kinds={},
        observation_feature_dim=2,
    )
    losses = policy.loss(
        {"features": torch.zeros(1, 2)}, torch.zeros(1, 2)
    )

    assert losses["attention_prior"].item() == 0.25
    assert losses["loss"].item() == pytest.approx(
        losses["trajectory"].item() + 0.25
    )

    tensor_policy = GenerativePolicy(
        encoder=AuxiliaryEncoder(),
        generator=SequenceGenerator(),
        observation_kinds={},
        observation_feature_dim=2,
    )
    assert tensor_policy.loss(
        {"features": torch.zeros(1, 2)}, torch.zeros(1, 2)
    ).item() == 0.25


def test_task_context_does_not_enter_encoder_observations():
    normalizer = Normalizer()
    normalizer.update_samples("action", torch.tensor([[-1.0], [1.0]]))
    normalizer.finalize()
    policy = GenerativePolicy(
        encoder=Encoder(output_dim=1),
        generator=Generator(),
        observation_kinds={},
        observation_feature_dim=1,
        action_normalizer=normalizer,
    )
    base = policy.sample({"features": torch.zeros(1, 1)})
    with_identity = policy.sample(
        {"features": torch.zeros(1, 1)},
        task_context={"robot_id": torch.ones(1, dtype=torch.long)},
    )
    torch.testing.assert_close(base, with_identity)


def test_execution_horizon_slices_generator_prediction():
    policy = GenerativePolicy(
        encoder=Encoder(output_dim=4),
        generator=SequenceGenerator(),
        observation_kinds={},
        observation_feature_dim=4,
        execution_horizon=2,
    )
    output = policy.predict_action({"features": torch.ones(3, 4)})
    assert output["action_pred"].shape == (3, 5, 4)
    assert output["action"].shape == (3, 2, 4)
