"""Conditional TB, replay snapshots and restart use the same trajectory labels."""

import json
import random

import pytest
import torch
from rdkit.Chem import QED

from rxnflow.config import Config, DataConfig, ModelConfig, RewardConfig, TrainingConfig
from rxnflow.envs.graph import GraphBatch, molecule_to_graph_data
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.gflownet.types import (
    Action,
    ActionKind,
    MoleculeState,
    Trajectory,
    Transition,
)
from rxnflow.reward import RewardFunction
from rxnflow.sampler import RxnFlowSampler
from rxnflow.trainer import RxnFlowTrainer


class TwoObjectiveReward(RewardFunction):
    objectives = ("qed", "size")

    def score(self, molecules):
        return torch.tensor(
            [[QED.qed(mol), min(mol.GetNumHeavyAtoms() / 20, 1)] for mol in molecules],
            dtype=torch.float32,
        ).reshape(-1, 2)


def config_for(env_dir, output_dir):
    return Config(
        data=DataConfig(env_dir=str(env_dir), max_atoms=20),
        reward=RewardConfig(exponent=[4.0, 128.0]),
        model=ModelConfig(hidden_dim=16, num_layers=1, block_dim=16),
        training=TrainingConfig(
            batch_size=4,
            replay_batch_size=4,
            replay_capacity=12,
            checkpoint_every=1,
            log_every=100,
            retrosynthesis_workers=0,
            log_z_learning_rate=0.001,
        ),
        output_dir=str(output_dir),
        device="cpu",
        seed=31,
    )


def test_sampling_configuration_is_independent_of_encoder_range(tmp_path):
    for exponent in (32.0, [1.0, 16.0], [4.0, 128.0]):
        config = Config(
            data=DataConfig(env_dir="example"),
            reward=RewardConfig(
                exponent=exponent,
                preferences=[0.25, 0.75],
            ),
        )
        path = tmp_path / "config.yaml"
        config.save(path)
        assert Config.from_file(path) == config
    for invalid in ([1], [64, 1], [1, 1], float("nan")):
        with pytest.raises(ValueError):
            RewardConfig(exponent=invalid).validate()


def test_condition_encoding_reaches_all_three_branches(prepared_env, tmp_path):
    trainer = RxnFlowTrainer(
        config_for(prepared_env, tmp_path / "encoding"), TwoObjectiveReward()
    )
    model = trainer.model
    beta = torch.tensor([1.0, 64.0, 128.0, 32.0])
    preferences = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    condition = model.encode_condition(beta, preferences)
    assert torch.isfinite(condition).all()
    # Fourier endpoints alias but the raw u coordinate distinguishes them.
    assert not torch.allclose(condition[0], condition[1])
    same_beta = model.encode_condition(torch.full((2,), 32.0), torch.eye(2))
    assert not torch.allclose(same_beta[0], same_beta[1])
    graphs = GraphBatch.from_graphs([molecule_to_graph_data(None, 20, 0)] * 4)
    embeddings = model.encode_graphs(graphs, condition)
    assert not torch.allclose(embeddings[0], embeddings[1])
    torch.testing.assert_close(
        model.temperature(condition), torch.full((4, len(trainer.env.action_names)), 0.2)
    )
    # The output layers start constant; once their weights move, both heads
    # must expose beta and preference, rather than a global learned scalar.
    with torch.no_grad():
        model.temperature_head[-1].weight.fill_(0.01)
        model.log_z[-1].weight.fill_(0.01)
    for head in (model.temperature, model.log_z):
        for inputs in (condition[:2], same_beta):
            values = head(inputs)
            assert not torch.allclose(values[0], values[1])
    loss = (
        embeddings.square().mean()
        + model.temperature(condition).mean()
        + model.log_z(condition).mean()
    )
    loss.backward()
    for module in (
        model.beta_encoder,
        model.preference_encoder,
        model.graph_condition,
        model.temperature_head,
        model.log_z,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )


def test_replay_snapshots_keep_conditions_and_objectives():
    trajectory = Trajectory(
        [],
        "CC",
        beta=64.0,
        preferences=[0.25, 0.75],
        objective_rewards=[0.8, 0.2],
        reward=0.35,
    )
    buffer = ReplayBuffer(2)
    expected = trajectory.to_dict()
    buffer.add([trajectory])
    trajectory.preferences[0] = 1.0
    trajectory.objective_rewards[0] = 0.0
    trajectory.beta = 1.0
    restored = ReplayBuffer(2)
    restored.load_state_dict(json.loads(json.dumps(buffer.state_dict())))
    sample = restored.sample(1, random.Random(1))[0]
    assert sample.to_dict() == expected
    sample.preferences[0] = 0.0
    assert restored.sample(1, random.Random(1))[0].to_dict() == expected


def test_tb_uses_stored_conditions_and_objective_vector(
    prepared_env, tmp_path, monkeypatch
):
    trainer = RxnFlowTrainer(
        config_for(prepared_env, tmp_path / "tb"), TwoObjectiveReward()
    )
    transition = Transition(MoleculeState(), Action(ActionKind.FIRST_BLOCK), "CC", -0.5)
    trajectories = [
        Trajectory(
            [transition, transition],
            "CC",
            beta=2.0,
            preferences=[1.0, 0.0],
            objective_rewards=[0.25, 0.5],
            reward=999.0,
        ),
        Trajectory(
            [transition],
            "CC",
            beta=3.0,
            preferences=[0.0, 1.0],
            objective_rewards=[0.25, 0.5],
            reward=999.0,
        ),
    ]

    def probabilities(states, actions, beta, preferences):
        torch.testing.assert_close(beta, torch.tensor([2.0, 2.0, 3.0]))
        torch.testing.assert_close(
            preferences, torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        return torch.tensor([-1.0, -2.0, -3.0])

    monkeypatch.setattr(trainer.policy, "action_log_probabilities", probabilities)
    loss, _ = trainer._loss(trajectories, num_fresh=1)
    expected = (
        torch.tensor([-3.0, -3.0])
        - torch.tensor([-1.0, -0.5])
        - torch.tensor([2.0, 3.0]) * torch.tensor([0.25, 0.5]).log()
    )
    torch.testing.assert_close(loss, expected.square().mean())


def test_multiobjective_training_restart_and_fixed_condition_sampling(
    prepared_env, tmp_path
):
    config = config_for(prepared_env, tmp_path / "training")
    trainer = RxnFlowTrainer(config, TwoObjectiveReward())
    checkpoint = trainer.run(2)
    before = trainer.replay.state_dict()
    trainer.run(1)
    expected_model = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    expected_replay = trainer.replay.state_dict()
    restarted = RxnFlowTrainer(config, TwoObjectiveReward(), restart=checkpoint)
    assert restarted.replay.state_dict() == before
    restarted_checkpoint = restarted.run(1)
    for key, value in restarted.model.state_dict().items():
        torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
    assert restarted.replay.state_dict() == expected_replay
    for trajectory in restarted.replay.sample(100, random.Random(0)):
        assert 4 <= trajectory.beta <= 128
        assert sum(trajectory.preferences) == pytest.approx(1)
        assert len(trajectory.objective_rewards) == 2
        assert trajectory.reward == pytest.approx(
            sum(
                w * r
                for w, r in zip(
                    trajectory.preferences, trajectory.objective_rewards, strict=True
                )
            ),
            abs=1e-6,
        )
    sampler = RxnFlowSampler(restarted_checkpoint, reward=TwoObjectiveReward())
    results = sampler.sample(2, beta=32.0, preferences=[0.25, 0.75], seed=5)
    for result in results:
        assert result.metadata["beta"] == 32.0
        assert result.metadata["preferences"] == [0.25, 0.75]
        scores = result.metadata["objective_rewards"]
        assert result.reward == pytest.approx(
            0.25 * scores["qed"] + 0.75 * scores["size"]
        )
    with pytest.raises(ValueError, match="preferences"):
        sampler.sample(1, beta=32.0, preferences=[1.0])
