import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from model_reference import get_synthon_logits, get_unirxn_logits
from numpy.typing import NDArray
from rdkit import Chem

from examples.qed import QEDReward
from rxnflow import __version__
from rxnflow.config import (
    Config,
    DataConfig,
    ModelConfig,
    RewardConfig,
    SubsamplingConfig,
    TrainingConfig,
)
from rxnflow.core.types import (
    Action,
    ActionType,
    State,
    Trajectory,
    Transition,
)
from rxnflow.reward import RewardFunction
from rxnflow.sampler import RxnFlowSampler
from rxnflow.trainer import RxnFlowTrainer


class CarbonReward(RewardFunction):
    objectives = ("score",)

    def score(self, mols: list[Chem.Mol]) -> NDArray[np.float32]:
        return np.array(
            [
                float(sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())) / 10
                for mol in mols
            ],
            dtype=np.float32,
        ).reshape(-1, 1)


def tiny_config(env_dir: Path) -> Config:
    return Config(
        data=DataConfig(env_dir=str(env_dir), max_atoms=20),
        reward=RewardConfig(beta=("fixed", [1.0])),
        subsampling=SubsamplingConfig(
            sampling_ratio=0.5, min_sampling=1, importance_temp=1.0
        ),
        model=ModelConfig(num_emb=32, num_layers=1, dropout=0.0),
        training=TrainingConfig(
            num_online=2,
            num_replay=1,
            replay_capacity=16,
            learning_rate=1e-3,
            checkpoint_every=1,
            log_every=1,
            retrosynthesis_workers=0,
        ),
    )


def test_training_restart_sampling_and_output_formats(
    prepared_env: Path, tmp_path: Path
) -> None:
    config = tiny_config(prepared_env)
    checkpoint = RxnFlowTrainer(
        config, QEDReward(), output_dir=tmp_path / "run", device="cpu", seed=7
    ).run(1)
    assert checkpoint.is_file()
    assert checkpoint.parent == (tmp_path / "run") / "checkpoints"
    assert checkpoint.name == "step_000001.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["rxnflow_version"] == __version__
    assert payload["run"] == {
        "output_dir": str(tmp_path / "run"),
        "device": "cpu",
        "seed": 7,
    }
    assert not {"output_dir", "device", "seed", "run"} & payload["config"].keys()

    with pytest.raises(FileExistsError, match="already exists"):
        RxnFlowTrainer(config, QEDReward(), output_dir=tmp_path / "run")

    with pytest.raises(ValueError, match="reward implementation"):
        RxnFlowTrainer(
            config,
            CarbonReward(),
            output_dir=tmp_path / "invalid_reward",
            device="cpu",
            seed=7,
        ).run(1, resume_from_checkpoint=checkpoint)

    restarted = RxnFlowTrainer(
        config,
        QEDReward(),
        output_dir=tmp_path / "resumed",
        device="cpu",
        seed=7,
    )
    restarted_checkpoint = restarted.run(1, resume_from_checkpoint=checkpoint)
    assert restarted.step == 2
    assert restarted_checkpoint.is_file()

    records = [
        json.loads(line)
        for directory in (tmp_path / "run", tmp_path / "resumed")
        for line in (directory / "training.jsonl").read_text().splitlines()
    ]
    sample_paths = sorted(
        [
            path
            for directory in (tmp_path / "run", tmp_path / "resumed")
            for path in (directory / "samples").glob("*.jsonl")
        ],
        key=lambda path: path.name,
    )
    assert [path.name for path in sample_paths] == [
        "step_000001.jsonl",
        "step_000002.jsonl",
    ]
    samples = [
        json.loads(line)
        for path in sample_paths
        for line in path.read_text().splitlines()
    ]
    assert [row["step"] for row in records] == [1, 2]
    assert [row["num_replay"] for row in records] == [0, 1]
    assert len(samples) == 4  # Fresh attempts only, never duplicate replay entries.
    assert {(row["step"], row["sample"]) for row in samples} == {
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }
    for record in records:
        online_trajs = [row for row in samples if row["step"] == record["step"]]
        assert record["reward"] == pytest.approx(
            sum(row["reward"] for row in online_trajs) / len(online_trajs)
        )
        # This fixture always selects FirstSynthon; the path log omits it while
        # the training length metric still counts all selected actions.
        assert record["traj_lens"] == pytest.approx(
            sum(len(row["traj"]) + 1 for row in online_trajs) / len(online_trajs)
        )
        assert record["num_online"] == len(online_trajs)
        assert (
            not {"num_reactionss", "action_counts", "mean_reactions", "invalid_reasons"}
            & record.keys()
        )
        assert record["iteration_time"] >= record["sampling_time"] >= 0
        assert record["policy_grad_norm"] >= 0
        assert record["grad_norm"] >= record["policy_grad_norm"]
        assert record["policy_grad_clipped"] == float(record["policy_grad_norm"] > 100)
        assert record["batch_entropy"] == -record["traj_log_p_F"]
    # Sample logs contain readable paths; full replay serialization stays in
    # checkpoints. The parent state is a SMILES string, not a state dictionary.
    for row in samples:
        assert "steps" not in row
        for transition in row["traj"]:
            assert set(transition) == {"state", "reaction", "synthon_smiles"}
            assert isinstance(transition["state"], str) and transition["state"]
            assert transition["reaction"] != "FIRST_SYNTHON"
            if transition["reaction"] in restarted.env.uni_reactions:
                assert transition["synthon_smiles"] is None
            else:
                assert transition["reaction"] in restarted.env.bi_reactions
                assert transition["synthon_smiles"]
            if transition["synthon_smiles"] is not None:
                assert any(
                    transition["synthon_smiles"] in library.smiles
                    for library in restarted.env.synthons.values()
                )

    sampler = RxnFlowSampler(restarted_checkpoint)
    results = sampler.sample(3, seed=11, beta=("fixed", [1.0]))
    assert len(results) == 3
    assert all(result.metadata["preferences"] == [1.0] for result in results)
    assert all(result.smiles and result.traj for result in results)
    assert all("workflow" not in result.to_dict() for result in results)
    for result in results:
        assert set(result.to_dict()) == {"smiles", "traj", "metadata"}
        assert set(result.metadata) == {"beta", "preferences"}
        assert result.traj[-1]["product_smiles"] == result.smiles

    smi_path = tmp_path / "samples.smi"
    csv_path = tmp_path / "samples.csv"
    json_path = tmp_path / "samples.json"
    sampler.write(results, smi_path)
    sampler.write(results, csv_path)
    sampler.write(results, json_path)
    assert len(smi_path.read_text().splitlines()) == 3
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]["traj"])[0]["action_type"] == "FIRST_SYNTHON"
    assert "*" not in rows[0]["smiles"]
    structured = json.loads(json_path.read_text())
    assert structured[0]["traj"]
    assert set(rows[0]) == {"smiles", "traj", "beta", "preferences"}
    assert float(rows[0]["beta"]) == structured[0]["metadata"]["beta"]
    assert json.loads(rows[0]["preferences"]) == structured[0]["metadata"]["preferences"]
    assert "workflow" not in structured[0]


def test_trajectory_balance_uses_backward_probability(
    prepared_env: Path, tmp_path: Path
) -> None:
    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    trainer.model._logZ[-1].bias.data.zero_()
    trainer.policy.log_prob = lambda states, actions, beta, preferences: (
        trainer.model._logZ[-1].bias.expand(len(states)) * 0
    )
    trajectory = Trajectory(
        steps=[
            Transition(
                state=State(),
                action=Action(ActionType.FIRST_SYNTHON),
                product_smiles="[1*]N",
                log_p_B=-2.0,
            )
        ],
        final_smiles="N",
        reward=1.0,
        beta=1.0,
        preferences=[1.0],
        objective_rewards=[1.0],
    )
    loss, _ = trainer.compute_batch_losses([trajectory], num_online=1)
    assert torch.isclose(loss, torch.tensor(4.0))


def test_tb_diagnostics_separate_online_replay_and_invalid(prepared_env, tmp_path):
    config = tiny_config(prepared_env)
    config.training.reward_floor = math.exp(-4)
    trainer = RxnFlowTrainer(
        config, QEDReward(), output_dir=tmp_path / "run", device="cpu", seed=7
    )
    trainer.model._logZ[-1].bias.data.zero_()
    trainer.policy.log_prob = lambda states, actions, beta, preferences: (
        torch.tensor([-2.0, -3.0, -1.0]) + trainer.model._logZ[-1].bias * 0
    )
    batch = [
        Trajectory(
            steps=[Transition(State(), Action(ActionType.FIRST_SYNTHON), "C", pb)],
            final_smiles="C",
            reward=reward,
            valid=valid,
            beta=1.0,
            preferences=[1.0],
            objective_rewards=[reward],
        )
        for pb, reward, valid in [
            (-0.5, math.exp(-1), True),
            (-1.0, 0.0, False),
            (-0.25, math.exp(-2), True),
        ]
    ]
    loss, info = trainer.compute_batch_losses(batch, num_online=2)
    # Hand-calculated residuals: -0.5, 2, 1.25. The invalid reward uses the floor.
    expected = {
        "loss": 1.9375,
        "logZ": 0.0,
        "batch_entropy": 2.0,
        "traj_log_p_F": -2.0,
        "traj_log_p_B": -1.75 / 3,
        "scaled_log_R": -7 / 3,
        "tb_residual": 2.75 / 3,
        "online_loss": 2.125,
        "replay_loss": 1.5625,
        "valid_losses": 0.90625,
        "invalid_losses": 4.0,
        "invalid_logprob": -3.0,
        "invalid_trajectories": 1 / 3,
    }
    for key, value in expected.items():
        assert not info[key].requires_grad
        assert info[key].item() == pytest.approx(value)
    loss.backward()
    assert trainer.model._logZ[-1].bias.grad.item() == pytest.approx(5.5 / 3)

    # No valid samples, replay, or transitions: diagnostics remain finite.
    loss, info = trainer.compute_batch_losses(
        [
            Trajectory(
                [], "", valid=False, beta=1.0, preferences=[1.0], objective_rewards=[0.0]
            )
        ],
        num_online=1,
    )
    assert loss.item() == pytest.approx(16.0)
    assert info["valid_losses"].item() == info["replay_loss"].item() == 0
    assert all(torch.isfinite(value) for value in info.values())


def test_restart_reproduces_next_update_with_dropout(
    prepared_env: Path, tmp_path: Path
) -> None:
    config = tiny_config(prepared_env)
    config.model.dropout = 0.2
    trainer = RxnFlowTrainer(
        config, QEDReward(), output_dir=tmp_path / "run", device="cpu", seed=7
    )
    first_checkpoint = trainer.run(1)
    trainer.run(1)
    expected = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    expected_schedule = trainer.lr_scheduler.state_dict()
    expected_rates = [group["lr"] for group in trainer.optimizer.param_groups]
    expected_ema = {
        key: value.clone() for key, value in trainer.sampling_model.state_dict().items()
    }

    restarted = RxnFlowTrainer(
        config,
        QEDReward(),
        output_dir=tmp_path / "restarted",
        device="cpu",
        seed=99,
    )
    restarted.run(1, resume_from_checkpoint=first_checkpoint)
    assert restarted.lr_scheduler.state_dict() == expected_schedule
    assert [group["lr"] for group in restarted.optimizer.param_groups] == expected_rates
    for key, value in restarted.model.state_dict().items():
        assert torch.equal(value, expected[key]), key
    for key, value in restarted.sampling_model.state_dict().items():
        assert torch.equal(value, expected_ema[key]), key


def test_oriented_synthon_scoring_and_observed_action_log_probability(
    prepared_env, tmp_path
):
    from rxnflow.core.errors import NoValidActions

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    state = State.from_smiles("[3*]C")
    from rxnflow.core.errors import InvalidTransition

    spec = (ActionType.BIRXN_LINKER, "amide_coupling_synthon_first", "1-1")
    assert any(
        subspace.action_type == spec[0] and subspace.name == (spec[1], spec[2])
        for subspace in trainer.env.get_action_space(state)
    )
    actions = []
    for i in range(len(trainer.env.synthons["1-1"])):
        action = Action(*spec, i)
        try:
            trainer.env.step(state, action)
        except InvalidTransition:
            continue
        actions.append(action)
        break
    assert actions
    probability = trainer.policy.log_prob_single(
        state, actions[0], beta=torch.ones(1), preferences=torch.ones(1, 1)
    )
    assert torch.isfinite(probability) and probability <= 0
    (-probability).backward()
    assert trainer.model.mlp_synthon[0].weight.grad is not None
    with pytest.raises(NoValidActions):
        trainer.policy.sample_action(
            State.from_smiles("[33*]NCC", trainer.env.max_reactions - 1),
            1.0,
            0.0,
            beta=torch.ones(1),
            preferences=torch.ones(1, 1),
        )


def test_subsampling_precedes_budget_mask_without_candidate_reactions(
    prepared_env, tmp_path, monkeypatch
):
    import math

    from rxnflow.envs.features import PROPERTY_NAMES

    config = tiny_config(prepared_env)
    config.property_penalty = {"mw": 100.0}
    trainer = RxnFlowTrainer(
        config, QEDReward(), output_dir=tmp_path / "run", device="cpu", seed=7
    )
    env = trainer.env
    name = next(n for n in env.brick_types if len(env.synthons[n]) > 1)
    mw = PROPERTY_NAMES.index("mw")
    for library in env.synthons.values():
        library.properties[:, mw] = 200.0
    target = len(env.synthons[name]) - 1
    env.synthons[name].properties[target, mw] = 50.0
    monkeypatch.setattr(
        env,
        "_apply_action",
        lambda *args: pytest.fail("candidate scoring executed chemistry"),
    )
    original = env.get_synthon_mask

    def mask(properties, library_name, indices=None):
        assert indices is not None
        return original(properties, library_name, indices)

    monkeypatch.setattr(env, "get_synthon_mask", mask)
    found = missed = 0
    for seed in range(20):
        trainer.policy.rng.bit_generator.state = np.random.default_rng(
            seed
        ).bit_generator.state
        group = trainer.policy.forward(
            [env.initial_state()],
            beta=torch.ones(len([env.initial_state()])),
            preferences=torch.ones(len([env.initial_state()]), 1),
        ).action_logits[0]
        valid = torch.isfinite(group.logits[0])
        # Masking retains every sampled column, even when all actions are invalid.
        assert group.logits.shape[1] == len(group.subspace.sample_indices)
        assert group.logits.shape[1] > 1
        if not valid.any():
            missed += 1
            continue
        found += 1
        position = int(valid.nonzero().flatten()[0])
        assert valid.sum() == 1
        action = group.subspace.action_at(position)
        assert action.library_name == name and action.synthon_index == target
        count = trainer.policy.subsampling[name].num_sampling
        assert group.log_importance[position].item() == pytest.approx(
            math.log(len(env.synthons[name]) / count)
        )
    assert found and missed


def test_failed_selected_action_is_retained_for_tb(
    prepared_env: Path, tmp_path: Path, monkeypatch
) -> None:
    config = tiny_config(prepared_env)
    config.data.max_atoms = 5
    trainer = RxnFlowTrainer(
        config, QEDReward(), output_dir=tmp_path / "run", device="cpu", seed=7
    )
    # Reactant budget fits (4 + 1), but the amidation inserts two more atoms.
    state = State.from_smiles("[1*]NCCN")
    index = trainer.env.synthons["3"].smiles.index("*C")
    action = Action(
        ActionType.BIRXN_BRICK,
        reaction="amide_coupling_state_first",
        library_name="3",
        synthon_index=index,
    )
    monkeypatch.setattr(trainer.env, "initial_state", lambda: state)
    monkeypatch.setattr(
        trainer.sampling_policy,
        "sample_actions",
        lambda states, *args: [action] * len(states),
    )
    trajectory = trainer.sampling_policy.sample_from_model_single(
        analyze_backward=False, beta=torch.ones(1), preferences=torch.ones(1, 1)
    )
    assert not trajectory.valid
    assert trajectory.invalid_reason == "invalid_transition"
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action == action
    assert trajectory.steps[0].product_smiles == ""
    trainer._assign_rewards([trajectory])
    assert trajectory.reward == 0
    loss, _ = trainer.compute_batch_losses([trajectory], num_online=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert (
        trainer.model.action_heads[ActionType.BIRXN_BRICK.name][0].weight.grad.abs().sum()
        > 0
    )


@pytest.mark.parametrize("sampling_ratio", [0.35, 1.0])
def test_batched_scores_and_gradients_match_scalar_reference(
    prepared_env, tmp_path, sampling_ratio
):
    from rxnflow.envs.graph import GraphBatch, molecule_to_graph_data

    config = tiny_config(prepared_env)
    config.subsampling.sampling_ratio = sampling_ratio
    trainer = RxnFlowTrainer(
        config, QEDReward(), output_dir=tmp_path / "run", device="cpu", seed=7
    )
    model = trainer.model.eval()
    states = [
        State(),
        State.from_smiles("[3*]C"),
        State.from_smiles("[3*]C", num_reactions=trainer.env.max_reactions - 1),
        State.from_smiles("[11*]C", num_synthons=2, num_reactions=1),
        State.from_smiles("[33*]NCC"),
        State.from_smiles("[3*]C", num_synthons=2, num_reactions=1),
    ]
    categorical = trainer.policy.forward(
        states, beta=torch.ones(len(states)), preferences=torch.ones(len(states), 1)
    )
    # Molecular embeddings must not reveal the path's reaction/synthon counts.
    torch.testing.assert_close(categorical.graph_emb[1], categorical.graph_emb[2])
    torch.testing.assert_close(categorical.graph_emb[1], categorical.graph_emb[-1])
    reference, actual = [], []
    selected_actions, selected_rows = [], []
    for row, state in enumerate(states):
        embedding = model.graph_embedding(
            GraphBatch.from_graphs(
                [molecule_to_graph_data(state.mol, trainer.env.max_atoms)]
            ),
            cond_info=_condition(
                model,
                len(
                    GraphBatch.from_graphs(
                        [molecule_to_graph_data(state.mol, trainer.env.max_atoms)]
                    ).node_mask
                ),
            ),
        )
        available = {subspace.name for subspace in trainer.env.get_action_space(state)}
        for group in categorical.action_logits:
            if group.subspace.name not in available:
                assert torch.isneginf(group.logits[row]).all()
            for column in range(group.logits.shape[1]):
                if not torch.isfinite(group.logits[row, column]):
                    continue
                action = group.subspace.action_at(column)
                score = (
                    get_unirxn_logits(
                        model,
                        embedding,
                        action.reaction,
                        logit_scale=model.logit_scale(_condition(model, 1)),
                    )
                    if action.action_type.is_unirxn
                    else get_synthon_logits(
                        model,
                        embedding,
                        group.subspace.name[0],
                        action.library_name,
                        torch.tensor([action.synthon_index]),
                        logit_scale=model.logit_scale(_condition(model, 1)),
                    )[0]
                )
                reference.append(score)
                actual.append(group.logits[row, column])
                selected_actions.append(action)
                selected_rows.append(row)
    actual, expected = torch.stack(actual), torch.stack(reference)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    # Numerator scoring must split the same reaction into brick/linker heads
    # exactly as denominator scoring does, without changing the action order.
    indices = torch.tensor(selected_rows)
    observed = trainer.policy.get_action_logits(
        categorical.graph_emb[indices], selected_actions, categorical.logit_scale[indices]
    )
    torch.testing.assert_close(observed, actual, atol=3e-6, rtol=3e-5)
    assert {action.action_type for action in selected_actions} == set(ActionType)
    params = tuple(model.parameters())
    a = torch.autograd.grad(actual.square().mean(), params, allow_unused=True)
    b = torch.autograd.grad(expected.square().mean(), params, allow_unused=True)
    for left, right in zip(a, b, strict=True):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)


def test_batch_shares_library_subsamples_and_handles_dead_ends(
    prepared_env, tmp_path, monkeypatch
):
    from rxnflow.gflownet.policy import SubsamplingPolicy

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    trainer.model.eval()
    draws = []
    original = SubsamplingPolicy.sample

    def sample(self):
        draws.append(id(self))
        return original(self)

    monkeypatch.setattr(SubsamplingPolicy, "sample", sample)
    queries = []
    forward_mdp = trainer.model.forward_mdp

    def record_query(graph_emb, name, logit_scale, action_type):
        queries.append(name)
        return forward_mdp(graph_emb, name, logit_scale, action_type)

    monkeypatch.setattr(trainer.model, "forward_mdp", record_query)
    initial = trainer.env.initial_state()
    categorical = trainer.policy.forward(
        [initial, initial],
        beta=torch.ones(len([initial, initial])),
        preferences=torch.ones(len([initial, initial]), 1),
    )
    assert len(draws) == len(set(draws))
    assert queries == ["first_synthon"]  # Shared across all first-synthon libraries.
    assert len(categorical.action_logits) == len(trainer.env.initial_action_space)
    assert all(s.sample_indices is None for s in trainer.env.initial_action_space)
    for group in categorical.action_logits:
        torch.testing.assert_close(group.logits[0], group.logits[1])
    # Unary-only batches take the same sampling path, without synthon features or RNG.
    draws.clear()
    before = trainer.rng.bit_generator.state
    unary = trainer.policy.forward(
        [State.from_smiles("[33*]NCC")], beta=torch.ones(1), preferences=torch.ones(1, 1)
    )
    assert draws == [id(trainer.policy.subsampling[None])]
    assert trainer.rng.bit_generator.state == before
    assert len(unary.action_logits) == 1
    assert unary.action_logits[0].subspace.sample_indices.tolist() == [0]
    assert unary.action_logits[0].log_importance.tolist() == [0.0]
    assert unary.sample(1.0, 0.0, 1.0)[0].action_type == ActionType.UNIRXN_TRANSFORM
    dead = State.from_smiles("[33*]NCC", trainer.env.max_reactions - 1)
    choices = trainer.policy.sample_actions(
        [dead, initial],
        1.0,
        0.0,
        beta=torch.ones(len([dead, initial])),
        preferences=torch.ones(len([dead, initial]), 1),
    )
    assert choices[0] is None and choices[1].action_type == ActionType.FIRST_SYNTHON


def test_only_selected_actions_are_materialized(prepared_env, tmp_path, monkeypatch):
    import rxnflow.core.types as policy_module

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    initial = trainer.env.initial_state()
    observed = Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=0)
    constructed = []

    def record_action(*args, **kwargs):
        action = Action(*args, **kwargs)
        constructed.append(action)
        return action

    monkeypatch.setattr(policy_module, "Action", record_action)
    probabilities = trainer.policy.log_prob(
        [initial, initial],
        [observed, observed],
        beta=torch.ones(len([initial, initial])),
        preferences=torch.ones(len([initial, initial]), 1),
    )
    assert torch.isfinite(probabilities).all()
    assert not constructed
    selected = trainer.policy.sample_actions(
        [initial, initial],
        1.0,
        0.0,
        beta=torch.ones(len([initial, initial])),
        preferences=torch.ones(len([initial, initial]), 1),
    )
    assert constructed == selected
    assert len(constructed) == 2


def test_observed_edge_outside_subsample_matches_reference_normalizer(
    prepared_env, tmp_path, monkeypatch
):
    import math

    from rxnflow.gflownet.policy import SubsamplingPolicy

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    trainer.model.eval()
    monkeypatch.setattr(
        SubsamplingPolicy,
        "sample",
        lambda self: (np.array([0]), math.log(self.num_actions)),
    )
    state = trainer.env.initial_state()
    actions = [
        Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=i)
        for i in (0, 1)
    ]
    categorical = trainer.policy.forward(
        [state, state],
        beta=torch.ones(len([state, state])),
        preferences=torch.ones(len([state, state]), 1),
    )
    assert all(
        p.subspace.sample_indices.tolist() == [0] for p in categorical.action_logits
    )
    monkeypatch.setattr(trainer.policy, "forward", lambda *args: categorical)
    actual = trainer.policy.log_prob(
        [state, state],
        actions,
        beta=torch.ones(len([state, state])),
        preferences=torch.ones(len([state, state]), 1),
    )
    numerator = torch.stack(
        [
            get_synthon_logits(
                trainer.model,
                categorical.graph_emb[i : i + 1],
                "first_synthon",
                "1",
                torch.tensor([a.synthon_index]),
                logit_scale=trainer.model.logit_scale(_condition(trainer.model, 1)),
            )[0]
            for i, a in enumerate(actions)
        ]
    )
    denominator = torch.logsumexp(
        torch.cat([p.logits + p.log_importance for p in categorical.action_logits], 1),
        1,
    )
    expected = (numerator - denominator).clamp(max=0)
    torch.testing.assert_close(actual, expected)
    params = tuple(trainer.model.parameters())
    a = torch.autograd.grad(actual.sum(), params, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(expected.sum(), params, allow_unused=True)
    for left, right in zip(a, b, strict=True):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)


def test_sampler_loads_checkpoint_once_on_cpu(prepared_env, tmp_path, monkeypatch):
    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    checkpoint = trainer.save_checkpoint()
    assert (
        checkpoint.read_bytes() == (trainer.checkpoint_dir / "latest.ckpt").read_bytes()
    )
    original = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs.get("map_location"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    sampler = RxnFlowSampler(checkpoint)
    assert calls == ["cpu"]
    assert "retro_analyzer" not in sampler.env.__dict__


def test_reverse_results_overlap_forward_and_terminal_batch_is_drained(
    prepared_env, tmp_path, monkeypatch
):
    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    policy = trainer.policy
    events, pending = [], []
    initial = trainer.env.initial_state()
    middle = State.from_smiles("[1*]N")
    terminal = State.from_smiles("NC", num_reactions=1, terminated=True)
    first = Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=0)
    last = Action(
        ActionType.BIRXN_BRICK,
        reaction="amide_coupling_synthon_first",
        library_name="3",
        synthon_index=0,
    )

    def choose(states, *args):
        events.append("forward")
        return [first if state == initial else last for state in states]

    def submit(key, smiles, depth, known):
        events.append("submit")
        # Reverse alternatives may use more reactions than the observed prefix.
        assert depth == trainer.env.max_reactions
        # A parent's routes must be collected before extending the next edge.
        assert known[0][0][1] == ("" if smiles == middle.smiles else middle.smiles)
        if smiles == terminal.smiles:
            assert known[0][1:] == [(first, "")]
        pending.append((key, known))

    def result():
        events.append("collect")
        result = list(pending)
        pending.clear()
        return result

    analyzer = trainer.env.retro_analyzer
    monkeypatch.setattr(policy, "sample_actions", choose)
    monkeypatch.setattr(
        trainer.env,
        "step",
        lambda state, action: middle if state == initial else terminal,
    )
    monkeypatch.setattr(analyzer, "submit", submit)
    monkeypatch.setattr(analyzer, "result", result)
    monkeypatch.setattr(policy, "calc_bck_logprob", lambda *args: -0.5)
    trajectories = policy.sample_from_model(
        1, beta=torch.ones(1), preferences=torch.ones(1, 1)
    )
    assert events == [
        "forward",
        "collect",
        "submit",
        "forward",
        "collect",
        "submit",
        "collect",
    ]
    assert trajectories[0].valid and not pending
    assert [step.log_p_B for step in trajectories[0].steps] == [-0.5, -0.5]


def _condition(model, count):
    return model.encode_cond(torch.ones(count), torch.ones(count, 1))


def test_training_error_closes_analyzer_and_allows_reuse(
    prepared_env, tmp_path, monkeypatch
):
    from unittest.mock import Mock

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        CarbonReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    analyzer = Mock()
    trainer.env.__dict__["retro_analyzer"] = analyzer
    with monkeypatch.context() as patch:
        patch.setattr(
            trainer.condition_sampler,
            "sample",
            Mock(side_effect=RuntimeError("interrupted")),
        )
        with pytest.raises(RuntimeError, match="interrupted"):
            trainer.run(1)
    analyzer.close.assert_called_once()
    assert "retro_analyzer" not in trainer.env.__dict__
    trainer.close()
    analyzer.close.assert_called_once()
    trainer.run(1)
    assert "retro_analyzer" not in trainer.env.__dict__


def test_extracted_ema_model_matches_checkpoint(prepared_env, tmp_path, capsys):
    import subprocess
    import sys

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    checkpoint = trainer.save_checkpoint()
    extracted = tmp_path / "model.pt"
    subprocess.run(
        [
            sys.executable,
            "scripts/extract_model.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(extracted),
        ],
        check=True,
    )
    payload = torch.load(extracted, map_location="cpu", weights_only=False)
    assert set(payload) == {
        "config",
        "sampling_model",
        "objectives",
        "templates",
        "rxnflow_version",
    }
    full = RxnFlowSampler(checkpoint)
    compact = RxnFlowSampler(extracted)
    for name, value in trainer.sampling_model.state_dict().items():
        torch.testing.assert_close(compact.model.state_dict()[name], value)
    expected = full.sample(2, seed=11, beta=("fixed", [1.0]))
    actual = compact.sample(2, seed=11, beta=("fixed", [1.0]))
    assert [r.to_dict() for r in actual] == [r.to_dict() for r in expected]
    output = capsys.readouterr().out
    assert "Training reward config:" in output
    assert "Sampling settings:" in output


def test_sampling_rejects_untrained_conditions(prepared_env, tmp_path):
    config = tiny_config(prepared_env)
    config.reward.moo_preferences = ("fixed", [1.0])
    trainer = RxnFlowTrainer(
        config, QEDReward(), output_dir=tmp_path / "run", device="cpu", seed=7
    )
    sampler = RxnFlowSampler(trainer.save_checkpoint())
    for beta in (("fixed", [2.0]), ("uniform", [1.0, 2.0])):
        with pytest.raises(ValueError, match="fixed-beta"):
            sampler.sample(1, beta=beta)
    with pytest.raises(ValueError, match="fixed-preference"):
        sampler.sample(1, beta=("fixed", [1.0]), preferences=("dirichlet", [1.0]))
    sampler.config.reward.beta = ("uniform", [1.0, 4.0])
    with pytest.raises(ValueError, match="training range"):
        sampler.sample(1, beta=("uniform", [2.0, 5.0]))
    sampler.config.reward.moo_preferences = ("none", [])
    with pytest.raises(ValueError, match="conditioning mode"):
        sampler.sample(1, beta=("fixed", [2.0]), preferences=("fixed", [1.0]))
    # A conditioned model can use fixed beta values within its training range.
    assert len(sampler.sample(1, seed=11, beta=("fixed", [2.0]))) == 1


def test_sampling_catalog_replacement_checks_templates(prepared_env, tmp_path):
    from rxnflow.envs.prepare import convert_stage, features_stage

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env),
        QEDReward(),
        output_dir=tmp_path / "run",
        device="cpu",
        seed=7,
    )
    checkpoint = trainer.save_checkpoint()
    catalog = tmp_path / "catalog.smi"
    catalog.write_text("CC(=O)O\tacid\nCCN\tamine\n")
    replacement = tmp_path / "replacement"
    convert_stage(
        catalog, replacement, Path("tests/fixtures/templates.yaml"), min_library_size=1
    )
    features_stage(replacement)
    sampler = RxnFlowSampler(checkpoint, env_dir=replacement)
    assert sampler.env.signature != trainer.env.signature
    assert sampler.env.synthon_type_to_index == trainer.env.synthon_type_to_index
    assert len(sampler.sample(1, beta=("fixed", [1.0]), seed=11)) == 1

    for name in ("reaction", "synthon", "exclude_smarts"):
        original = sampler.env.templates[name]
        changed = dict(sampler.env.templates)
        changed[name] = {"different": True}
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["templates"] = changed
        incompatible = tmp_path / f"{name}.ckpt"
        torch.save(payload, incompatible)
        with pytest.raises(ValueError, match="definitions differ"):
            RxnFlowSampler(incompatible, env_dir=replacement)
        assert sampler.env.templates[name] == original
    # YAML comments are not part of the chemical definitions.
    with (replacement / "synthon.yaml").open("a") as handle:
        handle.write("\n# catalog-specific comment\n")
    assert (
        RxnFlowSampler(checkpoint, env_dir=replacement).env.templates
        == trainer.env.templates
    )
