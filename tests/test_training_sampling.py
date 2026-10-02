import csv
import json
import math
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from rxnflow import __version__
from rxnflow.config import (
    Config,
    DataConfig,
    ModelConfig,
    RewardConfig,
    SubsamplingConfig,
    TrainingConfig,
)
from rxnflow.gflownet.types import (
    Action,
    ActionKind,
    MoleculeState,
    Trajectory,
    Transition,
)
from rxnflow.reward import QEDReward, RewardFunction
from rxnflow.sampler import RxnFlowSampler
from rxnflow.trainer import RxnFlowTrainer


class CarbonReward(RewardFunction):
    def score(self, molecules: list[Chem.Mol]) -> list[float]:
        return [
            float(sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())) / 10
            for mol in molecules
        ]


def tiny_config(env_dir: Path, output_dir: Path) -> Config:
    return Config(
        data=DataConfig(env_dir=str(env_dir), max_atoms=20),
        reward=RewardConfig(exponent=1.0),
        subsampling=SubsamplingConfig(
            sampling_ratio=0.5, min_sampling=1, importance_temp=1.0
        ),
        model=ModelConfig(hidden_dim=32, num_layers=1, dropout=0.0),
        training=TrainingConfig(
            steps=1,
            batch_size=2,
            replay_batch_size=1,
            replay_capacity=16,
            learning_rate=1e-3,
            checkpoint_every=1,
            log_every=1,
            retrosynthesis_workers=0,
        ),
        output_dir=str(output_dir),
        seed=7,
        device="cpu",
    )


def test_training_restart_sampling_and_output_formats(
    prepared_env: Path, tmp_path: Path
) -> None:
    config = tiny_config(prepared_env, tmp_path / "run")
    checkpoint = RxnFlowTrainer(config, QEDReward()).run()
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["rxnflow_version"] == __version__

    with pytest.raises(ValueError, match="reward implementation"):
        RxnFlowTrainer(config, CarbonReward(), restart=checkpoint)

    restarted = RxnFlowTrainer(config, QEDReward(), restart=checkpoint)
    restarted_checkpoint = restarted.run(1)
    assert restarted.step == 2
    assert restarted_checkpoint.is_file()

    records = [
        json.loads(line)
        for line in (Path(config.output_dir) / "training.jsonl").read_text().splitlines()
    ]
    samples = [
        json.loads(line)
        for line in (Path(config.output_dir) / "samples.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in records] == [1, 2]
    assert [row["replay_count"] for row in records] == [0, 1]
    assert len(samples) == 4  # Fresh attempts only, never duplicate replay entries.
    assert {(row["step"], row["sample"]) for row in samples} == {
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }
    for record in records:
        fresh = [row for row in samples if row["step"] == record["step"]]
        assert record["mean_reward"] == pytest.approx(
            sum(row["reward"] for row in fresh) / len(fresh)
        )
        assert sum(record["reaction_counts"].values()) == len(fresh)
        assert sum(record["action_counts"].values()) == sum(
            len(row["steps"]) for row in fresh
        )
        assert sum(record["invalid_reasons"].values()) == sum(
            not row["valid"] for row in fresh
        )
        assert record["policy_grad_norm"] >= 0
        assert record["grad_norm"] >= record["policy_grad_norm"]
        assert record["policy_grad_clipped"] == float(record["policy_grad_norm"] > 100)
        assert record["batch_entropy"] == -record["mean_log_pf"]
    # Log serialization remains usable after replay eviction/restart.
    for row in samples:
        value = {key: val for key, val in row.items() if key not in ("step", "sample")}
        assert Trajectory.from_dict(value).to_dict() == value

    sampler = RxnFlowSampler(restarted_checkpoint, reward=CarbonReward())
    results = sampler.sample(3, seed=11)
    assert len(results) == 3
    assert all(
        result.smiles and result.trajectory and result.reward is not None
        for result in results
    )
    assert all("workflow" not in result.to_dict() for result in results)
    assert all(len(result.intermediates) == len(result.trajectory) for result in results)
    for result in results:
        assert [
            step["product_smiles"] for step in result.trajectory
        ] == result.intermediates
        assert result.intermediates[-1] == result.smiles

    smi_path = tmp_path / "samples.smi"
    csv_path = tmp_path / "samples.csv"
    json_path = tmp_path / "samples.json"
    sampler.write(results, smi_path)
    sampler.write(results, csv_path)
    sampler.write(results, json_path)
    assert len(smi_path.read_text().splitlines()) == 3
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]["trajectory"])[0]["kind"] == "FIRST_BLOCK"
    assert "*" not in rows[0]["smiles"]
    structured = json.loads(json_path.read_text())
    assert structured[0]["intermediates"]
    assert "workflow" not in structured[0]


def test_trajectory_balance_uses_backward_probability(
    prepared_env: Path, tmp_path: Path
) -> None:
    trainer = RxnFlowTrainer(tiny_config(prepared_env, tmp_path / "tb"), QEDReward())
    trainer.model.log_z.data.zero_()
    trainer.policy.action_log_probabilities = lambda states, actions: (
        trainer.model.log_z.expand(len(states)) * 0
    )
    trajectory = Trajectory(
        steps=[
            Transition(
                state=MoleculeState(),
                action=Action(ActionKind.FIRST_BLOCK),
                product_smiles="[1*]N",
                log_pb=-2.0,
            )
        ],
        final_smiles="N",
        reward=1.0,
    )
    loss, _ = trainer._loss([trajectory], num_fresh=1)
    assert torch.isclose(loss, torch.tensor(4.0))


def test_tb_diagnostics_separate_fresh_replay_and_invalid(prepared_env, tmp_path):
    config = tiny_config(prepared_env, tmp_path / "diagnostics")
    config.reward.floor = math.exp(-4)
    trainer = RxnFlowTrainer(config, QEDReward())
    trainer.model.log_z.data.zero_()
    trainer.policy.action_log_probabilities = lambda states, actions: (
        torch.tensor([-2.0, -3.0, -1.0]) + trainer.model.log_z * 0
    )
    batch = [
        Trajectory(
            steps=[Transition(MoleculeState(), Action(ActionKind.FIRST_BLOCK), "C", pb)],
            final_smiles="C",
            reward=reward,
            valid=valid,
        )
        for pb, reward, valid in [
            (-0.5, math.exp(-1), True),
            (-1.0, 0.0, False),
            (-0.25, math.exp(-2), True),
        ]
    ]
    loss, info = trainer._loss(batch, num_fresh=2)
    # Hand-calculated residuals: -0.5, 2, 1.25. The invalid reward uses the floor.
    expected = {
        "loss": 1.9375,
        "log_z": 0.0,
        "batch_entropy": 2.0,
        "mean_log_pf": -2.0,
        "mean_log_pb": -1.75 / 3,
        "mean_log_reward": -7 / 3,
        "mean_tb_residual": 2.75 / 3,
        "fresh_loss": 2.125,
        "replay_loss": 1.5625,
        "valid_loss": 0.90625,
        "invalid_loss": 4.0,
        "invalid_logprob": -3.0,
        "batch_invalid_fraction": 1 / 3,
    }
    for key, value in expected.items():
        assert not info[key].requires_grad
        assert info[key].item() == pytest.approx(value)
    loss.backward()
    assert trainer.model.log_z.grad.item() == pytest.approx(5.5 / 3)

    # No valid samples, replay, or transitions: diagnostics remain finite.
    loss, info = trainer._loss([Trajectory([], "", valid=False)], num_fresh=1)
    assert loss.item() == pytest.approx(16.0)
    assert info["valid_loss"].item() == info["replay_loss"].item() == 0
    assert all(torch.isfinite(value) for value in info.values())


def test_restart_reproduces_next_update_with_dropout(
    prepared_env: Path, tmp_path: Path
) -> None:
    config = tiny_config(prepared_env, tmp_path / "dropout")
    config.model.dropout = 0.2
    trainer = RxnFlowTrainer(config, QEDReward())
    first_checkpoint = trainer.run()
    trainer.run(1)
    expected = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    expected_schedule = trainer.lr_scheduler.state_dict()
    expected_rates = [group["lr"] for group in trainer.optimizer.param_groups]
    expected_ema = {
        key: value.clone() for key, value in trainer.sampling_model.state_dict().items()
    }

    restarted = RxnFlowTrainer(config, QEDReward(), restart=first_checkpoint)
    restarted.run(1)
    assert restarted.lr_scheduler.state_dict() == expected_schedule
    assert [group["lr"] for group in restarted.optimizer.param_groups] == expected_rates
    for key, value in restarted.model.state_dict().items():
        assert torch.equal(value, expected[key]), key
    for key, value in restarted.sampling_model.state_dict().items():
        assert torch.equal(value, expected_ema[key]), key


def test_oriented_block_scoring_and_observed_action_log_probability(
    prepared_env, tmp_path
):
    from rxnflow.gflownet.policy import NoValidActions

    trainer = RxnFlowTrainer(tiny_config(prepared_env, tmp_path / "sites"), QEDReward())
    state = MoleculeState.from_smiles("[3*]C")
    group = next(
        g
        for g in trainer.env.available_groups(state)
        if g.name == "rxn1_block_first" and g.block_type == "1-1"
    )
    actions = next(
        values
        for i in range(len(trainer.env.blocks["1-1"]))
        if (values := trainer.env.outcomes(state, group, i))
    )
    probability = trainer.policy.action_log_probability(state, actions[0])
    assert torch.isfinite(probability) and probability <= 0
    (-probability).backward()
    assert trainer.model.block_encoder[0].weight.grad is not None
    with pytest.raises(NoValidActions):
        trainer.policy.choose_action(
            MoleculeState.from_smiles("[33*]NCC", trainer.env.max_reactions - 1), 1.0, 0.0
        )


def test_subsampling_precedes_budget_mask_without_candidate_reactions(
    prepared_env, tmp_path, monkeypatch
):
    import math

    from rxnflow.envs.chemistry.features import PROPERTY_NAMES

    config = tiny_config(prepared_env, tmp_path / "budget")
    config.property_penalty = {"mw": 100.0}
    trainer = RxnFlowTrainer(config, QEDReward())
    env = trainer.env
    name = next(n for n in env.brick_types if len(env.blocks[n]) > 1)
    mw = PROPERTY_NAMES.index("mw")
    for library in env.blocks.values():
        library.properties[:, mw] = 200.0
    target = len(env.blocks[name]) - 1
    env.blocks[name].properties[target, mw] = 50.0
    monkeypatch.setattr(
        env, "outcomes", lambda *args: pytest.fail("candidate scoring executed chemistry")
    )
    original = env.block_mask

    def mask(properties, block_type, indices=None):
        assert indices is not None
        return original(properties, block_type, indices)

    monkeypatch.setattr(env, "block_mask", mask)
    found = missed = 0
    for seed in range(20):
        trainer.policy.generator.manual_seed(seed)
        group = trainer.policy.candidate_batch([env.initial_state()]).action_groups[0]
        valid = torch.isfinite(group.logits[0])
        # Masking retains every sampled column, even when all actions are invalid.
        assert group.logits.shape[1] == sum(map(len, group.block_indices))
        assert group.logits.shape[1] > 1
        if not valid.any():
            missed += 1
            continue
        found += 1
        position = int(valid.nonzero().flatten()[0])
        assert valid.sum() == 1
        action = group.action_at(position)
        assert action.block_type == name and action.block_index == target
        count = trainer.policy.block_subsamplers[name].count
        assert group.log_importance[position].item() == pytest.approx(
            math.log(len(env.blocks[name]) / count)
        )
    assert found and missed


def test_failed_selected_action_is_retained_for_tb(
    prepared_env: Path, tmp_path: Path, monkeypatch
) -> None:
    config = tiny_config(prepared_env, tmp_path / "invalid")
    config.data.max_atoms = 5
    trainer = RxnFlowTrainer(config, QEDReward())
    # Reactant budget fits (4 + 1), but the amidation inserts two more atoms.
    state = MoleculeState.from_smiles("[1*]NCCN")
    index = trainer.env.blocks["3"].smiles.index("*C")
    action = Action(
        ActionKind.BI_REACTION,
        reaction="rxn1_state_first",
        block_type="3",
        block_index=index,
    )
    monkeypatch.setattr(trainer.env, "initial_state", lambda: state)
    monkeypatch.setattr(
        trainer.sampling_policy,
        "choose_actions",
        lambda states, *args: [action] * len(states),
    )
    trajectory = trainer.sampling_policy.rollout(analyze_backward=False)
    assert not trajectory.valid
    assert "graph-capacity" in trajectory.invalid_reason
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action == action
    assert trajectory.steps[0].product_smiles == ""
    trainer._assign_rewards([trajectory])
    assert trajectory.reward == 0
    loss, _ = trainer._loss([trajectory], num_fresh=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert trainer.model.bi_reaction_head[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("sampling_ratio", [0.35, 1.0])
def test_batched_scores_and_gradients_match_scalar_reference(
    prepared_env, tmp_path, sampling_ratio
):
    from rxnflow.envs.graph import GraphBatch, molecule_to_graph_data

    config = tiny_config(prepared_env, tmp_path / "batched")
    config.subsampling.sampling_ratio = sampling_ratio
    trainer = RxnFlowTrainer(config, QEDReward())
    model = trainer.model.eval()
    states = [
        MoleculeState(),
        MoleculeState.from_smiles("[3*]C"),
        MoleculeState.from_smiles("[11*]C"),
        MoleculeState.from_smiles("[33*]NCC"),
    ]
    categorical = trainer.policy.candidate_batch(states)
    reference, actual = [], []
    for row, state in enumerate(states):
        embedding = model.encode_graphs(
            GraphBatch.from_graphs(
                [
                    molecule_to_graph_data(
                        state.mol, trainer.env.max_atoms, state.reaction_count
                    )
                ]
            )
        )
        for group in categorical.action_groups:
            for column in range(group.logits.shape[1]):
                if not torch.isfinite(group.logits[row, column]):
                    continue
                action = group.action_at(column)
                score = (
                    model.score_scalar(embedding, action.reaction)
                    if action.kind == ActionKind.UNI_REACTION
                    else model.score_blocks(
                        embedding,
                        group.name,
                        action.block_type,
                        torch.tensor([action.block_index]),
                    )[0]
                )
                reference.append(score)
                actual.append(group.logits[row, column])
    actual, expected = torch.stack(actual), torch.stack(reference)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
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
    from rxnflow.gflownet.subsampling import BlockSubsampler

    trainer = RxnFlowTrainer(tiny_config(prepared_env, tmp_path / "shared"), QEDReward())
    trainer.model.eval()
    draws = []
    original = BlockSubsampler.sample

    def sample(self, generator):
        draws.append(id(self))
        return original(self, generator)

    monkeypatch.setattr(BlockSubsampler, "sample", sample)
    initial = trainer.env.initial_state()
    categorical = trainer.policy.candidate_batch([initial, initial])
    assert len(draws) == len(set(draws))
    for group in categorical.action_groups:
        torch.testing.assert_close(group.logits[0], group.logits[1])
    dead = MoleculeState.from_smiles("[33*]NCC", trainer.env.max_reactions - 1)
    choices = trainer.policy.choose_actions([dead, initial], 1.0, 0.0)
    assert choices[0] is None and choices[1].kind == ActionKind.FIRST_BLOCK


def test_only_selected_actions_are_materialized(prepared_env, tmp_path, monkeypatch):
    import rxnflow.gflownet.categorical as policy_module

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env, tmp_path / "index-actions"), QEDReward()
    )
    initial = trainer.env.initial_state()
    observed = Action(ActionKind.FIRST_BLOCK, block_type="1", block_index=0)
    constructed = []

    def record_action(*args, **kwargs):
        action = Action(*args, **kwargs)
        constructed.append(action)
        return action

    monkeypatch.setattr(policy_module, "Action", record_action)
    probabilities = trainer.policy.action_log_probabilities(
        [initial, initial], [observed, observed]
    )
    assert torch.isfinite(probabilities).all()
    assert not constructed
    selected = trainer.policy.choose_actions([initial, initial], 1.0, 0.0)
    assert constructed == selected
    assert len(constructed) == 2


def test_observed_edge_outside_subsample_matches_reference_normalizer(
    prepared_env, tmp_path, monkeypatch
):
    import math

    from rxnflow.gflownet.subsampling import BlockSubsample, BlockSubsampler

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env, tmp_path / "numerator"), QEDReward()
    )
    trainer.model.eval()
    monkeypatch.setattr(
        BlockSubsampler,
        "sample",
        lambda self, generator: BlockSubsample(torch.tensor([0]), math.log(self.size)),
    )
    state = trainer.env.initial_state()
    actions = [
        Action(ActionKind.FIRST_BLOCK, block_type="1", block_index=i) for i in (0, 1)
    ]
    categorical = trainer.policy.candidate_batch([state, state])
    assert all(
        indices.tolist() == [0]
        for p in categorical.action_groups
        for indices in p.block_indices
    )
    monkeypatch.setattr(trainer.policy, "candidate_batch", lambda *args: categorical)
    actual = trainer.policy.action_log_probabilities([state, state], actions)
    numerator = torch.stack(
        [
            trainer.model.score_blocks(
                categorical.embeddings[i : i + 1],
                "first_block",
                "1",
                torch.tensor([a.block_index]),
            )[0]
            for i, a in enumerate(actions)
        ]
    )
    denominator = torch.logsumexp(
        torch.cat([p.logits + p.log_importance for p in categorical.action_groups], 1), 1
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
        tiny_config(prepared_env, tmp_path / "single-load"), QEDReward()
    )
    checkpoint = trainer.save_checkpoint()
    assert (
        checkpoint.read_bytes()
        == (trainer.output_dir / "checkpoint_latest.pt").read_bytes()
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
    from rxnflow.envs.retrosynthesis import RetrosynthesisTree

    trainer = RxnFlowTrainer(tiny_config(prepared_env, tmp_path / "overlap"), QEDReward())
    policy = trainer.policy
    events, pending = [], []
    initial = trainer.env.initial_state()
    middle = MoleculeState.from_smiles("[1*]N")
    terminal = MoleculeState.from_smiles("NC", reaction_count=1, terminated=True)
    first = Action(ActionKind.FIRST_BLOCK, block_type="1", block_index=0)
    last = Action(
        ActionKind.BI_REACTION,
        reaction="rxn1_block_first",
        block_type="3",
        block_index=0,
    )

    def choose(states, *args):
        events.append("forward")
        return [first if state == initial else last for state in states]

    def submit(key, smiles, depth, known):
        events.append("submit")
        # A parent's tree must be collected before the child's reverse search.
        assert known[0][1].smiles == ("" if smiles == middle.smiles else middle.smiles)
        pending.append((key, RetrosynthesisTree(smiles, known)))

    def result():
        events.append("collect")
        result = list(pending)
        pending.clear()
        return result

    analyzer = trainer.env.retro_analyzer
    monkeypatch.setattr(policy, "choose_actions", choose)
    monkeypatch.setattr(
        trainer.env,
        "step",
        lambda state, action: middle if state == initial else terminal,
    )
    monkeypatch.setattr(analyzer, "submit", submit)
    monkeypatch.setattr(analyzer, "result", result)
    monkeypatch.setattr(analyzer, "tree_log_probability", lambda *args: -0.5)
    trajectories = policy.rollouts(1)
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
    assert [step.log_pb for step in trajectories[0].steps] == [-0.5, -0.5]
