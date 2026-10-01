import csv
import json
from pathlib import Path

import pytest
import torch

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
    ActionKind,
    MoleculeState,
    RxnAction,
    Sample,
    Trajectory,
    TrajectoryStep,
)
from rxnflow.reward import QEDReward, RewardFunction
from rxnflow.sampler import RxnFlowSampler
from rxnflow.trainer import RxnFlowTrainer


class CarbonReward(RewardFunction):
    def score(self, samples: list[Sample]) -> list[float]:
        return [
            float(sum(atom.GetAtomicNum() == 6 for atom in sample.mol.GetAtoms())) / 10
            for sample in samples
        ]


def tiny_config(env_dir: Path, output_dir: Path) -> Config:
    return Config(
        data=DataConfig(env_dir=str(env_dir), max_atoms=20),
        reward=RewardConfig(exponent=1.0),
        subsampling=SubsamplingConfig(
            sampling_ratio=0.5, min_sampling=1, importance_temp=1.0
        ),
        model=ModelConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0),
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

    sampler = RxnFlowSampler(restarted_checkpoint, reward=CarbonReward())
    results = sampler.sample(3, seed=11)
    assert len(results) == 3
    assert all(
        result.smiles and result.trajectory and result.reward is not None
        for result in results
    )
    assert all("workflow" not in result.to_dict() for result in results)
    assert all(len(result.intermediates) == len(result.trajectory) for result in results)

    smi_path = tmp_path / "samples.smi"
    csv_path = tmp_path / "samples.csv"
    json_path = tmp_path / "samples.json"
    sampler.write(results, smi_path)
    sampler.write(results, csv_path)
    sampler.write(results, json_path)
    assert len(smi_path.read_text().splitlines()) == 3
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]["trajectory"])[0]["type"] == "FIRST_BLOCK"
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
            TrajectoryStep(
                state=MoleculeState(),
                action=RxnAction(ActionKind.FIRST_BLOCK, "[1*]N"),
                product_smiles="[1*]N",
                log_backward=-2.0,
            )
        ],
        final_smiles="N",
        reward=1.0,
    )
    assert torch.isclose(trainer._loss([trajectory]), torch.tensor(4.0))


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
    prepared_env: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from rxnflow.gflownet.policy import NoValidActions

    trainer = RxnFlowTrainer(tiny_config(prepared_env, tmp_path / "sites"), QEDReward())
    policy = trainer.policy
    state = MoleculeState.from_smiles("[3*]C")
    group = next(
        g
        for g in trainer.env.available_groups(state)
        if g.name == "rxn1_block_first" and g.block_type == "1-1"
    )
    library = trainer.env.blocks["1-1"]
    # Direction is part of the block index. Require that row when a fresh
    # subsample would otherwise omit the observed action from the TB denominator.
    actions = next(
        values
        for i in range(len(library))
        if (values := trainer.env.outcomes(state, group, i))
    )
    candidates = policy.candidates(state, required=actions[0])
    assert candidates.action_at(candidates.required_position) == replace(
        actions[0], product_smiles=""
    )
    probability = policy.action_log_probability(state, actions[0])
    assert torch.isfinite(probability) and probability <= 0
    (-probability).backward()
    assert trainer.model.block_encoder[0].weight.grad.abs().sum() > 0
    # Missing a terminal action on the last step is a dead end, never a cap.
    with pytest.raises(NoValidActions):
        policy.candidates(
            MoleculeState.from_smiles("[33*]NCC", trainer.env.max_reactions - 1)
        )
    with pytest.raises(ValueError, match="not available"):
        trainer.env.step(state, replace(actions[0], product_smiles="CC"))


def test_subsampling_precedes_budget_mask_without_candidate_reactions(
    prepared_env: Path, tmp_path: Path, monkeypatch
) -> None:
    from rxnflow.envs.chemistry.features import PROPERTY_NAMES

    config = tiny_config(prepared_env, tmp_path / "budget")
    config.property_penalty = {"mw": 100.0}
    trainer = RxnFlowTrainer(config, QEDReward())
    env = trainer.env
    name = next(name for name in env.brick_types if len(env.blocks[name]) > 1)
    # Exactly one row passes. Sampling may miss it: masking must not refill
    # the draw, and surviving rows retain full-library inclusion weights.
    mw = PROPERTY_NAMES.index("mw")
    for library in env.blocks.values():
        library.properties[:, mw] = 200.0
    target = len(env.blocks[name]) - 1
    env.blocks[name].properties[target, mw] = 50.0

    def unexpected_reaction(*args, **kwargs):
        raise AssertionError("candidate scoring executed chemistry")

    monkeypatch.setattr(env, "outcomes", unexpected_reaction)
    import math

    from rxnflow.gflownet.policy import NoValidActions

    original_mask = env.block_mask

    def sampled_mask(properties, block_type, indices=None):
        assert indices is not None, "policy must only mask sampled rows"
        return original_mask(properties, block_type, indices)

    monkeypatch.setattr(env, "block_mask", sampled_mask)
    found = missed = 0
    for seed in range(20):
        trainer.policy.generator.manual_seed(seed)
        try:
            candidates = trainer.policy.candidates(env.initial_state())
        except NoValidActions:
            missed += 1
            continue
        found += 1
        assert candidates.logits.numel() == 1
        assert candidates.action_at(0).block_type == name
        assert candidates.action_at(0).block_index == target
        size = len(env.blocks[name])
        count = min(
            size,
            max(
                config.subsampling.min_sampling,
                math.ceil(size * config.subsampling.sampling_ratio),
            ),
        )
        assert candidates.log_importance.item() == pytest.approx(math.log(size / count))
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
    action = RxnAction(
        ActionKind.BI_REACTION,
        "",
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
    loss = trainer._loss([trajectory])
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
    with torch.no_grad():
        model.logit_temperature.add_(
            torch.linspace(-0.7, 0.9, len(model.logit_temperature))
        )
    states = [
        MoleculeState(),
        MoleculeState.from_smiles("[3*]C"),
        MoleculeState.from_smiles("[11*]C"),
        MoleculeState.from_smiles("[33*]NCC"),
    ]
    calls = {"graph": 0, "blocks": 0}

    def count_graph(*args):
        calls["graph"] += 1

    def count_blocks(*args):
        calls["blocks"] += 1

    graph_hook = model.graph_encoder.register_forward_hook(count_graph)
    block_hook = model.fingerprint_encoder.register_forward_hook(count_blocks)
    batched = trainer.policy.candidate_batch(states)
    graph_hook.remove()
    block_hook.remove()
    assert calls == {"graph": 1, "blocks": 1}
    reference = []
    for state, candidates in zip(states, batched, strict=True):
        embedding = model.encode_graphs(
            GraphBatch.from_graphs(
                [
                    molecule_to_graph_data(
                        state.mol, trainer.env.max_atoms, state.reaction_count
                    )
                ]
            )
        )
        logits = []
        for position in range(candidates.logits.numel()):
            action = candidates.action_at(position)
            if action.kind == ActionKind.UNI_REACTION:
                score = model.score_scalar(embedding, action.reaction)
            else:
                score = model.score_blocks(
                    embedding,
                    "first_block"
                    if action.kind == ActionKind.FIRST_BLOCK
                    else action.reaction,
                    action.block_type,
                    torch.tensor([action.block_index]),
                )[0]
            logits.append(score)
        reference.append(torch.stack(logits))
        torch.testing.assert_close(candidates.logits, reference[-1], atol=2e-6, rtol=2e-5)
    batch_loss = sum(torch.logsumexp(value.logits, 0) for value in batched)
    batch_loss.backward()
    expected = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    sum(torch.logsumexp(value, 0) for value in reference).backward()
    actual = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], atol=1e-5, rtol=1e-4, msg=name
        )


def test_batch_shares_library_subsamples_and_handles_dead_ends(
    prepared_env, tmp_path, monkeypatch
):
    trainer = RxnFlowTrainer(
        tiny_config(prepared_env, tmp_path / "batch-masks"), QEDReward()
    )
    policy = trainer.policy
    initial = trainer.env.initial_state()
    calls = []
    original = trainer.env.block_mask

    def record_mask(properties, name, indices=None):
        assert indices is not None
        calls.append(name)
        return original(properties, name, indices)

    monkeypatch.setattr(trainer.env, "block_mask", record_mask)
    rng = policy.generator.get_state()
    batch = policy.candidate_batch([initial, initial])
    assert len(calls) == len(set(calls))
    policy.generator.set_state(rng)
    single = [policy.candidates(initial)] * 2
    for actual, expected in zip(batch, single, strict=True):
        assert [actual.action_at(i) for i in range(actual.logits.numel())] == [
            expected.action_at(i) for i in range(expected.logits.numel())
        ]
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.log_importance, expected.log_importance)
    # Different observed rows in the same library must share one conditional
    # draw, with both rows retained in both state denominators.
    from rxnflow.gflownet.subsampling import BlockSubsampler

    draws = []
    original_sample = BlockSubsampler.sample

    def record_sample(self, generator, required_indices=()):
        draws.append((id(self), set(required_indices)))
        return original_sample(self, generator, required_indices)

    monkeypatch.setattr(BlockSubsampler, "sample", record_sample)
    observed = [
        RxnAction(ActionKind.FIRST_BLOCK, "", block_type="1", block_index=i)
        for i in (0, 1)
    ]
    shared = policy.candidate_batch([initial, initial], observed)
    assert len(draws) == len({key for key, _ in draws})
    assert any(rows == {0, 1} for _, rows in draws)
    assert [shared[0].action_at(i) for i in range(shared[0].logits.numel())] == [
        shared[1].action_at(i) for i in range(shared[1].logits.numel())
    ]
    for result in shared:
        for action in observed:
            index = [result.action_at(i) for i in range(result.logits.numel())].index(
                action
            )
            assert result.log_importance[index] == 0
    dead = MoleculeState.from_smiles("[33*]NCC", trainer.env.max_reactions - 1)
    choices = policy.choose_actions([dead, initial], 1.0, 0.0)
    assert choices[0] is None and choices[1].kind == ActionKind.FIRST_BLOCK


def test_only_selected_actions_are_materialized(prepared_env, tmp_path, monkeypatch):
    import rxnflow.gflownet.policy as policy_module

    trainer = RxnFlowTrainer(
        tiny_config(prepared_env, tmp_path / "index-actions"), QEDReward()
    )
    initial = trainer.env.initial_state()
    observed = RxnAction(ActionKind.FIRST_BLOCK, "", block_type="1", block_index=0)
    constructed = []

    def record_action(*args, **kwargs):
        action = RxnAction(*args, **kwargs)
        constructed.append(action)
        return action

    monkeypatch.setattr(policy_module, "RxnAction", record_action)
    probabilities = trainer.policy.action_log_probabilities(
        [initial, initial], [observed, observed]
    )
    assert torch.isfinite(probabilities).all()
    assert not constructed
    selected = trainer.policy.choose_actions([initial, initial], 1.0, 0.0)
    assert constructed == selected
    assert len(constructed) == 2


def test_segmented_log_probabilities_match_scalar_reduction(
    prepared_env, tmp_path, monkeypatch
):
    trainer = RxnFlowTrainer(
        tiny_config(prepared_env, tmp_path / "segments"), QEDReward()
    )
    states = [trainer.env.initial_state()] * 2
    actions = [
        RxnAction(ActionKind.FIRST_BLOCK, "", block_type="1", block_index=i)
        for i in (0, 1)
    ]
    candidates = trainer.policy.candidate_batch(states, actions)
    monkeypatch.setattr(
        trainer.policy, "candidate_batch", lambda *args, **kwargs: candidates
    )
    actual = trainer.policy.action_log_probabilities(states, actions)
    expected = torch.stack(
        [
            value.logits[value.required_position]
            - torch.logsumexp(value.logits + value.log_importance, 0)
            for value in candidates
        ]
    )
    torch.testing.assert_close(actual, expected)
    inputs = [value.logits for value in candidates]
    actual_grad = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), inputs)
    for first, second in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(first, second)


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
