import csv
import json
from pathlib import Path

import pytest
import torch

from rxnflow import (
    Config,
    DataConfig,
    QEDReward,
    RewardConfig,
    RewardFunction,
    RxnFlowSampler,
    RxnFlowTrainer,
    Sample,
    SubsamplingConfig,
    __version__,
)
from rxnflow.config import ModelConfig, TrainingConfig
from rxnflow.types import (
    ActionKind,
    MoleculeState,
    RxnAction,
    Trajectory,
    TrajectoryStep,
)


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

    sampler = RxnFlowSampler(config, restarted_checkpoint, reward=CarbonReward())
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
    trainer.runtime.action_log_probability = lambda state, action: trainer.model.log_z * 0
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
    assert torch.isclose(trainer._loss([trajectory]), torch.tensor(1.5))


def test_site_outcome_scoring_and_observed_action_log_probability(
    prepared_env: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from rxnflow.gflownet.runtime import NoValidActions

    trainer = RxnFlowTrainer(tiny_config(prepared_env, tmp_path / "sites"), QEDReward())
    runtime = trainer.runtime
    state = MoleculeState("[3*]C")
    group = next(
        g
        for g in trainer.env.available_groups(state)
        if g.name == "rxn1_b0" and g.block_type == "1-1"
    )
    library = trainer.env.blocks["1-1"]
    # Pick a linker with non-equivalent amines, then require the selected block
    # in a fresh TB denominator even when normal subsampling would miss it.
    actions = next(
        values
        for i in range(len(library))
        if len(values := trainer.env.outcomes(state, group, i)) == 2
    )
    candidates = runtime.candidates(state, required=actions[0])
    positions = [candidates.actions.index(action) for action in actions]
    assert not torch.isclose(
        candidates.logits[positions[0]], candidates.logits[positions[1]]
    )
    probability = runtime.action_log_probability(state, actions[1])
    assert torch.isfinite(probability) and probability <= 0
    (-probability).backward()
    assert trainer.model.outcome_encoder[0].weight.grad.abs().sum() > 0
    # Missing a terminal action on the last step is a dead end, never a cap.
    with pytest.raises(NoValidActions):
        runtime.candidates(MoleculeState("[33*]NCC", trainer.env.max_reactions - 1))
    with pytest.raises(ValueError, match="not available"):
        trainer.env.step(state, replace(actions[0], product_smiles="CC"))
