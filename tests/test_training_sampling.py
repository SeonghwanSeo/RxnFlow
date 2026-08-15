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

    smi_path = tmp_path / "samples.smi"
    csv_path = tmp_path / "samples.csv"
    json_path = tmp_path / "samples.json"
    sampler.write(results, smi_path)
    sampler.write(results, csv_path)
    sampler.write(results, json_path)
    assert len(smi_path.read_text().splitlines()) == 3
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]["trajectory"])[0]["type"] == "SET_WORKFLOW"
    structured = json.loads(json_path.read_text())
    assert structured[0]["intermediates"]
