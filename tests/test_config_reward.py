from pathlib import Path

import pytest
import yaml

from rxnflow import (
    Config,
    DataConfig,
    QEDReward,
    RewardConfig,
    RewardFunction,
    Sample,
    SubsamplingConfig,
    evaluate_rewards,
)
from rxnflow.config import TrainingConfig
from rxnflow.core.replay import ReplayBuffer
from rxnflow.types import Trajectory


class AtomCountReward(RewardFunction):
    def score(self, samples: list[Sample]) -> list[float]:
        return [float(sample.mol.GetNumHeavyAtoms()) for sample in samples]


class BrokenReward(RewardFunction):
    def score(self, samples: list[Sample]) -> list[float]:
        return [-1.0 for _ in samples]


class ScaledAtomCountReward(RewardFunction):
    def __init__(self, scale: float, options: dict[str, bool]):
        self.scale = scale
        self.options = options

    def score(self, samples: list[Sample]) -> list[float]:
        sign = 1.0 if self.options["positive"] else -1.0
        return [sign * self.scale * sample.mol.GetNumHeavyAtoms() for sample in samples]


def test_config_round_trip_and_validation(tmp_path: Path) -> None:
    config = Config(
        data=DataConfig(env_dir="example", max_atoms=50),
        property_penalty={"mw": 500.0},
    )
    path = tmp_path / "config.yaml"
    config.save(path)
    assert Config.from_file(path) == config
    assert config.to_dict()["data"]["max_atoms"] == 50
    saved = yaml.safe_load(path.read_text())
    assert saved["run"]["output_dir"] == "runs/rxnflow"
    assert saved["reward"]["exponent"] == 32.0
    assert saved["property_penalty"] == {"mw": 500.0}
    assert "property_limits" not in saved["data"]
    assert saved["training"]["learning_rate"] == 1e-4

    minimal = tmp_path / "minimal.yaml"
    minimal.write_text(
        "data:\n  env_dir: example\n"
        "run:\n  output_dir: run\n"
        "reward:\n  exponent: 8\n  settings:\n    scale: 2\n    options:\n      positive: true\n"
        "property_penalty:\n  tpsa: 140\n"
        "training:\n  steps: 25\n"
    )
    loaded = Config.from_file(minimal)
    assert loaded.training.steps == 25
    assert loaded.training.learning_rate == 1e-4
    assert loaded.output_dir == "run"
    assert loaded.reward.exponent == 8
    assert loaded.property_penalty == {"tpsa": 140.0}
    reward = ScaledAtomCountReward(**loaded.reward.settings)
    assert evaluate_rewards(reward, ["CCO"])[0] == [6.0]
    with pytest.raises(ValueError):
        DataConfig(max_atoms=0).validate()
    with pytest.raises(ValueError):
        SubsamplingConfig(sampling_ratio=1.1).validate()
    with pytest.raises(ValueError):
        SubsamplingConfig(importance_temp=-0.1).validate()
    with pytest.raises(ValueError):
        RewardConfig(exponent=0).validate()
    with pytest.raises(ValueError):
        RewardConfig(floor=0).validate()
    with pytest.raises(ValueError, match="mapping"):
        RewardConfig(settings=[]).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown property_penalty"):
        Config(data=DataConfig(env_dir="example"), property_penalty={"unknown": 1}).validate()
    with pytest.raises(ValueError, match="must be positive"):
        Config(data=DataConfig(env_dir="example"), property_penalty={"mw": 0}).validate()

    invalid_settings = tmp_path / "invalid-settings.yaml"
    invalid_settings.write_text("data:\n  env_dir: example\nreward:\n  settings: []\n")
    with pytest.raises(ValueError, match="reward.settings must be a mapping"):
        Config.from_file(invalid_settings)

    root_run_field = tmp_path / "root-run-field.yaml"
    root_run_field.write_text("data:\n  env_dir: example\noutput_dir: run\n")
    with pytest.raises(ValueError, match="must be configured under run"):
        Config.from_file(root_run_field)


def test_checked_in_minimal_and_complete_configs_load() -> None:
    root = Path(__file__).parents[1]
    minimal = Config.from_file(root / "configs" / "qed.yaml")
    complete = Config.from_file(root / "configs" / "template.yaml")
    assert minimal.reward == RewardConfig(exponent=32.0)
    assert complete.reward == RewardConfig(exponent=32.0, floor=1e-4, settings={})
    assert minimal.property_penalty == {"mw": 500.0}
    assert complete.property_penalty == {}
    assert minimal.training.batch_size == 128
    assert minimal.training.replay_batch_size == 128
    assert evaluate_rewards(QEDReward(**minimal.reward.settings), ["CCO"])[0][0] > 0
    assert list(complete.to_file_dict()) == [
        "data",
        "run",
        "reward",
        "property_penalty",
        "subsampling",
        "model",
        "training",
    ]


def test_zero_replay_capacity_disables_storage() -> None:
    TrainingConfig(replay_capacity=0).validate()
    buffer = ReplayBuffer(0)
    buffer.add([Trajectory(steps=[], final_smiles="CC")])
    assert len(buffer) == 0
    assert buffer.state_dict()["items"] == []


def test_qed_and_custom_reward_alignment() -> None:
    values, _ = evaluate_rewards(QEDReward(), ["CCO", None])
    assert 0 < values[0] <= 1
    assert values[1] == 0
    custom, _ = evaluate_rewards(AtomCountReward(), ["CCO"])
    assert custom == [3.0]
    filtered, _ = evaluate_rewards(
        AtomCountReward(),
        ["CCO", "CCCC"],
        sample_filter=lambda sample: sample.mol.GetNumHeavyAtoms() <= 3,
    )
    assert filtered == [3.0, 0.0]
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_rewards(BrokenReward(), ["CCO"])
