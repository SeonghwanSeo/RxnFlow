from pathlib import Path

import pytest
import torch
import yaml
from rdkit import Chem

from examples.qed import QEDReward
from rxnflow.config import (
    Config,
    DataConfig,
    GenerationConfig,
    RewardConfig,
    SubsamplingConfig,
    TrainingConfig,
)
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.gflownet.types import Trajectory
from rxnflow.reward import RewardFunction, evaluate_rewards


class AtomCountReward(RewardFunction):
    objectives = ("score",)

    def score(self, molecules: list[Chem.Mol]) -> torch.Tensor:
        return torch.tensor(
            [float(mol.GetNumHeavyAtoms()) for mol in molecules], dtype=torch.float32
        ).reshape(-1, 1)


class BrokenReward(RewardFunction):
    objectives = ("score",)

    def score(self, molecules: list[Chem.Mol]) -> torch.Tensor:
        return torch.tensor([-1.0 for _ in molecules], dtype=torch.float32).reshape(-1, 1)


class ScaledAtomCountReward(RewardFunction):
    objectives = ("score",)

    def __init__(self, scale: float, options: dict[str, bool]):
        self.scale = scale
        self.options = options

    def score(self, molecules: list[Chem.Mol]) -> torch.Tensor:
        sign = 1.0 if self.options["positive"] else -1.0
        return torch.tensor(
            [sign * self.scale * mol.GetNumHeavyAtoms() for mol in molecules],
            dtype=torch.float32,
        ).reshape(-1, 1)


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
    assert saved["reward"]["beta"] == "fixed(32.0)"
    assert saved["property_penalty"] == {"mw": 500.0}
    assert "property_limits" not in saved["data"]
    assert saved["training"]["learning_rate"] == 1e-4
    assert saved["generation"] == {"min_reactions": 1, "max_reactions": 3}

    minimal = tmp_path / "minimal.yaml"
    minimal.write_text(
        "data:\n  env_dir: example\n"
        "run:\n  output_dir: run\n"
        "reward:\n  beta: '8'\n  settings:\n    scale: 2\n    options:\n      positive: true\n"
        "property_penalty:\n  tpsa: 140\n"
        "training:\n  steps: 25\n"
    )
    loaded = Config.from_file(minimal)
    assert loaded.training.steps == 25
    assert loaded.training.learning_rate == 1e-4
    assert loaded.output_dir == "run"
    assert loaded.reward.beta == ("fixed", [8.0])
    assert loaded.property_penalty == {"tpsa": 140.0}
    reward = ScaledAtomCountReward(**loaded.reward.settings)
    assert evaluate_rewards(reward, [Chem.MolFromSmiles("CCO")])[0].tolist() == [[6.0]]
    with pytest.raises(ValueError):
        DataConfig(max_atoms=0).validate()
    with pytest.raises(ValueError):
        SubsamplingConfig(sampling_ratio=1.1).validate()
    with pytest.raises(ValueError):
        SubsamplingConfig(importance_temp=-0.1).validate()
    with pytest.raises(ValueError):
        RewardConfig(beta=0).validate()
    with pytest.raises(ValueError):
        RewardConfig(floor=0).validate()
    with pytest.raises(ValueError, match="at least"):
        GenerationConfig(min_reactions=2, max_reactions=1).validate()
    with pytest.raises(ValueError, match="mapping"):
        RewardConfig(settings=[]).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown property_penalty"):
        Config(
            data=DataConfig(env_dir="example"), property_penalty={"unknown": 1}
        ).validate()
    Config(
        data=DataConfig(env_dir="example"),
        property_penalty={"rings": 0, "hbd": 0, "logp": -1.0},
    ).validate()
    for bound in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="must be finite"):
            Config(
                data=DataConfig(env_dir="example"), property_penalty={"mw": bound}
            ).validate()

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
    assert minimal.reward == RewardConfig(beta=("fixed", [32.0]))
    assert complete.reward == RewardConfig(
        beta=("fixed", [32.0]), floor=1e-4, settings={}
    )
    assert minimal.property_penalty == {"mw": 500.0}
    assert complete.property_penalty == {}
    assert minimal.training.batch_size == 64
    assert minimal.training.replay_batch_size == 64
    assert (
        evaluate_rewards(
            QEDReward(**minimal.reward.settings), [Chem.MolFromSmiles("CCO")]
        )[0][0]
        > 0
    )
    assert list(complete.to_file_dict()) == [
        "data",
        "run",
        "reward",
        "property_penalty",
        "subsampling",
        "generation",
        "model",
        "training",
    ]


def test_zero_replay_capacity_disables_storage() -> None:
    TrainingConfig(replay_capacity=0).validate()
    buffer = ReplayBuffer(0)
    buffer.add(
        [
            Trajectory(
                steps=[],
                final_smiles="CC",
                beta=1.0,
                preferences=[1.0],
                objective_rewards=[0.0],
            )
        ]
    )
    assert len(buffer) == 0
    assert buffer.state_dict()["items"] == []


def test_qed_and_custom_reward_alignment() -> None:
    values, _ = evaluate_rewards(QEDReward(), [Chem.MolFromSmiles("CCO"), None])
    assert 0 < values[0] <= 1
    assert values[1] == 0
    custom, _ = evaluate_rewards(AtomCountReward(), [Chem.MolFromSmiles("CCO")])
    assert custom.tolist() == [[3.0]]

    class FilteredReward(AtomCountReward):
        def filter_object(self, mol):
            return mol.GetNumHeavyAtoms() <= 3

    filtered, _ = evaluate_rewards(
        FilteredReward(),
        [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCC")],
    )
    assert filtered.tolist() == [[3.0], [0.0]]
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_rewards(BrokenReward(), [Chem.MolFromSmiles("CCO")])


def test_replay_wraparound_matches_fifo_and_restarts() -> None:
    import random
    from collections import deque

    buffer = ReplayBuffer(7)
    reference = deque(maxlen=7)
    for start in range(0, 30, 5):
        items = [
            Trajectory(
                steps=[],
                final_smiles=str(i),
                beta=1.0,
                preferences=[1.0],
                objective_rewards=[0.0],
            )
            for i in range(start, start + 5)
        ]
        buffer.add(items)
        reference.extend(items)
        assert buffer.sample(20, random.Random(0)) == list(reference)
        assert buffer.sample(3, random.Random(11)) == random.Random(11).sample(
            list(reference), 3
        )
    restored = ReplayBuffer(7)
    restored.load_state_dict(buffer.state_dict())
    next_item = Trajectory(
        steps=[], final_smiles="new", beta=1.0, preferences=[1.0], objective_rewards=[0.0]
    )
    buffer.add([next_item])
    restored.add([next_item])
    assert restored.sample(4, random.Random(7)) == buffer.sample(4, random.Random(7))


def test_replay_stores_serializable_snapshots_and_restores_molecules() -> None:
    import json
    import random

    from rxnflow.gflownet.types import Action, ActionType, State, Transition

    state = State.from_smiles("[1*]N[C@@H](C)C/C=C/C", reaction_count=1)
    trajectory = Trajectory(
        steps=[
            Transition(
                state,
                Action(ActionType.UNI_REACTION, reaction="convert"),
                product_smiles="CC",
                log_p_B=-0.7,
            ),
        ],
        final_smiles="CC",
        reward=0.5,
        beta=1.0,
        preferences=[1.0],
        objective_rewards=[0.5],
    )
    expected = trajectory.to_dict()
    buffer = ReplayBuffer(2)
    buffer.add([trajectory])
    # JSON round-trip proves the entire stored trajectory is plain data.
    saved = json.loads(json.dumps(buffer.state_dict()))
    assert saved["items"] == [expected]
    trajectory.reward = 0.1
    trajectory.steps[0].log_p_B = -5.0

    restored = ReplayBuffer(2)
    restored.load_state_dict(saved)
    sampled = restored.sample(1, random.Random(0))[0]
    assert sampled.to_dict() == expected
    assert sampled.steps[0].state.mol is not state.mol
    assert sampled.steps[0].state.smiles == state.smiles
    sampled.steps[0].state.mol.GetAtomWithIdx(0).SetIsotope(9)
    sampled.reward = 0.0
    assert restored.sample(1, random.Random(0))[0].to_dict() == expected
