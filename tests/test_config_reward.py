from pathlib import Path

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray
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
from rxnflow.core.types import Trajectory
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.reward import RewardFunction


class AtomCountReward(RewardFunction):
    objectives = ("score",)

    def score(self, mols: list[Chem.Mol]) -> NDArray[np.float32]:
        return np.array(
            [float(mol.GetNumHeavyAtoms()) for mol in mols], dtype=np.float32
        ).reshape(-1, 1)


class BrokenReward(RewardFunction):
    objectives = ("score",)

    def score(self, mols: list[Chem.Mol]) -> NDArray[np.float32]:
        return np.array([-1.0 for _ in mols], dtype=np.float32).reshape(-1, 1)


class ScaledAtomCountReward(RewardFunction):
    objectives = ("score",)

    def __init__(self, scale: float, options: dict[str, bool]):
        self.scale = scale
        self.options = options

    def score(self, mols: list[Chem.Mol]) -> NDArray[np.float32]:
        sign = 1.0 if self.options["positive"] else -1.0
        return np.array(
            [sign * self.scale * mol.GetNumHeavyAtoms() for mol in mols],
            dtype=np.float32,
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
    assert "run" not in saved
    assert saved["reward"]["beta"] == "fixed(32.0)"
    assert saved["property_penalty"] == {"mw": 500.0}
    assert "property_limits" not in saved["data"]
    assert saved["training"]["learning_rate"] == 1e-4
    assert saved["generation"] == {
        "min_synthons": 2,
        "max_synthons": 3,
        "min_reactions": 1,
        "max_reactions": 3,
    }

    minimal = tmp_path / "minimal.yaml"
    minimal.write_text(
        "data:\n  env_dir: example\n"
        "reward:\n  beta: '8'\n  settings:\n    scale: 2\n    options:\n      positive: true\n"
        "property_penalty:\n  tpsa: 140\n"
        "training:\n  num_online: 25\n"
    )
    loaded = Config.from_file(minimal)
    assert loaded.training.num_online == 25
    assert loaded.training.learning_rate == 1e-4
    assert loaded.reward.beta == ("fixed", [8.0])
    assert loaded.property_penalty == {"tpsa": 140.0}
    reward = ScaledAtomCountReward(**loaded.reward.settings)
    assert reward.run([Chem.MolFromSmiles("CCO")]).tolist() == [[6.0]]
    with pytest.raises(ValueError):
        DataConfig(max_atoms=0).validate()
    with pytest.raises(ValueError):
        SubsamplingConfig(sampling_ratio=1.1).validate()
    with pytest.raises(ValueError):
        SubsamplingConfig(importance_temp=-0.1).validate()
    with pytest.raises(ValueError):
        RewardConfig(beta=0).validate()
    with pytest.raises(ValueError):
        TrainingConfig(reward_floor=0).validate()
    with pytest.raises(ValueError, match="min_reactions"):
        GenerationConfig(max_reactions=0).validate()
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

    obsolete_run = tmp_path / "obsolete-run.yaml"
    obsolete_run.write_text("data:\n  env_dir: example\nrun:\n  seed: 0\n")
    with pytest.raises(ValueError, match="unknown configuration fields"):
        Config.from_file(obsolete_run)

    root_run_field = tmp_path / "root-run-field.yaml"
    root_run_field.write_text("data:\n  env_dir: example\noutput_dir: run\n")
    with pytest.raises(ValueError, match="unknown configuration fields"):
        Config.from_file(root_run_field)


def test_checked_in_minimal_and_complete_configs_load() -> None:
    root = Path(__file__).parents[1]
    minimal = Config.from_file(root / "configs" / "qed.yaml")
    complete = Config.from_file(root / "configs" / "template.yaml")
    assert minimal.reward == RewardConfig(beta=("fixed", [32.0]))
    assert complete.reward == RewardConfig(beta=("fixed", [32.0]), settings={})
    assert minimal.property_penalty == {"mw": 500.0}
    assert complete.property_penalty == {}
    assert minimal.training.num_online == 64
    assert minimal.training.num_replay == 64
    assert QEDReward(**minimal.reward.settings).run([Chem.MolFromSmiles("CCO")])[0][0] > 0
    assert list(complete.to_file_dict()) == [
        "data",
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
    values = QEDReward().run([Chem.MolFromSmiles("CCO"), None])
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float32
    assert values.shape == (2, 1)
    assert 0 < values[0] <= 1
    assert values[1] == 0
    custom = AtomCountReward().run([Chem.MolFromSmiles("CCO")])
    assert custom.tolist() == [[3.0]]
    np.testing.assert_array_equal(AtomCountReward()([Chem.MolFromSmiles("CCO")]), custom)

    class FilteredReward(AtomCountReward):
        def score(self, mols):
            values = super().score(mols)
            values[values > 3] = 0
            return values

    filtered = FilteredReward().run(
        [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCC")]
    )
    assert filtered.tolist() == [[3.0], [0.0]]
    empty = QEDReward().run([])
    assert empty.shape == (0, 1) and empty.dtype == np.float32
    rejected = FilteredReward().run([None, Chem.MolFromSmiles("CCCC")])
    np.testing.assert_array_equal(rejected, np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        BrokenReward().run([Chem.MolFromSmiles("CCO")])


def test_replay_wraparound_matches_fifo_and_restarts() -> None:
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
        assert buffer.sample(20, np.random.default_rng(0)) == list(reference)
        indices = np.random.default_rng(11).choice(len(reference), 3, replace=False)
        assert buffer.sample(3, np.random.default_rng(11)) == [
            reference[i] for i in indices
        ]
    restored = ReplayBuffer(7)
    restored.load_state_dict(buffer.state_dict())
    next_item = Trajectory(
        steps=[], final_smiles="new", beta=1.0, preferences=[1.0], objective_rewards=[0.0]
    )
    buffer.add([next_item])
    restored.add([next_item])
    assert restored.sample(4, np.random.default_rng(7)) == buffer.sample(
        4, np.random.default_rng(7)
    )


def test_replay_stores_serializable_snapshots_and_restores_molecules() -> None:
    import json

    from rxnflow.core.types import Action, ActionType, State, Transition

    state = State.from_smiles("[1*]N[C@@H](C)C/C=C/C", num_reactions=1)
    trajectory = Trajectory(
        steps=[
            Transition(
                state,
                Action(ActionType.UNIRXN_TRANSFORM, reaction="convert"),
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
    sampled = restored.sample(1, np.random.default_rng(0))[0]
    assert sampled.to_dict() == expected
    assert sampled.steps[0].state.mol is not state.mol
    assert sampled.steps[0].state.smiles == state.smiles
    sampled.steps[0].state.mol.GetAtomWithIdx(0).SetIsotope(9)
    sampled.reward = 0.0
    assert restored.sample(1, np.random.default_rng(0))[0].to_dict() == expected
