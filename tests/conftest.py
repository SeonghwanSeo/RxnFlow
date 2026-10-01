from pathlib import Path

import pytest

from rxnflow.envs.prepare import convert_stage, features_stage


@pytest.fixture(scope="session")
def prepared_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(__file__).parents[1]
    env_dir = tmp_path_factory.mktemp("prepared") / "env"
    convert_stage(
        root / "tests/fixtures/enamine_stock.smi",
        env_dir,
        root / "data/templates",
    )
    features_stage(env_dir)
    return env_dir
