from pathlib import Path

import pytest

from rxnflow.data import prepare_all


@pytest.fixture()
def prepared_env(tmp_path: Path) -> Path:
    fixtures = Path(__file__).parent / "fixtures"
    env_dir = tmp_path / "env"
    prepare_all(fixtures / "raw", env_dir, [fixtures / "templates"])
    return env_dir
