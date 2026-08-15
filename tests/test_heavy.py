import os
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from rxnflow import (
    Config,
    DataConfig,
    QEDReward,
    RewardConfig,
    RxnFlowSampler,
    RxnFlowTrainer,
    SubsamplingConfig,
)
from rxnflow.config import ModelConfig, TrainingConfig
from rxnflow.data import prepare_all
from rxnflow.envs import SynthesisEnv
from rxnflow.policy import TieredActionSpace


def _representative_raw(
    source: Path, destination: Path, template_dirs: list[Path], limit: int
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    filenames: set[str] = set()
    for template_dir in template_dirs:
        specs = yaml.safe_load((template_dir / "synthon.yaml").read_text())
        filenames.update(spec["smi_file"] for spec in specs)
    for filename in filenames:
        source_path = source / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination_path = destination / filename
        with (
            source_path.open(encoding="utf-8-sig") as reader,
            destination_path.open("w", encoding="utf-8") as writer,
        ):
            for line_number, line in enumerate(reader):
                if line_number > limit:
                    break
                writer.write(line)
    for optional in ("salts.txt", "id_blacklist.txt"):
        if (source / optional).is_file():
            shutil.copy2(source / optional, destination / optional)
    return destination


@pytest.mark.heavy
def test_representative_emolecules_and_longer_smoke(tmp_path: Path) -> None:
    raw = os.environ.get("RXNFLOW_RAW_DATA")
    configured_env = os.environ.get("RXNFLOW_ENV_DIR")
    if configured_env:
        env_dir = Path(configured_env)
    elif raw:
        env_dir = tmp_path / "emolecules_env"
        root = Path(__file__).parents[1]
        template_dirs = [
            root / "data/templates/emolecules/synple",
            root / "data/templates/emolecules/explore",
        ]
        raw_path = Path(raw)
        if os.environ.get("RXNFLOW_FULL_PREPARE") != "1":
            limit = int(os.environ.get("RXNFLOW_REPRESENTATIVE_LIMIT", "200"))
            raw_path = _representative_raw(
                raw_path, tmp_path / "representative_raw", template_dirs, limit
            )
        prepare_all(
            raw_path,
            env_dir,
            template_dirs,
        )
    else:
        pytest.skip("set RXNFLOW_ENV_DIR or RXNFLOW_RAW_DATA for heavy validation")

    env = SynthesisEnv(env_dir, max_atoms=50)
    assert env.workflows and env.blocks
    million_tiers = torch.arange(1_000_000) % 5 + 1
    sampled = TieredActionSpace(
        million_tiers, SubsamplingConfig(sampling_ratio=0.01, min_sampling=50)
    ).sample(torch.Generator().manual_seed(0))
    assert len(sampled.indices) == 10_000

    steps = int(os.environ.get("RXNFLOW_HEAVY_STEPS", "10"))
    config = Config(
        data=DataConfig(env_dir=str(env_dir), max_atoms=50),
        reward=RewardConfig(exponent=1.0),
        subsampling=SubsamplingConfig(sampling_ratio=0.001, min_sampling=50),
        model=ModelConfig(hidden_dim=64, num_heads=4, num_layers=2),
        training=TrainingConfig(
            steps=steps,
            batch_size=4,
            replay_batch_size=4,
            replay_capacity=100,
            checkpoint_every=steps,
            log_every=max(1, steps // 2),
        ),
        output_dir=str(tmp_path / "heavy_run"),
        seed=0,
        device="auto",
    )
    checkpoint = RxnFlowTrainer(config, QEDReward()).run()
    assert RxnFlowSampler(config, checkpoint).sample(4)
