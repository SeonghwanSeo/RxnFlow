import os
from pathlib import Path

import pytest
import torch

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
from rxnflow.envs import SynthesisEnv
from rxnflow.envs.prepare import convert_stage, features_stage
from rxnflow.gflownet.subsampling import UniformActionSpace


def _representative_stock(source: Path, destination: Path, limit: int) -> Path:
    with (
        source.open(encoding="utf-8") as reader,
        destination.open("w", encoding="utf-8") as writer,
    ):
        for line_number, line in enumerate(reader):
            if line_number >= limit:
                break
            writer.write(line)
    return destination


@pytest.mark.heavy
def test_representative_enamine_and_longer_smoke(tmp_path: Path) -> None:
    stock = os.environ.get("RXNFLOW_ENAMINE_STOCK")
    configured_env = os.environ.get("RXNFLOW_ENV_DIR")
    if configured_env:
        env_dir = Path(configured_env)
    elif stock:
        root = Path(__file__).parents[1]
        stock_path = Path(stock)
        if os.environ.get("RXNFLOW_FULL_PREPARE") != "1":
            limit = int(os.environ.get("RXNFLOW_REPRESENTATIVE_LIMIT", "500"))
            stock_path = _representative_stock(
                stock_path, tmp_path / "enamine_stock.smi", limit
            )
        env_dir = tmp_path / "enamine_env"
        convert_stage(stock_path, env_dir, root / "data/templates")
        features_stage(env_dir)
    else:
        pytest.skip("set RXNFLOW_ENV_DIR or RXNFLOW_ENAMINE_STOCK")

    env = SynthesisEnv(env_dir, max_atoms=50, retrosynthesis_workers=0)
    assert env.bi_reactions and env.blocks and env.brick_types
    sampled = UniformActionSpace(
        1_000_000,
        SubsamplingConfig(sampling_ratio=0.01, min_sampling=50),
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
            retrosynthesis_workers=4,
        ),
        output_dir=str(tmp_path / "heavy_run"),
        seed=0,
        device="auto",
    )
    checkpoint = RxnFlowTrainer(config, QEDReward()).run()
    assert RxnFlowSampler(config, checkpoint).sample(4)
