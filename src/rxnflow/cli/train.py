"""Train or restart local RxnFlow with the bundled QED reward."""

from __future__ import annotations

import argparse
from pathlib import Path

from rxnflow.config import Config
from rxnflow.reward import QEDReward
from rxnflow.trainer import RxnFlowTrainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args(argv)
    config = Config.from_file(args.config)
    reward = QEDReward(**config.reward.settings)
    trainer = RxnFlowTrainer(config, reward, restart=args.restart)
    try:
        checkpoint = trainer.run(args.steps)
        print(checkpoint)
    finally:
        if "retro_analyzer" in trainer.env.__dict__:
            trainer.env.retro_analyzer.close()


if __name__ == "__main__":
    main()
