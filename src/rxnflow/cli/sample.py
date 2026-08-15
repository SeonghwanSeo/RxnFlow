"""Sample molecules locally from an RxnFlow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from rxnflow import QEDReward, RxnFlowSampler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("smi", "csv", "json"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--qed", action="store_true", help="include QED in structured JSON results"
    )
    args = parser.parse_args(argv)
    reward = QEDReward() if args.qed else None
    sampler = RxnFlowSampler.from_checkpoint(
        args.checkpoint, reward=reward, device=args.device
    )
    results = sampler.sample(args.num_samples, args.temperature, args.seed)
    sampler.write(results, args.output, args.format)
    print(f"wrote {len(results)} valid samples to {args.output}")


if __name__ == "__main__":
    main()
