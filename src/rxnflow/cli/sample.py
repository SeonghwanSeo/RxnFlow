"""Sample molecules locally from an RxnFlow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from rxnflow.reward import QEDReward
from rxnflow.sampler import RxnFlowSampler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("smi", "csv", "json"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        help="additional sampling temperature; independent of beta",
    )
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument(
        "--preferences",
        type=float,
        nargs="+",
        required=True,
        help="weights in the checkpoint objective order; sum to 1",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--qed", action="store_true", help="include QED in structured JSON results"
    )
    args = parser.parse_args(argv)
    reward = QEDReward() if args.qed else None
    sampler = RxnFlowSampler(args.checkpoint, reward=reward, device=args.device)
    results = sampler.sample(
        args.num_samples,
        args.temperature,
        args.seed,
        beta=args.beta,
        preferences=args.preferences,
    )
    sampler.write(results, args.output, args.format)
    print(f"wrote {len(results)} valid samples to {args.output}")


if __name__ == "__main__":
    main()
