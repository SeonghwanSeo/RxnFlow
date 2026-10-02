"""Sample molecules locally from an RxnFlow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from rxnflow.config import parse_distribution
from rxnflow.sampler import RxnFlowSampler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("smi", "csv", "json"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
        help="softmax temperature (default: 1); independent of reward exponent beta",
    )
    parser.add_argument(
        "--beta", required=True, help="reward exponent: 32 or uniform(1,64)"
    )
    parser.add_argument(
        "--preferences",
        default="uniform",
        help="uniform (default), dirichlet(alpha), or fixed(w1,...)",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    sampler = RxnFlowSampler(args.checkpoint, device=args.device)
    results = sampler.sample(
        args.num_samples,
        args.sampling_temperature,
        args.seed,
        beta=parse_distribution(args.beta),
        preferences=parse_distribution(args.preferences),
    )
    sampler.write(results, args.output, args.format)
    print(f"wrote {len(results)} valid samples to {args.output}")


if __name__ == "__main__":
    main()
