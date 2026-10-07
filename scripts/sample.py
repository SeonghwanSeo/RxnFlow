"""Sample molecules locally from an RxnFlow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rxnflow.config import parse_distribution
from rxnflow.sampler import RxnFlowSampler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="training checkpoint or extracted sampling model",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="output file path"
    )
    parser.add_argument(
        "--format", choices=("smi", "json", "jsonl"),
        help="output format (default: inferred from the output file extension)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="trajectory attempts (default: 1000); invalid results are excluded without retrying",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="trajectories per batch (default: 64)"
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0,
        help="softmax temperature (default: 1); independent of reward exponent beta",
    )
    parser.add_argument(
        "--beta",
        default=None,
        help="defaults to checkpoint setting; reward exponent: 32 or uniform(1,64)",
    )
    parser.add_argument(
        "--preferences",
        default=None,
        help=(
            "defaults to checkpoint setting; none, uniform, dirichlet(alpha), or "
            "fixed(w1,...)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, help="random seed; omitted leaves sampling RNGs unreset"
    )
    parser.add_argument(
        "--device", help="device, e.g. cpu or cuda; omitted selects CUDA when available"
    )
    parser.add_argument(
        "--env-dir", type=Path,
        help="override the checkpoint's prepared catalog; template definitions must match",
    )
    args = parser.parse_args(argv)
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sampler = RxnFlowSampler(args.checkpoint, device=device, env_dir=args.env_dir)
    results = sampler.sample(
        args.num_samples,
        softmax_temperature=args.softmax_temperature,
        seed=args.seed,
        beta=parse_distribution(args.beta) if args.beta is not None else None,
        batch_size=args.batch_size,
        preferences=parse_distribution(args.preferences)
        if args.preferences is not None
        else None,
    )
    sampler.write(results, args.output, output_format=args.format)
    print(f"wrote {len(results)} valid samples to {args.output}")


if __name__ == "__main__":
    main()
