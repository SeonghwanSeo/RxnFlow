"""Sample molecules locally from an RxnFlow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rxnflow.sampler import RxnFlowSampler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample molecules from an RxnFlow checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Training checkpoint or extracted sampling model",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument(
        "--format",
        choices=("smi", "json", "jsonl"),
        help="Output format (default: inferred from the output file extension)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help=(
            "Trajectory attempts (default: 1000); "
            "invalid results are excluded without retrying"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Trajectories per batch (default: 64)"
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (default: 1); independent of reward exponent beta",
    )
    parser.add_argument(
        "--beta",
        "--task-beta",
        default=None,
        help=(
            "Reward exponent: scalar, "
            "uniform(low,high) or log_uniform(low,high) "
            "(default: checkpoint setting)"
        ),
    )
    parser.add_argument(
        "--prior-beta",
        default=None,
        help="Prior guidance exponent (default: checkpoint setting)",
    )
    parser.add_argument(
        "--preference",
        default=None,
        help=(
            "Objective preference: none, dirichlet, dirichlet(alpha) or "
            "fixed(w1,...) (default: checkpoint setting)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed; omitted leaves sampling RNGs unreset"
    )
    parser.add_argument(
        "--device", help="Device, e.g. cpu or cuda; omitted selects CUDA when available"
    )
    parser.add_argument(
        "--env-dir",
        type=Path,
        help=(
            "Override the checkpoint's prepared environment; "
            "reaction, synthon and exclusion definitions must match"
        ),
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
        beta=args.beta,
        prior_beta=args.prior_beta,
        preference=args.preference,
        batch_size=args.batch_size,
        softmax_temperature=args.softmax_temperature,
        seed=args.seed,
    )
    sampler.write(results, args.output, output_format=args.format)
    print(f"wrote {len(results)} valid samples to {args.output}")


if __name__ == "__main__":
    main()
