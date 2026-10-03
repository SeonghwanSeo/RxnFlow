"""Prepare a synthon environment from an eMolecules or Enamine catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from rxnflow.envs.prepare import convert_stage, features_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--building-blocks", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Environment preparation YAML configuration",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Preparation processes (default: 1)"
    )
    parser.add_argument(
        "--min-library-size",
        type=int,
        default=10,
        help="Minimum unique oriented synthons per library (default: 10)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=30,
        help="Maximum heavy atoms per completed synthon, excluding dummies (default: 30)",
    )
    parser.add_argument(
        "--druglikeness-threshold",
        type=float,
        help="Filter source building blocks with DeepDL extended (0-100; e.g. 60); omitted disables filtering",
    )
    parser.add_argument(
        "--druglikeness-device", default="cpu", help="DeepDL device: cpu or cuda:0"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be at least 1")
    if args.env_dir.exists():
        raise SystemExit(f"{args.env_dir} already exists")
    convert_stage(
        args.building_blocks,
        args.env_dir,
        args.config,
        num_workers=args.num_workers,
        min_library_size=args.min_library_size,
        max_atoms=args.max_atoms,
        druglikeness_threshold=args.druglikeness_threshold,
        druglikeness_device=args.druglikeness_device,
    )
    features_stage(args.env_dir, args.num_workers)


if __name__ == "__main__":
    main()
