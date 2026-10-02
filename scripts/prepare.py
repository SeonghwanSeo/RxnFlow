"""Prepare an Enamine synthon environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from rxnflow.envs.prepare import convert_stage, features_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--building-blocks", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Preparation processes (default: 1)"
    )
    parser.add_argument(
        "--min-library-size",
        type=int,
        default=1,
        help="Minimum unique oriented synthons per library (default: 1)",
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
        args.template_dir,
        num_workers=args.num_workers,
        min_library_size=args.min_library_size,
    )
    features_stage(args.env_dir, args.num_workers)


if __name__ == "__main__":
    main()
