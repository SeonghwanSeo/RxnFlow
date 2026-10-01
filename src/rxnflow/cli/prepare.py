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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.env_dir.exists():
        raise SystemExit(f"{args.env_dir} already exists")
    convert_stage(args.building_blocks, args.env_dir, args.template_dir)
    features_stage(args.env_dir)


if __name__ == "__main__":
    main()
