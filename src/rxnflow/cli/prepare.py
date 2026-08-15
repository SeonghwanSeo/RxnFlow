"""Command-line interface for eMolecules environment preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from rxnflow.data import convert_stage, features_stage, prepare_all, reorder_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "convert", "features", "reorder"))
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--raw-data-dir", type=Path)
    parser.add_argument(
        "--template-dir", type=Path, action="append", dest="template_dirs"
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage in {"all", "convert", "reorder"} and args.raw_data_dir is None:
        raise SystemExit("--raw-data-dir is required for all, convert, and reorder")
    if args.stage in {"all", "convert"} and not args.template_dirs:
        raise SystemExit("at least one --template-dir is required for all and convert")
    if args.stage == "convert":
        convert_stage(
            raw_data_dir=args.raw_data_dir,
            env_dir=args.env_dir,
            template_dirs=args.template_dirs,
            force=args.force,
        )
    elif args.stage == "features":
        features_stage(args.env_dir, args.force)
    elif args.stage == "reorder":
        reorder_stage(
            raw_data_dir=args.raw_data_dir,
            env_dir=args.env_dir,
            force=args.force,
        )
    else:
        prepare_all(
            raw_data_dir=args.raw_data_dir,
            env_dir=args.env_dir,
            template_dirs=args.template_dirs,
            force=args.force,
        )


if __name__ == "__main__":
    main()
