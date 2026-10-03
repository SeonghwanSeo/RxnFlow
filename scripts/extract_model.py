"""Extract config and EMA weights for sampling without training state."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("output must differ from the training checkpoint")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = {
        key: payload[key]
        for key in (
            "config",
            "sampling_model",
            "objectives",
            "templates",
            "rxnflow_version",
        )
    }
    torch.save(model, args.output)
    print(f"wrote EMA sampling model to {args.output}")


if __name__ == "__main__":
    main()
