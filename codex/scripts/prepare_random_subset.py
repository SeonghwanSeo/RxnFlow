"""Build a reproducible local Enamine subset without loading the full stock."""

import argparse
import hashlib
import json
import platform
import random
import resource
import time
from pathlib import Path

import numpy as np

from rxnflow.envs.prepare import convert_stage, features_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("data/templates"))
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.count > 0
    if args.sample.exists() or args.env_dir.exists():
        raise SystemExit("sample and env-dir must both be new paths")

    started = time.perf_counter()
    rng = random.Random(args.seed)
    reservoir = []
    source_hash = hashlib.sha256()
    total = 0
    # Uniform sampling without replacement across the entire stock, including
    # late records. Source line numbers retain the exact sampling provenance.
    with args.source.open("rb") as source:
        for total, line in enumerate(source, 1):
            source_hash.update(line)
            if total <= args.count:
                reservoir.append((total, line))
            else:
                index = rng.randrange(total)
                if index < args.count:
                    reservoir[index] = (total, line)
    if total < args.count:
        raise ValueError(f"stock contains only {total} records")
    reservoir.sort()
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    sample_bytes = b"".join(line for _, line in reservoir)
    args.sample.write_bytes(sample_bytes)
    sampled_at = time.perf_counter()
    print(f"Sampled {args.count}/{total} records with seed {args.seed}", flush=True)

    convert_stage(args.sample, args.env_dir, args.template_dir)
    converted_at = time.perf_counter()
    print(f"Conversion complete in {converted_at - sampled_at:.2f}s", flush=True)
    features_stage(args.env_dir)
    completed_at = time.perf_counter()
    manifest = json.loads((args.env_dir / "prepare_manifest.json").read_text())
    counts = manifest["stages"]["convert"]["block_counts"]
    source_ids = set(json.loads((args.env_dir / "building_blocks.json").read_text()))
    represented_ids = set()
    for path in (args.env_dir / "blocks").glob("*.smi"):
        with path.open() as handle:
            for line in handle:
                represented_ids.update(json.loads(line.split("\t")[1]))
    assert represented_ids <= source_ids
    with np.load(args.env_dir / "bb_feature.npz") as arrays:
        feature_bytes = sum(arrays[key].nbytes for key in arrays.files)
    report = {
        "host": platform.node(),
        "source": str(args.source.resolve()),
        "source_sha256": source_hash.hexdigest(),
        "source_records": total,
        "sample": str(args.sample.resolve()),
        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "sample_records": args.count,
        "seed": args.seed,
        "algorithm": "uniform reservoir sampling without replacement",
        "source_line_numbers": [number for number, _ in reservoir],
        "template_sha256": {
            name: hashlib.sha256((args.template_dir / name).read_bytes()).hexdigest()
            for name in ("synthon.yaml", "reaction.yaml")
        },
        "cleaned_source_ids": len(source_ids),
        "represented_source_ids": len(represented_ids),
        "cleaned_sources_without_synthon": len(source_ids - represented_ids),
        "libraries": len(counts),
        "brick_rows": sum(count for name, count in counts.items() if "-" not in name),
        "linker_rows": sum(count for name, count in counts.items() if "-" in name),
        "sampling_seconds": sampled_at - started,
        "conversion_seconds": converted_at - sampled_at,
        "features_seconds": completed_at - converted_at,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "feature_array_bytes": feature_bytes,
        "feature_file_bytes": (args.env_dir / "bb_feature.npz").stat().st_size,
    }
    (args.env_dir / "subset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "source_line_numbers"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
