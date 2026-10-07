"""Calculate and store synthon fingerprints and properties."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm

from rxnflow.envs.features import FINGERPRINT_DIM, PROPERTY_DIM, synthon_feature_row

logger = logging.getLogger(__name__)


def preprocess_features(env_dir: str | Path, num_workers: int = 1) -> None:
    """Write properties, fingerprints, and atom counts in each library's row order."""
    started = perf_counter()
    logger.info("Preparing features: env=%s, workers=%s", env_dir, num_workers)
    env_path = Path(env_dir)
    synthon_dir = env_path / "action_spaces"
    total = 0
    paths = sorted(synthon_dir.glob("*.smi"))
    output = env_path / "synthon_features.npz"
    context = get_context("spawn").Pool(num_workers) if num_workers > 1 else nullcontext()
    # Stream one library at a time to bound feature memory usage.
    with context as pool, ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        with archive.open("format.npy", "w") as entry:
            np.lib.format.write_array(
                entry, np.array("rxnflow-synthon-feature"), allow_pickle=False
            )
        for path in paths:
            with path.open() as f:
                smiles = [line.split("\t", 1)[0] for line in f]
            count = len(smiles)
            total += count
            logger.info("Features for library %s: %d synthons", path.stem, count)
            properties = np.empty((count, PROPERTY_DIM), dtype=np.float32)
            fingerprints = np.empty((count, FINGERPRINT_DIM), dtype=np.uint8)
            heavy_atoms = np.empty(count, dtype=np.uint8)
            rows = (
                pool.imap(synthon_feature_row, smiles, chunksize=256)
                if pool is not None
                else map(synthon_feature_row, smiles)
            )
            for index, (prop, fingerprint, atom_count) in enumerate(
                tqdm(
                    rows,
                    total=count,
                    desc=f"Features {path.stem}",
                    unit="synthon",
                )
            ):
                properties[index] = prop
                fingerprints[index] = fingerprint
                heavy_atoms[index] = atom_count
            logger.info("Writing features for library %s", path.stem)
            for name, array in (
                ("smiles", np.array(smiles)),
                ("properties", properties),
                ("fingerprints", fingerprints),
                ("heavy_atoms", heavy_atoms),
            ):
                with archive.open(
                    f"{path.stem}/{name}.npy", "w", force_zip64=True
                ) as entry:
                    np.lib.format.write_array(entry, array, allow_pickle=False)
    logger.info(
        "Feature preparation complete: %d synthons, %s libraries (%.1fs)",
        total,
        len(paths),
        perf_counter() - started,
    )
