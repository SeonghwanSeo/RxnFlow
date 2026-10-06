"""Run preparation stages and resume from their completion markers."""

import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

import yaml

from rxnflow.__version__ import __version__

from .build_action_space import build_action_space
from .convert_synthons import convert_synthons
from .prepare_metadata import prepare_metadata
from .preprocess_features import preprocess_features
from .read_blocks import read_blocks

logger = logging.getLogger(__name__)


def write_signature(env_dir: str | Path) -> None:
    """Publish the signature after action libraries and features are complete."""
    env_path = Path(env_dir)
    logger.info("Computing environment signature")
    digest = hashlib.sha256()
    for name in (
        "config.yaml",
        "action_space.json",
        "synthon.yaml",
        "reaction.yaml",
        "building_blocks.smi",
        "synthon_features.npz",
    ):
        with (env_path / name).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    if (env_path / "exclude_smarts.yaml").exists():
        digest.update((env_path / "exclude_smarts.yaml").read_bytes())
    for path in sorted((env_path / "action_spaces").glob("*.smi")):
        digest.update(path.stem.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)

    for path in sorted((env_path / "synthons").glob("*.tsv")):
        digest.update(path.name.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)

    synthesis = {}
    for name in ("config", "reaction", "synthon", "exclude_smarts"):
        template = env_path / f"{name}.yaml"
        synthesis[name] = (
            yaml.safe_load(template.read_text()) if template.is_file() else None
        )
    signature = {
        "rxnflow_version": __version__,
        "synthesis": synthesis,
        "content_sha256": digest.hexdigest(),
    }
    path = env_path / "signature.json"
    path.write_text(json.dumps(signature, indent=2) + "\n")
    logger.info("Environment ready: %s", env_path)


def prepare_environment(
    building_blocks: str | Path,
    env_dir: str | Path,
    config_path: str | Path,
    num_workers: int = 1,
    min_library_size: int = 10,
    max_atoms: int = 30,
    druglikeness_threshold: float | None = None,
    druglikeness_device: str = "cpu",
) -> None:
    """Resume with unchanged inputs/settings; use a new directory for new settings."""
    if num_workers < 1:
        raise ValueError("The num_workers value must be at least 1")
    env_path = Path(env_dir)
    env_path.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    logger.info("Preparing environment: %s (workers=%d)", env_path, num_workers)

    done = env_path / "prepare_metadata.done"
    if not done.exists():
        stage_started = perf_counter()
        logger.info("Stage 1/6: prepare_metadata")
        prepare_metadata(config_path, env_path)
        done.touch()
        logger.info(
            "Stage 1/6 complete: prepare_metadata (%.1fs)",
            perf_counter() - stage_started,
        )
    else:
        logger.info("Stage 1/6 skipped: prepare_metadata already complete")

    done = env_path / "read_blocks.done"
    if not done.exists():
        stage_started = perf_counter()
        logger.info("Stage 2/6: read_blocks")
        read_blocks(
            building_blocks,
            env_path,
            druglikeness_threshold=druglikeness_threshold,
            druglikeness_device=druglikeness_device,
        )
        done.touch()
        logger.info(
            "Stage 2/6 complete: read_blocks (%.1fs)", perf_counter() - stage_started
        )
    else:
        logger.info("Stage 2/6 skipped: read_blocks already complete")

    done = env_path / "convert_synthons.done"
    if not done.exists():
        stage_started = perf_counter()
        logger.info("Stage 3/6: convert_synthons")
        convert_synthons(env_path, num_workers=num_workers, max_atoms=max_atoms)
        done.touch()
        logger.info(
            "Stage 3/6 complete: convert_synthons (%.1fs)", perf_counter() - stage_started
        )
    else:
        logger.info("Stage 3/6 skipped: convert_synthons already complete")

    done = env_path / "build_action_space.done"
    if not done.exists():
        stage_started = perf_counter()
        logger.info("Stage 4/6: build_action_space")
        build_action_space(env_path, min_library_size=min_library_size)
        done.touch()
        logger.info(
            "Stage 4/6 complete: build_action_space (%.1fs)",
            perf_counter() - stage_started,
        )
    else:
        logger.info("Stage 4/6 skipped: build_action_space already complete")

    done = env_path / "preprocess_features.done"
    if not done.exists():
        stage_started = perf_counter()
        logger.info("Stage 5/6: preprocess_features")
        preprocess_features(env_path, num_workers=num_workers)
        done.touch()
        logger.info(
            "Stage 5/6 complete: preprocess_features (%.1fs)",
            perf_counter() - stage_started,
        )
    else:
        logger.info("Stage 5/6 skipped: preprocess_features already complete")

    done = env_path / "write_signature.done"
    if not done.exists():
        stage_started = perf_counter()
        logger.info("Stage 6/6: write_signature")
        write_signature(env_path)
        done.touch()
        logger.info(
            "Stage 6/6 complete: write_signature (%.1fs)",
            perf_counter() - stage_started,
        )
    else:
        logger.info("Stage 6/6 skipped: write_signature already complete")

    logger.info(
        "Environment preparation complete: %s (elapsed %.1fs)",
        env_path,
        perf_counter() - started,
    )
