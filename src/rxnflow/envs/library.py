"""Aligned synthon libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rxnflow.envs.features import FINGERPRINT_DIM, PROPERTY_DIM


@dataclass
class SynthonLibrary:
    name: str  # Catalog identifier, e.g. "7" or "7-8".
    smiles: list[str]
    synthon_ids: list[str]
    properties: NDArray[np.float32]
    fingerprints: NDArray[np.uint8]
    heavy_atoms: NDArray[np.uint8]
    synthon_types: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        # One type for a brick; ordered attachment/remaining types for a linker.
        self.synthon_types = tuple(map(int, self.name.split("-")))

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def is_brick(self) -> bool:
        return len(self.synthon_types) == 1

    @property
    def is_linker(self) -> bool:
        return len(self.synthon_types) == 2

    @property
    def attachment_type(self) -> int:
        return self.synthon_types[0]

    def validate(self) -> None:
        if self.fingerprints.dtype != np.uint8:
            raise ValueError("fingerprints must be uint8; regenerate the environment")
        if self.heavy_atoms.dtype != np.uint8:
            raise ValueError("heavy_atoms must be uint8; regenerate the environment")
        count = len(self.smiles)
        if count == 0:
            raise ValueError(f"synthon library {self.name!r} is empty")
        if len(self.synthon_ids) != count:
            raise ValueError(f"identifier alignment failed for {self.name}")
        expected = {
            "properties": (count, PROPERTY_DIM),
            "fingerprints": (count, FINGERPRINT_DIM),
            "heavy_atoms": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{self.name}.{name} has shape {tuple(value.shape)}, expected {shape}"
                )
        # Preparation validates molecules, attachment labels and feature values.
        # Runtime checks only schema/shape, without rescanning every feature or
        # reparsing every molecule in the trusted prepared catalog.


def read_smiles_file(path: Path) -> tuple[list[str], list[str]]:
    smiles: list[str] = []
    synthon_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected two tab-separated fields"
                )
            if not fields[0] or not fields[1]:
                raise ValueError(f"{path}:{line_number}: missing SMILES or identifier")
            smiles.append(fields[0])
            synthon_ids.append(fields[1])
    return smiles, synthon_ids


def load_synthon_libraries(env_dir: Path) -> dict[str, SynthonLibrary]:
    """Load aligned NumPy features and synthon IDs without reparsing molecules."""
    feature_path = env_dir / "synthon_features.npz"
    synthon_dir = env_dir / "action_spaces"
    if not feature_path.is_file() or not synthon_dir.is_dir():
        raise FileNotFoundError(
            "prepared environment requires action_spaces/*.smi and synthon_features.npz"
        )
    libraries: dict[str, SynthonLibrary] = {}
    with np.load(feature_path) as arrays:
        if arrays["format"].item() != "rxnflow-synthon-feature":
            raise ValueError(
                "unsupported synthon_features.npz format; regenerate the environment"
            )
        files = sorted(synthon_dir.glob("*.smi"))
        feature_types = {key.split("/")[0] for key in arrays.files if key != "format"}
        if {path.stem for path in files} != feature_types:
            raise ValueError(
                "SMILES files and synthon_features.npz synthon types are not aligned"
            )
        for path in files:
            smiles, synthon_ids = read_smiles_file(path)
            # SMILES and features share the row order assigned during preparation.
            # Keep the catalog in NumPy; only sampled rows become model tensors.
            library = SynthonLibrary(
                name=path.stem,
                smiles=smiles,
                synthon_ids=synthon_ids,
                properties=arrays[f"{path.stem}/properties"],
                fingerprints=arrays[f"{path.stem}/fingerprints"],
                heavy_atoms=arrays[f"{path.stem}/heavy_atoms"],
            )
            library.validate()
            libraries[path.stem] = library
    if not libraries:
        raise ValueError(f"no synthon libraries in {synthon_dir}")
    return libraries
