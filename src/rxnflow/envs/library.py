"""Aligned synthon building-block libraries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from rxnflow.envs.chemistry.features import FINGERPRINT_DIM, PROPERTY_DIM


@dataclass
class BlockLibrary:
    block_type: str
    smiles: list[str]
    identifiers: list[list[str]]
    properties: Tensor
    fingerprints: Tensor
    heavy_atoms: Tensor

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def site_types(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.block_type.split("-"))

    @property
    def is_brick(self) -> bool:
        return len(self.site_types) == 1

    def validate(self) -> None:
        count = len(self.smiles)
        if count == 0:
            raise ValueError(f"building-block type {self.block_type!r} is empty")
        if len(self.identifiers) != count:
            raise ValueError(f"identifier alignment failed for {self.block_type}")
        expected = {
            "properties": (count, PROPERTY_DIM),
            "fingerprints": (count, FINGERPRINT_DIM),
            "heavy_atoms": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{self.block_type}.{name} has shape {tuple(value.shape)}, expected {shape}"
                )
        if (
            not torch.isfinite(self.properties).all()
            or not torch.isfinite(self.fingerprints).all()
        ):
            raise ValueError(f"non-finite block features in {self.block_type}")
        if (self.heavy_atoms < 0).any():
            raise ValueError(f"negative heavy-atom count in {self.block_type}")


def read_smiles_file(path: Path) -> tuple[list[str], list[list[str]]]:
    smiles: list[str] = []
    identifiers: list[list[str]] = []
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
            ids = json.loads(fields[1])
            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(value, str) for value in ids)
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON array of block IDs"
                )
            identifiers.append(ids)
    return smiles, identifiers


def load_block_libraries(env_dir: Path) -> dict[str, BlockLibrary]:
    feature_path = env_dir / "bb_feature.npz"
    block_dir = env_dir / "blocks"
    if not feature_path.is_file() or not block_dir.is_dir():
        raise FileNotFoundError(
            "prepared environment requires blocks/*.smi and bb_feature.npz"
        )
    libraries: dict[str, BlockLibrary] = {}
    with np.load(feature_path) as arrays:
        if arrays["format"].item() != "rxnflow-bb-feature":
            raise ValueError(
                "unsupported bb_feature.npz format; regenerate the environment"
            )
        files = sorted(block_dir.glob("*.smi"))
        feature_types = {key.split("/")[0] for key in arrays.files if key != "format"}
        if {path.stem for path in files} != feature_types:
            raise ValueError(
                "SMILES files and bb_feature.npz block types are not aligned"
            )
        for path in files:
            smiles, identifiers = read_smiles_file(path)
            if arrays[f"{path.stem}/smiles"].tolist() != smiles:
                raise ValueError(f"SMILES and feature rows are not aligned: {path}")
            # The preparation/chemistry boundary is NumPy; the model-facing
            # library owns CPU tensors for indexed action scoring.
            library = BlockLibrary(
                block_type=path.stem,
                smiles=smiles,
                identifiers=identifiers,
                properties=torch.from_numpy(arrays[f"{path.stem}/properties"]),
                fingerprints=torch.from_numpy(arrays[f"{path.stem}/fingerprints"]),
                heavy_atoms=torch.from_numpy(arrays[f"{path.stem}/heavy_atoms"]).long(),
            )
            library.validate()
            libraries[path.stem] = library
    if not libraries:
        raise ValueError(f"no building-block libraries in {block_dir}")
    return libraries
