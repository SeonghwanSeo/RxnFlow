"""Aligned building-block library loading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from rxnflow.chemistry import FINGERPRINT_DIM, PROPERTY_DIM

TIER_PATTERN = re.compile(r"^Tier\s+(\d+)$", re.IGNORECASE)


@dataclass
class BlockLibrary:
    block_type: str
    smiles: list[str]
    identifiers: list[str]
    tiers: Tensor
    properties: Tensor
    fingerprints: Tensor
    heavy_atoms: Tensor

    def __len__(self) -> int:
        return len(self.smiles)

    def validate(self) -> None:
        count = len(self.smiles)
        if count == 0:
            raise ValueError(f"building-block type {self.block_type!r} is empty")
        if len(self.identifiers) != count:
            raise ValueError(f"identifier alignment failed for {self.block_type}")
        expected = {
            "tiers": (count,),
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
        if (self.tiers <= 0).any():
            raise ValueError(f"tiers must be positive in {self.block_type}")
        if (self.heavy_atoms < 0).any():
            raise ValueError(f"negative heavy-atom count in {self.block_type}")


def read_smiles_file(path: Path) -> tuple[list[str], list[int], list[str]]:
    smiles: list[str] = []
    tiers: list[int] = []
    identifiers: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected three tab-separated fields"
                )
            match = TIER_PATTERN.match(fields[1].strip())
            if match is None:
                raise ValueError(f"{path}:{line_number}: invalid tier {fields[1]!r}")
            if not fields[0] or not fields[2]:
                raise ValueError(f"{path}:{line_number}: missing SMILES or identifier")
            smiles.append(fields[0])
            tiers.append(int(match.group(1)))
            identifiers.append(fields[2])
    return smiles, tiers, identifiers


def load_block_libraries(env_dir: Path) -> dict[str, BlockLibrary]:
    feature_path = env_dir / "bb_feature.pt"
    smiles_dir = env_dir / "smiles"
    if not feature_path.is_file() or not smiles_dir.is_dir():
        raise FileNotFoundError(
            "prepared environment requires smiles/*.smi and bb_feature.pt"
        )
    data = torch.load(feature_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or data.get("format") != "rxnflow-bb-feature-v1":
        raise ValueError("unsupported bb_feature.pt format; regenerate the environment")
    block_data = data.get("blocks")
    if not isinstance(block_data, dict):
        raise ValueError("bb_feature.pt is missing the blocks mapping")

    files = sorted(smiles_dir.glob("*.smi"))
    file_types = {path.stem for path in files}
    if file_types != set(block_data):
        raise ValueError("SMILES files and bb_feature.pt block types are not aligned")
    libraries: dict[str, BlockLibrary] = {}
    for path in files:
        smiles, tiers, identifiers = read_smiles_file(path)
        feature = block_data[path.stem]
        library = BlockLibrary(
            block_type=path.stem,
            smiles=smiles,
            identifiers=identifiers,
            tiers=torch.as_tensor(feature["tiers"], dtype=torch.long),
            properties=torch.as_tensor(feature["properties"], dtype=torch.float32),
            fingerprints=torch.as_tensor(feature["fingerprints"], dtype=torch.float32),
            heavy_atoms=torch.as_tensor(feature["heavy_atoms"], dtype=torch.long),
        )
        library.validate()
        if not torch.equal(library.tiers, torch.tensor(tiers, dtype=torch.long)):
            raise ValueError(f"tier alignment failed for {path.stem}")
        libraries[path.stem] = library
    return libraries
