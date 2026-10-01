"""Small serializable domain types shared by training and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Any

from rdkit import Chem


class ActionKind(IntEnum):
    FIRST_BLOCK = 0
    UNI_REACTION = 1
    BI_REACTION = 2


@dataclass(frozen=True)
class MoleculeState:
    """An RDKit molecular graph and trajectory metadata.

    Treat mol as read-only: reactions return new molecules, and edits such as
    restoring a first brick's isotope operate on a copy. Model tensors and
    descriptors read this same molecule; SMILES only serialize/identify it.
    """

    mol: Chem.Mol | None = field(default=None, repr=False)
    reaction_count: int = 0
    terminated: bool = False

    @cached_property
    def smiles(self) -> str:
        return "" if self.mol is None else Chem.MolToSmiles(self.mol)

    @classmethod
    def from_smiles(
        cls, smiles: str, reaction_count: int = 0, terminated: bool = False
    ) -> MoleculeState:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if smiles and mol is None:
            raise ValueError(f"invalid state SMILES: {smiles}")
        return cls(mol, reaction_count, terminated)


@dataclass(frozen=True)
class RxnAction:
    kind: ActionKind
    # Empty during policy scoring; filled after the selected reaction executes.
    # Canonical product SMILES serve trajectory serialization and reverse search.
    product_smiles: str
    reaction: str | None = None
    block_type: str | None = None
    block_index: int | None = None


@dataclass
class TrajectoryStep:
    state: MoleculeState
    action: RxnAction
    product_smiles: str
    log_backward: float = 0.0


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    final_smiles: str
    reward: float = 0.0
    valid: bool = True
    invalid_reason: str | None = None


@dataclass
class SamplingResult:
    smiles: str
    trajectory: list[dict[str, Any]]
    intermediates: list[str]
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sample:
    """A valid molecular sample with canonical SMILES and an RDKit molecule."""

    smiles: str
    mol: Chem.Mol = field(repr=False, compare=False)

    @classmethod
    def from_smiles(cls, smiles: str) -> Sample | None:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return cls.from_mol(mol)

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Sample | None:
        normalized = Chem.RemoveHs(Chem.Mol(mol))
        smiles = Chem.MolToSmiles(normalized, canonical=True)
        if not smiles:
            return None
        return cls(smiles=smiles, mol=normalized)


SampleInput = Sample | Chem.Mol | str | None


def as_sample(value: SampleInput) -> Sample | None:
    """Normalize a public sample input without burdening reward authors."""

    if value is None:
        return None
    if isinstance(value, Sample):
        return value
    if isinstance(value, str):
        return Sample.from_smiles(value)
    if isinstance(value, Chem.Mol):
        return Sample.from_mol(value)
    raise TypeError(f"unsupported sample type: {type(value).__name__}")
