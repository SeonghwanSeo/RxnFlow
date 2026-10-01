"""Small serializable domain types shared by training and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


class ActionKind(IntEnum):
    FIRST_BLOCK = 0
    UNI_REACTION = 1
    BI_REACTION = 2


@dataclass(frozen=True)
class MoleculeState:
    smiles: str = ""
    reaction_count: int = 0
    terminated: bool = False


@dataclass(frozen=True)
class RxnAction:
    kind: ActionKind
    # A canonical product identifies the selected reaction outcome/site. Symmetry
    # equivalent matches yielding the same product share an action.
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
