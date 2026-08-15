"""Small serializable domain types shared by training and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


class ActionKind(IntEnum):
    SET_WORKFLOW = 0
    FIRST_BLOCK = 1
    UNI_REACTION = 2
    BI_REACTION = 3


@dataclass(frozen=True)
class MoleculeState:
    smiles: str = ""
    workflow_index: int = -1
    protocol_order: int = -1


@dataclass(frozen=True)
class RxnAction:
    kind: ActionKind
    workflow_index: int
    protocol_order: int = -1
    block_type: str | None = None
    block_index: int | None = None


@dataclass
class TrajectoryStep:
    state: MoleculeState
    action: RxnAction
    product_smiles: str


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    final_smiles: str
    reward: float = 0.0
    valid: bool = True


@dataclass
class SamplingResult:
    smiles: str
    workflow: str
    trajectory: list[dict[str, Any]]
    intermediates: list[str]
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
