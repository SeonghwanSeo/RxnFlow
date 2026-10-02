"""Small serializable domain types shared by training and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Any, Literal

import numpy as np
from rdkit import Chem


class ActionType(IntEnum):
    FIRST_SYNTHON = 0
    UNI_REACTION = 1
    BI_REACTION = 2


@dataclass(frozen=True)
class State:
    """An RDKit molecular graph and trajectory metadata.

    Treat mol as read-only: reactions return new molecules, and edits such as
    restoring a first brick's isotope operate on a copy. Model tensors and
    descriptors read this same molecule; SMILES only serialize/identify it.
    """

    mol: Chem.Mol | None = field(default=None, repr=False)
    num_reactions: int = 0
    terminated: bool = False
    num_synthons: int = 0

    @cached_property
    def smiles(self) -> str:
        return "" if self.mol is None else Chem.MolToSmiles(self.mol)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        num_reactions: int = 0,
        terminated: bool = False,
        num_synthons: int = 1,
    ) -> State:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if smiles and mol is None:
            raise ValueError(f"invalid state SMILES: {smiles}")
        return cls(mol, num_reactions, terminated, num_synthons if mol is not None else 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "num_reactions": self.num_reactions,
            "num_synthons": self.num_synthons,
            "terminated": self.terminated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> State:
        return cls.from_smiles(**data)


@dataclass(frozen=True)
class Action:
    """A selected reaction and optional oriented catalog row."""

    action_type: ActionType
    reaction: str | None = None
    synthon_type: str | None = None
    synthon_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.name,
            "reaction": self.reaction,
            "synthon_type": self.synthon_type,
            "synthon_index": self.synthon_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        data = data.copy()
        data["action_type"] = ActionType[data.pop("action_type")]
        return cls(**data)


# Reverse-ordered edges: (forward action, parent SMILES), ending at the empty
# state's SMILES "" through FirstSynthon. The number of entries is the route length.
BackwardTrajectory = list[tuple[Action, str]]


@dataclass
class ActionSubspace:
    """One (reaction, library) pair, independent of policy logits.

    A unary reaction uses library=None and has one action. num_actions is the
    full library size; sample_indices=None selects that full range. A sampled
    subspace holds one array mapping columns back to original synthon indices.
    """

    name: tuple[str, str | None]
    action_type: ActionType
    num_actions: int
    sample_indices: np.ndarray | None = None

    def action_at(self, column: int) -> Action:
        reaction, library = self.name
        synthon_index = None
        if library is not None:
            synthon_index = (
                column
                if self.sample_indices is None
                else int(self.sample_indices[column])
            )
        return Action(
            self.action_type,
            reaction=None if self.action_type == ActionType.FIRST_SYNTHON else reaction,
            synthon_type=library,
            synthon_index=synthon_index,
        )


ActionSpace = list[ActionSubspace]


@dataclass
class Transition:
    """One observed action, its product, and the estimated backward log probability."""

    state: State
    action: Action
    product_smiles: str
    log_p_B: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "product_smiles": self.product_smiles,
            "log_p_B": self.log_p_B,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transition:
        return cls(
            state=State.from_dict(data["state"]),
            action=Action.from_dict(data["action"]),
            product_smiles=data["product_smiles"],
            log_p_B=data["log_p_B"],
        )


InvalidReason = Literal["no_valid_action", "invalid_transition", "max_reactions"]


@dataclass
class Trajectory:
    steps: list[Transition]
    final_smiles: str
    # Conditions are fixed for the entire trajectory, including replay.
    beta: float
    preferences: list[float]
    objective_rewards: list[float] = field(default_factory=list)
    reward: float = 0.0
    valid: bool = True
    invalid_reason: InvalidReason | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "final_smiles": self.final_smiles,
            "beta": self.beta,
            "preferences": list(self.preferences),
            "objective_rewards": list(self.objective_rewards),
            "reward": self.reward,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            steps=[Transition.from_dict(step) for step in data["steps"]],
            final_smiles=data["final_smiles"],
            beta=data["beta"],
            preferences=list(data["preferences"]),
            objective_rewards=list(data["objective_rewards"]),
            reward=data["reward"],
            valid=data["valid"],
            invalid_reason=data["invalid_reason"],
        )


@dataclass
class SamplingResult:
    smiles: str
    trajectory: list[dict[str, Any]]
    intermediates: list[str]
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
