"""Small serializable domain types shared by training and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Any

import numpy as np
from rdkit import Chem


class ActionType(IntEnum):
    FIRST_BLOCK = 0
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
    reaction_count: int = 0
    terminated: bool = False

    @cached_property
    def smiles(self) -> str:
        return "" if self.mol is None else Chem.MolToSmiles(self.mol)

    @classmethod
    def from_smiles(
        cls, smiles: str, reaction_count: int = 0, terminated: bool = False
    ) -> State:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if smiles and mol is None:
            raise ValueError(f"invalid state SMILES: {smiles}")
        return cls(mol, reaction_count, terminated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "reaction_count": self.reaction_count,
            "terminated": self.terminated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> State:
        return cls.from_smiles(**data)


@dataclass(frozen=True)
class Action:
    # A(s, a) -> s'
    action_type: ActionType
    reaction: str | None = None
    block_type: str | None = None
    block_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.name,
            "reaction": self.reaction,
            "block_type": self.block_type,
            "block_index": self.block_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        data = data.copy()
        data["action_type"] = ActionType[data.pop("action_type")]
        return cls(**data)


@dataclass
class ActionSubspace:
    """One reaction and its libraries, independent of policy logits.

    sample_indices=None denotes the full libraries. Otherwise each index array
    maps sampled columns to original block rows. Unary actions have no libraries
    and occupy column zero. library_sizes always stores the full library sizes.
    """

    name: str
    action_type: ActionType
    libraries: list[str]
    library_sizes: list[int]
    sample_indices: list[np.ndarray] | None = None

    def action_at(self, column: int) -> Action:
        block_type = None
        block_index = None
        for i, (name, size) in enumerate(
            zip(self.libraries, self.library_sizes, strict=True)
        ):
            indices = None if self.sample_indices is None else self.sample_indices[i]
            count = size if indices is None else len(indices)
            if column < count:
                block_type = name
                block_index = column if indices is None else int(indices[column])
                break
            column -= count
        return Action(
            self.action_type,
            reaction=None if self.action_type == ActionType.FIRST_BLOCK else self.name,
            block_type=block_type,
            block_index=block_index,
        )


ActionSpace = list[ActionSubspace]


@dataclass
class Transition:
    # T(s, a, s')
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
    invalid_reason: str | None = None

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
