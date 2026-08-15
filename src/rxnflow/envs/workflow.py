"""Synthesis protocol and workflow definitions."""

from __future__ import annotations

from dataclasses import dataclass

from rxnflow.types import ActionKind

from .reaction import Reaction


@dataclass
class Protocol:
    name: str
    kind: ActionKind
    block_type: str | None = None
    forward: str | None = None

    def __post_init__(self) -> None:
        if self.kind == ActionKind.FIRST_BLOCK:
            if self.block_type is None or self.forward is not None:
                raise ValueError(f"invalid FirstBlock protocol {self.name}")
        elif self.kind == ActionKind.UNI_REACTION:
            if self.block_type is not None or self.forward is None:
                raise ValueError(f"invalid UniRxn protocol {self.name}")
        elif self.kind == ActionKind.BI_REACTION:
            if self.block_type is None or self.forward is None:
                raise ValueError(f"invalid BiRxn protocol {self.name}")
        else:
            raise ValueError(f"invalid protocol action kind: {self.kind}")
        self.reaction = Reaction(self.forward) if self.forward is not None else None


@dataclass(frozen=True)
class Workflow:
    identifier: str
    name: str
    protocols: tuple[Protocol, ...]
