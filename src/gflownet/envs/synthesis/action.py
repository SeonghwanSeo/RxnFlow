import enum
import re
from functools import cached_property

from gflownet.envs.synthesis.reaction import Reaction


class ReactionActionType(enum.Enum):
    # Forward actions
    Stop = enum.auto()
    ReactUni = enum.auto()
    ReactBi = enum.auto()
    AddFirstReactant = enum.auto()

    # Backward actions
    BckReactUni = enum.auto()
    BckReactBi = enum.auto()
    BckRemoveFirstReactant = enum.auto()

    @cached_property
    def cname(self) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.name).lower()

    @cached_property
    def mask_name(self) -> str:
        return self.cname + "_mask"

    @cached_property
    def is_backward(self) -> bool:
        return self.name.startswith("Bck")


""" TYPE_IDX, IS_STOP, RXN_IDX, BLOCK_IDX, BLOCK_IS_FIRST"""
ReactionActionIdx = tuple[int, int, int, int]


def get_action_idx(
    type_idx: int,
    rxn_idx: int | None = None,
    block_idx: int | None = None,
    block_is_first: bool | None = None,
) -> ReactionActionIdx:
    _rxn_idx = -1 if rxn_idx is None else rxn_idx
    _block_idx = -1 if block_idx is None else block_idx
    _block_is_first = -1 if block_is_first is None else int(block_is_first)
    return (type_idx, _rxn_idx, _block_idx, _block_is_first)


class ReactionAction:
    def __init__(
        self,
        action: ReactionActionType,
        reaction: Reaction | None = None,
        block: str | None = None,
        block_idx: int | None = None,
        block_is_first: bool | None = None,
    ):
        """A single graph-building action

        Parameters
        ----------
        action: GraphActionType
            the action type
        reaction: Reaction, optional
        block: str, optional
            the block smi object
        block_local_idx: int, optional
            the block idx
        block_is_first: bool, optional
        """
        self.action = action
        self.reaction = reaction
        self.block = block
        self.block_idx = block_idx
        self.block_is_first: bool | None = block_is_first

    def __str__(self):
        return str(self.action)
