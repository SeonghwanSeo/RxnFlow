import enum
import re
from functools import cached_property
from typing import Iterable, List, Tuple, Optional
from rdkit.Chem import Mol

from gflownet.envs.synthesis.utils import Reaction


class ReactionActionType(enum.Enum):
    # Forward actions
    Stop = enum.auto()
    ReactUni = enum.auto()
    ReactBi = enum.auto()
    AddFirstReactant = enum.auto()
    AddReactant = enum.auto()

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
ReactionActionIdx = Tuple[int, int, int, int, int]


def get_action_idx(
    type_idx: int,
    is_stop: bool = False,
    rxn_idx: Optional[int] = None,
    block_idx: Optional[int] = None,
    block_is_first: Optional[bool] = None,
) -> ReactionActionIdx:
    _is_stop = int(is_stop)
    _rxn_idx = -1 if rxn_idx is None else rxn_idx
    _block_idx = -1 if block_idx is None else block_idx
    _block_is_first = -1 if block_is_first is None else int(block_is_first)
    return (type_idx, _is_stop, _rxn_idx, _block_idx, _block_is_first)


class ReactionAction:
    def __init__(
        self,
        action: ReactionActionType,
        reaction: Optional[Reaction] = None,
        block: Optional[str] = None,
        block_local_idx: Optional[int] = None,
        block_is_first: Optional[bool] = None,
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
        self.block_local_idx = block_local_idx
        self.block_is_first: Optional[bool] = block_is_first

    def __str__(self):
        return str(self.action)


class ForwardAction(ReactionAction):
    def __init__(
        self,
        action: ReactionActionType,
        reaction: Optional[Reaction] = None,
        block: Optional[str] = None,
        block_local_idx: Optional[int] = None,
        block_is_first: Optional[bool] = None,
    ):
        assert action in (
            ReactionActionType.Stop,
            ReactionActionType.AddFirstReactant,
            ReactionActionType.ReactUni,
            ReactionActionType.ReactBi,
        )
        super().__init__(action, reaction, block, block_local_idx, block_is_first)


class BackwardAction(ReactionAction):
    def __init__(
        self,
        action: ReactionActionType,
        reaction: Optional[Reaction] = None,
        block: Optional[str] = None,
        block_local_idx: Optional[int] = None,
        block_is_first: Optional[bool] = None,
    ):
        assert action in (
            ReactionActionType.Stop,
            ReactionActionType.BckRemoveFirstReactant,
            ReactionActionType.BckReactUni,
            ReactionActionType.BckReactBi,
        )
        super().__init__(action, reaction, block, block_local_idx, block_is_first)

    pass


class RetroSynthesisTree:
    def __init__(self, branches: List = []):
        self.branches: List[tuple[BackwardAction, RetroSynthesisTree]] = branches

    def iteration(self, prev_traj: List[BackwardAction] = []) -> Iterable[List[BackwardAction]]:
        if len(self.branches) > 0:
            for bck_action, subtree in self.branches:
                if bck_action.action is ReactionActionType.BckRemoveFirstReactant:
                    yield prev_traj + [bck_action]
                else:
                    for traj in subtree.iteration(prev_traj + [bck_action]):
                        yield traj

    def __len__(self):
        return len(self.branches)

    def length_distribution(self, max_len: int) -> List[int]:
        lengths = list(self.iteration_length())
        return [sum(length == _t for length in lengths) for _t in range(0, max_len + 1)]

    def iteration_length(self, prev_len: int = 0) -> Iterable[int]:
        if len(self.branches) > 0:
            for _, subtree in self.branches:
                yield from subtree.iteration_length(prev_len + 1)
        else:
            yield prev_len
