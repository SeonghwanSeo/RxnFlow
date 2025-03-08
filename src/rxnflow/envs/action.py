import enum
import re
from dataclasses import dataclass
from functools import cached_property

from .reaction import Reaction


class RxnActionType(enum.Enum):
    # Forward actions
    Stop = enum.auto()
    UniRxn = enum.auto()
    BiRxn = enum.auto()
    FirstBlock = enum.auto()

    # Backward actions
    BckStop = enum.auto()
    BckUniRxn = enum.auto()
    BckBiRxn = enum.auto()
    BckFirstBlock = enum.auto()

    @cached_property
    def cname(self) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.name).lower()

    @cached_property
    def mask_name(self) -> str:
        return self.cname + "_mask"

    @cached_property
    def is_backward(self) -> bool:
        return self.name.startswith("Bck")


class Protocol:
    def __init__(
        self,
        name: str,
        action: RxnActionType,
        rxn: Reaction | None = None,
    ):
        self.name: str = name
        self.action: RxnActionType = action
        self._rxn: Reaction | None = rxn

    def __str__(self) -> str:
        return self.name

    @property
    def rxn(self) -> Reaction:
        assert self._rxn is not None
        return self._rxn


@dataclass()
class RxnAction:
    """A single graph-building action

    Parameters
    ----------
    action: GraphActionType
        the action type
    protocol: str
        protocol name
    block: str, optional
        the block smi object
    """

    action: RxnActionType
    _protocol: str | None = None
    _block: str | None = None
    _block_idx: int | None = None

    def __repr__(self):
        return f"<{str(self)}>"

    def __str__(self):
        return f"<{self.action}> {self._protocol} - {self._block}({self._block_idx})"

    @property
    def is_fwd(self) -> bool:
        return self.action in (RxnActionType.FirstBlock, RxnActionType.UniRxn, RxnActionType.BiRxn, RxnActionType.Stop)

    @property
    def protocol(self) -> str:
        assert self._protocol is not None
        return self._protocol

    @property
    def block(self) -> str:
        assert self._block is not None
        return self._block

    @property
    def block_idx(self) -> int:
        assert self._block_idx is not None
        return self._block_idx
