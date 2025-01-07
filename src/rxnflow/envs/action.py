from dataclasses import dataclass
import re
import enum
from functools import cached_property

from .reaction import Reaction


class RxnActionType(enum.Enum):
    # Forward actions
    FirstBlock = enum.auto()
    UniRxn = enum.auto()  # DUMMY
    BiRxn = enum.auto()
    Stop = enum.auto()

    # Backward actions
    BckFirstBlock = enum.auto()
    BckUniRxn = enum.auto()
    BckBiRxn = enum.auto()

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
        block_types: list[str] | None = None,
        state_type: int | None = None,
        forward: Reaction | None = None,
        reverse: Reaction | None = None,
        is_block_first: bool | None = None,
    ):
        self.name: str = name
        self.action: RxnActionType = action
        if action is RxnActionType.FirstBlock:
            assert block_types is not None
        elif action is RxnActionType.UniRxn:
            raise ValueError(action)
        elif action is RxnActionType.BiRxn:
            assert block_types is not None
            assert state_type is not None and forward is not None and reverse is not None
        elif action is RxnActionType.Stop:
            pass
        self._block_types: list[str] | None = block_types
        self._state_type: int | None = state_type
        self._rxn_forward: Reaction | None = forward
        self._rxn_reverse: Reaction | None = reverse

    def __str__(self) -> str:
        return self.name

    @property
    def rxn_forward(self) -> Reaction:
        assert self._rxn_forward is not None
        return self._rxn_forward

    @property
    def rxn_reverse(self) -> Reaction:
        assert self._rxn_reverse is not None
        return self._rxn_reverse

    @property
    def state_type(self) -> int:
        assert self._state_type is not None
        return self._state_type

    @property
    def block_types(self) -> list[str]:
        assert self._block_types is not None
        return self._block_types


@dataclass()
class RxnAction:
    """A single graph-building action

    Parameters
    ----------
    action: GraphActionType
        the action type
    protocol: Protocol
        synthesis protocol
    block_idx: int, optional
        the block idx
    block: str, optional
        the block smi object
    block_str: int, optional
        the block idx
    """

    action: RxnActionType
    _protocol: str | None = None
    _block: str | None = None
    _block_type: str | None = None
    _block_idx: int | None = None

    def __repr__(self):
        return f"<{str(self)}>"

    def __str__(self):
        return f"{self.action}: {self._protocol}, {self._block}"

    @property
    def hash_key(self) -> tuple[RxnActionType, str | None, str | None, int | None]:
        return (self.action, self._protocol, self._block_type, self._block_idx)

    def __eq__(self, value) -> bool:
        return self.hash_key == value.hash_key

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
    def block_type(self) -> str:
        assert self._block_type is not None
        return self._block_type

    @property
    def block_idx(self) -> int:
        assert self._block_idx is not None
        return self._block_idx
