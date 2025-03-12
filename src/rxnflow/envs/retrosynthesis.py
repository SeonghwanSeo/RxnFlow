import concurrent.futures
import copy
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Self

from rdkit import Chem

from rxnflow.envs.action import Protocol, RxnAction, RxnActionType


class RetroSynthesisTree:
    smi: str
    branches: list[tuple[RxnAction, Self]]

    def __init__(self, smi: str, branches: list[tuple[RxnAction, Self]] | None = None):
        self.smi = smi
        self.branches = branches if branches is not None else []
        self._height: int | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.branches) == 0

    def __len__(self):
        return len(self.branches)

    def height(self) -> int:
        if self._height is None:
            self._height = max(self.iteration_depth(0))
        return self._height

    def iteration_depth(self, prev_len: int = 0) -> Iterable[int]:
        if self.is_leaf:
            yield prev_len
        elif not self.is_leaf:
            for _, subtree in self.branches:
                yield from subtree.iteration_depth(prev_len + 1)

    def print(self, indent=0):
        print(" " * indent + "SMILES: " + self.smi)
        for action, child in self.branches:
            print(" " * (indent + 2) + "- ACTION:", action)
            if not child.is_leaf:
                child.print(indent + 4)

    def iteration(self, prev_traj: list[RxnAction] | None = None) -> Iterable[list[RxnAction]]:
        prev_traj = prev_traj if prev_traj else []
        if self.is_leaf:
            yield prev_traj
        else:
            for action, subtree in self.branches:
                yield from subtree.iteration(prev_traj + [action])


class RetroSyntheticAnalyzer:
    def __init__(
        self,
        protocols: list[Protocol],
        blocks: list[str],
        approx: bool = True,
        max_decomposes: int = 2,
    ):
        self.protocols: list[Protocol] = protocols
        self.approx: bool = approx  # Fast analyzing
        self.__cache_success: Cache = Cache(100_000)
        self.__cache_fail: Cache = Cache(1_000_000)
        self.max_decomposes: int = max_decomposes

        # For Fast Search
        self.__block_search: dict[int, dict[str, int]] = {}
        for idx, smi in enumerate(blocks):
            smi_len = len(smi)
            self.__block_search.setdefault(smi_len, dict())[smi] = idx

        # temporary
        self.__min_depth: int
        self.__max_depth: int

    def run(
        self,
        mol: str | Chem.Mol,
        max_rxns: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:

        if isinstance(mol, Chem.Mol):
            smiles = Chem.MolToSmiles(mol)
        else:
            smiles = mol

        self.__max_depth = self.__min_depth = max_rxns + 1  # 1: AddFirstBlock
        if known_branches is not None:
            for _, tree in known_branches:
                self.__min_depth = min(self.__min_depth, min(tree.iteration_depth()) + 1)
        res = self.__dfs(smiles, 1, known_branches)
        del self.__max_depth, self.__min_depth
        return res

    def block_search(self, smi: str) -> int | None:
        assert isinstance(smi, str)
        prefix_block_set = self.__block_search.get(len(smi), None)
        if prefix_block_set is None:
            return None
        return prefix_block_set.get(smi, None)

    # For entire run
    def from_cache(self, smi: str, depth: int) -> tuple[bool, RetroSynthesisTree | None]:
        is_cached, _ = self.__cache_fail.get(smi, depth)
        if is_cached:
            return True, None
        is_cached, cached_tree = self.__cache_success.get(smi, depth)
        if is_cached:
            return True, cached_tree
        return False, None

    def to_cache(self, smi: str, depth: int, cache: RetroSynthesisTree | None):
        if cache is None:
            self.__cache_fail.update(smi, depth, None)
        else:
            self.__cache_success.update(smi, depth, cache)

    # Check tree depth
    def check_depth(self, depth: int) -> bool:
        if depth > self.__max_depth:
            return False
        if self.approx and (depth > self.__min_depth):
            return False
        return True

    def __dfs(
        self,
        smiles: str,
        depth: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        # Check state
        if (not self.check_depth(depth)) or (len(smiles) == 0):
            return None

        # Load cache
        is_cached, cached_tree = self.from_cache(smiles, depth)
        if is_cached:
            return cached_tree

        # convert mol
        mol: Chem.Mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if known_branches is None:
            known_branches = []
        known_protocols: set[str] = set(action.protocol for action, _ in known_branches)
        branches: list[tuple[RxnAction, RetroSynthesisTree]] = known_branches.copy()

        # Run
        is_block = False
        for protocol in self.protocols:
            # pass if the protocol is in known branches
            if protocol.name in known_protocols:
                continue

            # run retrosynthesis
            if protocol.action is RxnActionType.FirstBlock:
                block_idx = self.block_search(smiles)
                if block_idx is not None:
                    bck_action = RxnAction(RxnActionType.FirstBlock, protocol.name, smiles, block_idx)
                    branches.append((bck_action, RetroSynthesisTree("")))
                    self.__min_depth = depth
                    is_block = True
            elif protocol.action is RxnActionType.UniRxn:
                if not self.check_depth(depth + 1):
                    continue
                for child_smi, *_ in protocol.rxn.reverse_smi(mol)[: self.max_decomposes]:
                    child_tree = self.__dfs(child_smi, depth + 1)
                    if child_tree is not None:
                        bck_action = RxnAction(RxnActionType.UniRxn, protocol.name)
                        branches.append((bck_action, child_tree))
            elif protocol.action is RxnActionType.BiRxn:
                if not self.check_depth(depth + 1):
                    continue
                for child_smi, block_smi in protocol.rxn.reverse_smi(mol)[: self.max_decomposes]:
                    block_idx = self.block_search(block_smi)
                    if block_idx is not None:
                        child_tree = self.__dfs(child_smi, depth + 1)
                        if child_tree is not None:
                            bck_action = RxnAction(RxnActionType.BiRxn, protocol.name, block_smi, block_idx)
                            branches.append((bck_action, child_tree))

        # return None if retrosynthetically inaccessible
        if len(branches) == 0:
            result = None
        else:
            result = RetroSynthesisTree(smiles, branches)

        # update cache
        # if self.approx is True, we don't save cache for building blocks
        if not (self.approx and is_block):
            self.to_cache(smiles, depth, result)
        return result


class Cache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache_valid: dict[str, tuple[int, RetroSynthesisTree]] = {}
        self.cache_invalid: dict[str, int] = {}

    def update(self, smiles: str, height: int, tree: RetroSynthesisTree | None):
        if tree is not None:
            flag, cache = self.get(smiles, height)
            if flag is False:
                if len(self.cache_valid) >= self.max_size:
                    self.cache_valid.popitem()
                self.cache_valid[smiles] = (height, tree)
        else:
            self.cache_invalid[smiles] = max(self.cache_invalid.get(smiles, -1), height)

    def get(self, smiles: str, height: int) -> tuple[bool, RetroSynthesisTree | None]:
        cache = self.cache_valid.get(smiles, None)
        if cache is not None:
            cached_height, cached_tree = cache
            if height <= cached_height:
                return True, cached_tree
        cached_height = self.cache_invalid.get(smiles, -1)
        if height <= cached_height:
            return True, None
        else:
            return False, None


class MultiRetroSyntheticAnalyzer:
    def __init__(self, analyzer, num_workers: int = 4):
        self.pool = ProcessPoolExecutor(num_workers, initializer=self._init_worker, initargs=(analyzer,))
        self.futures = []

    @classmethod
    def create(
        cls,
        protocols: list[Protocol],
        blocks: list[str],
        approx: bool = True,
        max_decomposes: int = 2,
        num_workers: int = 4,
    ):
        analyzer = RetroSyntheticAnalyzer(protocols, blocks, approx, max_decomposes)
        return cls(analyzer, num_workers)

    def _init_worker(self, base_analyzer):
        global analyzer
        analyzer = copy.deepcopy(base_analyzer)

    def init(self):
        self.result()

    def terminate(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def submit(
        self,
        key: int,
        mol: str | Chem.Mol,
        max_rxns: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]],
    ):
        self.futures.append(self.pool.submit(self._worker, key, mol, max_rxns, known_branches))

    def result(self) -> list[tuple[int, RetroSynthesisTree]]:
        try:
            done, _ = concurrent.futures.wait(self.futures, return_when=concurrent.futures.FIRST_EXCEPTION)
            result = [future.result() for future in done]
            self.futures = []
            return result
        except Exception as e:
            print("Error during Retrosynthesis analysis")
            raise e

    @staticmethod
    def _worker(
        key: int,
        mol: str | Chem.Mol,
        max_step: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]],
    ) -> tuple[int, RetroSynthesisTree]:
        global analyzer
        res = analyzer.run(mol, max_step, known_branches)
        return key, res
