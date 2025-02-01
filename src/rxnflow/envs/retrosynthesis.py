import copy
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from rdkit import Chem
from collections.abc import Iterable

from rxnflow.envs.action import Protocol, RxnActionType, RxnAction


class RetroSynthesisTree:
    def __init__(self, smi: str, branches: list | None = None):
        self.smi: str = smi
        self.branches: list[tuple[RxnAction, RetroSynthesisTree]] = branches if branches else []

    @property
    def is_leaf(self) -> bool:
        return len(self.branches) == 0

    def iteration(self, prev_traj: list[RxnAction] | None = None) -> Iterable[list[RxnAction]]:
        prev_traj = prev_traj if prev_traj else []
        if self.is_leaf:
            yield prev_traj
        else:
            for bck_action, subtree in self.branches:
                yield from subtree.iteration(prev_traj + [bck_action])

    def __len__(self):
        return len(self.branches)

    @cached_property
    def height(self) -> int:
        return max(self.iteration_depth(0))

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


class RetroSyntheticAnalyzer:
    def __init__(
        self,
        protocols: list[Protocol],
        blocks: list[str],
        approx: bool = True,
    ):
        self.protocols: list[Protocol] = protocols
        self.approx: bool = approx  # Fast analyzing
        self.__cache: Cache = Cache(100_000)  # For Caching

        # For Fast Search
        self.__max_block_smi_len: int = 0
        self.__prefix_len = 5
        self.__block_search: dict[str, dict[str, int]] = {}
        for idx, smi in enumerate(blocks):
            self.__max_block_smi_len = max(self.__max_block_smi_len, len(smi))
            prefix = smi[: self.__prefix_len]
            self.__block_search.setdefault(prefix, dict())[smi] = idx

        # temporary
        self.__min_depth: int
        self.__max_depth: int

    def block_search(self, smi: str) -> int | None:
        if len(smi) > self.__max_block_smi_len:
            return False
        prefix = smi[: self.__prefix_len]
        prefix_block_set = self.__block_search.get(prefix, None)
        if prefix_block_set is not None:
            return prefix_block_set.get(smi, None)
        return None

    def from_cache(self, smi: str, depth: int) -> tuple[bool, RetroSynthesisTree | None]:
        return self.__cache.get(smi, self.__max_depth - depth)

    def to_cache(self, smi: str, depth: int, cache: RetroSynthesisTree | None):
        return self.__cache.update(smi, self.__max_depth - depth, cache)

    def run(
        self,
        mol: str | Chem.Mol,
        max_rxns: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        if isinstance(mol, str):
            smi = mol
            mol = Chem.MolFromSmiles(mol)
        else:
            mol = mol
            smi = Chem.MolToSmiles(mol)

        self.__max_depth = self.__min_depth = max_rxns + 1  # 1: AddFirstBlock
        if known_branches is not None:
            for _, tree in known_branches:
                self.__min_depth = min(self.__min_depth, min(tree.iteration_depth()) + 1)
        res = self.__dfs(smi, mol, 1, known_branches)
        del self.__max_depth, self.__min_depth
        return res

    def check_depth(self, depth: int) -> bool:
        if depth > self.__max_depth:
            return False
        if self.approx and (depth > self.__min_depth):
            return False
        return True

    def __dfs(
        self,
        smiles: str,
        mol: Chem.Mol,
        depth: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        if (not self.check_depth(depth)) or (mol is None):
            return None

        # NOTE: Load cache
        is_cached, cached_tree = self.from_cache(smiles, depth)
        if is_cached:
            return cached_tree

        known_branches = known_branches if known_branches is not None else []
        known_protocols: set[str] = set(action.protocol for action, _ in known_branches)
        branches: list[tuple[RxnAction, RetroSynthesisTree]] = known_branches.copy()

        tmp_cache = dict()

        for protocol in self.protocols:
            if protocol.name in known_protocols:
                continue
            if protocol.action is RxnActionType.FirstBlock:
                block_idx = self.block_search(smiles)
                if block_idx is not None:
                    bck_action = RxnAction(RxnActionType.FirstBlock, protocol.name, smiles, block_idx)
                    branches.append((bck_action, RetroSynthesisTree("")))
                    self.__min_depth = depth

            elif protocol.action is RxnActionType.UniRxn:
                if not self.check_depth(depth + 1):
                    continue
                for child_mol, *_ in protocol.rxn.reverse(mol)[:2]:
                    child_smi = Chem.MolToSmiles(child_mol)
                    if child_smi in tmp_cache:
                        child_tree = tmp_cache[child_smi]
                    else:
                        child_tree = self.__dfs(child_smi, child_mol, depth + 1)
                        tmp_cache[child_smi] = child_tree
                    if child_tree is not None:
                        bck_action = RxnAction(RxnActionType.BiRxn, protocol.name)
                        branches.append((bck_action, child_tree))

            elif protocol.action is RxnActionType.BiRxn:
                if not self.check_depth(depth + 1):
                    continue
                for child_mol, block_mol in protocol.rxn.reverse(mol)[:2]:
                    block_smi = Chem.MolToSmiles(block_mol)
                    block_idx = self.block_search(block_smi)
                    if block_idx is not None:
                        child_smi = Chem.MolToSmiles(child_mol)
                        if child_smi in tmp_cache:
                            child_tree = tmp_cache[child_smi]
                        else:
                            child_tree = self.__dfs(child_smi, child_mol, depth + 1)
                        tmp_cache[child_smi] = child_tree
                        if child_tree is not None:
                            bck_action = RxnAction(RxnActionType.BiRxn, protocol.name, block_smi, block_idx)
                            branches.append((bck_action, child_tree))

        # return if retrosynthetically inaccessible
        if len(branches) == 0:
            return None

        # insert cache when possible actions are not only firstblock
        result = RetroSynthesisTree(smiles, branches)
        if not (len(branches) == 1 and branches[0][0].action is RxnActionType.BckFirstBlock):
            self.to_cache(smiles, depth, result)
        return result


class MultiRetroSyntheticAnalyzer:
    def __init__(self, analyzer, num_workers: int = 4):
        self.pool = ProcessPoolExecutor(num_workers, initializer=self._init_worker, initargs=(analyzer,))
        self.futures = []

    def _init_worker(self, base_analyzer):
        global analyzer
        analyzer = copy.deepcopy(base_analyzer)

    def init(self):
        self.result()

    def result(self) -> list[tuple[int, RetroSynthesisTree]]:
        result = [future.result() for future in self.futures]
        self.futures = []
        return result

    def submit(
        self,
        key: int,
        mol: str | Chem.Mol,
        max_rxns: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]],
    ):
        self.futures.append(self.pool.submit(self._worker, key, mol, max_rxns, known_branches))

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
