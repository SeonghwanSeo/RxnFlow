import copy
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from rdkit import Chem
from collections.abc import Iterable, Hashable

from rxnflow.envs.action import RxnActionType, RxnAction, Protocol
from rxnflow.envs.workflow import Workflow


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
    def __init__(self, blocks: dict[str, list[str]], workflow: Workflow, approx: bool = True):
        self.protocols: list[Protocol] = workflow.protocols
        self.approx: bool = approx

        self._block_dict: dict[str, dict[str, int]] = {
            block_type: {block: i for i, block in enumerate(blocks)} for block_type, blocks in blocks.items()
        }
        self._cache: Cache = Cache(100_000)  # only save valid cases
        self.__min_depth: int
        self.__max_depth: int

    def block_search(self, block_types: list[str], block_smi: str) -> tuple[str, int] | tuple[None, None]:
        idx = None
        for t in block_types:
            idx = self._block_dict[t].get(block_smi, None)
            if idx is not None:
                return t, idx
        return None, None

    def from_cache(self, smi: str, depth: int) -> tuple[bool, RetroSynthesisTree | None]:
        return self._cache.get(smi, self.__max_depth - depth)

    def to_cache(self, smi: str, depth: int, cache: RetroSynthesisTree | None):
        return self._cache.update(smi, self.__max_depth - depth, cache)

    def run(
        self,
        mol: str | Chem.Mol,
        max_rxns: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        self.__max_depth = self.__min_depth = max_rxns + 1  # 1: AddFirstBlock
        if known_branches is not None:
            for _, tree in known_branches:
                self.__min_depth = min(self.__min_depth, min(tree.iteration_depth()) + 1)
        res = self.__dfs(mol, 0, known_branches)
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
        mol: Chem.Mol,
        depth: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        if (not self.check_depth(depth)) or (mol is None):
            return None
        smiles: str = Chem.MolToSmiles(mol)

        # NOTE: Define the molecule's state
        connecting_parts = [int(atom.GetIsotope()) for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
        is_complete = len(connecting_parts) == 0
        is_brick = len(connecting_parts) == 1

        # NOTE: Load cache
        is_cached, cached_tree = self.from_cache(smiles, depth)
        if is_cached:
            return cached_tree

        known_branches = known_branches if known_branches is not None else []
        known_actions: set[Hashable] = {action.hash_key for action, _ in known_branches}
        branches: list[tuple[RxnAction, RetroSynthesisTree]] = known_branches.copy()

        for protocol in self.protocols:
            if protocol.action is RxnActionType.FirstBlock:
                if not is_brick:
                    continue  # the state should be brick after first blcok addition
                block_type, block_idx = self.block_search(protocol.block_types, smiles)
                if block_idx is not None:
                    bck_action = RxnAction(RxnActionType.BckFirstBlock, protocol.name, smiles, block_type, block_idx)
                    if bck_action.hash_key in known_actions:
                        continue
                    branches.append((bck_action, RetroSynthesisTree("")))
                    self.__min_depth = depth

            elif protocol.action is RxnActionType.BiRxn:
                if not self.check_depth(depth + 1):
                    continue
                if ("brick" in protocol.name) and (not is_complete):
                    continue  # the state should be complete molecule after brick addition
                if ("linker" in protocol.name) and (not is_brick):
                    continue  # the state should be brick after linker addition

                for child_mol, block_mol in protocol.rxn_reverse(mol):
                    block_smi = Chem.MolToSmiles(block_mol)
                    block_type, block_idx = self.block_search(protocol.block_types, block_smi)
                    if block_idx is not None:
                        bck_action = RxnAction(RxnActionType.BckBiRxn, protocol.name, block_smi, block_type, block_idx)
                        if bck_action.hash_key in known_actions:
                            continue
                        child = self.__dfs(child_mol, depth + 1)
                        if child is not None:
                            branches.append((bck_action, child))

        if len(branches) == 0:
            result = None
        else:
            result = RetroSynthesisTree(smiles, branches)
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
