import copy
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from collections.abc import Iterable

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

    def depth(self) -> int:
        return max(self._iteration_length(0))

    def _iteration_length(self, prev_len: int) -> Iterable[int]:
        if self.is_leaf:
            yield prev_len
        elif not self.is_leaf:
            for _, subtree in self.branches:
                yield from subtree._iteration_length(prev_len + 1)

    def print(self, indent=0):
        print(" " * indent + "SMILES: " + self.smi)
        for action, child in self.branches:
            print(" " * (indent + 2) + "- ACTION:", action)
            if not child.is_leaf:
                child.print(indent + 4)


class RetroSyntheticAnalyzer:
    def __init__(self, blocks: dict[str, list[str]], workflow: Workflow):
        self.protocols: list[Protocol] = workflow.protocols

        self.__block_set: set[str] = set(sum(blocks.values(), start=()))
        self.__block_search: dict[str, dict[str, int]] = {
            block_type: {block: i for i, block in enumerate(blocks)} for block_type, blocks in blocks.items()
        }
        self.use_cache: bool = True
        self.__cache_nonblock: Cache = Cache(100_000)
        self.__cache_block: Cache = Cache(100_000)

        self.__min_depth: int
        self.__max_depth: int

    def from_cache(self, smiles: str, depth: int) -> RetroSynthesisTree | None:
        if self.use_cache:
            cache = self.__cache_block.get(smiles, depth)
            if cache is None:
                cache = self.__cache_nonblock.get(smiles, depth)
            return cache

    def to_cache(self, smiles: str, depth: int, is_block: bool, tree: RetroSynthesisTree):
        if self.use_cache:
            if is_block:
                return self.__cache_block.update(smiles, depth, tree)
            else:
                return self.__cache_nonblock.update(smiles, depth, tree)

    def run(
        self,
        mol: Chem.Mol,
        max_depth: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
        approx: bool = True,
    ) -> RetroSynthesisTree | None:
        """Retrosynthetic Analyze
          - A     => depth = 1
          - A'    => depth = 2 (UniRxn: A -> A')
          - A-B   => depth = 2
          - A-B'  => depth = 3 (UniRxn: B -> B')
          - A-B-C => depth = 3

        Parameters
        ----------
        mol : Chem.Mol
            State molecule
        max_depth : int
            maximum depth to retrosynthesis
        known_branches : list[tuple[RxnAction, RetroSynthesisTree]] | None
            known branches
        approx : bool
            if True, it is allowed for the analyzer to skip long-length trajectories

        Returns
        -------
        retro_tree: RetroSynthesisTree
        """
        smiles = Chem.MolToSmiles(mol)
        self.__min_depth = max_depth
        self.__max_depth = max_depth

        # TODO: replace dfs algorithm to bfs
        return self.__dfs(mol, smiles, 1, known_branches, approx)

    def __dfs(
        self,
        mol: Chem.Mol,
        smiles: str,
        depth: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
        approx: bool = True,
    ) -> RetroSynthesisTree | None:
        if (depth > self.__max_depth) or (mol is None) or (smiles == ""):
            return None
        cached_tree = self.from_cache(smiles, depth)
        if cached_tree is not None:
            return cached_tree
        if approx and (depth > self.__min_depth):
            return None

        if known_branches is None:
            known_branches = []
        passing_protocols: set[str] = set(action.protocol.name for action, _ in known_branches)
        branches: list[tuple[RxnAction, RetroSynthesisTree]] = known_branches.copy()

        is_block: bool = False
        for protocol in self.protocols:
            if protocol.name in passing_protocols:
                continue
            if protocol.action is RxnActionType.FirstBlock:
                block_idx = self.block_search(protocol.block_type, smiles)
                if block_idx is not None:
                    bck_action = RxnAction(RxnActionType.BckFirstBlock, protocol, smiles, block_idx)
                    branches.append((bck_action, RetroSynthesisTree("")))
                    is_block = True
                    self.__min_depth = min(depth, self.__min_depth)

            elif protocol.action is RxnActionType.UniRxn:
                if approx and is_block:
                    continue
                child_set = set()
                for rs in protocol.rxn_reverse(mol):
                    assert len(rs) == 1
                    child_mol = rs[0]
                    child_smi = Chem.MolToSmiles(child_mol)
                    if child_smi in child_set:
                        continue
                    child_set.add(child_smi)
                    child_tree = self.__dfs(child_mol, child_smi, depth + 1)
                    if child_tree is not None:
                        bck_action = RxnAction(RxnActionType.BckUniRxn, protocol)
                        branches.append((bck_action, child_tree))

            elif protocol.action is RxnActionType.BiRxn:
                if approx and is_block:
                    continue
                child_set = set()
                for rs in protocol.rxn_reverse(mol):
                    assert len(rs) == 2
                    child_mol, block_mol = rs
                    child_smi = Chem.MolToSmiles(child_mol)
                    if child_smi in child_set:
                        continue
                    child_set.add(child_smi)
                    block_smi = Chem.MolToSmiles(block_mol)
                    block_idx = self.block_search(protocol.block_type, block_smi)
                    if block_idx is None:
                        continue
                    child_tree = self.__dfs(child_mol, child_smi, depth + 1)
                    if child_tree is not None:
                        bck_action = RxnAction(RxnActionType.BckBiRxn, protocol, block_smi, block_idx)
                        branches.append((bck_action, child_tree))

        tree = RetroSynthesisTree(smiles, branches)
        self.to_cache(smiles, depth, is_block, tree)
        return tree

    def block_search(self, block_type: str, block_smi: str) -> int | None:
        return self.__block_search[block_type].get(block_smi, None)


class MultiRetroSyntheticAnalyzer:
    def __init__(self, analyzer: RetroSyntheticAnalyzer, num_workers: int = 4):
        self.pool = ProcessPoolExecutor(num_workers, initializer=self._init_worker, initargs=(analyzer,))
        self.futures = []

    def init(self):
        self.result()

    def result(self):
        result = [future.result() for future in self.futures]
        self.futures = []
        return result

    def submit(self, key: int, mol: Chem.Mol, depth, known_branches: list[tuple[RxnAction, RetroSynthesisTree]]):
        self.futures.append(self.pool.submit(self._worker, key, mol, depth, known_branches))

    def _init_worker(self, base_analyzer):
        global analyzer
        analyzer = copy.deepcopy(base_analyzer)

    @staticmethod
    def _worker(
        key: int, mol: Chem.Mol, depth: int, known_branches: list[tuple[RxnAction, RetroSynthesisTree]]
    ) -> tuple[int, RetroSynthesisTree]:
        global analyzer
        return key, analyzer.run(mol, depth, known_branches)


class Cache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: dict[str, tuple[int, RetroSynthesisTree]] = {}

    def update(self, smiles: str, depth: int, tree: RetroSynthesisTree):
        cache = self.cache.get(smiles, None)
        if cache is None:
            if len(self.cache) >= self.max_size:
                self.cache.popitem()
            self.cache[smiles] = (depth, tree)
        else:
            cached_depth, cached_branch = cache
            if depth > cached_depth:
                self.cache[smiles] = (depth, tree)

    def get(self, smiles: str, depth: int) -> RetroSynthesisTree | None:
        cache = self.cache.get(smiles, None)
        if cache is not None:
            cached_depth, cached_tree = cache
            if depth <= cached_depth:
                return cached_tree
