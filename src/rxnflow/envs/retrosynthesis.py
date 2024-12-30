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


RetroSynthesisBranch = list[tuple[RxnAction, RetroSynthesisTree]]


class Cache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: dict[str, RetroSynthesisBranch] = {}

    def update(self, smiles: str, branch: RetroSynthesisBranch):
        cache = self.cache.get(smiles, None)
        if cache is None:
            if len(self.cache) >= self.max_size:
                self.cache.popitem()
            self.cache[smiles] = branch

    def get(self, smiles: str) -> RetroSynthesisBranch | None:
        return self.cache.get(smiles, None)


class RetroSyntheticAnalyzer:
    def __init__(self, blocks: dict[str, list[str]], workflow: Workflow):
        self.protocols: list[Protocol] = workflow.protocols
        self.reverse_workflows: dict[Protocol, set[Protocol]] = {protocol: set() for protocol in self.protocols}
        for protocol_traj in workflow.root.iteration():
            for i in range(1, len(protocol_traj)):
                protocol, prev_protocol = protocol_traj[i], protocol_traj[i - 1]
                self.reverse_workflows[protocol].add(prev_protocol)

        self.__block_search: dict[str, dict[str, int]] = {
            block_type: {block: i for i, block in enumerate(blocks)} for block_type, blocks in blocks.items()
        }
        self.use_cache = False

        self.__cache: Cache = Cache(100_000)

    def from_cache(self, smi: str) -> RetroSynthesisBranch | None:
        if self.use_cache:
            return self.__cache.get(smi)

    def to_cache(self, smi: str, branch: RetroSynthesisBranch):
        if self.use_cache:
            return self.__cache.update(smi, branch)

    def run(
        self,
        mol: Chem.Mol,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree:
        return self._retrosynthesis(mol, known_branches)

    def _retrosynthesis(
        self,
        mol: Chem.Mol,
        known_branches: RetroSynthesisBranch | None = None,
    ) -> RetroSynthesisTree:
        known_branches = known_branches if known_branches else []
        branches = known_branches.copy()

        # TODO: add known branches
        smiles = Chem.MolToSmiles(mol)
        branches.extend(self.__dfs(mol, smiles))
        return RetroSynthesisTree(smiles, branches)

    def __dfs(
        self,
        mol: Chem.Mol,
        smiles: str,
        available_protocols: Iterable[Protocol] | None = None,
    ) -> list[tuple[RxnAction, RetroSynthesisTree]]:
        cached_branches = self.from_cache(smiles)
        if cached_branches is not None:
            return cached_branches

        branches = []
        protocols = available_protocols if available_protocols else self.protocols
        for protocol in protocols:
            if protocol.action is RxnActionType.Stop:
                continue
            if protocol.action is RxnActionType.FirstBlock:
                assert protocol.block_type is not None
                block_idx = self.block_search(protocol.block_type, smiles)
                bck_action = RxnAction(RxnActionType.BckFirstBlock, protocol, smiles, block_idx)
                if block_idx is not None:
                    branches.append((bck_action, RetroSynthesisTree("")))
            elif protocol.action is RxnActionType.UniRxn:
                assert protocol.rxn_reverse is not None
                if protocol.rxn_reverse.is_reactant(mol, 0):
                    for rs in protocol.rxn_reverse(mol):
                        assert len(rs) == 1
                        state_mol = rs[0]
                        state_smi = Chem.MolToSmiles(state_mol)
                        _branches = self.__dfs(state_mol, Chem.MolToSmiles(state_mol), self.reverse_workflows[protocol])
                        if len(_branches) > 0:
                            bck_action = RxnAction(RxnActionType.BckUniRxn, protocol)
                            branches.append((bck_action, RetroSynthesisTree(state_smi, _branches)))
            elif protocol.action is RxnActionType.BiRxn:
                assert protocol.block_type is not None
                assert protocol.rxn_reverse is not None
                if protocol.rxn_reverse.is_reactant(mol, 0):
                    for rs in protocol.rxn_reverse(mol):
                        assert len(rs) == 2
                        state_mol, block = rs
                        block_smi = Chem.MolToSmiles(block)
                        block_idx = self.block_search(protocol.block_type, block_smi)
                        if block_idx is None:
                            continue
                        state_smi = Chem.MolToSmiles(state_mol)
                        _branches = self.__dfs(state_mol, state_smi, self.reverse_workflows[protocol])
                        if len(_branches) > 0:
                            bck_action = RxnAction(RxnActionType.BckBiRxn, protocol, block_smi, block_idx)
                            branches.append((bck_action, RetroSynthesisTree(state_smi, _branches)))

        self.to_cache(smiles, branches)
        return branches

    def block_search(self, block_type: str, block_smi: str) -> int | None:
        return self.__block_search[block_type].get(block_smi, None)


class MultiRetroSyntheticAnalyzer:
    def __init__(self, analyzer, num_workers: int = 4):
        self.pool = ProcessPoolExecutor(num_workers, initializer=self._init_worker, initargs=(analyzer,))
        self.futures = []

    def init(self):
        self.result()

    def result(self):
        result = [future.result() for future in self.futures]
        self.futures = []
        return result

    def submit(self, key: int, mol: Chem.Mol, known_branches: list[tuple[RxnAction, RetroSynthesisTree]]):
        self.futures.append(self.pool.submit(self._worker, key, mol, known_branches))

    def _init_worker(self, base_analyzer):
        global analyzer
        analyzer = copy.deepcopy(base_analyzer)

    @staticmethod
    def _worker(
        key: int, mol: Chem.Mol, known_branches: list[tuple[RxnAction, RetroSynthesisTree]]
    ) -> tuple[int, RetroSynthesisTree]:
        global analyzer
        return key, analyzer.run(mol, known_branches)
