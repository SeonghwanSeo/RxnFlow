import copy
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from collections.abc import Iterable

from gflownet.envs.synthesis.reaction import Reaction
from gflownet.envs.synthesis.action import ReactionActionType, ReactionAction


class RetroSynthesisTree:
    def __init__(self, smi: str, branches: list = []):
        self.smi: str = smi
        self.branches: list[tuple[ReactionAction, RetroSynthesisTree]] = branches

    @property
    def is_leaf(self) -> bool:
        return len(self.branches) == 0

    def iteration(
        self, prev_traj: list[ReactionAction] = [], max_len: int | None = None
    ) -> Iterable[list[ReactionAction]]:
        max_len = max_len if max_len else 100
        prev_len = len(prev_traj)
        if self.is_leaf and prev_len <= max_len:
            yield prev_traj
        elif (not self.is_leaf) and (prev_len < max_len):
            for bck_action, subtree in self.branches:
                yield from subtree.iteration(prev_traj + [bck_action], max_len)

    def __len__(self):
        return len(self.branches)

    def depth(self, max_len: int) -> int:
        return min(self.iteration_length(0, max_len))

    def length_distribution(self, max_len: int) -> list[int]:
        lengths = list(self.iteration_length(0, max_len))
        return [sum(length == _t for length in lengths) for _t in range(0, max_len + 1)]

    def iteration_length(self, prev_len: int, max_len: int) -> Iterable[int]:
        if self.is_leaf and prev_len <= max_len:
            yield prev_len
        elif (not self.is_leaf) and (prev_len < max_len):
            for _, subtree in self.branches:
                yield from subtree.iteration_length(prev_len + 1, max_len)

    def filtering(self, block_set: set[str] | None):
        if block_set is None:
            return self
        filtered_branches = []
        for action, subtree in self.branches:
            if action.action is ReactionActionType.BckRemoveFirstReactant:
                assert subtree.is_leaf
                if action.block not in block_set:
                    continue
            else:
                if action.action is ReactionActionType.BckReactBi and action.block not in block_set:
                    continue
                subtree = subtree.filtering(block_set)
                if subtree.is_leaf:
                    continue
            filtered_branches.append((action, subtree))
        return RetroSynthesisTree(self.smi, filtered_branches)


RetroSynthesisBranch = list[tuple[ReactionAction, RetroSynthesisTree]]


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

    def submit(
        self, key: int, mol: Chem.Mol, max_depth: int, known_branches: list[tuple[ReactionAction, RetroSynthesisTree]]
    ):
        self.futures.append(self.pool.submit(self._worker, key, mol, max_depth, known_branches))

    def _init_worker(self, base_analyzer):
        global analyzer
        analyzer = copy.deepcopy(base_analyzer)

    @staticmethod
    def _worker(
        key: int, mol: Chem.Mol, max_depth: int, known_branches: list[tuple[ReactionAction, RetroSynthesisTree]]
    ) -> tuple[int, RetroSynthesisTree]:
        global analyzer
        return key, analyzer.run(mol, max_depth, known_branches)


class RetroSyntheticAnalyzer:
    def __init__(self, env):
        self.reactions: list[Reaction] = env.reactions
        self.unimolecular_reactions: list[Reaction] = env.unimolecular_reactions
        self.bimolecular_reactions: list[Reaction] = env.bimolecular_reactions

        # For Fast Search
        self.__max_block_smi_len: int = 0
        self.__prefix_len = 5
        self.__building_block_search: dict[str, set[str]] = {}

        for smi in env.building_blocks:
            self.__max_block_smi_len = max(self.__max_block_smi_len, len(smi))
            prefix = smi[: self.__prefix_len]
            self.__building_block_search.setdefault(prefix, set()).add(smi)

        self.__cache: Cache = Cache(100_000)
        self.__block_cache: Cache = Cache(1_000_000)

    def is_block(self, smi: str) -> bool:
        if len(smi) > self.__max_block_smi_len:
            return False
        prefix = smi[: self.__prefix_len]
        prefix_block_set = self.__building_block_search.get(prefix, None)
        if prefix_block_set is not None:
            return smi in prefix_block_set
        return False

    def load_cache(self, smi: str, depth: int, is_block: bool) -> RetroSynthesisBranch | None:
        if is_block:
            return self.__block_cache.get(smi, depth)
        else:
            return self.__cache.get(smi, depth)

    def update_cache(self, smi: str, depth: int, branch: RetroSynthesisBranch, is_block: bool):
        if is_block:
            return self.__block_cache.update(smi, depth, branch)
        else:
            return self.__cache.update(smi, depth, branch)

    def run(
        self,
        mol: Chem.Mol,
        max_depth: int,
        known_branches: list[tuple[ReactionAction, RetroSynthesisTree]] = [],
    ) -> RetroSynthesisTree:
        return self._retrosynthesis(mol, max_depth, known_branches)

    def _retrosynthesis(
        self,
        mol: Chem.Mol,
        max_depth: int,
        known_branches: RetroSynthesisBranch = [],
    ) -> RetroSynthesisTree:
        pass_bck_remove = False
        pass_bck_reactuni = []
        pass_bck_reactbi = []

        branches = known_branches.copy()
        for bck_action, subtree in known_branches:
            if bck_action.action is ReactionActionType.BckRemoveFirstReactant:
                pass_bck_remove = True
            elif bck_action.action is ReactionActionType.BckReactUni:
                pass_bck_reactuni.append(bck_action.reaction)
            elif bck_action.action is ReactionActionType.BckReactBi:
                pass_bck_reactbi.append(bck_action.reaction)
                for next_bck_action, _ in subtree.branches:
                    if next_bck_action.action is ReactionActionType.BckRemoveFirstReactant:
                        _ba1 = ReactionAction(
                            ReactionActionType.BckRemoveFirstReactant,
                            block=bck_action.block,
                            block_idx=bck_action.block_idx,
                        )
                        _ba2 = ReactionAction(
                            ReactionActionType.BckReactBi,
                            reaction=bck_action.reaction,
                            block=next_bck_action.block,
                            block_idx=next_bck_action.block_idx,
                            block_is_first=not (bck_action.block_is_first),
                        )
                        _rdmol = Chem.MolFromSmiles(bck_action.block)
                        _rt = self._retrosynthesis(_rdmol, max_depth - 1, [(_ba1, RetroSynthesisTree(""))])
                        branches.append((_ba2, _rt))
                        break
            else:
                raise ValueError(bck_action)

        smiles = Chem.MolToSmiles(mol)
        branches.extend(
            self.__dfs_retrosynthesis(mol, smiles, max_depth, pass_bck_remove, pass_bck_reactuni, pass_bck_reactbi)
        )
        return RetroSynthesisTree(smiles, branches)

    def __dfs_retrosynthesis(
        self,
        mol: Chem.Mol,
        smiles: str,
        max_depth: int,
        pass_bck_remove: bool = False,
        pass_bck_reactuni: list = [],
        pass_bck_reactbi: list = [],
    ) -> list[tuple[ReactionAction, RetroSynthesisTree]]:
        if max_depth == 0:
            return []
        if mol.GetNumAtoms() == 0:
            return []

        is_block = pass_bck_remove or self.is_block(smiles)

        cached_branches = self.load_cache(smiles, max_depth, is_block)
        if cached_branches is not None:
            return cached_branches

        branches = []
        if (not pass_bck_remove) and is_block:
            branches.append(
                (ReactionAction(ReactionActionType.BckRemoveFirstReactant, block=smiles), RetroSynthesisTree(""))
            )
        if max_depth == 1:
            return branches

        try:
            kekulized_mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(kekulized_mol, clearAromaticFlags=True)
        except Exception:
            kekulized_mol = None

        def _run_reaction(reaction: Reaction, mol: Chem.Mol, kekulized_mol: Chem.Mol):
            parents = reaction.run_reverse_reactants(mol)
            if (parents is None) and (kekulized_mol is not None):
                parents = reaction.run_reverse_reactants(kekulized_mol)
            return parents

        for reaction in self.unimolecular_reactions:
            if reaction in pass_bck_reactuni:
                continue
            parents = _run_reaction(reaction, mol, kekulized_mol)
            if parents is None:
                continue
            assert len(parents) == 1
            _branches = self.__dfs_retrosynthesis(mol, Chem.MolToSmiles(mol), max_depth - 1)
            if len(_branches) > 0:
                bck_action = ReactionAction(ReactionActionType.BckReactUni, reaction)
                branches.append((bck_action, RetroSynthesisTree(smiles, _branches)))

        for reaction in self.bimolecular_reactions:
            if reaction in pass_bck_reactbi:
                continue
            parents = _run_reaction(reaction, mol, kekulized_mol)
            if parents is None:
                continue
            assert len(parents) == 2
            parents_smi = Chem.MolToSmiles(parents[0]), Chem.MolToSmiles(parents[1])
            for i, j in [(0, 1), (1, 0)]:
                block_smi = parents_smi[i]
                if self.is_block(block_smi):
                    _branches = self.__dfs_retrosynthesis(parents[j], parents_smi[j], max_depth - 1)
                    if len(_branches) > 0:
                        bck_action = ReactionAction(
                            ReactionActionType.BckReactUni, reaction, block_smi, block_is_first=(i == 0)
                        )
                        branches.append((bck_action, RetroSynthesisTree(smiles, _branches)))

        self.update_cache(smiles, max_depth, branches, is_block)
        return branches


class Cache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: dict[str, tuple[int, RetroSynthesisBranch]] = {}

    def update(self, smiles: str, depth: int, branch: RetroSynthesisBranch):
        cache = self.cache.get(smiles, None)
        if cache is None:
            if len(self.cache) >= self.max_size:
                self.cache.popitem()
            self.cache[smiles] = (depth, branch)
        else:
            cached_depth, cached_branch = cache
            if depth > cached_depth:
                self.cache[smiles] = (depth, branch)

    def get(self, smiles: str, depth: int) -> RetroSynthesisBranch | None:
        cache = self.cache.get(smiles, None)
        if cache is not None:
            cached_depth, cached_branch = cache
            if depth <= cached_depth:
                return cached_branch
