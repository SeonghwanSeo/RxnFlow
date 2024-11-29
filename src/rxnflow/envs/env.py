from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol as RDMol

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv
from .reaction import Reaction
from .action import RxnAction, RxnActionType
from .retrosynthesis import RetroSyntheticAnalyzer

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")


class SynthesisEnv(GraphBuildingEnv):
    """Molecules and reaction templates environment. The new (initial) state are Empty Molecular Graph.

    This environment specifies how to obtain new molecules from applying reaction templates to current molecules. Works by
    having the agent select a reaction template. Masks ensure that only valid templates are selected.
    """

    def __init__(self, env_dir: str | Path):
        """A reaction template and building block environment instance"""
        self.env_dir = env_dir = Path(env_dir)
        reaction_template_path = env_dir / "template.txt"
        building_block_path = env_dir / "building_block.smi"
        pre_computed_building_block_mask_path = env_dir / "bb_mask.npy"
        pre_computed_building_block_fp_path = env_dir / "bb_fp_2_1024.npy"
        pre_computed_building_block_desc_path = env_dir / "bb_desc.npy"

        with reaction_template_path.open() as file:
            REACTION_TEMPLATES = file.readlines()
        with building_block_path.open() as file:
            lines = file.readlines()
            BUILDING_BLOCKS = [ln.split()[0] for ln in lines]
            BUILDING_BLOCK_IDS = [ln.strip().split()[1] for ln in lines]

        self.reactions = [Reaction(template=t.strip()) for t in REACTION_TEMPLATES]  # Reaction objects
        self.uni_rxns = [r for r in self.reactions if r.num_reactants == 1]  # rdKit reaction objects
        self.bi_rxns = [r for r in self.reactions if r.num_reactants == 2]
        self.num_uni_rxns = len(self.uni_rxns)
        self.num_bi_rxns = len(self.bi_rxns)

        self.building_blocks: list[str] = BUILDING_BLOCKS
        self.building_block_ids: list[str] = BUILDING_BLOCK_IDS
        self.num_building_blocks: int = len(BUILDING_BLOCKS)

        self.building_block_mask: NDArray[np.bool_] = np.load(pre_computed_building_block_mask_path)
        self.building_block_features: tuple[NDArray[np.bool_], NDArray[np.float32]] = (
            np.load(pre_computed_building_block_fp_path),
            np.load(pre_computed_building_block_desc_path),
        )

        self.num_total_actions = 1 + self.num_uni_rxns + int(self.building_block_mask.sum())
        self.retrosynthetic_analyzer: RetroSyntheticAnalyzer = RetroSyntheticAnalyzer(self)

    def new(self) -> Graph:
        return Graph()

    def step(self, mol: RDMol, action: RxnAction) -> Chem.Mol:
        """Applies the action to the current state and returns the next state.

        Args:
            mol (Chem.Mol): Current state as an RDKit mol.
            action tuple[int, Optional[int], Optional[int]]: Action indices to apply to the current state.
            (ActionType, reaction_template_idx, reactant_idx)

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)

        if action.action is RxnActionType.Stop:
            return mol
        elif action.action == RxnActionType.FirstBlock:
            assert isinstance(action.block, str)
            return Chem.MolFromSmiles(action.block)
        elif action.action is RxnActionType.UniRxn:
            assert isinstance(action.reaction, Reaction)
            p = action.reaction.run_reactants((mol,), safe=False)
            assert p is not None, "reaction is Fail"
            return p
        elif action.action is RxnActionType.BiRxn:
            assert isinstance(action.reaction, Reaction)
            assert isinstance(action.block, str)
            if action.block_is_first:
                p = action.reaction.run_reactants((Chem.MolFromSmiles(action.block), mol), safe=False)
            else:
                p = action.reaction.run_reactants((mol, Chem.MolFromSmiles(action.block)), safe=False)
            assert p is not None, "reaction is Fail"
            return p
        if action.action == RxnActionType.BckFirstBlock:
            return Chem.Mol()
        elif action.action is RxnActionType.BckUniRxn:
            assert isinstance(action.reaction, Reaction)
            reactant = action.reaction.run_reverse_reactants(mol)
            assert isinstance(reactant, Chem.Mol)
            return reactant
        elif action.action is RxnActionType.BckBiRxn:
            assert isinstance(action.reaction, Reaction)
            reactants = action.reaction.run_reverse_reactants(mol)
            assert isinstance(reactants, list) and len(reactants) == 2
            reactant = reactants[1] if action.block_is_first else reactants[0]
            return reactant
        else:
            raise ValueError(action.action)

    def parents(self, mol: RDMol, max_depth: int = 4) -> list[tuple[RxnAction, str]]:
        """list possible parents of molecule `mol`

        Parameters
        ----------
        mol: Chem.Mol
            molecule

        Returns
        -------
        parents: list[Pair(RxnAction, str)]
            The list of parent-action pairs
        """
        retro_tree = self.retrosynthetic_analyzer.run(mol, max_depth)
        return [(action, subtree.smi) for action, subtree in retro_tree.branches]

    def count_backward_transitions(self, mol: RDMol, check_idempotent: bool = False):
        """Counts the number of parents of molecule (by default, without checking for isomorphisms)"""
        # We can count actions backwards easily, but only if we don't check that they don't lead to
        # the same parent. To do so, we need to enumerate (unique) parents and count how many there are:
        return len(self.parents(mol))

    def reverse(self, g: str | RDMol | Graph | None, ra: RxnAction) -> RxnAction:
        if ra.action == RxnActionType.Stop:
            return ra
        if ra.action == RxnActionType.FirstBlock:
            return RxnAction(RxnActionType.BckFirstBlock, None, ra.block, ra.block_idx)
        elif ra.action == RxnActionType.BckFirstBlock:
            return RxnAction(RxnActionType.FirstBlock, None, ra.block, ra.block_idx)
        elif ra.action == RxnActionType.UniRxn:
            return RxnAction(RxnActionType.BckUniRxn, ra.reaction)
        elif ra.action == RxnActionType.BckUniRxn:
            return RxnAction(RxnActionType.UniRxn, ra.reaction)
        elif ra.action == RxnActionType.BiRxn:
            return RxnAction(RxnActionType.BckBiRxn, ra.reaction, ra.block, ra.block_idx, ra.block_is_first)
        elif ra.action == RxnActionType.BckBiRxn:
            return RxnAction(RxnActionType.BiRxn, ra.reaction, ra.block, ra.block_idx, ra.block_is_first)
        else:
            raise ValueError(ra)
