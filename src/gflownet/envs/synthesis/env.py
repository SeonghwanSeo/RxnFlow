from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv
from gflownet.envs.synthesis.reaction import Reaction
from gflownet.envs.synthesis.action import ReactionAction, ReactionActionType
from gflownet.envs.synthesis.retrosynthesis import RetroSyntheticAnalyzer

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
        self.unimolecular_reactions = [r for r in self.reactions if r.num_reactants == 1]  # rdKit reaction objects
        self.bimolecular_reactions = [r for r in self.reactions if r.num_reactants == 2]
        self.num_unimolecular_rxns = len(self.unimolecular_reactions)
        self.num_bimolecular_rxns = len(self.bimolecular_reactions)

        self.building_blocks: list[str] = BUILDING_BLOCKS
        self.building_block_ids: list[str] = BUILDING_BLOCK_IDS
        self.num_building_blocks: int = len(BUILDING_BLOCKS)

        self.building_block_mask: NDArray[np.bool_] = np.load(pre_computed_building_block_mask_path)
        self.building_block_features: tuple[NDArray[np.bool_], NDArray[np.float32]] = (
            np.load(pre_computed_building_block_fp_path),
            np.load(pre_computed_building_block_desc_path),
        )

        self.num_total_actions = 1 + self.num_unimolecular_rxns + int(self.building_block_mask.sum())
        self.retrosynthetic_analyzer: RetroSyntheticAnalyzer = RetroSyntheticAnalyzer(self)

    def new(self) -> Graph:
        return Graph()

    def step(self, mol: Chem.Mol, action: ReactionAction) -> Chem.Mol:
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

        if action.action is ReactionActionType.Stop:
            return mol
        elif action.action == ReactionActionType.AddFirstReactant:
            assert isinstance(action.block, str)
            return Chem.MolFromSmiles(action.block)
        elif action.action is ReactionActionType.ReactUni:
            assert isinstance(action.reaction, Reaction)
            p = action.reaction.run_reactants((mol,), safe=False)
            assert p is not None, "reaction is Fail"
            return p
        elif action.action is ReactionActionType.ReactBi:
            assert isinstance(action.reaction, Reaction)
            assert isinstance(action.block, str)
            if action.block_is_first:
                p = action.reaction.run_reactants((Chem.MolFromSmiles(action.block), mol), safe=False)
            else:
                p = action.reaction.run_reactants((mol, Chem.MolFromSmiles(action.block)), safe=False)
            assert p is not None, "reaction is Fail"
            return p
        if action.action == ReactionActionType.BckRemoveFirstReactant:
            return Chem.Mol()
        elif action.action is ReactionActionType.BckReactUni:
            assert isinstance(action.reaction, Reaction)
            reactant = action.reaction.run_reverse_reactants(mol)
            assert isinstance(reactant, Chem.Mol)
            return reactant
        elif action.action is ReactionActionType.BckReactBi:
            assert isinstance(action.reaction, Reaction)
            reactants = action.reaction.run_reverse_reactants(mol)
            assert isinstance(reactants, list) and len(reactants) == 2
            reactant = reactants[1] if action.block_is_first else reactants[0]
            return reactant
        else:
            raise ValueError(action.action)

    def parents(self, mol: Chem.Mol, max_depth: int = 4) -> list[tuple[ReactionAction, Chem.Mol]]:
        """list possible parents of molecule `mol`

        Parameters
        ----------
        mol: Chem.Mol
            molecule

        Returns
        -------
        parents: list[Pair(ReactionAction, Chem.Mol)]
            The list of parent-action pairs
        """
        retro_tree = self.retrosynthetic_analyzer.run(mol, max_depth)
        return [(action, subtree.mol) for action, subtree in retro_tree.branches]

    def count_backward_transitions(self, mol: Chem.Mol):
        """Counts the number of parents of molecule (by default, without checking for isomorphisms)"""
        # We can count actions backwards easily, but only if we don't check that they don't lead to
        # the same parent. To do so, we need to enumerate (unique) parents and count how many there are:
        return len(self.parents(mol))

    def reverse(self, g: str | Chem.Mol | Graph | None, ra: ReactionAction) -> ReactionAction:
        if ra.action == ReactionActionType.Stop:
            return ra
        if ra.action == ReactionActionType.AddFirstReactant:
            return ReactionAction(ReactionActionType.BckRemoveFirstReactant, None, ra.block, ra.block_idx)
        elif ra.action == ReactionActionType.BckRemoveFirstReactant:
            return ReactionAction(ReactionActionType.AddFirstReactant, None, ra.block, ra.block_idx)
        elif ra.action == ReactionActionType.ReactUni:
            return ReactionAction(ReactionActionType.BckReactUni, ra.reaction)
        elif ra.action == ReactionActionType.BckReactUni:
            return ReactionAction(ReactionActionType.ReactUni, ra.reaction)
        elif ra.action == ReactionActionType.ReactBi:
            return ReactionAction(ReactionActionType.BckReactBi, ra.reaction, ra.block, ra.block_idx, ra.block_is_first)
        elif ra.action == ReactionActionType.BckReactBi:
            return ReactionAction(ReactionActionType.ReactBi, ra.reaction, ra.block, ra.block_idx, ra.block_is_first)
        else:
            raise ValueError(ra)
