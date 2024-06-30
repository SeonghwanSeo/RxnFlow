from pathlib import Path
import warnings

from tqdm import tqdm
import numpy as np
from rdkit import Chem, RDLogger

from typing import List, Set, Tuple, Union

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv
from gflownet.envs.synthesis.utils import Reaction
from gflownet.envs.synthesis.action import (
    ReactionAction,
    ReactionActionType,
    BackwardAction,
    ForwardAction,
)

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")


class SynthesisEnv(GraphBuildingEnv):
    """Molecules and reaction templates environment. The new (initial) state are Empty Molecular Graph.

    This environment specifies how to obtain new molecules from applying reaction templates to current molecules. Works by
    having the agent select a reaction template. Masks ensure that only valid templates are selected.
    """

    def __init__(self, env_dir: Union[str, Path]):
        """A reaction template and building block environment instance"""
        self.env_dir = env_dir = Path(env_dir)
        reaction_template_path = env_dir / "template.txt"
        building_block_path = env_dir / "building_block.smi"
        pre_computed_building_block_mask_path = env_dir / "precompute_bb_mask.npy"

        with reaction_template_path.open() as file:
            REACTION_TEMPLATES = file.readlines()
        with building_block_path.open() as file:
            lines = file.readlines()
            BUILDING_BLOCKS = [ln.split()[0] for ln in lines]
            BUILDING_BLOCK_IDS = [ln.strip().split()[1] for ln in lines]
        PRECOMPUTED_BB_MASKS = np.load(pre_computed_building_block_mask_path)

        self.reactions = [Reaction(template=t.strip()) for t in REACTION_TEMPLATES]  # Reaction objects
        self.unimolecular_reactions = [r for r in self.reactions if r.num_reactants == 1]  # rdKit reaction objects
        self.bimolecular_reactions = [r for r in self.reactions if r.num_reactants == 2]
        self.building_blocks: List[str] = BUILDING_BLOCKS
        self.building_block_ids: List[str] = BUILDING_BLOCK_IDS
        self.building_block_set: Set[str] = set(BUILDING_BLOCKS)
        self.precomputed_bb_masks = PRECOMPUTED_BB_MASKS

    def new(self) -> Graph:
        return Graph()

    def step(self, mol: Chem.Mol, action: ReactionAction) -> Chem.Mol:
        """Applies the action to the current state and returns the next state.

        Args:
            mol (Chem.Mol): Current state as an RDKit mol.
            action Tuple[int, Optional[int], Optional[int]]: Action indices to apply to the current state.
            (ActionType, reaction_template_idx, reactant_idx)

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)

        if action.action is ReactionActionType.Stop:
            return mol
        elif isinstance(action, ForwardAction):
            if action.action == ReactionActionType.AddFirstReactant:
                assert isinstance(action.block, Chem.Mol)
                return action.block
            elif action.action is ReactionActionType.ReactUni:
                assert isinstance(action.reaction, Reaction)
                p = action.reaction.run_reactants((mol,))
                assert p is not None, "reaction is Fail"
                return p
            elif action.action is ReactionActionType.ReactBi:
                assert isinstance(action.reaction, Reaction)
                assert isinstance(action.block, Chem.Mol)
                p = action.reaction.run_reactants((mol, action.block))
                assert p is not None, "reaction is Fail"
                return p
            else:
                raise ValueError
        elif isinstance(action, BackwardAction):
            if action.action == ReactionActionType.BckRemoveFirstReactant:
                return Chem.Mol()
            elif action.action is ReactionActionType.BckReactUni:
                assert isinstance(action.reaction, Reaction)
                reactant = action.reaction.run_reverse_reactants((mol,))
                assert isinstance(reactant, Chem.Mol)
                return reactant
            elif action.action is ReactionActionType.BckReactBi:
                assert isinstance(action.reaction, Reaction)
                reactants = action.reaction.run_reverse_reactants((mol))
                assert isinstance(reactants, list) and len(reactants) == 2
                reactant_mol1, reactant_mol2 = reactants
                selected_mol = reactant_mol2 if action.block_is_first else reactant_mol1
                rw_mol = Chem.RWMol(selected_mol)

                atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == "*"]
                for idx in sorted(atoms_to_remove, reverse=True):
                    # Remove atoms in reverse order to avoid reindexing issues
                    rw_mol.ReplaceAtom(idx, Chem.Atom("H"))
                atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == "[CH]"]
                for idx in sorted(atoms_to_remove, reverse=True):
                    # Remove atoms in reverse order to avoid reindexing issues
                    rw_mol.ReplaceAtom(idx, Chem.Atom("C"))
                try:
                    rw_mol.UpdatePropertyCache()
                except Chem.rdchem.AtomValenceException as e:
                    warnings.warn(f"{e}: Reaction {action.reaction.template}, product {Chem.MolToSmiles(selected_mol)}")
                return rw_mol

            else:
                raise ValueError(action.action)
        else:
            raise ValueError(action.action)

    def parents(self, mol: Chem.Mol) -> List[Tuple[ReactionAction, Chem.Mol]]:
        """List possible parents of graph `g`

        Parameters
        ----------
        g: Graph
            graph

        Returns
        -------
        parents: List[Pair(GraphAction, Graph)]
            The list of parent-action pairs that lead to `g`.
        """
        parents: List[Tuple[ReactionAction, Chem.Mol]] = []
        # Count node degrees

        mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        Chem.Kekulize(mol1, clearAromaticFlags=True)

        for reaction in self.unimolecular_reactions:
            if reaction.is_product(mol):
                parent_mol = reaction.run_reverse_reactants((mol,))
            elif reaction.is_product(mol1):
                parent_mol = reaction.run_reverse_reactants((mol1,))
            else:
                continue
            assert isinstance(parent_mol, Chem.Mol)
            parents.append((BackwardAction(ReactionActionType.BckReactUni, reaction=reaction), parent_mol))

        for reaction in self.bimolecular_reactions:
            try:
                if reaction.is_product(mol):
                    parent_mols = reaction.run_reverse_reactants((mol,))
                elif reaction.is_product(mol1):
                    parent_mols = reaction.run_reverse_reactants((mol1,))
                else:
                    continue
                assert isinstance(parent_mols, list) and len(parent_mols) == 2
                reactant_mol1, reactant_mol2 = parent_mols
                is_block1 = Chem.MolToSmiles(reactant_mol1) in self.building_block_set
                is_block2 = Chem.MolToSmiles(reactant_mol2) in self.building_block_set
                if is_block1:
                    action = BackwardAction(ReactionActionType.BckReactUni, reaction=reaction, block_is_first=True)
                    parents.append((action, reactant_mol2))
                if is_block2:
                    action = BackwardAction(ReactionActionType.BckReactUni, reaction=reaction, block_is_first=False)
                    parents.append((action, reactant_mol1))
            except Exception as e:
                continue
        smiles = Chem.MolToSmiles(mol)
        if smiles in self.building_block_set:
            parents.append((BackwardAction(ReactionActionType.BckRemoveFirstReactant), Chem.Mol()))
        return parents

    def count_backward_transitions(self, mol: Chem.Mol):
        """Counts the number of parents of g (by default, without checking for isomorphisms)"""
        # We can count actions backwards easily, but only if we don't check that they don't lead to
        # the same parent. To do so, we need to enumerate (unique) parents and count how many there are:
        return len(self.parents(mol))

    def reverse(self, mol: Chem.Mol, ra: ReactionAction):
        if ra.action == ReactionActionType.Stop:
            return ra
        elif isinstance(ra, ForwardAction):
            if ra.action == ReactionActionType.AddFirstReactant:
                return BackwardAction(ReactionActionType.BckRemoveFirstReactant)
            elif ra.action == ReactionActionType.ReactUni:
                return BackwardAction(ReactionActionType.BckReactUni, reaction=ra.reaction)
            elif ra.action == ReactionActionType.ReactBi:
                return BackwardAction(
                    ReactionActionType.BckReactBi, reaction=ra.reaction, block_is_first=ra.block_is_first
                )
            else:
                raise ValueError(ra)
        elif isinstance(ra, BackwardAction):
            raise NotImplementedError
        else:
            ValueError(ra)
