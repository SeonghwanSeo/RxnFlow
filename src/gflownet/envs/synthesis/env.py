from pathlib import Path
import warnings

import numpy as np
from rdkit import Chem, RDLogger

from typing import List, Set, Tuple, Union, Optional

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv
from gflownet.envs.synthesis.utils import Reaction
from gflownet.envs.synthesis.action import (
    ReactionAction,
    ReactionActionType,
    BackwardAction,
    ForwardAction,
    RetroSynthesisTree,
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
        self.num_building_blocks: int = len(BUILDING_BLOCKS)
        self.precomputed_bb_masks = PRECOMPUTED_BB_MASKS
        # self.num_actions = len(self.unimolecular_reactions) + self.precomputed_bb_masks.any(-1).sum().item()
        self.num_average_possible_actions = len(self.building_blocks) // 2

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
                assert isinstance(action.block, str)
                return Chem.MolFromSmiles(action.block)
            elif action.action is ReactionActionType.ReactUni:
                assert isinstance(action.reaction, Reaction)
                p = action.reaction.run_reactants((mol,))
                assert p is not None, "reaction is Fail"
                return p
            elif action.action is ReactionActionType.ReactBi:
                assert isinstance(action.reaction, Reaction)
                assert isinstance(action.block, str)
                p = action.reaction.run_reactants((mol, Chem.MolFromSmiles(action.block)))
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

    def reverse(self, ga: Union[str, Chem.Mol, Graph, None], ra: ReactionAction) -> ReactionAction:
        if ra.action == ReactionActionType.Stop:
            if isinstance(ra, ForwardAction):
                return BackwardAction(ReactionActionType.Stop)
            else:
                return ForwardAction(ReactionActionType.Stop)
        elif isinstance(ra, ForwardAction):
            if ra.action == ReactionActionType.AddFirstReactant:
                return BackwardAction(ReactionActionType.BckRemoveFirstReactant, None, ra.block, ra.block_local_idx)
            elif ra.action == ReactionActionType.ReactUni:
                return BackwardAction(ReactionActionType.BckReactUni, ra.reaction)
            elif ra.action == ReactionActionType.ReactBi:
                return BackwardAction(
                    ReactionActionType.BckReactBi, ra.reaction, ra.block, ra.block_local_idx, ra.block_is_first
                )
            else:
                raise ValueError(ra)
        else:
            if ra.action == ReactionActionType.BckRemoveFirstReactant:
                return ForwardAction(ReactionActionType.AddFirstReactant, None, ra.block, ra.block_local_idx)
            elif ra.action == ReactionActionType.BckReactUni:
                return ForwardAction(ReactionActionType.ReactUni, ra.reaction)
            elif ra.action == ReactionActionType.BckReactBi:
                return ForwardAction(
                    ReactionActionType.ReactBi, ra.reaction, ra.block, ra.block_local_idx, ra.block_is_first
                )
            else:
                raise ValueError(ra)

    def count_reverse_traj_lengths(
        self,
        mol: Chem.Mol,
        max_len: int,
        block_set: Optional[Set[str]] = None,
        known_childs: List[Tuple[BackwardAction, List[List[BackwardAction]]]] = [],
    ) -> List[int]:
        trajectories = self.get_reverse_trajectories(mol, max_len, block_set, known_childs)
        traj_lens = [sum(len(traj) == _t for traj in trajectories) for _t in range(0, max_len + 1)]
        return traj_lens

    def get_reverse_trajectories(
        self,
        mol: Chem.Mol,
        max_len: int,
        block_set: Optional[Set[str]] = None,
        known_childs: List[Tuple[BackwardAction, List[List[BackwardAction]]]] = [],
    ) -> List[List[BackwardAction]]:
        if mol.GetNumAtoms() == 0:
            return []
        if max_len == 0:
            return []
        if block_set is None:
            block_set = self.building_block_set

        trajectories: List[List[BackwardAction]] = []
        pass_bck_remove = False
        pass_bck_reactuni = []
        pass_bck_reactbi = []

        for bck_action, subtrajs in known_childs:
            if bck_action.action is ReactionActionType.BckRemoveFirstReactant:
                pass_bck_remove = True
                subtrajs = [[]]
            elif bck_action.action is ReactionActionType.BckReactUni:
                pass_bck_reactuni.append(bck_action.reaction)
            elif bck_action.action is ReactionActionType.BckReactBi:
                pass_bck_reactbi.append(bck_action.reaction)
            else:
                raise ValueError(bck_action)
            trajectories.extend([[bck_action] + subtraj for subtraj in subtrajs if len(subtraj) < max_len])

        smiles = Chem.MolToSmiles(mol)

        if not pass_bck_remove:
            if smiles in block_set:
                trajectories.append([BackwardAction(ReactionActionType.BckRemoveFirstReactant, block=smiles)])

        if max_len <= 1:
            return trajectories

        mol1 = Chem.MolFromSmiles(smiles)
        if mol1 is not None:
            Chem.Kekulize(mol1, clearAromaticFlags=True)

        for reaction in self.unimolecular_reactions:
            if reaction in pass_bck_reactuni:
                continue
            bck_action = BackwardAction(ReactionActionType.BckReactUni, reaction)
            try:
                if reaction.is_product(mol):
                    parent_mol = reaction.run_reverse_reactants((mol,))
                elif mol1 is not None and reaction.is_product(mol1):
                    parent_mol = reaction.run_reverse_reactants((mol1,))
                else:
                    continue
                assert parent_mol is not None
                assert isinstance(parent_mol, Chem.Mol)
            except Exception:
                continue
            for subtraj in self.get_reverse_trajectories(parent_mol, max_len - 1, block_set):
                trajectories.append([bck_action] + subtraj)

        for reaction in self.bimolecular_reactions:
            if reaction in pass_bck_reactbi:
                continue
            try:
                if reaction.is_product(mol):
                    parent_mols = reaction.run_reverse_reactants((mol,))
                elif mol1 is not None and reaction.is_product(mol1):
                    parent_mols = reaction.run_reverse_reactants((mol1,))
                else:
                    continue
                assert isinstance(parent_mols, list) and len(parent_mols) == 2
                reactant_mol1, reactant_mol2 = parent_mols
                reactant_smi1, reactant_smi2 = Chem.MolToSmiles(reactant_mol1), Chem.MolToSmiles(reactant_mol2)
            except Exception:
                continue
            for block, parent, block_order in [(reactant_smi1, reactant_mol2, 1), (reactant_smi2, reactant_mol1, 2)]:
                if block in block_set:
                    bck_action = BackwardAction(
                        ReactionActionType.BckReactUni, reaction, block, None, block_is_first=(block_order == 1)
                    )
                    for subtraj in self.get_reverse_trajectories(parent, max_len - 1, block_set):
                        trajectories.append([bck_action] + subtraj)
        return trajectories

    def retrosynthesis(
        self,
        mol: Chem.Mol,
        max_len: int,
        block_set: Optional[Set[str]] = None,
        known_branches: List[tuple[BackwardAction, RetroSynthesisTree]] = [],
    ) -> RetroSynthesisTree:
        if block_set is None:
            block_set = self.building_block_set

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
                for traj in subtree.iteration():
                    if traj[0].action == ReactionActionType.BckRemoveFirstReactant:
                        _ba1 = BackwardAction(
                            ReactionActionType.BckRemoveFirstReactant,
                            block=bck_action.block,
                            block_local_idx=bck_action.block_local_idx,
                        )
                        _ba2 = BackwardAction(
                            ReactionActionType.BckReactBi,
                            reaction=bck_action.reaction,
                            block=traj[0].block,
                            block_local_idx=traj[0].block_local_idx,
                            block_is_first=not (bck_action.block_is_first),
                        )
                        _rdmol = Chem.MolFromSmiles(bck_action.block)
                        _rt = self.retrosynthesis(_rdmol, max_len - 1, known_branches=[(_ba1, RetroSynthesisTree())])
                        branches.append((_ba2, _rt))
                        break
            else:
                raise ValueError(bck_action)

        branches.extend(
            self._dfs_retrosynthesis(mol, max_len, block_set, pass_bck_remove, pass_bck_reactuni, pass_bck_reactbi)
        )
        return RetroSynthesisTree(branches)

    def _dfs_retrosynthesis(
        self,
        mol: Chem.Mol,
        max_len: int,
        block_set: Set[str],
        pass_bck_remove: bool = False,
        pass_bck_reactuni: list = [],
        pass_bck_reactbi: list = [],
    ) -> List[tuple[BackwardAction, RetroSynthesisTree]]:
        if mol.GetNumAtoms() == 0:
            return []
        if max_len == 0:
            return []

        smiles = Chem.MolToSmiles(mol)

        branches = []
        if (not pass_bck_remove) and (smiles in block_set):
            branches.append(
                (BackwardAction(ReactionActionType.BckRemoveFirstReactant, block=smiles), RetroSynthesisTree())
            )
        if max_len <= 1:
            return branches

        mol1 = Chem.MolFromSmiles(smiles)
        if mol1 is not None:
            Chem.Kekulize(mol1, clearAromaticFlags=True)
        for reaction in self.unimolecular_reactions:
            if reaction in pass_bck_reactuni:
                continue
            bck_action = BackwardAction(ReactionActionType.BckReactUni, reaction)
            try:
                if reaction.is_product(mol):
                    parent = reaction.run_reverse_reactants((mol,))
                elif mol1 is not None and reaction.is_product(mol1):
                    parent = reaction.run_reverse_reactants((mol1,))
                else:
                    continue
                assert parent is not None
                assert isinstance(parent, Chem.Mol)
            except Exception:
                continue
            _branches = self._dfs_retrosynthesis(parent, max_len - 1, block_set)
            if len(_branches) > 0:
                branches.append((bck_action, RetroSynthesisTree(_branches)))

        for reaction in self.bimolecular_reactions:
            if reaction in pass_bck_reactbi:
                continue
            try:
                if reaction.is_product(mol):
                    parent_mols = reaction.run_reverse_reactants((mol,))
                elif mol1 is not None and reaction.is_product(mol1):
                    parent_mols = reaction.run_reverse_reactants((mol1,))
                else:
                    continue
                assert isinstance(parent_mols, list) and len(parent_mols) == 2
                reactant_mol1, reactant_mol2 = parent_mols
                reactant_smi1, reactant_smi2 = Chem.MolToSmiles(reactant_mol1), Chem.MolToSmiles(reactant_mol2)
            except Exception:
                continue
            for block, parent, block_order in [(reactant_smi1, reactant_mol2, 1), (reactant_smi2, reactant_mol1, 2)]:
                if block in block_set:
                    bck_action = BackwardAction(
                        ReactionActionType.BckReactUni, reaction, block, None, block_is_first=(block_order == 1)
                    )
                    _branches = self._dfs_retrosynthesis(parent, max_len - 1, block_set)
                    if len(_branches) > 0:
                        branches.append((bck_action, RetroSynthesisTree(_branches)))
        return branches
