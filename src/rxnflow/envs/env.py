from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol as RDMol

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv

from .action import Protocol, RxnAction, RxnActionType
from .reaction import BiReaction, Reaction, UniReaction
from .retrosynthesis import MultiRetroSyntheticAnalyzer

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")


class MolGraph(Graph):
    def __init__(self, mol: str | Chem.Mol, **kwargs):
        super().__init__(**kwargs)
        self._mol: str | Chem.Mol = mol
        self.is_setup: bool = False

    def __repr__(self):
        return self.smi

    @cached_property
    def smi(self) -> str:
        if isinstance(self._mol, Chem.Mol):
            return Chem.MolToSmiles(self._mol)
        else:
            return self._mol

    @cached_property
    def mol(self) -> Chem.Mol:
        if isinstance(self._mol, Chem.Mol):
            return self._mol
        else:
            return Chem.MolFromSmiles(self._mol)


class SynthesisEnv(GraphBuildingEnv):
    """Molecules and reaction templates environment. The new (initial) state are Empty Molecular Graph.

    This environment specifies how to obtain new molecules from applying reaction templates to current molecules. Works by
    having the agent select a reaction template. Masks ensure that only valid templates are selected.
    """

    def __init__(self, env_dir: str | Path, num_workers: int = 4):
        """Environment for Synthesis-oriented generation

        Parameters
        ----------
        env_dir : str | Path
            root directory of synthesis environment
        num_workers : int
            number of workers for retrosynthetic analysis
        """
        """A reaction template and building block environment instance"""
        self.env_dir = env_dir = Path(env_dir)
        reaction_template_path = env_dir / "template.txt"
        building_block_path = env_dir / "building_block.smi"
        pre_computed_building_block_mask_path = env_dir / "bb_mask.npy"
        pre_computed_building_block_fp_path = env_dir / "bb_fp_2_1024.npy"
        pre_computed_building_block_desc_path = env_dir / "bb_desc.npy"

        # set protocol
        self.protocols: list[Protocol] = []
        self.protocols.append(Protocol("stop", RxnActionType.Stop))
        self.protocols.append(Protocol("firstblock", RxnActionType.FirstBlock))
        with reaction_template_path.open() as file:
            reaction_templates = [ln.strip() for ln in file.readlines()]
        for i, template in enumerate(reaction_templates):
            _rxn = Reaction(template)
            if _rxn.num_reactants == 1:
                rxn = UniReaction(template)
                self.protocols.append(Protocol(f"unirxn{i}", RxnActionType.UniRxn, _rxn))
            elif _rxn.num_reactants == 2:
                for block_is_first in [True, False]:  # this order is important
                    rxn = BiReaction(template, block_is_first)
                    self.protocols.append(Protocol(f"birxn{i}_{block_is_first}", RxnActionType.BiRxn, rxn))
        self.protocol_dict: dict[str, Protocol] = {protocol.name: protocol for protocol in self.protocols}
        self.stop_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.Stop]
        self.firstblock_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.FirstBlock]
        self.unirxn_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.UniRxn]
        self.birxn_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.BiRxn]

        # set building blocks
        with building_block_path.open() as file:
            lines = file.readlines()
            building_blocks = [ln.split()[0] for ln in lines]
            building_block_ids = [ln.strip().split()[1] for ln in lines]
        self.blocks: list[str] = building_blocks
        self.block_ids: list[str] = building_block_ids
        self.num_blocks: int = len(building_blocks)

        # set precomputed building block feature
        self.block_fp = np.load(pre_computed_building_block_fp_path)
        self.block_prop = np.load(pre_computed_building_block_desc_path)

        # set block mask
        block_mask: NDArray[np.bool_] = np.load(pre_computed_building_block_mask_path)
        self.birxn_block_indices: dict[str, np.ndarray] = {}
        for i, protocol in enumerate(self.birxn_list):
            self.birxn_block_indices[protocol.name] = np.where(block_mask[i])[0]
        self.num_total_actions = (
            1 + len(self.unirxn_list) + sum(indices.shape[0] for indices in self.birxn_block_indices.values())
        )

        self.retro_analyzer = MultiRetroSyntheticAnalyzer.create(self.protocols, self.blocks, num_workers=num_workers)

    def new(self) -> MolGraph:
        return MolGraph("")

    def step(self, g: MolGraph, action: RxnAction) -> MolGraph:
        """Applies the action to the current state and returns the next state.

        Args:
            mol (Chem.Mol): Current state as an RDKit mol.
            action tuple[int, Optional[int], Optional[int]]: Action indices to apply to the current state.
            (ActionType, reaction_template_idx, reactant_idx)

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        state_info = g.graph
        protocol = self.protocol_dict[action.protocol]

        if action.action is RxnActionType.Stop:
            return g
        elif action.action is RxnActionType.BckStop:
            return g

        elif action.action == RxnActionType.FirstBlock:
            obj = action.block
        elif action.action == RxnActionType.BckFirstBlock:
            obj = ""

        elif action.action is RxnActionType.UniRxn:
            ps = protocol.rxn.forward(g.mol, strict=True)
            assert len(ps) > 0, "reaction is Fail"
            obj = Chem.MolToSmiles(ps[0][0])
        elif action.action is RxnActionType.BckUniRxn:
            rs = protocol.rxn.reverse(g.mol)[0]
            assert len(rs) > 0, "reverse reaction is Fail"
            obj = Chem.MolToSmiles(rs[0])

        elif action.action is RxnActionType.BiRxn:
            block = Chem.MolFromSmiles(action.block)
            ps = protocol.rxn.forward(g.mol, block, strict=True)
            assert len(ps) > 0, "forward reaction is Fail"
            obj = Chem.MolToSmiles(ps[0][0])
        elif action.action is RxnActionType.BckBiRxn:
            rs = protocol.rxn.reverse(g.mol)[0]
            assert len(rs) > 0, "reverse reaction is Fail"
            obj = Chem.MolToSmiles(rs[0])

        else:
            raise ValueError(action.action)
        return MolGraph(obj, **state_info)

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
        raise NotImplementedError
        retro_tree = self.retrosynthetic_analyzer.run(mol, max_depth)
        return [(action, subtree.smi) for action, subtree in retro_tree.branches]

    def count_backward_transitions(self, mol: RDMol, check_idempotent: bool = False):
        """Counts the number of parents of molecule (by default, without checking for isomorphisms)"""
        # We can count actions backwards easily, but only if we don't check that they don't lead to
        # the same parent. To do so, we need to enumerate (unique) parents and count how many there are:
        return len(self.parents(mol))

    def reverse(self, g: str | RDMol | Graph | None, ra: RxnAction) -> RxnAction:
        if ra.action == RxnActionType.Stop:
            return RxnAction(RxnActionType.BckStop, ra.protocol)
        elif ra.action == RxnActionType.BckStop:
            return RxnAction(RxnActionType.Stop, ra.protocol)
        elif ra.action == RxnActionType.FirstBlock:
            return RxnAction(RxnActionType.BckFirstBlock, ra.protocol, ra.block, ra.block_idx)
        elif ra.action == RxnActionType.BckFirstBlock:
            return RxnAction(RxnActionType.FirstBlock, ra.protocol, ra.block, ra.block_idx)
        elif ra.action == RxnActionType.UniRxn:
            return RxnAction(RxnActionType.BckUniRxn, ra.protocol)
        elif ra.action == RxnActionType.BckUniRxn:
            return RxnAction(RxnActionType.UniRxn, ra.protocol)
        elif ra.action == RxnActionType.BiRxn:
            return RxnAction(RxnActionType.BckBiRxn, ra.protocol, ra.block, ra.block_idx)
        elif ra.action == RxnActionType.BckBiRxn:
            return RxnAction(RxnActionType.BiRxn, ra.protocol, ra.block, ra.block_idx)
        else:
            raise ValueError(ra)
