import torch
from pathlib import Path
from rdkit import Chem, RDLogger
from functools import cached_property

from torch import Tensor
from rdkit.Chem import Mol as RDMol

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv
from .action import RxnAction, RxnActionType
from .workflow import Workflow, Protocol
from .retrosynthesis import RetroSyntheticAnalyzer

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")


class MolGraph(Graph):
    def __init__(self, mol: str | Chem.Mol):
        self._mol: str | Chem.Mol = mol

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

    def __init__(self, env_dir: str | Path):
        """A reaction template and building block environment instance"""
        self.env_dir = env_dir = Path(env_dir)
        protocol_config_path = env_dir / "protocol.yaml"
        block_smiles_dir = env_dir / "blocks"
        block_feature_path = env_dir / "bb_feature.pt"

        # NOTE: Load Building Blocks & Feature
        self.blocks: dict[str, list[str]] = {}
        self.block_codes: dict[str, list[str]] = {}
        self.num_blocks: dict[str, int] = {}
        for file in block_smiles_dir.iterdir():
            key = file.stem
            with file.open() as f:
                lines = f.readlines()
            self.blocks[key] = [ln.split()[0] for ln in lines]
            self.block_codes[key] = [ln.strip().split()[1] for ln in lines]
            self.num_blocks[key] = len(lines)

        self.block_features: tuple[Tensor, Tensor] = torch.load(block_feature_path)

        # NOTE: Load Protocol
        self.workflow = Workflow(protocol_config_path)
        self.protocols: list[Protocol] = self.workflow.protocols
        self.protocol_dict: dict[str, Protocol] = self.workflow.protocol_dict

        self.retro_analyzer: RetroSyntheticAnalyzer = RetroSyntheticAnalyzer(self.blocks, self.workflow)

    def new(self) -> Graph:
        return MolGraph("")

    def step(self, mol: RDMol, action: RxnAction) -> Chem.Mol:
        """Applies the action to the current state and returns the next state.

        Args:
            mol (Chem.Mol): Current state as an RDKit mol.
            action tuple[int, Optional[int], Optional[int]]: Action indices to apply to the current state.
            (ActionType, reaction_template_idx, reactant_idx)

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        if action.action is RxnActionType.Stop:
            return mol
        elif action.action == RxnActionType.FirstBlock:
            return Chem.MolFromSmiles(action.block)
        elif action.action is RxnActionType.UniRxn:
            ps = action.reaction.forward(mol)
            return ps[0][0]
        elif action.action is RxnActionType.BiRxn:
            ps = action.reaction.forward(mol, Chem.MolFromSmiles(action.block))
            return ps[0][0]
        else:
            raise ValueError(action.action)

    def parents(self, mol: RDMol) -> list[tuple[RxnAction, str]]:
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

    def count_backward_transitions(self, mol: RDMol, max_step: int = False) -> int:
        raise NotImplementedError

    def reverse(self, g: str | RDMol | Graph | None, ra: RxnAction) -> RxnAction:
        if ra.action == RxnActionType.Stop:
            return ra
        if ra.action == RxnActionType.FirstBlock:
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
