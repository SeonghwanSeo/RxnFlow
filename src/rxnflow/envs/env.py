from functools import cached_property
import torch
from pathlib import Path
from rdkit import Chem, RDLogger

from torch import Tensor
from rdkit.Chem import Mol as RDMol

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv
from .action import RxnAction, RxnActionType
from .workflow import Workflow, Protocol
from .retrosynthesis import RetroSyntheticAnalyzer

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")


class MolGraph(Graph):
    def __init__(self, mol: str | Chem.Mol, **attr):
        super().__init__(**attr)
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

    def __init__(self, env_dir: str | Path):
        """A reaction template and building block environment instance"""
        self.env_dir = env_dir = Path(env_dir)
        workflow_config_path = env_dir / "workflow.yaml"
        block_smiles_dir = env_dir / "blocks"
        block_feature_path = env_dir / "bb_feature.pt"

        # NOTE: Load Building Blocks & Feature
        self.blocks: dict[str, list[str]] = {}
        self.block_codes: dict[str, list[str]] = {}
        for file in block_smiles_dir.iterdir():
            key = file.stem
            with file.open() as f:
                lines = f.readlines()
            self.blocks[key] = [ln.split()[0] for ln in lines]
            self.block_codes[key] = [ln.strip().split()[1] for ln in lines]

        self.block_features: dict[str, tuple[Tensor, Tensor]] = torch.load(block_feature_path, weights_only=False)
        self.block_types = sorted(list(self.blocks.keys()))
        self.num_block_types = len(self.block_types)

        # NOTE: Load Protocol
        self.workflow = Workflow(workflow_config_path)
        self.protocols: list[Protocol] = self.workflow.protocols
        self.protocol_dict: dict[str, Protocol] = {protocol.name: protocol for protocol in self.protocols}

        self.retro_analyzer: RetroSyntheticAnalyzer = RetroSyntheticAnalyzer(self.blocks, self.workflow)

    def new(self) -> MolGraph:
        return MolGraph("")

    def step(self, g: MolGraph, action: RxnAction) -> MolGraph:
        """Applies the action to the current state and returns the next state.

        Args:
            g (MolGraph): Current state as an RDKit mol.
            action tuple[int, Optional[int], Optional[int]]: Action indices to apply to the current state.
                (ActionType, reaction_template_idx, reactant_idx)

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        state_info = g.graph
        if action.action is RxnActionType.FirstBlock:
            return MolGraph(action.block, **state_info)
        elif action.action is RxnActionType.BiRxn:
            protocol = self.protocol_dict[action.protocol]
            ps = protocol.rxn_forward.forward(g.mol, Chem.MolFromSmiles(action.block), strict=True)
            assert len(ps) == 1, "reaction is Fail"
            return MolGraph(Chem.MolToSmiles(ps[0][0]), **state_info)
        else:
            # In our setup, Stop and UniRxn is invalid.
            raise ValueError(action.action)

    def parents(self, g: Graph) -> list[tuple[RxnAction, str]]:
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

    def count_backward_transitions(self, g: Graph, max_step: int) -> int:
        assert isinstance(g, MolGraph)
        retro_tree = self.retro_analyzer.run(g.smi, max_step)
        if retro_tree is None:
            return 1
        else:
            return max(len(retro_tree.branches), 1)  # Chirality is often converted.

    def reverse(self, g: str | RDMol | Graph | None, ra: RxnAction) -> RxnAction:
        if ra.action == RxnActionType.Stop:
            return ra
        if ra.action == RxnActionType.FirstBlock:
            return RxnAction(RxnActionType.BckFirstBlock, ra.protocol, ra.block, ra.block_type, ra.block_idx)
        elif ra.action == RxnActionType.BckFirstBlock:
            return RxnAction(RxnActionType.FirstBlock, ra.protocol, ra.block, ra.block_type, ra.block_idx)
        elif ra.action == RxnActionType.UniRxn:
            return RxnAction(RxnActionType.BckUniRxn, ra.protocol)
        elif ra.action == RxnActionType.BckUniRxn:
            return RxnAction(RxnActionType.UniRxn, ra.protocol)
        elif ra.action == RxnActionType.BiRxn:
            return RxnAction(RxnActionType.BckBiRxn, ra.protocol, ra.block, ra.block_type, ra.block_idx)
        elif ra.action == RxnActionType.BckBiRxn:
            return RxnAction(RxnActionType.BiRxn, ra.protocol, ra.block, ra.block_type, ra.block_idx)
        else:
            raise ValueError(ra)
