import torch
from pathlib import Path
from rdkit import Chem, RDLogger

from torch import Tensor
from rdkit.Chem import Mol as RDMol

from gflownet.envs.graph_building_env import Graph, GraphBuildingEnv
from .reaction import Reaction
from .action import RxnAction, RxnActionType
from .workflow import Workflow, Protocol
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
        workflow_config_path = env_dir / "workflow.yaml"
        protocol_config_path = env_dir / "protocol.yaml"
        block_smiles_dir = env_dir / "smiles"
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

        self.block_features: dict[str, tuple[Tensor, Tensor]] = torch.load(block_feature_path)
        self.block_types = sorted(list(self.blocks.keys()))
        self.num_block_types = len(self.block_types)

        # NOTE: Load Protocol
        self.workflow = Workflow(workflow_config_path, protocol_config_path)
        self.protocol_dict = self.workflow.protocol_dict

        self.protocols: list[Protocol] = self.workflow.protocols

        self.retro_analyzer: RetroSyntheticAnalyzer = RetroSyntheticAnalyzer(self.blocks, self.workflow)

    def new(self) -> Graph:
        return Graph(smi="")

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
            # NOTE: In our workflow, Stop is invalid.
            raise ValueError(action.action)
        elif action.action == RxnActionType.FirstBlock:
            assert isinstance(action.block, str)
            return Chem.MolFromSmiles(action.block)
        elif action.action is RxnActionType.UniRxn:
            assert action.reaction is not None
            ps = action.reaction.forward(mol)
            assert len(ps) == 1, "reaction is Fail"
            return ps[0][0]
        elif action.action is RxnActionType.BiRxn:
            assert isinstance(action.reaction, Reaction)
            assert isinstance(action.block, str)
            ps = action.reaction.forward(mol, Chem.MolFromSmiles(action.block))
            assert len(ps) == 1, "reaction is Fail"
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
        retro_tree = self.retro_analyzer.run(mol)
        return [(action, subtree.smi) for action, subtree in retro_tree.branches]

    def count_backward_transitions(self, mol: RDMol, check_idempotent: bool = False):
        return max(len(self.parents(mol)), 1)  # NOTE: Chirality is often converted.

    def reverse(self, g: str | RDMol | Graph | None, ra: RxnAction) -> RxnAction:
        if ra.action == RxnActionType.Stop:
            return ra
        if ra.action == RxnActionType.FirstBlock:
            return RxnAction(RxnActionType.BckFirstBlock, ra.protocol, ra.block, ra.block_code, ra.block_idx)
        elif ra.action == RxnActionType.BckFirstBlock:
            return RxnAction(RxnActionType.FirstBlock, ra.protocol, ra.block, ra.block_code, ra.block_idx)
        elif ra.action == RxnActionType.UniRxn:
            return RxnAction(RxnActionType.BckUniRxn, ra.protocol)
        elif ra.action == RxnActionType.BckUniRxn:
            return RxnAction(RxnActionType.UniRxn, ra.protocol)
        elif ra.action == RxnActionType.BiRxn:
            return RxnAction(RxnActionType.BckBiRxn, ra.protocol, ra.block, ra.block_code, ra.block_idx)
        elif ra.action == RxnActionType.BckBiRxn:
            return RxnAction(RxnActionType.BiRxn, ra.protocol, ra.block, ra.block_code, ra.block_idx)
        else:
            raise ValueError(ra)
