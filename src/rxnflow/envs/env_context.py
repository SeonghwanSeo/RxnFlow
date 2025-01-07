import json
from collections import OrderedDict
import numpy as np
import torch
import torch_geometric.data as gd
from rdkit.Chem import Crippen, rdMolDescriptors

from torch import Tensor
from rdkit.Chem import BondType, ChiralType, Mol as RDMol

from gflownet.envs.graph_building_env import GraphBuildingEnvContext, ActionIndex, Graph
from gflownet.utils.misc import get_worker_device
from rxnflow.envs.action import Protocol
from .building_block import NUM_BLOCK_FEATURES
from .action import RxnAction, RxnActionType
from .env import SynthesisEnv, MolGraph

"""
The method `create_masks` is implemented in a manner of hard-coding.
It would be not work well with additional synthesis workflows
"""

DEFAULT_ATOMS = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
DEFAULT_AROMATIC_TYPES = [True, False]
DEFAULT_CHARGE_RANGE = [-3, -2, -1, 0, 1, 2, 3]
DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]
DEFAULT_EXPL_H_RANGE = [0, 1, 2, 3, 4]  # for N


class SynthesisEnvContext(GraphBuildingEnvContext):
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(
        self,
        env: SynthesisEnv,
        num_cond_dim: int = 0,
        *args,
        atoms: list[str] = DEFAULT_ATOMS,
        chiral_types: list = DEFAULT_CHIRAL_TYPES,
        charges: list[int] = DEFAULT_CHARGE_RANGE,  # for aromatic
        aromatic_type: list[bool] = DEFAULT_AROMATIC_TYPES,  # for aromatic
        expl_H_range: list[int] = DEFAULT_EXPL_H_RANGE,  # for N
    ):
        """An env context for generating molecules by sequentially applying reaction templates.
        Contains functionalities to build molecular graphs, create masks for actions, and convert molecules to other representations.

        Args:
            atoms (list): list of atom symbols.
            chiral_types (list): list of chiral types.
            charges (list): list of charges.
            expl_H_range (list): list of explicit H counts.
            allow_5_valence_nitrogen (bool): Whether to allow N with valence of 5.
            num_cond_dim (int): The dimensionality of the observations' conditional information vector (if >0)
            reaction_templates (list): list of SMIRKS.
            blocks (list): list of SMILES strings of building blocks.
        """
        # NOTE: For Molecular Graph
        self.atom_attr_values = {
            "v": atoms,
            "chi": chiral_types,
            "charge": charges,
            "aromatic": aromatic_type,
            "expl_H": expl_H_range,
        }
        self.atom_attrs = sorted(self.atom_attr_values.keys())
        self.atom_attr_slice = [0] + list(np.cumsum([len(self.atom_attr_values[i]) for i in self.atom_attrs]))
        self.bond_attr_values = {
            "type": [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC],
        }
        self.bond_attrs = sorted(self.bond_attr_values.keys())
        self.bond_attr_slice = [0] + list(np.cumsum([len(self.bond_attr_values[i]) for i in self.bond_attrs]))
        self.num_node_dim = sum(len(v) for v in self.atom_attr_values.values())
        self.num_edge_dim = sum(len(v) for v in self.bond_attr_values.values())
        self.num_graph_dim = 8  # mw, tpsa, numhbd, numhba, mollogp, rotbonds, numrings
        self.num_cond_dim = num_cond_dim

        # NOTE: Action Type Order
        self.action_type_order: list[RxnActionType] = [
            RxnActionType.FirstBlock,
            RxnActionType.UniRxn,
            RxnActionType.BiRxn,
            RxnActionType.Stop,
        ]

        self.bck_action_type_order: list[RxnActionType] = [
            RxnActionType.BckFirstBlock,
            RxnActionType.BckUniRxn,
            RxnActionType.BckBiRxn,
            RxnActionType.Stop,
        ]

        # NOTE: For Molecular Reaction - Environment
        self.env: SynthesisEnv = env
        self.protocols: list[Protocol] = env.protocols
        self.protocol_dict: dict[str, Protocol] = env.protocol_dict
        self.protocol_to_idx: dict[Protocol, int] = {protocol: i for i, protocol in enumerate(self.protocols)}
        self.num_protocols = len(self.protocols)

        self.firstblock_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.FirstBlock]
        self.unirxn_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.UniRxn]
        self.birxn_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.BiRxn]
        self.stop_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.Stop]

        self.block_codes: dict[str, list[str]] = env.block_codes
        self.blocks: dict[str, list[str]] = env.blocks
        self.block_sets: dict[str, set[str]] = {k: set(v) for k, v in self.blocks.items()}
        self.block_types: list[str] = env.block_types
        self.num_block_types: int = len(env.block_types)
        self.num_blocks: dict[str, int] = env.num_blocks
        self.block_type_to_idx: dict[str, int] = {block_type: i for i, block_type in enumerate(self.block_types)}

        # NOTE: Setup Building Block Datas
        self.block_features: dict[str, tuple[Tensor, Tensor]] = env.block_features
        self.num_block_features = NUM_BLOCK_FEATURES

    def get_block_data(
        self,
        block_type: str,
        block_indices: np.ndarray | list[int] | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        if device is None:
            device = get_worker_device()
        fp, feat = self.block_features[block_type]
        if block_indices is not None:
            assert len(block_indices) <= fp.shape[0]
            if len(block_indices) < fp.shape[0]:
                fp, feat = fp[block_indices], feat[block_indices]
        feature = torch.cat([feat, fp.float()], dim=-1)
        return feature.to(device=device)

    def ActionIndex_to_GraphAction(self, g: gd.Data, aidx: ActionIndex, fwd: bool = True) -> RxnAction:
        protocol_idx, _, block_idx = aidx
        protocol: Protocol = self.protocols[protocol_idx]
        t = protocol.action
        if t in (RxnActionType.FirstBlock, RxnActionType.BckFirstBlock):
            block = self.blocks[protocol.block_type][block_idx]
            return RxnAction(t, protocol, block, block_idx)
        elif t in (RxnActionType.BiRxn, RxnActionType.BckBiRxn):
            block = self.blocks[protocol.block_type][block_idx]
            return RxnAction(t, protocol, block, block_idx)
        elif t in (RxnActionType.UniRxn, RxnActionType.BckUniRxn):
            return RxnAction(t, protocol)
        elif t is RxnActionType.Stop:
            return RxnAction(t, protocol)
        else:
            raise ValueError(t)

    def GraphAction_to_ActionIndex(self, g: gd.Data, action: RxnAction) -> ActionIndex:
        protocol_idx = self.protocol_to_idx[action.protocol]
        if action.action in (RxnActionType.FirstBlock, RxnActionType.BckFirstBlock):
            block_idx = action.block_idx
        elif action.action in (RxnActionType.BiRxn, RxnActionType.BckBiRxn):
            block_idx = action.block_idx
        elif action.action in (RxnActionType.UniRxn, RxnActionType.BckUniRxn):
            block_idx = 0
        elif action.action is RxnActionType.Stop:
            block_idx = 0
        else:
            raise ValueError(action)
        return ActionIndex(protocol_idx, 0, block_idx)

    def graph_to_Data(self, g: Graph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        x = np.zeros((max(1, len(g.nodes)), self.num_node_dim), dtype=np.float32)
        if len(g.nodes) == 0:
            x = np.zeros((1, self.num_node_dim), dtype=np.float32)
            x[0, -1] = 1
            edge_attr = np.zeros((0, self.num_edge_dim), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=np.int64)
            graph_attr = np.zeros(
                (1, self.num_graph_dim),
                dtype=np.float32,
            )
        else:
            for i, n in enumerate(g.nodes):
                ad = g.nodes[n]
                for k, sl in zip(self.atom_attrs, self.atom_attr_slice, strict=False):
                    idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                    x[i, sl + idx] = 1  # One-hot encode the attribute value

            edge_attr = np.zeros((len(g.edges) * 2, self.num_edge_dim), dtype=np.float32)
            for i, e in enumerate(g.edges):
                ad = g.edges[e]
                for k, sl in zip(self.bond_attrs, self.bond_attr_slice, strict=False):
                    if ad[k] in self.bond_attr_values[k]:
                        idx = self.bond_attr_values[k].index(ad[k])
                    else:
                        idx = 0
                    edge_attr[i * 2, sl + idx] = 1
                    edge_attr[i * 2 + 1, sl + idx] = 1
            edge_index = np.array([e for i, j in g.edges for e in [(i, j), (j, i)]]).reshape((-1, 2)).T.astype(np.int64)
            graph_attr = self.get_mol_features(self.graph_to_obj(g)).reshape(1, self.num_graph_dim)

        # NOTE: Add molecular properties (multi-modality)
        data: dict[str, np.ndarray] = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_attr=graph_attr,
            protocol_mask=self.create_masks(g).reshape(1, -1),
        )
        return gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})

    def get_mol_features(self, obj: RDMol) -> np.ndarray:
        props = [
            rdMolDescriptors.CalcExactMolWt(obj) / 500,
            rdMolDescriptors.CalcTPSA(obj) / 100,
            rdMolDescriptors.CalcNumHBA(obj) / 10,
            rdMolDescriptors.CalcNumHBD(obj) / 10,
            rdMolDescriptors.CalcNumRotatableBonds(obj) / 10,
            rdMolDescriptors.CalcNumAliphaticRings(obj) / 10,
            rdMolDescriptors.CalcNumAromaticRings(obj) / 10,
            Crippen.MolLogP(obj) / 10,
        ]
        return np.array(props, np.float32)

    def create_masks(self, g: Graph) -> np.ndarray:
        """Creates masks for reaction templates for a given molecule.

        Args:
            g (Graph): networkx Graph of the Molecule

        Returns:
            np.ndarry: Masks for invalid protocols.
        """
        mask_dict: dict[str, bool] = {protocol.name: False for protocol in self.protocols}
        mol = self.graph_to_obj(g)

        if mol.GetNumAtoms() == 0:
            # NOTE: always FirstBlock (initial state)
            for protocol in self.firstblock_list:
                mask_dict[protocol.name] = True
        else:
            if g.graph["allow_stop"]:
                for protocol in self.stop_list:
                    mask_dict[protocol.name] = True
            for protocol in self.unirxn_list + self.birxn_list:
                if protocol.rxn_forward.is_reactant(mol, 0):
                    mask_dict[protocol.name] = True
        mask = np.array([mask_dict[protocol.name] for protocol in self.protocols], dtype=np.bool_)
        return mask

    def collate(self, graphs: list[gd.Data]) -> gd.Batch:
        return gd.Batch.from_data_list(graphs, follow_batch=["x"])

    def obj_to_graph(self, obj: RDMol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = MolGraph(obj)
        for a in obj.GetAtoms():
            attrs = {
                "atomic_number": a.GetAtomicNum(),
                "chi": a.GetChiralTag(),
                "charge": a.GetFormalCharge(),
                "aromatic": a.GetIsAromatic(),
                "expl_H": a.GetNumExplicitHs(),
            }
            g.add_node(
                a.GetIdx(),
                v=a.GetSymbol(),
                **{attr: val for attr, val in attrs.items()},
            )
        for b in obj.GetBonds():
            attrs = {"type": b.GetBondType()}
            g.add_edge(
                b.GetBeginAtomIdx(),
                b.GetEndAtomIdx(),
                **{attr: val for attr, val in attrs.items()},
            )
        return g

    def graph_to_obj(self, g: Graph) -> RDMol:
        """Convert a Graph to an RDKit Mol"""
        assert isinstance(g, MolGraph)
        return g.mol

    def object_to_log_repr(self, g: Graph) -> str:
        """Convert a Graph to a string representation"""
        assert isinstance(g, MolGraph)
        return g.smi

    def traj_to_log_repr(self, traj: list[tuple[Graph | RDMol, RxnAction]]) -> str:
        """Convert a trajectory of (Graph, Action) to a trajectory of json representation"""
        traj_logs = self.read_traj(traj)
        repr_obj = []
        for i, (smiles, action_repr) in enumerate(traj_logs):
            repr_obj.append(OrderedDict([("step", i), ("smiles", smiles), ("action", action_repr)]))
        return json.dumps(repr_obj, sort_keys=False)

    def read_traj(self, traj: list[tuple[Graph, RxnAction]]) -> list[tuple[str, tuple[str, ...]]]:
        """Convert a trajectory of (Graph, Action) to a trajectory of tuple representation"""
        traj_repr = []
        for g, action in traj:
            if action.action is RxnActionType.FirstBlock:
                action_repr = (action.protocol.name, action.block)
            elif action.action is RxnActionType.UniRxn:
                action_repr = (action.protocol.name,)
            elif action.action is RxnActionType.BiRxn:
                action_repr = (action.protocol.name, action.block)
            elif action.action is RxnActionType.Stop:
                action_repr = (action.protocol.name,)
            else:
                raise ValueError(action.action)
            obj_repr = self.object_to_log_repr(g)
            traj_repr.append((obj_repr, action_repr))
        return traj_repr
