import json
from collections import OrderedDict

import numpy as np
import torch
import torch_geometric.data as gd

from torch import Tensor
from rdkit.Chem import BondType, ChiralType, Mol as RDMol
from rdkit.Chem import Crippen, rdMolDescriptors

from gflownet.envs.graph_building_env import GraphBuildingEnvContext, ActionIndex
from rxnflow.envs.action import Protocol
from .building_block import BLOCK_FP_DIM, BLOCK_PROPERTY_DIM
from .action import RxnAction, RxnActionType
from .env import MolGraph, SynthesisEnv


DEFAULT_ATOMS = ["*", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
DEFAULT_ATOM_CHARGE_RANGE = [-1, 0, 1]
DEFAULT_ATOM_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]
DEFAULT_ATOM_EXPL_H_RANGE = [0, 1]  # for N
DEFAULT_BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


class SynthesisEnvContext(GraphBuildingEnvContext):
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(self, env: SynthesisEnv, num_cond_dim: int = 0):
        """An env context for generating molecules by sequentially applying reaction templates.
        Contains functionalities to build molecular graphs, create masks for actions, and convert molecules to other representations.

        Args:
            num_cond_dim (int): The dimensionality of the observations' conditional information vector (if >0)
        """

        # NOTE: For Molecular Reaction - Environment
        self.env: SynthesisEnv = env
        self.protocols: list[Protocol] = env.protocols
        self.protocol_to_idx: dict[str, int] = {protocol.name: i for i, protocol in enumerate(self.protocols)}
        self.num_protocols = len(self.protocols)

        # NOTE: Protocols
        self.firstblock_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.FirstBlock]
        self.birxn_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.BiRxn]

        # NOTE: Building Blocks
        self.blocks: dict[str, list[str]] = env.blocks
        self.block_codes: dict[str, list[str]] = env.block_codes
        self.block_types: list[str] = env.block_types
        self.num_block_types: int = len(env.block_types)
        self.block_type_to_idx: dict[str, int] = {block_type: i for i, block_type in enumerate(self.block_types)}

        # NOTE: Setup Building Block Datas
        self.block_features: dict[str, tuple[Tensor, Tensor]] = env.block_features
        self.block_fp_dim: int = BLOCK_FP_DIM
        self.block_desc_dim: int = BLOCK_PROPERTY_DIM

        # NOTE: Setup State Molecular Type
        self.state_types = [protocol.state_type for protocol in self.birxn_list]

        # NOTE: For Molecular Graph
        self.atom_attr_values = {
            "v": DEFAULT_ATOMS,
            "chi": DEFAULT_ATOM_CHIRAL_TYPES,
            "charge": DEFAULT_ATOM_CHARGE_RANGE,
            "expl_H": DEFAULT_ATOM_EXPL_H_RANGE,
            "aromatic": [True, False],
            "isotope": self.state_types,
        }
        self.atom_attrs = sorted(self.atom_attr_values.keys())
        self.atom_attr_slice = [0] + list(np.cumsum([len(self.atom_attr_values[i]) for i in self.atom_attrs]))
        self.bond_attr_values = {
            "type": DEFAULT_BOND_TYPES,
        }
        self.bond_attrs = sorted(self.bond_attr_values.keys())
        self.bond_attr_slice = [0] + list(np.cumsum([len(self.bond_attr_values[i]) for i in self.bond_attrs]))
        self.num_node_dim = sum(len(v) for v in self.atom_attr_values.values())
        self.num_edge_dim = sum(len(v) for v in self.bond_attr_values.values())
        self.num_graph_dim = 8  # mw, tpsa, numhbd, numhba, mollogp, rotbonds, numrings

        # NOTE: For Condition
        self.num_cond_dim = num_cond_dim

        # NOTE: Action Type Order
        self.action_type_order: list[RxnActionType] = [
            RxnActionType.FirstBlock,
            RxnActionType.UniRxn,
            RxnActionType.BiRxn,
        ]

        self.bck_action_type_order: list[RxnActionType] = [
            RxnActionType.BckFirstBlock,
            RxnActionType.BckUniRxn,
            RxnActionType.BckBiRxn,
        ]

    def get_block_data(self, block_type: str, block_indices: torch.Tensor | int) -> tuple[Tensor, Tensor, Tensor]:
        """Get the block features for the given type and indices

        Parameters
        ----------
        block_type : str
            Block type
        block_indices : torch.Tensor | int
            Block indices for the given block type

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            typ: index tensor for given block type
            descs: molecular feature of blocks
            fp: molecular fingerprints of blocks
        """
        desc, fp = self.block_features[block_type]
        desc, fp = desc[block_indices], fp[block_indices]
        if fp.dim() == 1:
            desc, fp = desc.view(1, -1), fp.view(1, -1)
        type_idx = self.block_type_to_idx[block_type]  # NOTE: it is different to block_type_idx in aidx
        typ = torch.full((fp.shape[0],), type_idx, dtype=torch.long)
        return typ, desc, fp.to(torch.float32)

    def ActionIndex_to_GraphAction(self, g: gd.Data, aidx: ActionIndex, fwd: bool = True) -> RxnAction:
        protocol_idx, block_type_idx, block_idx = aidx
        protocol: Protocol = self.protocols[protocol_idx]
        t = protocol.action
        if t in (RxnActionType.FirstBlock, RxnActionType.BckFirstBlock):
            assert block_type_idx == 0
            block_type = protocol.block_types[0]
            block = self.blocks[block_type][block_idx]
            return RxnAction(t, protocol.name, block, block_type, block_idx)
        elif t in (RxnActionType.BiRxn, RxnActionType.BckBiRxn):
            block_type = protocol.block_types[block_type_idx]
            block = self.blocks[block_type][block_idx]
            return RxnAction(t, protocol.name, block, block_type, block_idx)
        else:
            raise ValueError(t)

    def GraphAction_to_ActionIndex(self, g: gd.Data, action: RxnAction) -> ActionIndex:
        protocol_idx = self.protocol_to_idx[action.protocol]
        protocol: Protocol = self.protocols[protocol_idx]
        if action.action in (RxnActionType.FirstBlock, RxnActionType.BckFirstBlock):
            block_type_idx = 0
            block_idx = action.block_idx
        elif action.action in (RxnActionType.BiRxn, RxnActionType.BckBiRxn):
            block_type_idx = protocol.block_types.index(action.block_type)
            block_idx = action.block_idx
        else:
            raise ValueError(action)
        return ActionIndex(protocol_idx, block_type_idx, block_idx)

    def graph_to_Data(self, g: MolGraph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        return gd.Data(**self._graph_to_data_dict(g))

    def _graph_to_data_dict(self, g: MolGraph) -> dict[str, Tensor]:
        """Convert a networkx Graph to a torch tensors"""
        assert isinstance(g, MolGraph)
        self.setup_graph(g)
        if len(g.nodes) == 0:
            x = torch.zeros((1, self.num_node_dim))
            x[0, -1] = 1
            edge_attr = torch.zeros((0, self.num_edge_dim))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph_attr = torch.zeros((self.num_graph_dim,))
        else:
            x = torch.zeros((len(g.nodes), self.num_node_dim))
            for i, n in enumerate(g.nodes):
                ad = g.nodes[n]
                for k, sl in zip(self.atom_attrs, self.atom_attr_slice, strict=False):
                    idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                    x[i, sl + idx] = 1  # One-hot encode the attribute value

            edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
            for i, e in enumerate(g.edges):
                ad = g.edges[e]
                for k, sl in zip(self.bond_attrs, self.bond_attr_slice, strict=False):
                    if ad[k] in self.bond_attr_values[k]:
                        idx = self.bond_attr_values[k].index(ad[k])
                    else:
                        idx = 0
                    edge_attr[i * 2, sl + idx] = 1
                    edge_attr[i * 2 + 1, sl + idx] = 1
            edge_index = torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).view(-1, 2).T
            # Add molecular properties (multi-modality)
            graph_attr = self.get_obj_features(self.graph_to_obj(g))

        return dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_attr=graph_attr.reshape(1, -1),
            protocol_mask=self.create_masks(g).reshape(1, -1),
        )

    def get_obj_features(self, obj: RDMol) -> Tensor:
        descs = [
            rdMolDescriptors.CalcExactMolWt(obj) / 500,
            rdMolDescriptors.CalcTPSA(obj) / 100,
            rdMolDescriptors.CalcNumHBA(obj) / 10,
            rdMolDescriptors.CalcNumHBD(obj) / 10,
            rdMolDescriptors.CalcNumRotatableBonds(obj) / 10,
            rdMolDescriptors.CalcNumAliphaticRings(obj) / 10,
            rdMolDescriptors.CalcNumAromaticRings(obj) / 10,
            Crippen.MolLogP(obj) / 10,
        ]
        return torch.tensor(descs, dtype=torch.float32)

    def create_masks(self, g: MolGraph) -> Tensor:
        """Creates masks for reaction templates for a given molecule.

        Args:
            g (Graph): networkx Graph of the Molecule

        Returns:
            np.ndarry: Masks for invalid protocols.
        """
        mol = self.graph_to_obj(g)
        is_last_step = g.graph["is_last_step"]

        mask_dict: dict[str, bool] = {protocol.name: False for protocol in self.protocols}
        if mol.GetNumAtoms() == 0:
            # NOTE: always FirstBlock (initial state)
            for protocol in self.firstblock_list:
                mask_dict[protocol.name] = True
        else:
            # NOTE: always BiRxn (later state)
            connecting_part = [int(atom.GetIsotope()) for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
            # TODO: it should be modified if we allowed linker as starting block
            assert len(connecting_part) == 1, "fragmentation part should be 1"
            pattern = connecting_part[0]
            for protocol in self.birxn_list:
                if is_last_step and "linker" in protocol.name:
                    continue
                if pattern == protocol.state_type:
                    mask_dict[protocol.name] = True
        mask = torch.tensor([mask_dict[protocol.name] for protocol in self.protocols], dtype=torch.bool)
        return mask

    def collate(self, graphs: list[gd.Data]) -> gd.Batch:
        return gd.Batch.from_data_list(graphs, follow_batch=["x"])

    def setup_graph(self, g: MolGraph):
        if not g.is_setup:
            obj = g.mol
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
            g.is_setup = True

    def obj_to_graph(self, obj: RDMol) -> MolGraph:
        """Convert an RDMol to a Graph"""
        g = MolGraph(obj)
        self.setup_graph(g)
        return g

    def graph_to_obj(self, g: MolGraph) -> RDMol:
        """Convert a Graph to an RDKit Mol"""
        return g.mol

    def object_to_log_repr(self, g: MolGraph) -> str:
        """Convert a Graph to a string representation"""
        return g.smi

    def traj_to_log_repr(self, traj: list[tuple[MolGraph | RDMol, RxnAction]]) -> str:
        """Convert a trajectory of (Graph, Action) to a trajectory of json representation"""
        traj_logs = self.read_traj(traj)
        repr_obj = []
        for i, (smiles, action_repr) in enumerate(traj_logs):
            repr_obj.append(OrderedDict([("step", i), ("smiles", smiles), ("action", action_repr)]))
        return json.dumps(repr_obj, sort_keys=False)

    def read_traj(self, traj: list[tuple[MolGraph, RxnAction]]) -> list[tuple[str, tuple[str, ...]]]:
        """Convert a trajectory of (Graph, Action) to a trajectory of tuple representation"""
        traj_repr = []
        for g, action in traj:
            obj_repr = self.object_to_log_repr(g)
            if action.action is RxnActionType.FirstBlock:
                action_repr = (action.protocol, action.block_type, action.block)
            elif action.action is RxnActionType.BiRxn:
                action_repr = (action.protocol, action.block_type, action.block)
            else:
                raise ValueError(action.action)
            traj_repr.append((obj_repr, action_repr))
        return traj_repr
