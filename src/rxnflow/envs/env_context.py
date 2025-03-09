import json
from collections import OrderedDict

import numpy as np
import torch
import torch_geometric.data as gd
from rdkit.Chem import BondType, ChiralType
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.envs.graph_building_env import ActionIndex, GraphBuildingEnvContext
from rxnflow.envs.building_block import MOL_PROPERTY_DIM, get_mol_features

from .action import Protocol, RxnAction, RxnActionType
from .env import MolGraph, SynthesisEnv

DEFAULT_ATOMS: list[str] = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
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
        self.protocol_dict: dict[str, Protocol] = env.protocol_dict
        self.protocol_to_idx: dict[str, int] = {protocol.name: i for i, protocol in enumerate(self.protocols)}
        self.num_protocols = len(self.protocols)

        # NOTE: Protocols
        self.stop_list: list[Protocol] = env.stop_list
        self.firstblock_list: list[Protocol] = env.firstblock_list
        self.unirxn_list: list[Protocol] = env.unirxn_list
        self.birxn_list: list[Protocol] = env.birxn_list
        self.protocol_type_dict: dict[RxnActionType, list[Protocol]] = {
            RxnActionType.Stop: self.stop_list,
            RxnActionType.FirstBlock: self.firstblock_list,
            RxnActionType.UniRxn: self.unirxn_list,
            RxnActionType.BiRxn: self.birxn_list,
        }

        # NOTE: Building Block
        self.blocks: list[str] = env.blocks
        self.num_blocks: int = len(self.blocks)
        self.block_fp: Tensor = torch.from_numpy(env.block_fp)
        self.block_prop: Tensor = torch.from_numpy(env.block_prop)
        self.block_fp_dim = self.block_fp.shape[-1]
        self.block_prop_dim = self.block_prop.shape[-1]

        # NOTE: For PB
        self.birxn_block_indices: dict[str, np.ndarray] = env.birxn_block_indices
        self.num_total_actions = (
            1 + len(self.unirxn_list) + sum(indices.shape[0] for indices in self.birxn_block_indices.values())
        )

        # NOTE: For Molecular Graph
        self.atom_attr_values = {
            "v": DEFAULT_ATOMS,
            "chi": DEFAULT_ATOM_CHIRAL_TYPES,
            "charge": DEFAULT_ATOM_CHARGE_RANGE,
            "expl_H": DEFAULT_ATOM_EXPL_H_RANGE,
            "aromatic": [True, False],
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
        self.num_cond_dim = num_cond_dim
        self.num_graph_dim = MOL_PROPERTY_DIM

        # NOTE: Action Type Order
        self.action_type_order: list[RxnActionType] = [
            RxnActionType.Stop,
            RxnActionType.UniRxn,
            RxnActionType.BiRxn,
            RxnActionType.FirstBlock,
        ]

        self.bck_action_type_order: list[RxnActionType] = [
            RxnActionType.BckStop,
            RxnActionType.BckUniRxn,
            RxnActionType.BckBiRxn,
            RxnActionType.BckFirstBlock,
        ]

    def get_block_data(
        self,
        block_indices: Tensor | int,
        device: str | torch.device | None = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Get the block features for the given type and indices

        Parameters
        ----------
        block_indices : Tensor | int
            Block indices for the given block type
        device: torch.device | None
            torch device

        Returns
        -------
        fp: Tensor
            molecular fingerprints of blocks
        prop: Tensor
            molecular properties of blocks
        """
        prop = self.block_prop[block_indices]
        fp = self.block_fp[block_indices]
        if fp.dim() == 1:
            prop, fp = prop.unsqueeze(0), fp.unsqueeze(0)
        fp = fp.to(dtype=torch.float32, device=device, non_blocking=True)
        prop = prop.to(dtype=torch.float32, device=device, non_blocking=True)
        return fp, prop

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
            graph_attr = torch.from_numpy(get_mol_features(self.graph_to_obj(g)))

        return dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_attr=graph_attr.reshape(1, -1),
            protocol_mask=self.create_masks(g).reshape(1, -1),
            sample_idx=g.graph["sample_idx"],
        )

    def create_masks(self, g: MolGraph) -> Tensor:
        """Creates masks for protocol for a given objecule.

        Args:
            obj (Chem.Mol)
                Molecule as a rdKit Mol object.

        Returns:
            mask (torch.Tensor):
                Masks for invalid actions.
        """
        obj = g.mol
        masks = torch.zeros((self.num_protocols,), dtype=torch.bool)
        possible_protocols: list[Protocol] = []
        if obj.GetNumAtoms() == 0:
            possible_protocols += self.firstblock_list
        else:
            if g.graph["allow_stop"]:
                possible_protocols += self.stop_list
            for protocol in self.unirxn_list:
                if protocol.rxn.is_reactant(obj):
                    possible_protocols.append(protocol)
            for protocol in self.birxn_list:
                if protocol.rxn.is_reactant(obj, 0):
                    possible_protocols.append(protocol)
        for protocol in possible_protocols:
            masks[self.protocol_to_idx[protocol.name]] = True
        return masks

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

    def ActionIndex_to_GraphAction(self, g: gd.Data, aidx: ActionIndex, fwd: bool = True) -> RxnAction:
        protocol_idx, _, block_idx = aidx
        protocol: Protocol = self.protocols[protocol_idx]
        t = protocol.action
        if t in (RxnActionType.Stop, RxnActionType.BckStop):
            return RxnAction(t, protocol.name)
        elif t in (RxnActionType.FirstBlock, RxnActionType.BckFirstBlock):
            block = self.blocks[block_idx]
            return RxnAction(t, protocol.name, block, block_idx)
        elif t in (RxnActionType.UniRxn, RxnActionType.BckUniRxn):
            return RxnAction(t, protocol.name)
        elif t in (RxnActionType.BiRxn, RxnActionType.BckBiRxn):
            block = self.blocks[block_idx]
            return RxnAction(t, protocol.name, block, block_idx)
        else:
            raise ValueError(t)

    def GraphAction_to_ActionIndex(self, g: gd.Data, action: RxnAction) -> ActionIndex:
        protocol_idx = self.protocol_to_idx[action.protocol]
        if action.action in (RxnActionType.Stop, RxnActionType.BckStop):
            block_idx = 0
        elif action.action in (RxnActionType.FirstBlock, RxnActionType.BckFirstBlock):
            block_idx = action.block_idx
        elif action.action in (RxnActionType.BiRxn, RxnActionType.BckBiRxn):
            block_idx = action.block_idx
        elif action.action in (RxnActionType.UniRxn, RxnActionType.BckUniRxn):
            block_idx = 0
        else:
            raise ValueError(action)
        return ActionIndex(protocol_idx, 0, block_idx)

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
            if action.action is RxnActionType.Stop:
                action_repr = ("Stop",)
            elif action.action is RxnActionType.FirstBlock:
                action_repr = ("FirstBlock", action.block)
            elif action.action is RxnActionType.UniRxn:
                rxn_template = self.protocol_dict[action.protocol].rxn.template
                action_repr = ("UniRxn", rxn_template)
            elif action.action is RxnActionType.BiRxn:
                rxn_template = self.protocol_dict[action.protocol].rxn.template
                action_repr = ("BiRxn", rxn_template, action.block)
            else:
                raise ValueError(action.action)
            traj_repr.append((obj_repr, action_repr))
        return traj_repr
