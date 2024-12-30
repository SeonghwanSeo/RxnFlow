import json
from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray
import torch
import torch_geometric.data as gd
from rdkit import Chem

from torch import Tensor
from rdkit.Chem import BondType, ChiralType, Mol as RDMol

from gflownet.envs.graph_building_env import GraphBuildingEnvContext, Graph, ActionIndex
from gflownet.utils.misc import get_worker_device
from rxnflow.envs.action import Protocol
from .building_block import NUM_BLOCK_FEATURES
from .action import RxnAction, RxnActionType
from .env import SynthesisEnv

"""
The method `create_masks` is implemented in a manner of hard-coding.
It would be not work well with additional synthesis workflows
"""

DEFAULT_ATOMS: list[str] = ["C", "N", "O", "F", "S", "Cl", "Br", "I", "At"]
DEFAULT_AROMATIC_TYPES = [True, False]
DEFAULT_CHARGE_RANGE = [-1, 0, 1]
DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]
DEFAULT_EXPL_H_RANGE = [0, 1]  # for N
DEFAULT_ISOTOPE = [0, 1, 2, 11, 12, 13, 14, 15]  # for At


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
        isotope_types: list[int] = DEFAULT_ISOTOPE,  # for At
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
            "isotope": isotope_types,
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

        # NOTE: For Molecular Reaction - Environment
        self.env: SynthesisEnv = env
        self.protocols: list[Protocol] = env.protocols
        self.protocol_dict: dict[str, Protocol] = env.protocol_dict
        self.protocol_to_idx: dict[Protocol, int] = {protocol: i for i, protocol in enumerate(self.protocols)}
        self.num_protocols = len(self.protocols)

        self.firstblock_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.FirstBlock]
        self.unirxn_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.UniRxn]
        self.birxn_list: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.BiRxn]
        self.stop: list[Protocol] = [p for p in self.protocols if p.action is RxnActionType.Stop]

        self.block_codes: dict[str, list[str]] = env.block_codes
        self.blocks: dict[str, list[str]] = env.blocks
        self.block_sets: dict[str, set[str]] = {k: set(v) for k, v in self.blocks.items()}
        self.block_types: list[str] = env.block_types
        self.num_block_types: int = len(env.block_types)
        self.num_blocks: dict[str, int] = env.num_blocks
        self.block_type_to_idx: dict[str, int] = {block_type: i for i, block_type in enumerate(self.block_types)}

        # NOTE: Setup Building Block Datas
        self.block_features: dict[str, tuple[Tensor, Tensor]] = env.block_features
        self.num_block_features = NUM_BLOCK_FEATURES + self.num_block_types

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
        block_type_idx: int = self.block_type_to_idx[block_type]
        if fp.dim() == 1:
            typ = torch.zeros((self.num_block_types,), dtype=torch.float32)
        else:
            typ = torch.zeros((fp.shape[0], self.num_block_types), dtype=torch.float32)
        typ[..., block_type_idx] = 1.0
        feature = torch.cat([feat, fp.float(), typ], dim=-1)
        return feature.to(device=device)

    def ActionIndex_to_GraphAction(self, g: gd.Data, aidx: ActionIndex, fwd: bool = True) -> RxnAction:
        protocol_idx, row_idx, col_idx = aidx
        assert row_idx == 0
        block_idx = col_idx  # readability

        protocol: Protocol = self.protocols[protocol_idx]
        t = protocol.action
        if t in (RxnActionType.FirstBlock, RxnActionType.BckFirstBlock):
            assert protocol.block_type is not None
            block = self.blocks[protocol.block_type][block_idx]
            code = self.block_codes[protocol.block_type][block_idx]
            return RxnAction(t, protocol, block, code, block_idx)

        elif t in (RxnActionType.UniRxn, RxnActionType.BckUniRxn):
            assert block_idx == 0
            return RxnAction(t, protocol)

        elif t in (RxnActionType.BiRxn, RxnActionType.BckBiRxn):
            assert protocol.block_type is not None
            block = self.blocks[protocol.block_type][col_idx]
            code = self.block_codes[protocol.block_type][block_idx]
            return RxnAction(t, protocol, block, code, col_idx)
        else:
            raise ValueError(t)

    def GraphAction_to_ActionIndex(self, g: gd.Data, action: RxnAction) -> ActionIndex:
        protocol_idx = self.protocol_to_idx[action.protocol]
        if action.action in (
            RxnActionType.FirstBlock,
            RxnActionType.BckFirstBlock,
            RxnActionType.BiRxn,
            RxnActionType.BckBiRxn,
        ):
            assert action.block_idx is not None
            block_idx = action.block_idx
        elif action.action in (RxnActionType.UniRxn, RxnActionType.BckUniRxn):
            assert action.block_idx is None
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

        data: dict[str, NDArray] = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            **self.create_masks(g),
        )
        return gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})

    def obj_to_graph(self, obj: RDMol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = Graph(smi=Chem.MolToSmiles(obj))
        mol = Chem.RemoveHs(obj)
        mol.UpdatePropertyCache()
        # Only set an attribute tag if it is not the default attribute
        for a in obj.GetAtoms():
            attrs = {
                "atomic_number": a.GetAtomicNum(),
                "chi": a.GetChiralTag(),
                "charge": a.GetFormalCharge(),
                "aromatic": a.GetIsAromatic(),
                "expl_H": a.GetNumExplicitHs(),
                "isotope": a.GetIsotope(),
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
        if "smi" in g.graph:
            return Chem.MolFromSmiles(g.graph["smi"])
        mp = Chem.RWMol()
        mp.BeginBatchEdit()
        for i in range(len(g.nodes)):
            d = g.nodes[i]
            a = Chem.Atom(d["v"])
            a.SetChiralTag(d["chi"])
            a.SetIsAromatic(d["aromatic"])
            a.SetFormalCharge(d["charge"])
            a.SetNumExplicitHs(d["expl_H"])
            a.SetIsotope(d["isotope"])
            mp.AddAtom(a)
        for e in g.edges:
            d = g.edges[e]
            mp.AddBond(e[0], e[1], d["type"])
        mp.CommitBatchEdit()
        Chem.SanitizeMol(mp)
        return Chem.MolFromSmiles(Chem.MolToSmiles(mp))

    def object_to_log_repr(self, g: Graph) -> str:
        """Convert a Graph to a string representation"""
        try:
            return g.graph["smi"]
        except Exception:
            return ""

    def traj_to_log_repr(self, traj: list[tuple[Graph | RDMol, RxnAction]]) -> str:
        """Convert a trajectory of (Graph, Action) to a trajectory of json representation"""
        traj_logs = self.read_traj(traj)
        repr_obj = []
        for i, (smiles, action_repr) in enumerate(traj_logs):
            repr_obj.append(
                OrderedDict(
                    [
                        ("step", i),
                        ("smiles", smiles),
                        ("action", action_repr),
                    ]
                )
            )
        return json.dumps(repr_obj, sort_keys=False)

    def read_traj(self, traj: list[tuple[Graph, RxnAction]]) -> list[tuple[str, tuple[str, ...]]]:
        """Convert a trajectory of (Graph, Action) to a trajectory of tuple representation"""
        traj_repr = []
        for g, action in traj:
            if action.action is RxnActionType.FirstBlock:
                assert action.block is not None
                assert action.block_code is not None
                action_repr = (action.protocol.name, action.block, action.block_code)
            elif action.action is RxnActionType.UniRxn:
                assert action.reaction is not None
                action_repr = (action.protocol.name,)
            elif action.action is RxnActionType.BiRxn:
                assert action.reaction is not None
                assert action.block is not None
                assert action.block_code is not None
                action_repr = (action.protocol.name, action.block, action.block_code)
            else:
                raise ValueError(action.action)
            obj_repr = self.object_to_log_repr(g)
            traj_repr.append((obj_repr, action_repr))
        return traj_repr

    def create_masks(self, g: Graph) -> dict[str, np.ndarray]:
        """Creates masks for reaction templates for a given molecule.

        Args:
            g (Graph): networkx Graph of the Molecule

        Returns:
            Dict[str, np.ndarry]: Masks for invalid protocols.
        """
        # NOTE: This is hard-coding.

        mask_dict: dict[str, np.ndarray] = {
            protocol.mask_name: np.zeros((1,), dtype=np.bool_) for protocol in self.protocols
        }
        mol = self.graph_to_obj(g)

        if mol.GetNumAtoms() == 0:
            # NOTE: always FirstBlock (initial state)
            for protocol in self.firstblock_list:
                mask_dict[protocol.mask_name].fill(True)
            return mask_dict

        connecting_part = [int(atom.GetIsotope()) for atom in mol.GetAtoms() if atom.GetSymbol() == "At"]
        assert len(connecting_part) in (1, 2)

        if len(connecting_part) == 1:
            label = connecting_part[0]
            if label in (1, 2):
                # NOTE: nboc group (after deboc, 1: primary amine, 2: secondary amine)
                # Always Second Reaction
                # Always BiRxn: abf_acid, ra-a1_aldehyde, ra-a2_aldehyde, ra-k1_ketond, ra-k2_ketone
                if label == 1:
                    possible_protocols = ["rxn2_abf1_acid", "rxn2_ra-a1_aldehyde", "rxn2_ra-k1_ketone"]
                else:
                    possible_protocols = ["rxn2_abf2_acid", "rxn2_ra-a2_aldehyde", "rxn2_ra-k2_ketone"]
            elif label == 11:
                # NOTE: Amine
                # Always First Reaction
                # abf_acid, ra-a2_aldehyde, ra-k2_ketone
                # TODO: use the info from Workflow
                smi = Chem.MolToSmiles(mol)
                possible_protocols = set()
                for block_type, blocks in self.blocks.items():
                    if block_type not in ("abf_amine", "ra-a2_amine", "ra-k2_amine"):
                        continue
                    if smi in blocks:
                        if block_type == "abf_amine":
                            possible_protocols.add("rxn0_abf_acid")
                        elif block_type == "ra-a2_amine":
                            possible_protocols.add("rxn0_ra-a2_aldehyde")
                        else:
                            possible_protocols.add("rxn0_ra-k2_ketone")
                assert len(possible_protocols) > 0, smi
            elif label in (12, 13, 14):
                # NOTE: Acid, Aldehyde, Ketone
                # Always First Reaction
                # abf_amine, ra-a2_amine, ra-k2_amine
                # TODO: Someone might want to consider *_nboc_* for adding blocks.
                if label == 12:
                    possible_protocols = ["rxn0_abf_amine"]
                elif label == 13:
                    possible_protocols = ["rxn0_ra-a2_amine"]
                else:
                    possible_protocols = ["rxn0_ra-k2_amine"]
            elif label == 15:
                # NOTE: snap_aldehyde
                # Always First Reaction
                # all unirxns are available (all unirxns are currently related to snap_aldehyde)
                possible_protocols = [
                    "rxn0_snap-diazepane",
                    "rxn0_snap-piperazine",
                    "rxn0_snap-morpholine",
                    "rxn0_snap-oxazepane",
                    "rxn0_snap-diazepane_end",
                    "rxn0_snap-piperazine_end",
                ]
            else:
                raise ValueError(label)
        else:
            # NOTE: Always First Reaction
            # Block with nboc (two connecting parts)
            label1, label2 = connecting_part
            label = label1 if label2 in (1, 2) else label2  # NOTE: select non-nboc label
            if label == 11:
                # NOTE: nboc_amine
                smi = Chem.MolToSmiles(mol)
                possible_protocols = set()
                for block_type, blocks in self.blocks.items():
                    if smi in blocks:
                        if block_type in ("abf_nboc1_amine", "abf_nboc2_amine"):
                            possible_protocols.add("rxn0_abf_acid")
                        elif block_type in ("ra-a2_nboc1_amine", "ra-a2_nboc2_amine"):
                            possible_protocols.add("rxn0_ra-a2_aldehyde")
                        elif block_type in ("ra-k2_nboc1_amine", "ra-k2_nboc2_amine"):
                            possible_protocols.add("rxn0_ra-k2_ketone")
                        else:
                            ValueError(block_type)
            elif label in (12, 13, 14):
                # NOTE: nboc_acid, nboc_aldehyde, nboc_ketone
                if label == 12:
                    possible_protocols = ["rxn0_abf_amine"]
                elif label == 13:
                    possible_protocols = ["rxn0_ra-a2_amine"]
                else:
                    possible_protocols = ["rxn0_ra-k2_amine"]
            elif label == 15:
                # NOTE: snap_nboc1_aldehyde, snap_nboc2_aldehyde
                possible_protocols = ["rxn0_snap-morpholine", "rxn0_snap-oxazepane"]
            else:
                raise ValueError(label)
        assert len(possible_protocols) > 0
        for name in possible_protocols:
            protocol = self.protocol_dict[name]
            mask_dict[protocol.mask_name].fill(True)
        return mask_dict

    def collate(self, graphs: list[gd.Data]) -> gd.Batch:
        return gd.Batch.from_data_list(graphs, follow_batch=["x"])


def protocol_name_to_mask(protocol_name: str, gbatch: gd.Batch):
    mask_name = protocol_name + "_mask"
    assert hasattr(gbatch, mask_name), f"Mask {mask_name} not found in graph data"
    return getattr(gbatch, mask_name)
