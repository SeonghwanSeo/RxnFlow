from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric.data as gd
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType, ChiralType
from tqdm import tqdm

from gflownet.envs.synthesis.utils import Reaction
from gflownet.envs.synthesis.env import SynthesisEnv, Graph
from gflownet.envs.synthesis.action import (
    ReactionAction,
    ReactionActionType,
    ReactionActionIdx,
    ForwardAction,
    BackwardAction,
)

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")

DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]

ATOMS: List[str] = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Sn",
    "Ca",
    "Na",
    "Ba",
    "Zn",
    "Rh",
    "Ag",
    "Li",
    "Yb",
    "K",
    "Fe",
    "Cs",
    "Bi",
    "Pd",
    "Cu",
    "Si",
]


class SynthesisEnvContext:
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(
        self,
        env: SynthesisEnv,
        num_cond_dim: int = 0,
        num_block_sampling: int = 6000,
        *args,
        atoms: List[str] = ATOMS,
        chiral_types: List = DEFAULT_CHIRAL_TYPES,
        charges: List[int] = [-3, -2, -1, 0, 1, 2, 3],
        expl_H_range: List[int] = [0, 1, 2, 3, 4],  # for N
        allow_explicitly_aromatic: bool = False,
        allow_5_valence_nitrogen: bool = False,
    ):
        """An env context for generating molecules by sequentially applying reaction templates.
        Contains functionalities to build molecular graphs, create masks for actions, and convert molecules to other representations.

        Args:
            atoms (list): List of atom symbols.
            chiral_types (list): List of chiral types.
            charges (list): List of charges.
            expl_H_range (list): List of explicit H counts.
            allow_explicitly_aromatic (bool): Whether to allow explicitly aromatic molecules.
            allow_5_valence_nitrogen (bool): Whether to allow N with valence of 5.
            num_cond_dim (int): The dimensionality of the observations' conditional information vector (if >0)
            reaction_templates (list): List of SMIRKS.
            building_blocks (list): List of SMILES strings of building blocks.
            precomputed_bb_masks (np.ndarray): Precomputed masks (for bimoelcular reactions) for building blocks and reaction templates.
        """
        # NOTE: For Molecular Graph
        self.atom_attr_values = {
            "v": atoms + ["*"],
            "chi": chiral_types,
            "charge": charges,
            "expl_H": expl_H_range,
            "fill_wildcard": [None] + atoms,  # default is, there is nothing
        }
        self.atom_attrs = sorted(self.atom_attr_values.keys())
        self.atom_attr_slice = [0] + list(np.cumsum([len(self.atom_attr_values[i]) for i in self.atom_attrs]))
        self.allow_explicitly_aromatic = allow_explicitly_aromatic
        aromatic_optional = [BondType.AROMATIC] if allow_explicitly_aromatic else []
        self.bond_attr_values = {
            "type": [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE] + aromatic_optional,
        }
        self.bond_attrs = sorted(self.bond_attr_values.keys())
        self.bond_attr_slice = [0] + list(np.cumsum([len(self.bond_attr_values[i]) for i in self.bond_attrs]))
        self.default_wildcard_replacement = "C"
        self.negative_attrs = ["fill_wildcard"]
        pt = Chem.GetPeriodicTable()

        # We'll handle nitrogen valence later explicitly in graph_to_Data;
        # wildcard atoms have 0 valence until filled in
        self._max_atom_valence = {
            **{a: max(pt.GetValenceList(a)) for a in atoms},
            "N": 3 if not allow_5_valence_nitrogen else 5,
            "*": 0,
        }
        self.num_node_dim = sum(len(v) for v in self.atom_attr_values.values())
        self.num_edge_dim = sum(len(v) for v in self.bond_attr_values.values())
        self.num_cond_dim = num_cond_dim

        # NOTE: For Molecular Reaction - Environment
        self.env: SynthesisEnv = env
        self.reactions: List[Reaction] = env.reactions
        self.unimolecular_reactions = env.unimolecular_reactions
        self.bimolecular_reactions = env.bimolecular_reactions
        self.num_unimolecular_rxns = len(self.unimolecular_reactions)
        self.num_bimolecular_rxns = len(self.bimolecular_reactions)

        self.building_blocks: List[str] = env.building_blocks
        self.building_block_mols: List[Chem.Mol] = env.building_block_mols
        self.num_building_blocks: int = len(self.building_blocks)
        self.building_block_datas: List[gd.Data] = [
            self.graph_to_Data_block(self.mol_to_graph(bb)) for bb in tqdm(self.building_block_mols)
        ]
        self.num_block_sampling: int = min(num_block_sampling, self.num_building_blocks)
        self.precomputed_bb_masks = env.precomputed_bb_masks

        self.primary_action_type_order = [
            ReactionActionType.Stop,
            ReactionActionType.ReactUni,
            ReactionActionType.ReactBi,
        ]
        self.secondary_action_type_order = [
            ReactionActionType.AddReactant,
            ReactionActionType.AddFirstReactant,
        ]
        self.action_type_order = self.primary_action_type_order + self.secondary_action_type_order

        self.primary_bck_action_type_order = [
            ReactionActionType.BckReactUni,
            ReactionActionType.BckReactBi,
            ReactionActionType.BckRemoveFirstReactant,
        ]
        self.secondary_bck_action_type_order = []
        self.bck_action_type_order = self.primary_bck_action_type_order + self.secondary_bck_action_type_order

    def sample_blocks(self) -> Tuple[List[int], gd.Batch]:
        if self.num_block_sampling == self.num_building_blocks:
            block_indices = list(range(self.num_building_blocks))
            block_g = gd.Batch.from_data_list(self.building_block_datas)
        else:
            assert self.num_block_sampling < self.num_building_blocks
            block_indices = list(range(self.num_block_sampling))
            # block_indices = np.random.choice(self.num_building_blocks, self.num_block_sampling, replace=False).tolist()
            # block_indices.sort()
            block_g = gd.Batch.from_data_list([self.building_block_datas[idx] for idx in block_indices])
        return block_indices, block_g

    def aidx_to_ReactionAction(
        self, g: gd.Data, action_idx: ReactionActionIdx, fwd: bool = True, block_indices: Optional[List[int]] = None
    ) -> ReactionAction:
        type_idx, is_stop, rxn_idx, block_local_idx, block_is_first = action_idx
        if fwd:
            t = self.action_type_order[type_idx]
            if is_stop:
                return ForwardAction(ReactionActionType.Stop)
            elif t is ReactionActionType.AddFirstReactant:
                assert block_local_idx >= 0 and block_indices is not None
                block_global_idx = block_indices[block_local_idx]
                building_block = self.building_block_mols[block_global_idx]
                return ForwardAction(t, block_local_idx=block_local_idx, block=building_block)
            elif t is ReactionActionType.ReactUni:
                assert rxn_idx >= 0
                reaction = self.unimolecular_reactions[rxn_idx]
                return ForwardAction(t, reaction=reaction)
            elif t is ReactionActionType.ReactBi:
                assert rxn_idx >= 0 and block_local_idx >= 0 and block_is_first >= 0 and block_indices is not None
                reaction = self.bimolecular_reactions[rxn_idx]
                block_global_idx = block_indices[block_local_idx]
                building_block = self.building_block_mols[block_global_idx]
                return ForwardAction(
                    t,
                    reaction=reaction,
                    block_local_idx=block_local_idx,
                    block=building_block,
                    block_is_first=(block_is_first == 1),
                )
            else:
                raise ValueError(t)
        else:
            t = self.bck_action_type_order[type_idx]
            if is_stop:
                return BackwardAction(ReactionActionType.Stop)
            elif t is ReactionActionType.BckRemoveFirstReactant:
                return BackwardAction(t)
            elif t is ReactionActionType.BckReactUni:
                assert rxn_idx >= 0
                reaction = self.unimolecular_reactions[rxn_idx]
                return BackwardAction(t, reaction=reaction)
            elif t is ReactionActionType.BckReactBi:
                assert rxn_idx >= 0 and block_is_first >= 0
                reaction = self.bimolecular_reactions[rxn_idx]
                return BackwardAction(t, reaction=reaction, block_is_first=(block_is_first == 1))
            else:
                raise ValueError(t)

    def ReactionAction_to_aidx(self, g: gd.Data, action: ReactionAction) -> ReactionActionIdx:
        type_idx = rxn_idx = block_is_first = block_local_idx = -1
        for u in [self.action_type_order, self.bck_action_type_order]:
            if action.action in u:
                type_idx = u.index(action.action)
                break
        # NOTE: -1: None
        assert type_idx >= 0
        if action.action is ReactionActionType.Stop:
            pass
        elif action.action is ReactionActionType.AddFirstReactant:
            assert isinstance(action, ForwardAction)
            assert action.block_local_idx is not None
            block_local_idx = action.block_local_idx
        elif action.action is ReactionActionType.ReactUni:
            assert isinstance(action, ForwardAction)
            assert action.reaction is not None
            rxn_idx = self.unimolecular_reactions.index(action.reaction)
        elif action.action is ReactionActionType.ReactBi:
            assert isinstance(action, ForwardAction)
            assert (
                action.reaction is not None and action.block_local_idx is not None and action.block_is_first is not None
            )
            rxn_idx = self.bimolecular_reactions.index(action.reaction)
            block_local_idx = action.block_local_idx
            block_is_first = int(action.block_is_first)
        elif action.action is ReactionActionType.BckRemoveFirstReactant:
            assert isinstance(action, BackwardAction)
            pass
        elif action.action is ReactionActionType.BckReactUni:
            assert isinstance(action, BackwardAction)
            assert action.reaction is not None
            rxn_idx = self.unimolecular_reactions.index(action.reaction)
        elif action.action is ReactionActionType.BckReactBi:
            assert isinstance(action, BackwardAction)
            assert action.reaction is not None and action.block_is_first is not None
            rxn_idx = self.bimolecular_reactions.index(action.reaction)
            block_is_first = action.block_is_first
        else:
            raise ValueError(action)
        is_stop = action.action is ReactionActionType.Stop
        return (type_idx, is_stop, rxn_idx, block_local_idx, block_is_first)

    def graph_to_Data(self, g: Graph, traj_idx: int, prev_action: Optional[ForwardAction]) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        x = np.zeros((max(1, len(g.nodes)), self.num_node_dim), dtype=np.float32)
        x[0, -1] = len(g.nodes) == 0  # If there are no nodes, set the last dimension to 1

        for i, n in enumerate(g.nodes):
            ad = g.nodes[n]
            for k, sl in zip(self.atom_attrs, self.atom_attr_slice):
                idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                x[i, sl + idx] = 1  # One-hot encode the attribute value

        edge_attr = np.zeros((len(g.edges) * 2, self.num_edge_dim), dtype=np.float32)
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for k, sl in zip(self.bond_attrs, self.bond_attr_slice):
                idx = self.bond_attr_values[k].index(ad[k])
                edge_attr[i * 2, sl + idx] = 1
                edge_attr[i * 2 + 1, sl + idx] = 1
        edge_index = np.array([e for i, j in g.edges for e in [(i, j), (j, i)]]).reshape((-1, 2)).T.astype(np.int64)

        react_uni_mask = self.create_masks(g, fwd=True, unimolecular=True)
        react_bi_mask = self.create_masks(g, fwd=True, unimolecular=False)
        bck_react_uni_mask = self.create_masks(g, fwd=False, unimolecular=True)
        bck_react_bi_mask = self.create_masks(g, fwd=False, unimolecular=False)

        # NOTE: prevent the rdkit bck-reaction error
        # if (R1, ...) -> P with rxn R,
        # There are cases where the reverse reaction R is not possible to P although P was performed with R.
        if prev_action is not None:
            if prev_action.action is ReactionActionType.ReactUni:
                assert prev_action.reaction is not None
                rxn_idx = self.unimolecular_reactions.index(prev_action.reaction)
                bck_react_uni_mask[0, rxn_idx] = True
            elif prev_action.action is ReactionActionType.ReactBi:
                assert prev_action.reaction is not None
                rxn_idx = self.bimolecular_reactions.index(prev_action.reaction)
                bck_react_bi_mask[0, rxn_idx] = True

        data = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            traj_idx=np.array([traj_idx], dtype=np.int32),
            # add attribute for masks
            react_uni_mask=react_uni_mask,
            react_bi_mask=react_bi_mask,
            bck_react_uni_mask=bck_react_uni_mask,
            bck_react_bi_mask=bck_react_bi_mask,
        )
        data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})
        return data

    def graph_to_Data_block(self, g: Graph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        x = np.zeros((max(1, len(g.nodes)), self.num_node_dim), dtype=np.float32)
        x[0, -1] = len(g.nodes) == 0  # If there are no nodes, set the last dimension to 1

        for i, n in enumerate(g.nodes):
            ad = g.nodes[n]
            for k, sl in zip(self.atom_attrs, self.atom_attr_slice):
                idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                x[i, sl + idx] = 1  # One-hot encode the attribute value

        edge_attr = np.zeros((len(g.edges) * 2, self.num_edge_dim), dtype=np.float32)
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for k, sl in zip(self.bond_attrs, self.bond_attr_slice):
                idx = self.bond_attr_values[k].index(ad[k])
                edge_attr[i * 2, sl + idx] = 1
                edge_attr[i * 2 + 1, sl + idx] = 1
        edge_index = np.array([e for i, j in g.edges for e in [(i, j), (j, i)]]).reshape((-1, 2)).T.astype(np.int64)

        data = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})
        return data

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        return gd.Batch.from_data_list(graphs, follow_batch=["edge_index"])

    def get_mol(self, smi: Union[str, Chem.Mol, Graph]) -> Chem.Mol:
        """
        A function that returns an `RDKit.Chem.Mol` object.

        Args:
            smi (str or RDKit.Chem.Mol or Graph): The query molecule, as either a
                SMILES string an `RDKit.Chem.Mol` object, or a Graph.

        Returns:
            RDKit.Chem.Mol
        """
        if isinstance(smi, str):
            # return Chem.MolFromSmiles(smi, replacements={"[2H]": "[H]"}) -> This is performed on pre-processing
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        elif isinstance(smi, Graph):
            return self.graph_to_mol(smi)
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")

    def mol_to_graph(self, mol: Chem.Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = Graph()
        mol = Chem.RemoveHs(mol)
        mol.UpdatePropertyCache()
        if not self.allow_explicitly_aromatic:
            # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
            Chem.Kekulize(mol, clearAromaticFlags=True)
        # Only set an attribute tag if it is not the default attribute
        for a in mol.GetAtoms():
            attrs = {
                "atomic_number": a.GetAtomicNum(),
                "chi": a.GetChiralTag(),
                "charge": a.GetFormalCharge(),
                "expl_H": a.GetNumExplicitHs(),
            }
            g.add_node(
                a.GetIdx(),
                v=a.GetSymbol(),
                **{attr: val for attr, val in attrs.items()},
                **({"fill_wildcard": None} if a.GetSymbol() == "*" else {}),
            )
        for b in mol.GetBonds():
            attrs = {"type": b.GetBondType()}
            g.add_edge(
                b.GetBeginAtomIdx(),
                b.GetEndAtomIdx(),
                **{attr: val for attr, val in attrs.items()},
            )
        return g

    def graph_to_mol(self, g: Graph) -> Chem.Mol:
        """Convert a Graph to an RDKit Mol"""
        mp = Chem.RWMol()
        mp.BeginBatchEdit()
        for i in range(len(g.nodes)):
            d = g.nodes[i]
            s = d.get("fill_wildcard", d["v"])
            a = Chem.Atom(s if s is not None else self.default_wildcard_replacement)
            if "chi" in d:
                a.SetChiralTag(d["chi"])
            if "charge" in d:
                a.SetFormalCharge(d["charge"])
            if "expl_H" in d:
                a.SetNumExplicitHs(d["expl_H"])
            if "no_impl" in d:
                a.SetNoImplicit(d["no_impl"])
            mp.AddAtom(a)
        for e in g.edges:
            d = g.edges[e]
            mp.AddBond(e[0], e[1], d.get("type", BondType.SINGLE))
        mp.CommitBatchEdit()
        Chem.SanitizeMol(mp)
        return Chem.MolFromSmiles(Chem.MolToSmiles(mp))

    def object_to_log_repr(self, g: Graph):
        """Convert a Graph to a string representation"""
        try:
            mol = self.graph_to_mol(g)
            assert mol is not None
            return Chem.MolToSmiles(mol)
        except Exception:
            return ""

    def has_n(self) -> bool:
        return False

    def log_n(self, g) -> float:
        return 0.0

    def traj_log_n(self, traj):
        return [self.log_n(g) for g, _ in traj]

    def traj_to_log_repr(self, traj: List[Tuple[Graph]]):
        """Convert a tuple of graph, action idx to a string representation, action idx"""
        smi_traj = []
        for i in traj:
            mol = self.graph_to_mol(i[0])
            assert mol is not None
            smi_traj.append((Chem.MolToSmiles(mol), i[1]))
        return str(smi_traj)

    def create_masks(self, smi: Union[str, Chem.Mol, Graph], fwd: bool = True, unimolecular: bool = True) -> np.ndarray:
        """Creates masks for reaction templates for a given molecule.

        Args:
            mol (Chem.Mol): Molecule as a rdKit Mol object.
            fwd (bool): Whether it is a forward or a backward step.
            unimolecular (bool): Whether it is a unimolecular or a bimolecular reaction.

        Returns:
            (torch.Tensor): Masks for invalid actions.
        """
        mol = self.get_mol(smi)
        if unimolecular:
            masks = np.ones(self.num_unimolecular_rxns, dtype=np.bool_)
            reactions = self.unimolecular_reactions
        else:
            masks = np.ones(self.num_bimolecular_rxns, dtype=np.bool_)
            reactions = self.bimolecular_reactions
        for idx, r in enumerate(reactions):
            if fwd:
                if not r.is_reactant(mol):
                    masks[idx] = False
            else:
                if r.is_product(mol):
                    continue
                mol_copy = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                Chem.Kekulize(mol_copy, clearAromaticFlags=True)
                if r.is_product(mol_copy):
                    continue
                masks[idx] = False
        return masks.reshape(1, -1)

    def create_masks_for_bb(self, smi: Union[str, Chem.Mol, Graph], bimolecular_rxn_idx: int) -> np.ndarray:
        """Create masks for building blocks for a given molecule."""
        mol = self.get_mol(smi)
        reaction = self.bimolecular_reactions[bimolecular_rxn_idx]
        reactants = reaction.rxn.GetReactants()
        mol_match_0 = mol.HasSubstructMatch(reactants[0])
        mol_match_1 = mol.HasSubstructMatch(reactants[1])
        assert (
            mol_match_0 or mol_match_1
        ), "Molecule does not match reaction template -- this should be verified at the reaction-selection step."

        masks = np.zeros((self.num_building_blocks, 2), dtype=np.bool_)
        for idx, bb in enumerate(self.building_block_mols):
            if mol_match_1 and bb.HasSubstructMatch(reactants[0]):
                masks[idx, 0] = True
            elif mol_match_0 and bb.HasSubstructMatch(reactants[1]):
                masks[idx, 1] = True
        return masks

    def create_masks_for_bb_from_precomputed(
        self, smi: Union[str, Chem.Mol, Graph], bimolecular_rxn_idx: int
    ) -> np.ndarray:
        """Creates masks for building blocks (for the 2nd reactant) for a given molecule and bimolecular reaction.
        Uses masks precomputed with data/building_blocks/precompute_bb_masks.py.

        Args:
            smi (Union[str, Chem.Mol, Graph]): Molecule as a rdKit Mol object.
            bimolecular_rxn_idx (int): Index of the bimolecular reaction.
        """
        mol = self.get_mol(smi)
        reaction = self.bimolecular_reactions[bimolecular_rxn_idx]
        reactants = reaction.rxn.GetReactants()

        precomputed_bb_masks = self.precomputed_bb_masks[:, bimolecular_rxn_idx, :]
        mol_mask = np.array(
            [  # we reverse the order of the reactants w.r.t BBs (i.e. reactants[1] first)
                [mol.HasSubstructMatch(reactants[1])] * self.num_building_blocks,
                [mol.HasSubstructMatch(reactants[0])] * self.num_building_blocks,
            ],
            dtype=np.bool_,
        )
        masks = mol_mask & precomputed_bb_masks
        return masks.transpose(1, 0).reshape(self.num_building_blocks, 2)
