import json
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import torch
import torch_geometric.data as gd
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType, ChiralType

from gflownet.envs.graph_building_env import GraphBuildingEnvContext, Graph
from gflownet.envs.synthesis.reaction import Reaction
from gflownet.envs.synthesis.env import SynthesisEnv
from gflownet.envs.synthesis.action import ReactionAction, ReactionActionType, ReactionActionIdx
from gflownet.envs.synthesis.building_block import get_block_features

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")

DEFAULT_ATOMS: list[str] = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]
DEFAULT_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
DEFAULT_EXPL_H_RANGE = [0, 1, 2, 3, 4]  # for N


class SynthesisEnvContext(GraphBuildingEnvContext):
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(
        self,
        env: SynthesisEnv,
        num_cond_dim: int = 0,
        fp_radius_building_block: int = 2,
        fp_nbits_building_block: int = 1024,
        *args,
        atoms: list[str] = DEFAULT_ATOMS,
        chiral_types: list = DEFAULT_CHIRAL_TYPES,
        charges: list[int] = DEFAULT_CHARGES,
        expl_H_range: list[int] = DEFAULT_EXPL_H_RANGE,  # for N
        allow_explicitly_aromatic: bool = False,
    ):
        """An env context for generating molecules by sequentially applying reaction templates.
        Contains functionalities to build molecular graphs, create masks for actions, and convert molecules to other representations.

        Args:
            atoms (list): list of atom symbols.
            chiral_types (list): list of chiral types.
            charges (list): list of charges.
            expl_H_range (list): list of explicit H counts.
            allow_explicitly_aromatic (bool): Whether to allow explicitly aromatic molecules.
            allow_5_valence_nitrogen (bool): Whether to allow N with valence of 5.
            num_cond_dim (int): The dimensionality of the observations' conditional information vector (if >0)
            reaction_templates (list): list of SMIRKS.
            building_blocks (list): list of SMILES strings of building blocks.
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
        self.num_node_dim = sum(len(v) for v in self.atom_attr_values.values())
        self.num_edge_dim = sum(len(v) for v in self.bond_attr_values.values())
        self.num_cond_dim = num_cond_dim

        # NOTE: Action Type Order
        self.action_type_order: list[ReactionActionType] = [
            ReactionActionType.Stop,
            ReactionActionType.ReactUni,
            ReactionActionType.ReactBi,
            ReactionActionType.AddFirstReactant,
        ]

        self.bck_action_type_order: list[ReactionActionType] = [
            ReactionActionType.BckReactUni,
            ReactionActionType.BckReactBi,
            ReactionActionType.BckRemoveFirstReactant,
        ]

        # NOTE: For Molecular Reaction - Environment
        self.env: SynthesisEnv = env
        self.reactions: list[Reaction] = env.reactions
        self.unimolecular_reactions = env.unimolecular_reactions
        self.bimolecular_reactions = env.bimolecular_reactions
        self.num_unimolecular_rxns = env.num_unimolecular_rxns
        self.num_bimolecular_rxns = env.num_bimolecular_rxns
        self.unimolecular_reaction_to_idx = {env: i for i, env in enumerate(env.unimolecular_reactions)}
        self.bimolecular_reaction_to_idx = {env: i for i, env in enumerate(env.bimolecular_reactions)}

        self.building_blocks: list[str] = env.building_blocks
        self.num_building_blocks: int = len(self.building_blocks)

        self.building_block_mask: NDArray[np.bool_] = env.building_block_mask
        self.allowable_birxn_mask: NDArray[np.bool_] = env.building_block_mask.any(-1)

        # # NOTE: Setup Building Block Datas
        self.building_block_features: tuple[NDArray[np.bool_], NDArray[np.float32]]
        if fp_nbits_building_block == 1024 and fp_radius_building_block == 2:
            self.building_block_features = env.building_block_features
        else:
            raise NotImplementedError("I implement it but do not check following code block is working")
            print("Fragment Data Construction...")
            self.building_block_features = (
                np.empty((self.num_building_blocks, 166 + 1024), dtype=np.bool_),
                np.empty((self.num_building_blocks, 8), dtype=np.float32),
            )
            for i, smi in enumerate(tqdm(self.building_blocks)):
                get_block_features(
                    smi,
                    fp_radius_building_block,
                    fp_nbits_building_block,
                    self.building_block_features[0][i],
                    self.building_block_features[1][i],
                )
        self.num_block_features = self.building_block_features[0].shape[-1] + self.building_block_features[1].shape[-1]

    def get_block_data(
        self, block_indices: torch.Tensor | list[int] | np.ndarray, device: torch.device
    ) -> torch.Tensor:
        if len(block_indices) >= self.num_building_blocks:
            fp = self.building_block_features[0]
            feat = self.building_block_features[1]
        else:
            fp = self.building_block_features[0][block_indices]
            feat = self.building_block_features[1][block_indices]
        out = torch.cat([torch.as_tensor(fp, dtype=torch.float32), torch.from_numpy(feat)], dim=-1)
        return out.to(device=device, dtype=torch.float)

    def aidx_to_GraphAction(self, action_idx: ReactionActionIdx, fwd: bool = True) -> ReactionAction:
        type_idx, rxn_idx, block_idx, block_is_first = action_idx
        if fwd:
            t = self.action_type_order[type_idx]
        else:
            t = self.bck_action_type_order[type_idx]

        if t is ReactionActionType.Stop:
            return ReactionAction(ReactionActionType.Stop)

        elif t in (ReactionActionType.AddFirstReactant, ReactionActionType.BckRemoveFirstReactant):
            assert block_idx >= 0
            building_block = self.building_blocks[block_idx]
            return ReactionAction(t, block_idx=block_idx, block=building_block)

        elif t in (ReactionActionType.ReactUni, ReactionActionType.BckReactUni):
            assert rxn_idx >= 0
            reaction = self.unimolecular_reactions[rxn_idx]
            return ReactionAction(t, reaction=reaction)

        elif t in (ReactionActionType.ReactBi, ReactionActionType.BckReactBi):
            assert rxn_idx >= 0 and block_idx >= 0 and block_is_first >= 0
            reaction = self.bimolecular_reactions[rxn_idx]
            building_block = self.building_blocks[block_idx]
            return ReactionAction(
                t,
                reaction=reaction,
                block_idx=block_idx,
                block=building_block,
                block_is_first=(block_is_first == 1),
            )
        else:
            raise ValueError(t)

    def GraphAction_to_aidx(self, action: ReactionAction) -> ReactionActionIdx:
        type_idx = -1
        for u in [self.action_type_order, self.bck_action_type_order]:
            if action.action in u:
                type_idx = u.index(action.action)
                break
        # NOTE: -1: None
        assert type_idx >= 0

        rxn_idx = block_is_first = block_idx = -1
        if action.action is ReactionActionType.Stop:
            pass
        elif action.action is ReactionActionType.AddFirstReactant:
            assert action.block_idx is not None
            block_idx = action.block_idx
        elif action.action is ReactionActionType.ReactUni:
            assert action.reaction is not None
            rxn_idx = self.unimolecular_reaction_to_idx[action.reaction]
        elif action.action is ReactionActionType.ReactBi:
            assert action.reaction is not None and action.block_idx is not None and action.block_is_first is not None
            rxn_idx = self.bimolecular_reaction_to_idx[action.reaction]
            block_idx = action.block_idx
            block_is_first = int(action.block_is_first)
        elif action.action is ReactionActionType.BckRemoveFirstReactant:
            assert action.block_idx is not None
            block_idx = action.block_idx
        elif action.action is ReactionActionType.BckReactUni:
            assert action.reaction is not None
            rxn_idx = self.unimolecular_reaction_to_idx[action.reaction]
        elif action.action is ReactionActionType.BckReactBi:
            assert action.reaction is not None and action.block_idx is not None and action.block_is_first is not None
            rxn_idx = self.bimolecular_reaction_to_idx[action.reaction]
            block_idx = action.block_idx
            block_is_first = action.block_is_first
        else:
            raise ValueError(action)
        return (type_idx, rxn_idx, block_idx, block_is_first)

    def graph_to_Data(self, g: Graph, traj_idx: int) -> gd.Data:
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
                if True and ad[k] in self.bond_attr_values[k]:
                    idx = self.bond_attr_values[k].index(ad[k])
                else:
                    idx = 0
                edge_attr[i * 2, sl + idx] = 1
                edge_attr[i * 2 + 1, sl + idx] = 1
        edge_index = np.array([e for i, j in g.edges for e in [(i, j), (j, i)]]).reshape((-1, 2)).T.astype(np.int64)
        data = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            traj_idx=np.array([traj_idx], dtype=np.int32),
        )

        # NOTE: add attribute for masks
        data["react_uni_mask"] = self.create_masks(g, fwd=True, unimolecular=True).reshape(1, -1)  # [1, Nrxn]
        data["react_bi_mask"] = self.create_masks(g, fwd=True, unimolecular=False).reshape(1, -1, 2)  # [1, Nrxn, 2]

        data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})
        return data

    def get_mol(self, smi: str | Chem.Mol | Graph) -> Chem.Mol:
        """
        A function that returns an `RDKit.Chem.Mol` object.

        Args:
            smi (str or RDKit.Chem.Mol or Graph): The query molecule, as either a
                SMILES string an `RDKit.Chem.Mol` object, or a Graph.

        Returns:
            RDKit.Chem.Mol
        """
        if isinstance(smi, str):
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

    def traj_to_log_repr(self, traj: list[tuple[Graph | Chem.Mol, ReactionAction]]) -> str:
        """Convert a trajectory of (Graph, Action) to a trajectory of json representation"""
        traj_logs = self.read_traj(traj)
        repr_obj = []
        for i, (action_repr, smiles) in enumerate(traj_logs):
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

    def read_traj(self, traj: list[tuple[Graph | Chem.Mol, ReactionAction]]) -> list[tuple[str, tuple[str, ...]]]:
        """Convert a trajectory of (Graph, Action) to a trajectory of tuple representation"""
        repr = []
        for mol, action in traj:
            if action.action is ReactionActionType.AddFirstReactant:
                assert action.block is not None
                action_repr = ("StartingBlock", action.block)
            elif action.action is ReactionActionType.ReactUni:
                assert action.reaction is not None
                action_repr = ("UniMolecularReaction", action.reaction.template)
            elif action.action is ReactionActionType.ReactBi:
                assert action.reaction is not None and action.block is not None
                action_repr = ("BiMolecularReaction", action.reaction.template, action.block)
            elif action.action is ReactionActionType.Stop:
                action_repr = ("Stop",)
            else:
                raise ValueError(action.action)
            smiles = Chem.MolToSmiles(self.get_mol(mol))
            repr.append((smiles, action_repr))
        return repr

    def create_masks(self, smi: str | Chem.Mol | Graph, fwd: bool = True, unimolecular: bool = True) -> np.ndarray:
        """Creates masks for reaction templates for a given molecule.

        Args:
            mol (Chem.Mol): Molecule as a rdKit Mol object.
            fwd (bool): Whether it is a forward or a backward step.
            unimolecular (bool): Whether it is a unimolecular or a bimolecular reaction.

        Returns:
            (np.ndarry): Masks for invalid actions.
        """
        mol = self.get_mol(smi)
        if fwd:
            if unimolecular:
                masks = np.zeros((self.num_unimolecular_rxns,), dtype=np.bool_)
                for idx, r in enumerate(self.unimolecular_reactions):
                    if r.is_reactant(mol):
                        masks[idx] = True
            else:
                masks = np.zeros((self.num_bimolecular_rxns, 2), dtype=np.bool_)
                for idx, r in enumerate(self.bimolecular_reactions):
                    # NOTE: block_is_first is True
                    if self.allowable_birxn_mask[idx, 0]:
                        masks[idx, int(True)] = r.is_reactant_second(mol)
                    # NOTE: block_is_second
                    if self.allowable_birxn_mask[idx, 1]:
                        masks[idx, int(False)] = r.is_reactant_first(mol)
        else:
            mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            Chem.Kekulize(mol1, clearAromaticFlags=True)
            if unimolecular:
                masks = np.zeros((self.num_unimolecular_rxns,), dtype=np.bool_)
                for idx, r in enumerate(self.unimolecular_reactions):
                    if r.is_product(mol) or r.is_product(mol1):
                        masks[idx] = True
            else:
                masks = np.ones((self.num_bimolecular_rxns,), dtype=np.bool_)
                for idx, r in enumerate(self.bimolecular_reactions):
                    if r.is_product(mol) or r.is_product(mol1):
                        masks[idx] = True
        return masks
