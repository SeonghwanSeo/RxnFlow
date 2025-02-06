from pathlib import Path
import tempfile
import numpy as np
import torch
import torch_geometric.data as gd
from rdkit import Chem, RDLogger
import torch_cluster

from gflownet.envs.graph_building_env import Graph
from gflownet.envs.synthesis.env import SynthesisEnv
from gflownet.sbdd.pocket.data import generate_protein_data
from gflownet.envs.synthesis.env_context import (
    SynthesisEnvContext,
    DEFAULT_ATOMS,
    DEFAULT_CHIRAL_TYPES,
    DEFAULT_CHARGES,
    DEFAULT_EXPL_H_RANGE,
)
from .docking import run_docking_list

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = D.reshape(-1, 1)

    RBF = np.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


class Synthesis3DEnvContext(SynthesisEnvContext):
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(
        self,
        env: SynthesisEnv,
        protein_path: str,
        protein_pdbqt_path: str,
        protein_center: tuple[float, float, float],
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
        super().__init__(
            env,
            num_cond_dim,
            fp_radius_building_block,
            fp_nbits_building_block,
            atoms=atoms,
            chiral_types=chiral_types,
            charges=charges,
            expl_H_range=expl_H_range,
            allow_explicitly_aromatic=allow_explicitly_aromatic,
        )
        self.num_node_dim += 20
        self.num_edge_dim += 16
        self.pocket_file = protein_path
        self.pocket_center = protein_center
        self.pocket_pdbqt_file: Path = Path(protein_pdbqt_path)
        self.protein_data: dict[str, np.ndarray] = self.generate_protein_data(protein_path, protein_center)

    def generate_protein_data(self, protein_path: str, center: tuple[float, float, float]) -> dict[str, np.ndarray]:
        data = generate_protein_data(protein_path, center, top_k=8)
        x = np.zeros((len(data["seq"]), self.num_node_dim), dtype=np.float32)
        u, v = data["edge_index"].numpy()
        pos = data["coords"].numpy()
        for char_idx in data["seq"]:
            x[-char_idx] = 1

        distance = np.linalg.norm((pos[u] - pos[v]), axis=-1)
        edge_attr = np.zeros((u.shape[0], self.num_edge_dim), dtype=np.float32)
        edge_attr[:, -16:] = _rbf(distance)
        return dict(
            x=x,
            pos=pos,
            seq=data["seq"].numpy(),
            edge_index=data["edge_index"].numpy(),
            edge_attr=edge_attr,
        )

    def graph_to_Data(self, g: Graph, traj_idx: int) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        if len(g.nodes) == 0:
            data = dict(
                x=np.zeros((1, self.num_node_dim), dtype=np.float32),
                pos=np.zeros((1, 3), dtype=np.float32),
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_attr=np.zeros((0, self.num_edge_dim), dtype=np.float32),
                traj_idx=np.array([0], dtype=np.int32),
            )
            data["x"][0, -1] = 1.0
            # NOTE: add attribute for masks
            data["react_uni_mask"] = self.create_masks(g, fwd=True, unimolecular=True).reshape(1, -1)  # [1, Nrxn]
            data["react_bi_mask"] = self.create_masks(g, fwd=True, unimolecular=False).reshape(1, -1, 2)  # [1, Nrxn, 2]
            data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})
            return data

        x = np.zeros((len(g.nodes), self.num_node_dim), dtype=np.float32)
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

        # NOTE: Protein-Ligand Graph
        lig_x = x
        lig_pos = np.stack([g.nodes[n]["pos"] for n in g.nodes], 0)
        lig_edge_attr = edge_attr
        lig_edge_index = edge_index

        prot_x = self.protein_data["x"]
        prot_pos = self.protein_data["pos"]
        prot_edge_attr = self.protein_data["edge_attr"]
        prot_edge_index = self.protein_data["edge_index"]
        prot_edge_index = prot_edge_index + len(g.nodes)

        u, v = torch_cluster.knn(torch.from_numpy(prot_pos).double(), torch.from_numpy(lig_pos).double(), 16).numpy()
        distance = np.linalg.norm((prot_pos[v] - lig_pos[u]), axis=-1)
        lp_edge_attr = np.zeros((u.shape[0], self.num_edge_dim), dtype=np.float32)
        lp_edge_attr[:, -16:] = _rbf(distance)
        v = v + len(g.nodes)
        lp_edge_index = np.stack([np.concatenate([u, v]), np.concatenate([v, u])])

        x = np.concatenate([lig_x, prot_x], 0)
        pos = np.concatenate([lig_pos, prot_pos], 0)
        edge_index = np.concatenate([lig_edge_index, prot_edge_index, lp_edge_index], 1)
        edge_attr = np.concatenate([lig_edge_attr, prot_edge_attr, lp_edge_attr, lp_edge_attr], 0)

        data = dict(
            x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, traj_idx=np.array([traj_idx], dtype=np.int32)
        )

        # NOTE: add attribute for masks
        data["react_uni_mask"] = self.create_masks(g, fwd=True, unimolecular=True).reshape(1, -1)  # [1, Nrxn]
        data["react_bi_mask"] = self.create_masks(g, fwd=True, unimolecular=False).reshape(1, -1, 2)  # [1, Nrxn, 2]

        data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})
        return data

    def mol_to_graph(self, mol: Chem.Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = Graph()
        if mol.GetNumAtoms() > 0:
            with tempfile.TemporaryDirectory() as dir:
                run_docking(mol, self.pocket_pdbqt_file, self.pocket_center, search_mode="fast", out_dir=dir)
                docking_files = Path(dir) / "ligand_out.sdf"
                mol = list(Chem.SDMolSupplier(str(docking_files)))[0]
            mol = Chem.RemoveHs(mol)
            pos = mol.GetConformer().GetPositions()
        else:
            pos = None

        mol.UpdatePropertyCache()
        if not self.allow_explicitly_aromatic:
            # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
            Chem.Kekulize(mol, clearAromaticFlags=True)
        # Only set an attribute tag if it is not the default attribute
        for idx, a in enumerate(mol.GetAtoms()):
            assert pos is not None
            attrs = {
                "atomic_number": a.GetAtomicNum(),
                "chi": a.GetChiralTag(),
                "charge": a.GetFormalCharge(),
                "expl_H": a.GetNumExplicitHs(),
            }
            g.add_node(
                a.GetIdx(),
                v=a.GetSymbol(),
                pos=pos[idx],
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

    def mol_to_graph_list(self, mols: list[Chem.Mol]) -> list[Graph]:
        """Convert an RDMol to a Graph"""
        if mols[0].GetNumAtoms() > 0:
            with tempfile.TemporaryDirectory() as dir:
                out_mols = run_docking_list(
                    mols, self.pocket_pdbqt_file, self.pocket_center, search_mode="fast", out_dir=dir
                )
        else:
            out_mols = mols
        del mols
        graphs = []
        for mol in out_mols:
            if mol is None:
                graphs.append(None)
                continue
            if mol.GetNumAtoms() > 0:
                mol = Chem.RemoveHs(mol)
                pos = mol.GetConformer().GetPositions()
            else:
                pos = None

            g = Graph()
            mol.UpdatePropertyCache()
            if not self.allow_explicitly_aromatic:
                # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
                Chem.Kekulize(mol, clearAromaticFlags=True)
            # Only set an attribute tag if it is not the default attribute
            for idx, a in enumerate(mol.GetAtoms()):
                assert pos is not None
                attrs = {
                    "atomic_number": a.GetAtomicNum(),
                    "chi": a.GetChiralTag(),
                    "charge": a.GetFormalCharge(),
                    "expl_H": a.GetNumExplicitHs(),
                }
                g.add_node(
                    a.GetIdx(),
                    v=a.GetSymbol(),
                    pos=pos[idx],
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
            graphs.append(g)
        return graphs

    def mol_to_graph(self, mol: Chem.Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = Graph()
        if mol.GetNumAtoms() > 0:
            with tempfile.TemporaryDirectory() as dir:
                run_docking(mol, self.pocket_pdbqt_file, self.pocket_center, search_mode="fast", out_dir=dir)
                docking_files = Path(dir) / "ligand_out.sdf"
                mol = list(Chem.SDMolSupplier(str(docking_files)))[0]
            mol = Chem.RemoveHs(mol)
            pos = mol.GetConformer().GetPositions()
        else:
            pos = None

        mol.UpdatePropertyCache()
        if not self.allow_explicitly_aromatic:
            # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
            Chem.Kekulize(mol, clearAromaticFlags=True)
        # Only set an attribute tag if it is not the default attribute
        for idx, a in enumerate(mol.GetAtoms()):
            assert pos is not None
            attrs = {
                "atomic_number": a.GetAtomicNum(),
                "chi": a.GetChiralTag(),
                "charge": a.GetFormalCharge(),
                "expl_H": a.GetNumExplicitHs(),
            }
            g.add_node(
                a.GetIdx(),
                v=a.GetSymbol(),
                pos=pos[idx],
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
