"""Padded molecular batches with directed neighbor features.

Each state stores its atoms and bonds. Batches pad the atom and neighbor axes;
masks exclude padding from message passing and pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from rxnflow.core.molecule import Molecule
from rxnflow.envs.features import heavy_atom_count, molecular_properties

# Atomic numbers and synthon labels 0..99 have dedicated slots; each
# vocabulary gets one overflow slot in NODE_FEATURE_DIM.
ATOM_TYPES = tuple(range(100))
SYNTHON_TYPES = tuple(range(100))
# Atom/type one-hots, degree, charge, four scalar features and chirality.
NODE_FEATURE_DIM = len(ATOM_TYPES) + 1 + 7 + 6 + 4 + len(SYNTHON_TYPES) + 1 + 3
# Distinct categories retain E/Z and cis/trans bond stereochemistry.
BOND_STEREO_TYPES = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
)
# Four bond types, stereo categories + unknown, conjugation and ring flags.
BOND_FEATURE_DIM = 4 + len(BOND_STEREO_TYPES) + 1 + 2


@dataclass
class GraphData:
    node_features: Tensor
    edge_index: Tensor
    bond_features: Tensor
    mol_features: Tensor
    size: Tensor


@dataclass
class GraphBatch:
    node_features: Tensor  # [B, N, D_node]
    node_mask: Tensor  # [B, N]
    num_nodes: Tensor  # [B]
    neighbor_index: Tensor  # [B, N, K], local source indices for each target atom
    neighbor_mask: Tensor  # [B, N, K]
    bond_features: Tensor  # [B, N, K, D_bond]
    mol_features: Tensor
    size: Tensor

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    @property
    def batch_size(self) -> int:
        return self.mol_features.shape[0]

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    @classmethod
    def from_list(cls, graphs: list[GraphData]) -> GraphBatch:
        assert graphs
        num_nodes = torch.tensor([len(graph.node_features) for graph in graphs])
        batch_size = len(graphs)
        max_nodes = int(num_nodes.max())
        max_neighbors = 0
        for graph in graphs:
            if graph.edge_index.numel():
                degrees = torch.bincount(graph.edge_index[1])
                max_neighbors = max(max_neighbors, int(degrees.max()))

        node_features = graphs[0].node_features.new_zeros(
            (batch_size, max_nodes, NODE_FEATURE_DIM)
        )
        node_mask = torch.arange(max_nodes)[None, :] < num_nodes[:, None]
        neighbor_index = torch.zeros(
            (batch_size, max_nodes, max_neighbors), dtype=torch.long
        )
        neighbor_mask = torch.zeros_like(neighbor_index, dtype=torch.bool)
        bond_features = graphs[0].bond_features.new_zeros(
            (batch_size, max_nodes, max_neighbors, BOND_FEATURE_DIM)
        )
        for index, graph in enumerate(graphs):
            count = len(graph.node_features)
            node_features[index, :count] = graph.node_features
            source, target = graph.edge_index
            # Each directed bond occupies one incoming-neighbor slot of its target.
            next_slot = [0] * count
            slots = []
            for atom in target.tolist():
                slots.append(next_slot[atom])
                next_slot[atom] += 1
            slots = torch.tensor(slots, dtype=torch.long)
            neighbor_index[index, target, slots] = source
            neighbor_mask[index, target, slots] = True
            bond_features[index, target, slots] = graph.bond_features

        return cls(
            node_features=node_features,
            node_mask=node_mask,
            num_nodes=num_nodes,
            neighbor_index=neighbor_index,
            neighbor_mask=neighbor_mask,
            bond_features=bond_features,
            mol_features=torch.stack([graph.mol_features for graph in graphs]),
            size=torch.stack([graph.size for graph in graphs]),
        )


def molecule_to_graph_data(molecule: Molecule | None) -> GraphData:
    """Encode one state without truncating atoms or including explicit hydrogens."""
    rdmol = None if molecule is None else molecule.rdmol

    if (
        rdmol is not None
        and sum(atom.GetAtomicNum() == 0 for atom in rdmol.GetAtoms()) > 1
    ):
        raise ValueError("a synthesis state can have at most one dummy handle")

    atoms = (
        []
        if rdmol is None
        else [atom for atom in rdmol.GetAtoms() if atom.GetAtomicNum() != 1]
    )
    node_features = np.zeros((len(atoms), NODE_FEATURE_DIM), dtype=np.float32)
    # Explicit isotope-labelled hydrogen bonds are omitted below; allocate at
    # most two directed edges per RDKit bond and retain only the filled rows.
    max_edges = 0 if rdmol is None else 2 * rdmol.GetNumBonds()
    edge_index = np.empty((2, max_edges), dtype=np.int64)
    bond_features = np.zeros((max_edges, BOND_FEATURE_DIM), dtype=np.float32)
    num_edges = 0

    if rdmol is not None:
        index_map = {atom.GetIdx(): index for index, atom in enumerate(atoms)}
        # 2. Fill atom categories/scalars in NumPy; wrap completed arrays once.
        degree_start = len(ATOM_TYPES) + 1
        charge_start = degree_start + 7
        scalar_start = charge_start + 6
        synthon_start = scalar_start + 4
        chiral_types = (
            Chem.ChiralType.CHI_UNSPECIFIED,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        )
        for index, atom in enumerate(atoms):
            number = atom.GetAtomicNum()
            isotope = atom.GetIsotope() if number == 0 else 0
            node_features[index, min(number, len(ATOM_TYPES))] = 1
            node_features[index, degree_start + min(atom.GetDegree(), 6)] = 1
            node_features[
                index, charge_start + max(-2, min(3, atom.GetFormalCharge())) + 2
            ] = 1
            node_features[index, scalar_start:synthon_start] = (
                float(atom.GetIsAromatic()),
                atom.GetMass() / 100.0 if number else 0.0,
                atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
                min(int(atom.GetHybridization()) / 8.0, 1.0),
            )
            node_features[index, synthon_start + min(isotope, len(SYNTHON_TYPES))] = 1
            # Chirality distinguishes enantiomers independently of synthon type.
            tag = atom.GetChiralTag()
            node_features[
                index, -3 + (chiral_types.index(tag) if tag in chiral_types else 0)
            ] = 1

        # 3. Store each bond in both directions, with shared chemistry features.
        for bond in rdmol.GetBonds():
            # Explicit isotopic H atoms can survive RDKit's RemoveHs, but the
            # model representation only allocates heavy-atom and dummy slots.
            if (
                bond.GetBeginAtomIdx() not in index_map
                or bond.GetEndAtomIdx() not in index_map
            ):
                continue
            begin = index_map[bond.GetBeginAtomIdx()]
            end = index_map[bond.GetEndAtomIdx()]
            feature = bond_features[num_edges]
            feature[:4] = (
                bond.GetBondType() == Chem.BondType.SINGLE,
                bond.GetBondType() == Chem.BondType.DOUBLE,
                bond.GetBondType() == Chem.BondType.TRIPLE,
                bond.GetBondType() == Chem.BondType.AROMATIC,
            )
            stereo = bond.GetStereo()
            stereo_index = (
                BOND_STEREO_TYPES.index(stereo)
                if stereo in BOND_STEREO_TYPES
                else len(BOND_STEREO_TYPES)
            )
            feature[4 + stereo_index] = 1
            feature[-2:] = bond.GetIsConjugated(), bond.IsInRing()
            edge_index[:, num_edges] = begin, end
            edge_index[:, num_edges + 1] = end, begin
            bond_features[num_edges + 1] = feature
            num_edges += 2

    mol_properties = molecular_properties(rdmol)
    atom_count = heavy_atom_count(rdmol)

    # 4. Retain only represented bonds and attach raw molecular descriptors.
    return GraphData(
        node_features=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index[:, :num_edges]),
        bond_features=torch.from_numpy(bond_features[:num_edges]),
        mol_features=torch.from_numpy(mol_properties),
        size=torch.tensor(atom_count, dtype=torch.long),
    )
