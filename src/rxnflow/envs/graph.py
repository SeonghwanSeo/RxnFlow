"""Fixed-shape model tensors derived from the state's RDKit molecule.

Every ``GraphData`` has length ``L = Config.data.max_atoms + 1``: heavy atoms
plus one reserved dummy-handle slot. RDKit excludes dummies from its heavy-atom
count, and a nonterminal linear synthesis state has exactly one. Batching only
stacks those tensors, producing nodes ``[B, L, node_dim]``, adjacency
``[B, L, L]``, and bonds ``[B, L, L, bond_dim]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from rxnflow.envs.chemistry.features import (
    heavy_atom_count,
    molecular_properties,
    normalize_molecular_properties,
)

# NOTE: For general/common usage, we allocate 100 atom & synthon types,
# which is more than enough for most practical applications.
ATOM_TYPES = tuple(range(100))
SYNTHON_TYPES = tuple(range(100))
# Atom/type one-hots, degree, charge and four continuous/boolean features.
NODE_FEATURE_DIM = len(ATOM_TYPES) + 1 + 7 + 6 + 4 + len(SYNTHON_TYPES) + 1
BOND_FEATURE_DIM = 7


@dataclass
class GraphData:
    node_features: Tensor
    node_mask: Tensor
    adjacency: Tensor
    bond_features: Tensor
    mol_features: Tensor
    reaction_count: int
    remaining_capacity: float


@dataclass
class GraphBatch:
    node_features: Tensor
    node_mask: Tensor
    adjacency: Tensor
    bond_features: Tensor
    mol_features: Tensor
    reaction_count: Tensor
    remaining_capacity: Tensor

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    @property
    def batch_size(self) -> int:
        return self.node_features.shape[0]

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    @classmethod
    def from_graphs(cls, graphs: list[GraphData]) -> GraphBatch:
        assert graphs
        return cls(
            node_features=torch.stack([graph.node_features for graph in graphs]),
            node_mask=torch.stack([graph.node_mask for graph in graphs]),
            adjacency=torch.stack([graph.adjacency for graph in graphs]),
            bond_features=torch.stack([graph.bond_features for graph in graphs]),
            mol_features=torch.stack([graph.mol_features for graph in graphs]),
            reaction_count=torch.tensor(
                [graph.reaction_count for graph in graphs], dtype=torch.long
            ),
            remaining_capacity=torch.tensor(
                [graph.remaining_capacity for graph in graphs], dtype=torch.float32
            ),
        )


def molecule_to_graph_data(
    mol: Chem.Mol | None,
    max_atoms: int,
    reaction_count: int,
    properties: np.ndarray | None = None,
) -> GraphData:
    atom_count = heavy_atom_count(mol)
    if atom_count > max_atoms:
        raise ValueError(
            f"molecule has {atom_count} heavy atoms, exceeding max_atoms={max_atoms}"
        )

    if mol is not None and sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()) > 1:
        raise ValueError("a synthesis state can have at most one dummy handle")
    capacity = max_atoms + 1
    node_features = np.zeros((capacity, NODE_FEATURE_DIM), dtype=np.float32)
    node_mask = np.zeros(capacity, dtype=np.bool_)
    adjacency = np.zeros((capacity, capacity), dtype=np.bool_)
    bond_features = np.zeros((capacity, capacity, BOND_FEATURE_DIM), dtype=np.float32)
    if mol is not None:
        atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
        index_map = {atom.GetIdx(): index for index, atom in enumerate(atoms)}
        # Fill plain arrays; creating/assigning a tiny Torch tensor per atom
        # and bond is substantially more work than wrapping each finished array.
        degree_start = len(ATOM_TYPES) + 1
        charge_start = degree_start + 7
        scalar_start = charge_start + 6
        synthon_start = scalar_start + 4
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
            node_mask[index] = True
        for bond in mol.GetBonds():
            # Explicit isotopic H atoms can survive RDKit's RemoveHs, but the
            # model representation only allocates heavy-atom and dummy slots.
            if (
                bond.GetBeginAtomIdx() not in index_map
                or bond.GetEndAtomIdx() not in index_map
            ):
                continue
            begin = index_map[bond.GetBeginAtomIdx()]
            end = index_map[bond.GetEndAtomIdx()]
            adjacency[begin, end] = adjacency[end, begin] = True
            feature = (
                bond.GetBondType() == Chem.BondType.SINGLE,
                bond.GetBondType() == Chem.BondType.DOUBLE,
                bond.GetBondType() == Chem.BondType.TRIPLE,
                bond.GetBondType() == Chem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing(),
                bond.GetStereo() != Chem.BondStereo.STEREONONE,
            )
            bond_features[begin, end] = bond_features[end, begin] = feature

    return GraphData(
        node_features=torch.from_numpy(node_features),
        node_mask=torch.from_numpy(node_mask),
        adjacency=torch.from_numpy(adjacency),
        bond_features=torch.from_numpy(bond_features),
        mol_features=torch.from_numpy(
            normalize_molecular_properties(
                molecular_properties(mol) if properties is None else properties
            )
        ),
        reaction_count=reaction_count,
        remaining_capacity=(max_atoms - atom_count) / max_atoms,
    )
