"""Fixed-shape molecular graph construction for model input.

Every ``GraphData`` has length ``L = Config.data.max_atoms + 1``: heavy atoms
plus one reserved dummy-handle slot. RDKit excludes dummies from its heavy-atom
count, and a nonterminal linear synthesis state has exactly one. Batching only
stacks those tensors, producing nodes ``[B, L, node_dim]``, adjacency
``[B, L, L]``, and bonds ``[B, L, L, bond_dim]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem
from torch import Tensor

from rxnflow.envs.chemistry.features import (
    heavy_atom_count,
    molecular_properties,
    normalize_molecular_properties,
    parse_molecule,
)

# NOTE: For general/common usage, we allocate 100 atom & synthon types,
# which is more than enough for most practical applications.
ATOM_TYPES = list(range(100))
SYNTHON_TYPES = list(range(100))
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


def _one_hot(value: int, choices: list[int]) -> list[float]:
    result = [0.0] * (len(choices) + 1)
    result[choices.index(value) if value in choices else -1] = 1.0
    return result


def _atom_features(atom: Chem.Atom) -> Tensor:
    degree = min(atom.GetDegree(), 6)
    charge = max(-2, min(3, atom.GetFormalCharge()))
    values = _one_hot(atom.GetAtomicNum(), ATOM_TYPES)
    values.extend([float(degree == index) for index in range(7)])
    values.extend([float(charge == index) for index in range(-2, 4)])
    values.extend(
        [
            float(atom.GetIsAromatic()),
            atom.GetMass() / 100.0 if atom.GetAtomicNum() else 0.0,
            atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
            min(int(atom.GetHybridization()) / 8.0, 1.0),
        ]
    )
    values.extend(
        _one_hot(atom.GetIsotope() if atom.GetAtomicNum() == 0 else 0, SYNTHON_TYPES)
    )
    return torch.tensor(values, dtype=torch.float32)


def _bond_features(bond: Chem.Bond) -> Tensor:
    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    values = [float(bond.GetBondType() == bond_type) for bond_type in bond_types]
    values.extend(
        [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
        ]
    )
    return torch.tensor(values, dtype=torch.float32)


def molecule_to_graph_data(
    smiles: str,
    max_atoms: int,
    reaction_count: int,
) -> GraphData:
    mol = parse_molecule(smiles)
    if smiles and mol is None:
        raise ValueError(f"invalid graph SMILES: {smiles}")
    atom_count = heavy_atom_count(mol)
    if atom_count > max_atoms:
        raise ValueError(
            f"molecule has {atom_count} heavy atoms, exceeding max_atoms={max_atoms}"
        )

    if mol is not None and sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()) > 1:
        raise ValueError("a synthesis state can have at most one dummy handle")
    capacity = max_atoms + 1
    node_features = torch.zeros((capacity, NODE_FEATURE_DIM), dtype=torch.float32)
    node_mask = torch.zeros(capacity, dtype=torch.bool)
    adjacency = torch.zeros((capacity, capacity), dtype=torch.bool)
    bond_features = torch.zeros(
        (capacity, capacity, BOND_FEATURE_DIM), dtype=torch.float32
    )
    if mol is not None:
        atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
        index_map = {atom.GetIdx(): index for index, atom in enumerate(atoms)}
        for index, atom in enumerate(atoms):
            node_features[index] = _atom_features(atom)
            node_mask[index] = True
        for bond in mol.GetBonds():
            begin = index_map[bond.GetBeginAtomIdx()]
            end = index_map[bond.GetEndAtomIdx()]
            adjacency[begin, end] = adjacency[end, begin] = True
            feature = _bond_features(bond)
            bond_features[begin, end] = bond_features[end, begin] = feature

    return GraphData(
        node_features=node_features,
        node_mask=node_mask,
        adjacency=adjacency,
        bond_features=bond_features,
        mol_features=torch.from_numpy(
            normalize_molecular_properties(molecular_properties(mol))
        ),
        reaction_count=reaction_count,
        remaining_capacity=(max_atoms - atom_count) / max_atoms,
    )
