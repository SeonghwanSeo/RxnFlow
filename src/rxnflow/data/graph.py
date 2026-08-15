"""Fixed-shape molecular graph construction for model input.

Every ``GraphData`` has length ``L = Config.data.max_atoms``. Batching only
stacks those tensors, producing nodes ``[B, L, node_dim]``, adjacency
``[B, L, L]``, and bonds ``[B, L, L, bond_dim]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem
from torch import Tensor

from rxnflow.chemistry import (
    heavy_atom_count,
    molecular_properties,
    normalize_molecular_properties,
    parse_molecule,
)

ATOM_TYPES = [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 85]
NODE_FEATURE_DIM = len(ATOM_TYPES) + 1 + 7 + 6 + 5
BOND_FEATURE_DIM = 8


@dataclass
class GraphData:
    node_features: Tensor
    node_mask: Tensor
    adjacency: Tensor
    bond_features: Tensor
    mol_features: Tensor
    workflow_index: int
    protocol_order: int
    action_kind: int
    remaining_capacity: float


@dataclass
class GraphBatch:
    node_features: Tensor
    node_mask: Tensor
    adjacency: Tensor
    bond_features: Tensor
    mol_features: Tensor
    workflow_index: Tensor
    protocol_order: Tensor
    action_kind: Tensor
    remaining_capacity: Tensor

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
            workflow_index=torch.tensor(
                [graph.workflow_index for graph in graphs], dtype=torch.long
            ),
            protocol_order=torch.tensor(
                [graph.protocol_order for graph in graphs], dtype=torch.long
            ),
            action_kind=torch.tensor(
                [graph.action_kind for graph in graphs], dtype=torch.long
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
            min(atom.GetMass() / 200.0, 2.0),
            min(atom.GetTotalNumHs(includeNeighbors=True) / 4.0, 1.0),
            min(atom.GetIsotope() / 200.0, 1.0),
            min(int(atom.GetHybridization()) / 8.0, 1.0),
        ]
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
            1.0,
        ]
    )
    return torch.tensor(values, dtype=torch.float32)


def molecule_to_graph_data(
    smiles: str,
    max_atoms: int,
    workflow_index: int,
    protocol_order: int,
    action_kind: int,
) -> GraphData:
    mol = parse_molecule(smiles)
    atom_count = heavy_atom_count(mol)
    if atom_count > max_atoms:
        raise ValueError(
            f"molecule has {atom_count} heavy atoms, exceeding max_atoms={max_atoms}"
        )

    node_features = torch.zeros((max_atoms, NODE_FEATURE_DIM), dtype=torch.float32)
    node_mask = torch.zeros(max_atoms, dtype=torch.bool)
    adjacency = torch.zeros((max_atoms, max_atoms), dtype=torch.bool)
    bond_features = torch.zeros(
        (max_atoms, max_atoms, BOND_FEATURE_DIM), dtype=torch.float32
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
        mol_features=normalize_molecular_properties(molecular_properties(mol)),
        workflow_index=workflow_index,
        protocol_order=protocol_order,
        action_kind=action_kind,
        remaining_capacity=(max_atoms - atom_count) / max_atoms,
    )
