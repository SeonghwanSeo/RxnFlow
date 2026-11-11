"""Molecular graph representation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from rxnflow.core.molecule import Molecule
from rxnflow.envs.features import (
    EDGE_DIM,
    NODE_DIM,
    PROPERTY_DIM,
    get_atom_features,
    get_bond_features,
    get_mol_features,
    heavy_atom_count,
)

GRAPH_DIM = PROPERTY_DIM

MAX_NEIGHBORS = 6


@dataclass
class GraphData:
    node_features: np.ndarray  # [N, D_node]
    edge_features: np.ndarray  # [N, K, D_edge]
    graph_features: np.ndarray  # [D_graph]
    neighbor_index: np.ndarray  # [N, K], local source indices for each target atom
    neighbor_mask: np.ndarray  # [N, K]
    num_heavy_atoms: int  # Scalar, excluding dummy handles (attachment points)

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @classmethod
    def empty(cls) -> GraphData:
        return cls(
            node_features=np.zeros((0, NODE_DIM), dtype=np.float32),
            edge_features=np.zeros((0, MAX_NEIGHBORS, EDGE_DIM), dtype=np.float32),
            graph_features=np.zeros(GRAPH_DIM, dtype=np.float32),
            neighbor_index=np.zeros((0, MAX_NEIGHBORS), dtype=np.int64),
            neighbor_mask=np.zeros((0, MAX_NEIGHBORS), dtype=np.bool_),
            num_heavy_atoms=0,
        )


@dataclass
class GraphBatch:
    node_features: Tensor  # [B, N, D_node]
    edge_features: Tensor  # [B, N, K, D_edge]
    graph_features: Tensor  # [B, D_graph]
    neighbor_index: Tensor  # [B, N, K]
    neighbor_mask: Tensor  # [B, N, K]
    node_mask: Tensor  # [B, N]
    num_nodes: Tensor  # [B]
    num_heavy_atoms: Tensor  # [B]

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
    def from_list(cls, graphs: list[GraphData]) -> GraphBatch:
        assert graphs
        B = len(graphs)
        num_nodes = np.array([graph.num_nodes for graph in graphs])
        N_max = num_nodes.max()
        node_mask = np.arange(N_max)[None, :] < num_nodes[:, None]

        neighbor_shape = (B, N_max, MAX_NEIGHBORS)
        node_features = np.zeros((B, N_max, NODE_DIM), dtype=np.float32)
        edge_features = np.zeros((*neighbor_shape, EDGE_DIM), dtype=np.float32)
        neighbor_index = np.zeros(neighbor_shape, dtype=np.int64)
        neighbor_mask = np.zeros(neighbor_shape, dtype=np.bool_)
        for i, g in enumerate(graphs):
            count = num_nodes[i]
            node_features[i, :count] = g.node_features
            edge_features[i, :count] = g.edge_features
            neighbor_index[i, :count] = g.neighbor_index
            neighbor_mask[i, :count] = g.neighbor_mask

        graph_features = np.stack([graph.graph_features for graph in graphs])
        num_heavy_atoms = np.array(
            [graph.num_heavy_atoms for graph in graphs], dtype=np.int64
        )

        return cls(
            node_features=torch.from_numpy(node_features),
            edge_features=torch.from_numpy(edge_features),
            graph_features=torch.from_numpy(graph_features),
            neighbor_index=torch.from_numpy(neighbor_index),
            neighbor_mask=torch.from_numpy(neighbor_mask),
            node_mask=torch.from_numpy(node_mask),
            num_nodes=torch.from_numpy(num_nodes),
            num_heavy_atoms=torch.from_numpy(num_heavy_atoms),
        )


def molecule_to_graph_data(molecule: Molecule | None) -> GraphData:
    """Encode one state without truncating atoms or including explicit hydrogens."""
    if molecule is None:
        return GraphData.empty()
    rdmol = molecule.rdmol

    atoms = [atom for atom in rdmol.GetAtoms() if atom.GetAtomicNum() != 1]
    if sum(atom.GetAtomicNum() == 0 for atom in atoms) > 1:
        raise ValueError("States can have at most one attachment point (dummy atom)")

    # Allocate the graph for heavy atoms and the dummy handle.
    num_nodes = len(atoms)
    node_features = np.empty((num_nodes, NODE_DIM), dtype=np.float32)
    neighbor_shape = (num_nodes, MAX_NEIGHBORS)
    edge_features = np.zeros((*neighbor_shape, EDGE_DIM), dtype=np.float32)
    neighbor_index = np.zeros(neighbor_shape, dtype=np.int64)
    neighbor_mask = np.zeros(neighbor_shape, dtype=np.bool_)

    # Fill node features.
    for index, atom in enumerate(atoms):
        get_atom_features(atom, out=node_features[index])

    # Fill edge features and neighbor indices.
    index_map = {atom.GetIdx(): index for index, atom in enumerate(atoms)}
    next_slot = [0] * num_nodes
    for bond in rdmol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Explicit isotopic H atoms can survive RDKit's RemoveHs, but the
        # model representation only allocates heavy-atom and dummy slots.
        if begin_idx not in index_map or end_idx not in index_map:
            continue
        begin, end = index_map[begin_idx], index_map[end_idx]
        begin_slot, end_slot = next_slot[begin], next_slot[end]
        if begin_slot == MAX_NEIGHBORS or end_slot == MAX_NEIGHBORS:
            warnings.warn(
                f"Atom {begin} or {end} has more than {MAX_NEIGHBORS} neighbors;"
                "truncating the graph representation",
                stacklevel=2,
            )
            continue

        feature = get_bond_features(bond, out=edge_features[begin, begin_slot])
        edge_features[end, end_slot] = feature
        neighbor_index[begin, begin_slot] = end
        neighbor_index[end, end_slot] = begin
        neighbor_mask[begin, begin_slot] = True
        neighbor_mask[end, end_slot] = True
        next_slot[begin] += 1
        next_slot[end] += 1

    # Compute graph features
    graph_features = get_mol_features(rdmol)
    num_heavy_atoms = heavy_atom_count(rdmol)

    return GraphData(
        node_features=node_features,
        edge_features=edge_features,
        graph_features=graph_features,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
        num_heavy_atoms=num_heavy_atoms,
    )
