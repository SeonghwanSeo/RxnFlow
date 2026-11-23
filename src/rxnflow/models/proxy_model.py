"""Atom MPNN for scalar molecular properties."""

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

MAX_NEIGHBORS = 6
NODE_DIM = 69
EDGE_DIM = 13


def two_layer_mlp(d_in: int, d_hid: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_out))


class MPNNConv(nn.Module):
    def __init__(self, dim: int = 256, p_dropout: float = 0.0):
        super().__init__()
        self.mlp_msg = two_layer_mlp(2 * dim, dim, dim)
        self.mlp_upd = two_layer_mlp(2 * dim, dim, dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        h0 = h

        batch_index = torch.arange(h.shape[0], device=h.device)[:, None, None]
        hij = h[batch_index, edge_index]  # [B, N, K, D]
        msg = self.mlp_msg(torch.cat([hij, e], dim=-1))  # [B, N, K, D]
        msg = msg * edge_mask[..., None]
        aggr = msg.sum(dim=-2)  # [B, N, D]

        h_upd = h0 + self.dropout(self.mlp_upd(torch.cat([h, aggr], dim=-1)))  # [B, N, D]
        h_upd = h_upd * node_mask[..., None]

        return h_upd


class MPNNModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        model_dim: int = 256,
        depth: int = 3,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, model_dim)
        self.edge_emb = nn.Linear(edge_dim, model_dim)
        self.layers = nn.ModuleList(
            [MPNNConv(model_dim, p_dropout) for _ in range(depth)]
        )
        self.head = two_layer_mlp(model_dim, model_dim, 1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        node_mask = batch["node_mask"].float()
        edge_mask = batch["edge_mask"].float()
        h = self.node_emb(batch["h"])
        e = self.edge_emb(batch["e"])
        for layer in self.layers:
            h = layer(h, e, batch["edge_index"], node_mask, edge_mask)
        h = h * node_mask[..., None]
        pooled = h.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(pooled).squeeze(-1)


class MolProxy(nn.Module):
    def __init__(self, model_dim: int = 256, depth: int = 3, p_dropout: float = 0.0):
        super().__init__()
        self.mpnn = MPNNModel(NODE_DIM, EDGE_DIM, model_dim, depth, p_dropout)
        self.register_buffer("score_mean", torch.tensor(0.0))
        self.register_buffer("score_std", torch.tensor(1.0))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def predict(self, mol: str | Chem.Mol) -> float:
        return self.predict_batch([mol])[0].item()

    @torch.no_grad()
    def predict_batch(self, mols: list[str | Chem.Mol]) -> np.ndarray:
        """Predict in the current mode; call eval() for inference."""
        if not mols:
            return np.empty(0, dtype=np.float32)
        graphs = [encode_mol(mol) for mol in mols]
        batch = collate(graphs)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        y = self(batch)
        return y.cpu().numpy()

    @torch.no_grad()
    def screening(self, smiles_list: list[str], batch_size: int = 64) -> np.ndarray:
        """Batch prepared SMILES and return scores in input order; call eval() first."""
        order = sorted(
            range(len(smiles_list)),
            key=lambda i: (len(smiles_list[i]), smiles_list[i]),
            reverse=True,
        )
        scores = np.empty(len(smiles_list), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            scores[indices] = self.predict_batch([smiles_list[i] for i in indices])
        return scores

    def forward(
        self, batch: dict[str, torch.Tensor], normalized: bool = False
    ) -> torch.Tensor:
        prediction = self.mpnn(batch)
        return prediction if normalized else prediction * self.score_std + self.score_mean


_ATOM_CHOICES = (
    (*range(1, 37), 53),
    (0, 1, 2, 3, 4, 5),
    (-2, -1, 0, 1, 2),
    (0, 1, 2, 3, 4),
    (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ),
    (
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ),
)
_ATOM_LOOKUPS = tuple(
    {value: i for i, value in enumerate(choices)} for choices in _ATOM_CHOICES
)
_BOND_TYPES = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
_BOND_STEREO = {
    Chem.BondStereo.STEREONONE: 0,
    Chem.BondStereo.STEREOANY: 1,
    Chem.BondStereo.STEREOZ: 2,
    Chem.BondStereo.STEREOE: 3,
    Chem.BondStereo.STEREOCIS: 4,
    Chem.BondStereo.STEREOTRANS: 5,
}


def encode_mol(mol: str | Chem.Mol) -> dict[str, torch.Tensor]:
    """Encode parsed atoms, retaining hydrogens left by MolFromSmiles."""

    def encode_atom(atom: Chem.Atom, out: np.ndarray) -> np.ndarray:
        values = (
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetHybridization(),
            atom.GetChiralTag(),
        )
        offset = 0
        for value, lookup in zip(values, _ATOM_LOOKUPS, strict=True):
            out[offset + lookup.get(value, len(lookup))] = 1
            offset += len(lookup) + 1
        out[offset : offset + 2] = [atom.GetIsAromatic(), atom.GetMass() * 0.01]
        return out

    def encode_bond(bond: Chem.Bond, out: np.ndarray) -> np.ndarray:
        bond_type = bond.GetBondType()
        if bond_type not in _BOND_TYPES:
            raise ValueError(f"Unknown bond type: {bond_type}")
        out[_BOND_TYPES[bond_type]] = 1
        out[4 + _BOND_STEREO.get(bond.GetStereo(), 6)] = 1
        out[11:13] = [bond.GetIsConjugated(), bond.IsInRing()]
        return out

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        raise ValueError("Invalid molecule")
    atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
    if not atoms or any(atom.GetAtomicNum() == 0 for atom in atoms):
        raise ValueError("Expected a nonempty molecule without dummy atoms")

    n = len(atoms)
    h = np.zeros((n, NODE_DIM), dtype=np.float32)
    e = np.zeros((n, MAX_NEIGHBORS, EDGE_DIM), dtype=np.float32)
    edge_index = np.full((n, MAX_NEIGHBORS), -1, dtype=np.int64)

    # Encode atoms
    for i, atom in enumerate(atoms):
        encode_atom(atom, out=h[i])

    # Encode bonds
    counts = [0] * n
    for bond_index in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(bond_index)
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        si, sj = counts[i], counts[j]
        if si >= MAX_NEIGHBORS or sj >= MAX_NEIGHBORS:
            target = i if si >= MAX_NEIGHBORS else j
            raise ValueError(f"Atom {target} has more than {MAX_NEIGHBORS} neighbors")
        encode_bond(bond, out=e[i, si])
        e[j, sj] = e[i, si]
        edge_index[i, si], edge_index[j, sj] = j, i
        counts[i], counts[j] = si + 1, sj + 1

    edge_mask = edge_index >= 0
    edge_index[~edge_mask] = 0  # avoid negative indices for padding

    return {
        "h": torch.from_numpy(h),
        "e": torch.from_numpy(e),
        "edge_index": torch.from_numpy(edge_index),
        "edge_mask": torch.from_numpy(edge_mask),
    }


def collate(graphs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad graphs on CPU and add a node mask."""
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    n = max(graph["h"].shape[0] for graph in graphs)
    b = len(graphs)
    batch = {
        "h": torch.zeros(b, n, NODE_DIM),
        "e": torch.zeros(b, n, MAX_NEIGHBORS, EDGE_DIM),
        "edge_index": torch.zeros(b, n, MAX_NEIGHBORS, dtype=torch.long),
        "node_mask": torch.zeros(b, n, dtype=torch.bool),
        "edge_mask": torch.zeros(b, n, MAX_NEIGHBORS, dtype=torch.bool),
    }
    for i, graph in enumerate(graphs):
        count = graph["h"].shape[0]
        for key in ("h", "e", "edge_index", "edge_mask"):
            batch[key][i, :count] = graph[key]
        batch["node_mask"][i, :count] = True
    return batch
