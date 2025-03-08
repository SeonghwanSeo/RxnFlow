import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
from Bio.PDB.PDBParser import PDBParser
from torch_geometric.data import Data

__all__ = ["generate_protein_graph", "generate_protein_data"]


@torch.no_grad()
def generate_protein_graph(
    protein_path: str | Path,
    center: tuple[float, float, float],
    pocket_radius: float = 20,
) -> Data:
    data = generate_protein_data(protein_path, center, pocket_radius)
    return Data(**{k: torch.from_numpy(v) for k, v in data.items()})


@torch.no_grad()
def generate_protein_data(
    protein_path: str | Path,
    center: tuple[float, float, float],
    pocket_radius: float = 20,
) -> dict[str, torch.Tensor]:
    center_t = torch.as_tensor(center).reshape(1, 3)
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("protein", protein_path)
    res_list = list(s.get_residues())
    clean_res_list = get_clean_res_list(res_list, ensure_ca_exist=True)
    coords, seq, node_s, node_v, edge_index, edge_s, edge_v = get_protein_feature(clean_res_list)
    distance = (coords - center_t).norm(dim=-1)
    node_mask = distance < pocket_radius
    masked_edge_index, masked_edge_s, masked_edge_v = get_protein_edge_features_and_index(
        edge_index, edge_s, edge_v, node_mask
    )
    return dict(
        center=center_t.to(torch.float),
        coords=coords[node_mask].to(torch.float),
        node_s=node_s[node_mask].to(torch.float),
        node_v=node_v[node_mask].to(torch.float),
        seq=seq[node_mask].to(torch.long),
        edge_index=masked_edge_index.to(torch.long),
        edge_s=masked_edge_s.to(torch.float),
        edge_v=masked_edge_v.to(torch.float),
    )


three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == " ":
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ("CA" in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res["CA"].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode):
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0).values
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = new_edge_inex[:, keepEdge].clone()
    input_protein_edge_s = protein_edge_s[keepEdge]
    input_protein_edge_v = protein_edge_v[keepEdge]
    return input_edge_idx, input_protein_edge_s, input_protein_edge_v


letter_to_num = {
    "C": 4,
    "D": 3,
    "S": 15,
    "Q": 5,
    "K": 11,
    "I": 9,
    "P": 14,
    "T": 16,
    "F": 13,
    "A": 0,
    "G": 7,
    "H": 8,
    "E": 6,
    "L": 10,
    "R": 1,
    "W": 17,
    "V": 19,
    "N": 2,
    "Y": 18,
    "M": 12,
}
num_to_letter = {v: k for k, v in letter_to_num.items()}


def get_protein_feature(res_list):
    res_list = [res for res in res_list if (("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res))]
    structure = {}
    structure["name"] = "placeholder"
    structure["seq"] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res["N"], res["CA"], res["C"], res["O"]]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure["coords"] = coords
    torch.set_num_threads(1)  # this reduce the overhead, and speed up the process for me.
    protein = featurize_as_graph(structure)
    x = (
        protein.x,
        protein.seq,
        protein.node_s,
        protein.node_v,
        protein.edge_index,
        protein.edge_s,
        protein.edge_v,
    )
    return x


@torch.no_grad()
def featurize_as_graph(protein, num_positional_embeddings=16, top_k=30, num_rbf=16):
    coords = torch.as_tensor(protein["coords"], dtype=torch.float32)
    seq = torch.as_tensor([letter_to_num[a] for a in protein["seq"]], dtype=torch.long)

    mask = torch.isfinite(coords.sum(dim=(1, 2)))
    coords[~mask] = np.inf

    X_ca = coords[:, 1]
    edge_index = torch_cluster.knn_graph(X_ca, k=top_k)

    pos_embeddings = _positional_embeddings(edge_index, num_positional_embeddings)
    E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)

    dihedrals = _dihedrals(coords)
    orientations = _orientations(X_ca)
    sidechains = _sidechains(coords)

    node_s = dihedrals
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

    data = Data(
        x=X_ca,
        seq=seq,
        node_s=node_s,
        node_v=node_v,
        edge_s=edge_s,
        edge_v=edge_v,
        edge_index=edge_index,
        mask=mask,
    )
    return data


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def _dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design

    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index, num_embeddings):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n, dim=-1))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec
