from pathlib import Path
import numpy as np
import torch
import torch_cluster
from Bio.PDB.PDBParser import PDBParser
from torch import Tensor


POCKET_NODE_DIM = 20
POCKET_EDGE_DIM = 32


@torch.no_grad()
def generate_protein_data(
    protein_path: str | Path,
    center: tuple[float, float, float],
    pocket_radius: float = 20,
    top_k: int = 20,
) -> dict[str, Tensor]:
    # load protein
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("protein", protein_path)
    res_list = list(s.get_residues())
    clean_res_list = get_clean_res_list(res_list, ensure_ca_exist=True)

    # get protein feature
    coords, seq, edge_index, edge_attr = get_protein_feature(clean_res_list, top_k)

    # get pocket feature
    center_t = torch.tensor(center, dtype=torch.float32).reshape(1, 3)
    distance = (coords - center_t).norm(dim=-1)
    node_mask = distance < pocket_radius
    masked_edge_index, masked_edge_attr = get_protein_edge_features_and_index(edge_index, edge_attr, node_mask)
    return dict(
        seq=seq[node_mask],
        pos=coords[node_mask],
        center=center_t,
        edge_index=masked_edge_index,
        edge_attr=masked_edge_attr,
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


def get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, keepNode):
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0).values
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = new_edge_inex[:, keepEdge].clone()
    input_protein_edge_s = protein_edge_s[keepEdge]
    return input_edge_idx, input_protein_edge_s


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


def get_protein_feature(res_list, top_k=30) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    data = featurize_protein(structure, top_k)
    return data["x"], data["seq"], data["edge_index"], data["edge_s"]


@torch.no_grad()
def featurize_protein(protein, top_k, num_positional_embeddings=16, num_rbf=16) -> dict[str, Tensor]:
    coords = torch.as_tensor(protein["coords"], dtype=torch.float32)
    seq = torch.as_tensor([letter_to_num[a] for a in protein["seq"]], dtype=torch.long)

    mask = torch.isfinite(coords.sum(dim=(1, 2)))
    coords[~mask] = torch.inf

    X_ca = coords[:, 1]
    edge_index = torch_cluster.knn_graph(X_ca, k=top_k)
    pos_embeddings = _positional_embeddings(edge_index, num_positional_embeddings)
    E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
    rbf_embeddings = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)
    edge_s = torch.cat([pos_embeddings, rbf_embeddings], dim=-1)
    return dict(
        x=X_ca,
        seq=seq,
        edge_s=torch.nan_to_num(edge_s),
        edge_index=edge_index,
        mask=mask,
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def _positional_embeddings(edge_index, num_embeddings):
    """From https://github.com/jingraham/neurips19-graph-protein-design"""
    num_embeddings = num_embeddings
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E
