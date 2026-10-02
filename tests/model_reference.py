"""Scalar reference calculations for checking batched policy scores and gradients."""

import torch
from torch.nn import functional as F

from rxnflow.core.types import ActionType


def get_unirxn_logits(
    model, graph_emb: torch.Tensor, action_name: str, logit_scale: torch.Tensor
) -> torch.Tensor:
    action_type = (
        ActionType.UNIRXN_TERMINAL
        if model.env.uni_reactions[action_name].output_type is None
        else ActionType.UNIRXN_TRANSFORM
    )
    return model.forward_mdp(graph_emb, action_name, logit_scale, action_type)[0, 0]


def get_synthon_emb(
    model, library_name: str, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    library = model.env.synthons[library_name]
    cpu_indices = indices.detach().cpu().to(torch.long).numpy()
    fp = torch.from_numpy(library.fingerprints[cpu_indices]).to(
        device, dtype=torch.float32
    )
    prop = torch.from_numpy(library.properties[cpu_indices]).to(device)
    type_index = model.env.library_to_index[library_name]
    library_indices = torch.full(
        (len(indices),), type_index, dtype=torch.long, device=device
    )
    return model.synthon_embedding(fp, prop, library_indices)


def get_synthon_logits(
    model,
    graph_emb: torch.Tensor,
    action_name: str,
    library_name: str,
    indices: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    assert indices.ndim == 1 and graph_emb.shape[0] == 1
    action_type = (
        ActionType.FIRST_SYNTHON
        if action_name == "first_synthon"
        else ActionType.BIRXN_BRICK
        if model.env.synthons[library_name].is_brick
        else ActionType.BIRXN_LINKER
    )
    state_emb = model.forward_mdp(graph_emb, action_name, logit_scale, action_type)
    synthon_emb = get_synthon_emb(model, library_name, indices, graph_emb.device)
    return F.normalize(synthon_emb, dim=-1) @ state_emb.squeeze(0)
