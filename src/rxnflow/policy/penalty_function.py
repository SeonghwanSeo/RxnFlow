"""Action penalty functions applied before policy scoring."""

from __future__ import annotations

import torch
from torch import Tensor

from rxnflow.envs.building_block import BlockLibrary


def block_penalty(
    library: BlockLibrary,
    indices: Tensor,
    current_heavy_atoms: int,
    max_heavy_atoms: int,
    current_mol_features: Tensor | None = None,
    mol_feature_limits: dict[int, float] | None = None,
) -> Tensor:
    """Evaluate the hard action penalty ``Omega (Ω)`` for sampled block indices.

    Feasible actions receive zero and infeasible actions receive negative
    infinity, so the result can be added to policy logits. The runtime uses the
    same values to avoid scoring hard-penalized blocks. The synthesis
    environment separately validates the exact reaction product because atom
    loss is not represented by this conservative pre-reaction calculation.
    """

    indices = indices.to(dtype=library.heavy_atoms.dtype, device="cpu")
    feasible = current_heavy_atoms + library.heavy_atoms[indices] <= max_heavy_atoms
    if current_mol_features is not None and mol_feature_limits:
        for feature_index, limit in mol_feature_limits.items():
            feasible &= (
                current_mol_features[feature_index]
                + library.properties[indices, feature_index]
                <= limit
            )
    penalty = torch.zeros(len(indices), dtype=torch.float32)
    return penalty.masked_fill(~feasible, -torch.inf)
