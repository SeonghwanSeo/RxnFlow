"""Condition and synthon encoders; weights are initialized by RxnFlowModel."""

import torch
from torch import nn

from rxnflow.envs.features import FINGERPRINT_DIM, PROPERTY_DIM, PROPERTY_SCALE

from .nn import mlp


class GFNConditionEncoder(nn.Module):
    """Encode GFlowNet conditions, i.e., temperature and preference vector."""

    def __init__(self, num_objectives: int, hidden_dim: int) -> None:
        super().__init__()
        self.emb_temp = mlp(1, hidden_dim, hidden_dim, 2)
        self.emb_pref = mlp(num_objectives, hidden_dim, hidden_dim, 2)

    def forward(
        self,
        temperature: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        """temperature: [B], preference: [B, num_objectives]; return [B, H]."""
        return self.emb_temp(temperature[:, None]) + self.emb_pref(preference)


class ExternalConditionEncoder(nn.Module):
    """Encode external conditions, e.g., a target protein embedding."""

    def __init__(self, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.emb_cond = mlp(condition_dim, hidden_dim, hidden_dim, 2)

    def forward(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """condition: [B, condition_dim]; return [B, hidden_dim]."""
        return self.emb_cond(condition)


class SynthonEncoder(nn.Module):
    """Fuse fingerprints, properties, and attachment/remaining type embeddings."""

    def __init__(self, num_types: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.register_buffer(
            "property_scale", torch.tensor(PROPERTY_SCALE), persistent=False
        )
        # The two positions share a vocabulary, but retain their ordered roles.
        self.emb_type = nn.Embedding(num_types, hidden_dim)
        self.lin_fp = nn.Linear(FINGERPRINT_DIM, hidden_dim)
        self.lin_prop = nn.Linear(PROPERTY_DIM, hidden_dim)
        # Normalize in the fusion MLP after the separate linear projections.
        self.mlp = mlp(4 * hidden_dim, hidden_dim, hidden_dim, num_layers, layernorm=True)

    def forward(
        self, fp: torch.Tensor, prop: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """fp: [N, F], prop: [N, P], types: [N, 2]; return [N, S].

        F/P are fingerprint/property dimensions; S is the synthon embedding size.
        prop contains raw descriptors in PROPERTY_NAMES order.
        """
        prop = prop / self.property_scale
        type_emb = self.emb_type(types).flatten(start_dim=1)
        return self.mlp(
            torch.cat(
                [
                    self.lin_fp(fp),
                    self.lin_prop(prop),
                    type_emb,
                ],
                dim=-1,
            )
        )
