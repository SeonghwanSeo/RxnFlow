"""Condition and synthon encoders; weights are initialized by RxnFlowModel."""

import math

import torch
from torch import nn

from rxnflow.envs.features import FINGERPRINT_DIM, PROPERTY_DIM, PROPERTY_SCALE

from .nn import mlp


class ConditionEncoder(nn.Module):
    """Encode beta and objective weights, including fixed equal weights."""

    def __init__(self, num_emb: int, num_objectives: int) -> None:
        super().__init__()
        # Normalized beta plus four sine/cosine pairs.
        self.emb_beta = mlp(9, num_emb, num_emb, 2)
        self.emb_preferences = mlp(num_objectives, num_emb, num_emb, 2)

    def forward(self, beta: torch.Tensor, preferences: torch.Tensor) -> torch.Tensor:
        """beta: [B], preferences: [B, num_objectives]; return [B, H]."""
        # Fixed coordinates are independent of the sampling range. The linear
        # term u distinguishes values that share the same periodic features.
        u = (beta[:, None] - 1.0) / 63.0
        frequencies = u.new_tensor((1.0, 2.0, 4.0, 8.0))
        angles = 2 * math.pi * u * frequencies
        features = torch.cat([u, angles.sin(), angles.cos()], dim=-1)
        return self.emb_beta(features) + self.emb_preferences(preferences)


class SynthonEncoder(nn.Module):
    """Fuse fingerprints, properties, and attachment/remaining type embeddings."""

    def __init__(self, num_types: int, num_emb: int, num_layers: int) -> None:
        super().__init__()
        self.register_buffer(
            "property_scale", torch.tensor(PROPERTY_SCALE), persistent=False
        )
        # The two positions share a vocabulary, but retain their ordered roles.
        self.emb_type = nn.Embedding(num_types, num_emb)
        self.lin_fp = nn.Linear(FINGERPRINT_DIM, num_emb)
        self.lin_prop = nn.Linear(PROPERTY_DIM, num_emb)
        # Normalize in the fusion MLP after the separate linear projections.
        self.mlp = mlp(4 * num_emb, num_emb, num_emb, num_layers, layernorm=True)

    def forward(
        self, fp: torch.Tensor, prop: torch.Tensor, site_indices: torch.Tensor
    ) -> torch.Tensor:
        """fp: [N, F], prop: [N, P], site_indices: [N, 2]; return [N, S].

        F/P are fingerprint/property dimensions; S is the synthon embedding size.
        """
        prop = prop / self.property_scale
        return self.mlp(
            torch.cat(
                [
                    self.lin_fp(fp),
                    self.lin_prop(prop),
                    self.emb_type(site_indices[:, 0]),
                    self.emb_type(site_indices[:, 1]),
                ],
                dim=-1,
            )
        )
