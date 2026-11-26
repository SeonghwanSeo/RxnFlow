"""Condition and synthon encoders for the reaction policy."""

import torch
from torch import nn

from rxnflow.envs.features import FINGERPRINT_DIM, PROPERTY_DIM, PROPERTY_SCALE

from .nn import Linear


class GFNConditionEncoder(nn.Module):
    """Embed only the GFlowNet inputs used by this policy."""

    def __init__(
        self,
        num_objectives: int,
        hidden_dim: int,
        use_cond_beta: bool = False,
        use_cond_prior: bool = False,
        use_cond_preference: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_beta = None
        if use_cond_beta:
            self.emb_beta = nn.Sequential(
                Linear(1, hidden_dim),
                nn.SiLU(),
                Linear(hidden_dim, hidden_dim, init="zero"),
            )
        self.emb_prior = None
        if use_cond_prior:
            self.emb_prior = nn.Sequential(
                Linear(1, hidden_dim),
                nn.SiLU(),
                Linear(hidden_dim, hidden_dim, init="zero"),
            )
        self.emb_pref = None
        if use_cond_preference:
            self.emb_pref = nn.Sequential(
                Linear(num_objectives, hidden_dim),
                nn.SiLU(),
                Linear(hidden_dim, hidden_dim, init="zero"),
            )

    def forward(
        self,
        beta: torch.Tensor,
        preference: torch.Tensor,
        prior_strength: torch.Tensor,
    ) -> torch.Tensor:
        embedding = beta.new_zeros((len(beta), self.hidden_dim))
        if self.emb_beta is not None:
            embedding = self.emb_beta(beta[:, None])
        if self.emb_prior is not None:
            embedding = embedding + self.emb_prior(prior_strength[:, None])
        if self.emb_pref is not None:
            embedding = embedding + self.emb_pref(preference)
        return embedding


class ExternalConditionEncoder(nn.Module):
    """Encode external conditions, e.g., a target protein embedding."""

    def __init__(self, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.emb_cond = nn.Sequential(
            Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            Linear(hidden_dim, hidden_dim),
        )

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
        nn.init.uniform_(self.emb_type.weight, a=-0.1, b=0.1)
        self.lin_fp = Linear(FINGERPRINT_DIM, hidden_dim)
        self.lin_prop = Linear(PROPERTY_DIM, hidden_dim)
        # Normalize in the fusion MLP after the separate linear projections.
        self.mlp = nn.Sequential()
        in_dim = 4 * hidden_dim
        for _ in range(num_layers - 1):
            self.mlp.extend(
                [
                    Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = hidden_dim
        self.mlp.append(Linear(in_dim, hidden_dim))

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
