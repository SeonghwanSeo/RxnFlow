"""RxnFlow dynamic reaction policy and state-flow model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rxnflow.config import ModelConfig
from rxnflow.core.types import ActionType
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.graph import EDGE_DIM, GRAPH_DIM, NODE_DIM, GraphBatch

from .encoders import ExternalConditionEncoder, GFNConditionEncoder, SynthonEncoder
from .mpnn import MPNN
from .nn import Linear


class RxnFlowModel(nn.Module):
    """RxnFlow model"""

    def __init__(
        self,
        env: SynthesisEnv,
        cfg: ModelConfig,
        num_objectives: int = 1,
        condition_dim: int | None = None,
        *,
        use_cond_beta: bool = False,
        use_cond_prior: bool = False,
        use_cond_preference: bool = False,
    ):
        super().__init__()
        self.num_objectives: int = num_objectives
        self.condition_dim: int | None = condition_dim

        self.use_cond_beta = use_cond_beta
        self.use_cond_prior = use_cond_prior
        self.use_cond_preference = use_cond_preference

        # State encoder
        self.state_encoder = MPNN(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            graph_dim=GRAPH_DIM,
            model_dim=cfg.state_dim,
            num_layers=cfg.num_state_layers,
        )

        # GFN condition encoders
        self.gfn_condition_encoder = GFNConditionEncoder(
            num_objectives,
            cfg.hidden_dim,
            use_cond_beta=use_cond_beta,
            use_cond_prior=use_cond_prior,
            use_cond_preference=use_cond_preference,
        )
        # External condition encoder
        if condition_dim is not None:
            self.condition_encoder = ExternalConditionEncoder(
                condition_dim, cfg.hidden_dim
            )
        else:
            self.condition_encoder = None
        cond_dim = 2 * cfg.hidden_dim

        state_dim = cfg.state_dim + cond_dim

        # Reaction name embeddings
        reaction_names = list(env.uni_reactions.keys()) + list(env.bi_reactions.keys())
        self.rxn_embedding = nn.ParameterDict(
            (name, nn.Parameter(torch.empty(cfg.hidden_dim))) for name in reaction_names
        )
        for embedding in self.rxn_embedding.values():
            nn.init.uniform_(embedding, a=-0.1, b=0.1)
        rxn_dim = cfg.hidden_dim

        # Synthon encoder
        self.synthon_encoder = SynthonEncoder(
            num_types=len(env.synthon_types) + 1,
            hidden_dim=cfg.synthon_dim,
            num_layers=cfg.num_synthon_layers,
        )
        synthon_dim = cfg.synthon_dim

        # Action heads
        hidden_dim = cfg.hidden_dim

        action_dims = {
            ActionType.FIRST_SYNTHON: (state_dim, synthon_dim),
            ActionType.UNIRXN_TRANSFORM: (state_dim + rxn_dim, 1),
            ActionType.UNIRXN_TERMINAL: (state_dim + rxn_dim, 1),
            ActionType.BIRXN_BRICK: (state_dim + rxn_dim, synthon_dim),
            ActionType.BIRXN_LINKER: (state_dim + rxn_dim, synthon_dim),
        }
        self.action_heads = nn.ModuleDict()
        for action, (in_dim, out_dim) in action_dims.items():
            head = nn.Sequential()
            for _ in range(cfg.num_action_layers - 1):
                head.append(Linear(in_dim, hidden_dim))
                head.append(nn.LayerNorm(hidden_dim))
                head.append(nn.LeakyReLU())
                in_dim = hidden_dim
            head.append(Linear(in_dim, out_dim, init="xavier_small"))
            self.action_heads[action.name] = head

        # Logit scaling (LogitGFN)
        self._logit_scale = nn.Sequential(
            Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(),
            Linear(hidden_dim, 1, init="zero"),
        )

        # Trajectory-balance (TB)
        self._logZ = nn.Sequential(
            Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(),
            Linear(hidden_dim, 1, init="zero"),
        )

    def logit_scale(self, cond: torch.Tensor) -> torch.Tensor:
        r"""[B, 2*hidden_dim] → [B] \in [0, ∞)."""
        return F.elu(self._logit_scale(cond)).squeeze(-1) + 1

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        """Predict logZ from the full GFN and external condition embedding."""
        return self._logZ(cond)

    def encode_cond(
        self,
        beta: torch.Tensor,
        preference: torch.Tensor,
        prior_strength: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode beta, preference, prior strength and optional external conditions.

        Return [B, 2*hidden_dim], with a zero external embedding when no external
        condition encoder is configured.
        """
        gfn_cond_emb = self.gfn_condition_encoder(beta, preference, prior_strength)
        if self.condition_encoder is not None:
            assert cond is not None, "external conditions must be provided"
            ext_cond_emb = self.condition_encoder(cond)
        else:
            ext_cond_emb = torch.zeros_like(gfn_cond_emb)
        return torch.cat([gfn_cond_emb, ext_cond_emb], dim=-1)

    def encode_state(self, batch: GraphBatch) -> torch.Tensor:
        """Encode a molecule graphs into state embeddings."""
        return self.state_encoder(batch)

    def encode_synthon(
        self,
        fp: torch.Tensor,
        prop: torch.Tensor,
        types: torch.Tensor,
    ) -> torch.Tensor:
        """fp: [n, d_fp], prop: [n, d_prop], types: [n, 2]
        return [n, d_synthon]."""
        return self.synthon_encoder(fp, prop, types)

    def forward_first_synthon(
        self,
        state_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Map conditioned state embeddings to first-synthon queries."""
        head = self.action_heads[ActionType.FIRST_SYNTHON.name]
        return head(state_emb)

    def forward_unirxn(
        self,
        state_emb: torch.Tensor,
        action_name: str,
        action_type: ActionType,
    ) -> torch.Tensor:
        """Score a unimolecular reaction from conditioned state embeddings."""
        assert state_emb.ndim in (1, 2)
        assert action_type.is_unirxn
        head = self.action_heads[action_type.name]
        rxn_emb = self.rxn_embedding[action_name]
        if state_emb.ndim == 2:
            rxn_emb = rxn_emb.expand(state_emb.shape[0], -1)
        return head(torch.cat([state_emb, rxn_emb], dim=-1))

    def forward_reactions(
        self,
        state_emb: torch.Tensor,
        action_names: list[str],
        action_type: ActionType,
    ) -> torch.Tensor:
        """Apply one shared action head to [reactions, states, features]."""
        assert state_emb.ndim == 2
        assert action_type.is_unirxn or action_type.is_birxn
        rxn_emb = torch.stack([self.rxn_embedding[name] for name in action_names])
        rxn_emb = rxn_emb[:, None, :].expand(-1, len(state_emb), -1)
        states = state_emb[None, :, :].expand(len(action_names), -1, -1)
        head = self.action_heads[action_type.name]
        return head(torch.cat([states, rxn_emb], dim=-1))

    def forward_birxn(
        self,
        state_emb: torch.Tensor,
        action_name: str,
        action_type: ActionType,
    ) -> torch.Tensor:
        """Form bimolecular synthon queries from conditioned state embeddings."""
        assert state_emb.ndim in (1, 2)
        assert action_type in (ActionType.BIRXN_BRICK, ActionType.BIRXN_LINKER)
        head = self.action_heads[action_type.name]
        rxn_emb = self.rxn_embedding[action_name]
        if state_emb.ndim == 2:
            rxn_emb = rxn_emb.expand(state_emb.shape[0], -1)
        return head(torch.cat([state_emb, rxn_emb], dim=-1))
