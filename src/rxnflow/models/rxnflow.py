"""RxnFlow dynamic reaction policy and state-flow model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rxnflow.config import ModelConfig
from rxnflow.core.types import ActionType
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import PROPERTY_DIM
from rxnflow.envs.graph import BOND_FEATURE_DIM, NODE_FEATURE_DIM, GraphBatch

from .encoders import ExternalConditionEncoder, GFNConditionEncoder, SynthonEncoder
from .mpnn import MPNN
from .nn import mlp


class RxnFlowModel(nn.Module):
    """RxnFlow model"""

    def __init__(
        self,
        env: SynthesisEnv,
        cfg: ModelConfig,
        num_objectives: int = 1,
        condition_dim: int | None = None,
    ):
        super().__init__()
        self.num_objectives: int = num_objectives
        self.condition_dim: int | None = condition_dim

        # State encoder
        self.state_encoder = MPNN(
            node_dim=NODE_FEATURE_DIM,
            edge_dim=BOND_FEATURE_DIM,
            graph_dim=PROPERTY_DIM,
            model_dim=cfg.state_dim,
            num_layers=cfg.num_state_layers,
        )

        # GFN condition encoders, i.e., temperature and preference vector
        self.gfn_condition_encoder = GFNConditionEncoder(num_objectives, cfg.hidden_dim)
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

        def _action_mlp(in_dim, out_dim):
            return mlp(in_dim, hidden_dim, out_dim, cfg.num_action_layers, layernorm=True)

        action_heads = {
            ActionType.FIRST_SYNTHON: _action_mlp(state_dim, synthon_dim),
            ActionType.UNIRXN_TRANSFORM: _action_mlp(state_dim + rxn_dim, 1),
            ActionType.UNIRXN_TERMINAL: _action_mlp(state_dim + rxn_dim, 1),
            ActionType.BIRXN_BRICK: _action_mlp(state_dim + rxn_dim, synthon_dim),
            ActionType.BIRXN_LINKER: _action_mlp(state_dim + rxn_dim, synthon_dim),
        }
        self.action_heads = nn.ModuleDict({k.name: v for k, v in action_heads.items()})

        # Logit scaling (LogitGFN)
        self._logit_scale = mlp(cond_dim, hidden_dim, 1, 2)

        # Trajectory-balance (TB)
        self._logZ = mlp(cond_dim, hidden_dim, 1, 2)

        self.init_weight()

    def init_weight(self) -> None:
        # Keep policy outputs nonzero so both dot-product branches receive
        # gradients from the first update. All Linear layers share this rule.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, nn.Embedding):
                # Keep categorical contributions small at initialization.
                nn.init.uniform_(module.weight, -0.1, 0.1)
        for embedding in self.rxn_embedding.values():
            nn.init.uniform_(embedding, -0.1, 0.1)
        # Start from unscaled logits and log Z = 0.
        nn.init.zeros_(self._logit_scale[-1].weight)
        nn.init.zeros_(self._logit_scale[-1].bias)
        nn.init.zeros_(self._logZ[-1].weight)
        nn.init.zeros_(self._logZ[-1].bias)

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
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode beta [B], preference [B, O] and optional cond [B, condition_dim].

        Return [B, 2*hidden_dim], with a zero external embedding when omitted.
        """
        gfn_cond_emb = self.gfn_condition_encoder(beta, preference)
        if self.condition_encoder is not None:
            assert cond is not None, "external conditions must be provided"
            cond_emb = self.condition_encoder(cond)
        else:
            cond_emb = torch.zeros_like(gfn_cond_emb)
        return torch.cat([gfn_cond_emb, cond_emb], dim=-1)

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
