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

from .encoders import ConditionEncoder, SynthonEncoder
from .mpnn import MPNN
from .nn import mlp


class RxnFlowModel(nn.Module):
    """RxnFlow model"""

    def __init__(
        self,
        env: SynthesisEnv,
        cfg: ModelConfig,
        num_objectives: int,
    ):
        super().__init__()
        # Action name to index mapping for embedding lookup
        self.action_to_index = dict(env.action_to_index)

        # Number of objectives for multi-objective optimization
        self.num_objectives = num_objectives

        # Condition encoders
        self.condition_encoder = ConditionEncoder(cfg.hidden_dim, num_objectives)
        cond_dim = cfg.hidden_dim

        # State graph encoder
        self.state_encoder = MPNN(
            x_dim=NODE_FEATURE_DIM,
            e_dim=BOND_FEATURE_DIM,
            g_dim=PROPERTY_DIM,
            cond_dim=cond_dim,
            model_dim=cfg.state_dim,
            num_layers=cfg.num_state_layers,
        )
        state_dim = 2 * cfg.state_dim

        # Action type embeddings
        self.emb_rxn = nn.Embedding(len(env.action_names), cfg.hidden_dim)
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
        action_heads = {}
        for action_type in ActionType:
            out_dim = 1 if action_type.is_unirxn else synthon_dim
            action_heads[action_type.name] = mlp(
                state_dim + rxn_dim,
                hidden_dim,
                out_dim,
                cfg.num_action_layers,
                layernorm=True,
                dropout=cfg.dropout,
            )
        self.action_heads = nn.ModuleDict(action_heads)

        # Logit scaling (LogitGFN)
        self._logit_scale = mlp(hidden_dim, hidden_dim, 1, 2)

        # Trajectory-balance (TB)
        self._logZ = mlp(hidden_dim, hidden_dim, 1, 2)

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
        # Start from unscaled logits and log Z = 0.
        nn.init.zeros_(self._logit_scale[-1].weight)
        nn.init.zeros_(self._logit_scale[-1].bias)
        nn.init.zeros_(self._logZ[-1].weight)
        nn.init.zeros_(self._logZ[-1].bias)

    def encode_cond(self, beta: torch.Tensor, preferences: torch.Tensor) -> torch.Tensor:
        """Encode beta and objective weights into one [bs, d_hid] condition."""
        return self.condition_encoder(beta, preferences)

    def encode_state(self, batch: GraphBatch, cond: torch.Tensor) -> torch.Tensor:
        """cond: [bs, d_hid]; return [bs, 2*d_state]."""
        return self.state_encoder(batch, cond)

    def logit_scale(self, cond: torch.Tensor) -> torch.Tensor:
        """[bs, b_hid] → [bs] \in [0, ∞)."""
        return F.elu(self._logit_scale(cond)).squeeze(-1) + 1

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        """[bs, d_hid] → [bs, 1]."""
        return self._logZ(cond)

    def encode_synthon(
        self, fp: torch.Tensor, prop: torch.Tensor, synthon_types: torch.Tensor
    ) -> torch.Tensor:
        """fp: [n, d_fp], prop: [n, d_prop], synthon_types: [n, 2]
        return [n, d_synthon]."""
        return self.synthon_encoder(fp, prop, synthon_types)

    def forward_mdp(
        self,
        action_type: ActionType,
        action_name: str,
        state_emb: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        - state_emb: [bs, 2*d_state].
        - logit_scale: [bs].

        Return [bs, 1] logits for unimolecular actions or
        [bs, d_synthon] logits for bimolecular actions.
        """
        index = self.action_to_index[action_name]
        rxn_emb = self.emb_rxn.weight[index].expand(state_emb.shape[0], -1)
        state_rxn_emb = torch.cat([state_emb, rxn_emb], dim=-1)
        head = self.action_heads[action_type.name]
        return head(state_rxn_emb) * logit_scale[:, None]
