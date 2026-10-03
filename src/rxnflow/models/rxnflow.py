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
    """Shapes use B=batch size, H=num_emb, S=num_synthon_emb."""

    def __init__(
        self,
        env: SynthesisEnv,
        cfg: ModelConfig,
        num_objectives: int,
    ):
        super().__init__()
        num_emb = cfg.num_emb
        self.action_to_index = dict(env.action_to_index)
        self.num_objectives = num_objectives
        # One condition is shared by the graph encoder, logit scale, and logZ.
        self.condition_encoder = ConditionEncoder(num_emb, num_objectives)
        self.state_encoder = MPNN(
            x_dim=NODE_FEATURE_DIM,
            e_dim=BOND_FEATURE_DIM,
            g_dim=PROPERTY_DIM,
            num_emb=num_emb,
            num_layers=cfg.num_layers,
        )
        self.emb_rxn = nn.Embedding(len(env.action_names), num_emb)
        self.synthon_encoder = SynthonEncoder(
            len(env.synthon_types) + 1, cfg.num_synthon_emb, cfg.num_mlp_layers_synthon
        )
        # Concatenate the 2H state and H reaction embeddings. Each head learns
        # their joint projection while the graph encoding is shared by reactions.
        # Unary heads emit scalar logits; FirstSynthon and binary heads emit
        # query vectors for the shared synthon embeddings. All actions still
        # compete in one categorical distribution.
        self.action_heads = nn.ModuleDict(
            {
                action_type.name: mlp(
                    3 * num_emb,
                    num_emb,
                    1 if action_type.is_unirxn else cfg.num_synthon_emb,
                    cfg.num_mlp_layers,
                    layernorm=True,
                    dropout=cfg.dropout,
                )
                for action_type in ActionType
            }
        )
        # One condition-dependent multiplier controls the scale of all action logits.
        self._logit_scale = mlp(num_emb, num_emb, 1, 2)
        self._logZ = mlp(num_emb, num_emb, 1, 2)
        self.init_weight()

    def init_weight(self) -> None:
        # Keep policy outputs nonzero so both dot-product branches receive
        # gradients from the first update. All Linear layers share this rule.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
        """Encode beta and objective weights into one [batch, num_emb] condition."""
        return self.condition_encoder(beta, preferences)

    def encode_state(self, batch: GraphBatch, cond_info: torch.Tensor) -> torch.Tensor:
        """cond_info: [B, H]; return [B, 2H]."""
        return self.state_encoder(batch, cond_info)

    def logit_scale(self, cond_info: torch.Tensor) -> torch.Tensor:
        """[B, H] → [B] ELU + 1 logit multipliers, initialized at 1."""
        return F.elu(self._logit_scale(cond_info)).squeeze(-1) + 1

    def logZ(self, cond_info: torch.Tensor) -> torch.Tensor:
        """[B, H] → [B, 1]."""
        return self._logZ(cond_info)

    def encode_synthon(
        self, fp: torch.Tensor, prop: torch.Tensor, site_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.synthon_encoder(fp, prop, site_indices)

    def forward_mdp(
        self,
        state_emb: torch.Tensor,
        action_name: str,
        logit_scale: torch.Tensor,
        action_type: ActionType,
    ) -> torch.Tensor:
        """state_emb: [B, 2H], logit_scale: [B].

        Return [B, 1] logits for UniReaction or [B, S] queries for synthon selection.
        """
        # Add reaction identity after graph encoding so all reactions share the GNN.
        index = self.action_to_index[action_name]
        rxn_emb = self.emb_rxn.weight[index].expand(state_emb.shape[0], -1)
        state_rxn_emb = torch.cat([state_emb, rxn_emb], dim=-1)
        head = self.action_heads[action_type.name]
        return head(state_rxn_emb) * logit_scale[:, None]
