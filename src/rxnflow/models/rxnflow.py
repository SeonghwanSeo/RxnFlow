"""RxnFlow dynamic reaction policy and state-flow model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rxnflow.config import ModelConfig
from rxnflow.core.types import ActionType
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import (
    FINGERPRINT_DIM,
    PROPERTY_DIM,
    PROPERTY_SCALE,
)
from rxnflow.envs.graph import BOND_FEATURE_DIM, NODE_FEATURE_DIM, GraphBatch

from .mpnn import MPNN
from .nn import mlp


class RxnFlowModel(nn.Module):
    def __init__(self, env: SynthesisEnv, cfg: ModelConfig, num_objectives: int):
        super().__init__()
        num_emb = cfg.num_emb
        self.env = env
        self.num_objectives = num_objectives
        # Condition encoders feed the virtual node, logit scale, and logZ head.
        # Beta has nine features: its normalized value plus four sine/cosine pairs.
        self.emb_beta = mlp(9, num_emb, num_emb, 2)
        self.emb_preferences = mlp(num_objectives, num_emb, num_emb, 2)
        self.cond2h = nn.Linear(num_emb, num_emb)
        self.register_buffer(
            "property_scale", torch.tensor(PROPERTY_SCALE), persistent=False
        )
        self.mpnn = MPNN(
            x_dim=NODE_FEATURE_DIM,
            e_dim=BOND_FEATURE_DIM,
            g_dim=PROPERTY_DIM,
            num_emb=num_emb,
            num_layers=cfg.num_layers,
            max_reactions=env.max_reactions,
            max_synthons=env.max_synthons,
        )
        # Mean pooling and the virtual node can have different scales.
        self.norm_mean = nn.LayerNorm(num_emb)
        self.norm_virtual = nn.LayerNorm(num_emb)
        self.emb_rxn = nn.Embedding(len(env.action_names), num_emb)
        self.emb_type = nn.Embedding(len(env.library_names), cfg.num_synthon_emb)
        # Project each feature, then normalize only in the fusion MLP.
        # Properties are scaled before projection in synthon_embedding.
        self.lin_fp = nn.Linear(FINGERPRINT_DIM, cfg.num_synthon_emb)
        self.lin_prop = nn.Linear(PROPERTY_DIM, cfg.num_synthon_emb)
        self.mlp_synthon = mlp(
            cfg.num_synthon_emb * 3,
            cfg.num_synthon_emb,
            cfg.num_synthon_emb,
            cfg.num_mlp_layers_synthon,
            layernorm=True,
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
        # Small categorical embeddings limit their initial contribution to fused inputs.
        nn.init.uniform_(self.emb_rxn.weight, -0.1, 0.1)
        nn.init.uniform_(self.emb_type.weight, -0.1, 0.1)
        # Start from unscaled logits and log Z = 0.
        nn.init.zeros_(self._logit_scale[-1].weight)
        nn.init.zeros_(self._logit_scale[-1].bias)
        nn.init.zeros_(self._logZ[-1].weight)
        nn.init.zeros_(self._logZ[-1].bias)

    def encode_cond(self, beta: torch.Tensor, preferences: torch.Tensor) -> torch.Tensor:
        """Encode beta and objective weights into one [batch, num_emb] condition."""
        # Fixed coordinates are independent of the sampling range. The linear
        # term u distinguishes values that share the same periodic features.
        u = (beta[:, None] - 1.0) / 63.0
        frequencies = u.new_tensor((1.0, 2.0, 4.0, 8.0))
        angles = 2 * math.pi * u * frequencies
        features = torch.cat([u, angles.sin(), angles.cos()], dim=-1)
        return self.emb_beta(features) + self.emb_preferences(preferences)

    def graph_embedding(self, batch: GraphBatch, cond_info: torch.Tensor) -> torch.Tensor:
        mean_emb, virtual_emb = self.mpnn(batch, self.cond2h(cond_info)).chunk(2, dim=-1)
        return torch.cat(
            [self.norm_mean(mean_emb), self.norm_virtual(virtual_emb)], dim=-1
        )

    def logit_scale(self, cond_info: torch.Tensor) -> torch.Tensor:
        """Map each condition to an ELU + 1 logit multiplier, initialized at 1."""
        return F.elu(self._logit_scale(cond_info)).squeeze(-1) + 1

    def logZ(self, cond_info: torch.Tensor) -> torch.Tensor:
        return self._logZ(cond_info)

    def get_unirxn_logits(
        self, graph_emb: torch.Tensor, action_name: str, logit_scale: torch.Tensor
    ) -> torch.Tensor:
        action_type = (
            ActionType.UNIRXN_TERMINAL
            if self.env.uni_reactions[action_name].output_type is None
            else ActionType.UNIRXN_TRANSFORM
        )
        return self.forward_mdp(graph_emb, action_name, logit_scale, action_type)[0, 0]

    def get_synthon_emb(
        self, synthon_type: str, indices: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        library = self.env.synthons[synthon_type]
        cpu_indices = indices.detach().cpu().to(torch.long).numpy()
        fp = torch.from_numpy(library.fingerprints[cpu_indices]).to(
            device, dtype=torch.float32
        )
        prop = torch.from_numpy(library.properties[cpu_indices]).to(device)
        type_index = self.env.library_to_index[synthon_type]
        synthon_types = torch.full(
            (len(indices),), type_index, dtype=torch.long, device=device
        )
        return self.synthon_embedding(fp, prop, synthon_types)

    def synthon_embedding(
        self, fp: torch.Tensor, prop: torch.Tensor, synthon_types: torch.Tensor
    ) -> torch.Tensor:
        prop = prop / self.property_scale
        return self.mlp_synthon(
            torch.cat(
                [
                    self.lin_fp(fp),
                    self.lin_prop(prop),
                    self.emb_type(synthon_types),
                ],
                dim=-1,
            )
        )

    def forward_mdp(
        self,
        graph_emb: torch.Tensor,
        action_name: str,
        logit_scale: torch.Tensor,
        action_type: ActionType,
    ) -> torch.Tensor:
        """Return a synthon-query vector or unary logit for one reaction and batch."""
        # Add reaction identity after graph encoding so all reactions share the GNN.
        index = self.env.action_to_index[action_name]
        rxn_emb = self.emb_rxn.weight[index].expand(graph_emb.shape[0], -1)
        state_rxn_emb = torch.cat([graph_emb, rxn_emb], dim=-1)
        head = self.action_heads[action_type.name]
        return head(state_rxn_emb) * logit_scale[:, None]

    def get_synthon_logits(
        self,
        graph_emb: torch.Tensor,
        action_name: str,
        synthon_type: str,
        indices: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        assert indices.ndim == 1 and graph_emb.shape[0] == 1
        action_type = (
            ActionType.FIRST_SYNTHON
            if action_name == "first_synthon"
            else ActionType.BIRXN_BRICK
            if self.env.synthons[synthon_type].is_brick
            else ActionType.BIRXN_LINKER
        )
        state_emb = self.forward_mdp(graph_emb, action_name, logit_scale, action_type)
        synthon_emb = self.get_synthon_emb(synthon_type, indices, graph_emb.device)
        return F.normalize(synthon_emb, dim=-1) @ state_emb.squeeze(0)
