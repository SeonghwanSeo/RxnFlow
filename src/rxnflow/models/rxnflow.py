"""RxnFlow dynamic reaction policy and state-flow model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rxnflow.config import ModelConfig
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
        # Fixed encoder coordinates, independent of the beta sampling range.
        # Keep u itself so periodic features never alias the whole encoding.
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
        )
        # Mean pooling and the virtual node can have different scales.
        self.norm_mean = nn.LayerNorm(num_emb)
        self.norm_virtual = nn.LayerNorm(num_emb)
        self.emb_rxn = nn.Embedding(len(env.action_names), num_emb)
        self.emb_type = nn.Embedding(len(env.block_types), cfg.num_block_emb)
        # Project each feature, then normalize only in the fusion MLP.
        # Properties are scaled before projection in block_embedding.
        self.lin_fp = nn.Linear(FINGERPRINT_DIM, cfg.num_block_emb)
        self.lin_prop = nn.Linear(PROPERTY_DIM, cfg.num_block_emb)
        self.mlp_block = mlp(
            cfg.num_block_emb * 3,
            cfg.num_block_emb,
            cfg.num_block_emb,
            cfg.num_mlp_layers_block,
            layernorm=True,
        )
        # Concatenate the 2H state and H reaction embeddings. Each head learns
        # their joint projection while the graph encoding is shared by reactions.
        self.mlp_firstblock = mlp(
            3 * num_emb,
            num_emb,
            cfg.num_block_emb,
            cfg.num_mlp_layers,
            layernorm=True,
            dropout=cfg.dropout,
        )
        self.mlp_birxn = mlp(
            3 * num_emb,
            num_emb,
            cfg.num_block_emb,
            cfg.num_mlp_layers,
            layernorm=True,
            dropout=cfg.dropout,
        )
        self.mlp_unirxn = mlp(
            3 * num_emb,
            num_emb,
            1,
            cfg.num_mlp_layers,
            layernorm=True,
            dropout=cfg.dropout,
        )
        # HSX/Logit-GFN: one positive condition-dependent scale for all actions.
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
        # HSX main initializes both reaction and block-type embeddings small.
        nn.init.uniform_(self.emb_rxn.weight, -0.1, 0.1)
        nn.init.uniform_(self.emb_type.weight, -0.1, 0.1)
        # Start from unscaled logits and log Z = 0.
        nn.init.zeros_(self._logit_scale[-1].weight)
        nn.init.zeros_(self._logit_scale[-1].bias)
        nn.init.zeros_(self._logZ[-1].weight)
        nn.init.zeros_(self._logZ[-1].bias)

    def encode_cond(self, beta: Tensor, preferences: Tensor) -> Tensor:
        u = (beta[:, None] - 1.0) / 63.0
        frequencies = u.new_tensor((1.0, 2.0, 4.0, 8.0))
        angles = 2 * math.pi * u * frequencies
        features = torch.cat([u, angles.sin(), angles.cos()], dim=-1)
        return self.emb_beta(features) + self.emb_preferences(preferences)

    def graph_embedding(self, batch: GraphBatch, cond_info: Tensor) -> Tensor:
        mean_emb, virtual_emb = self.mpnn(batch, self.cond2h(cond_info)).chunk(2, dim=-1)
        return torch.cat(
            [self.norm_mean(mean_emb), self.norm_virtual(virtual_emb)], dim=-1
        )

    def logit_scale(self, cond_info: Tensor) -> Tensor:
        """HSX ELU + 1: positive scalar per condition, without fixed bounds."""
        return F.elu(self._logit_scale(cond_info)).squeeze(-1) + 1

    def logZ(self, cond_info: Tensor) -> Tensor:
        return self._logZ(cond_info)

    def get_unirxn_logits(
        self, graph_emb: Tensor, action_name: str, logit_scale: Tensor
    ) -> Tensor:
        return self.forward_mdp(graph_emb, action_name, logit_scale)[0, 0]

    def get_block_emb(
        self, block_type: str, indices: Tensor, device: torch.device
    ) -> Tensor:
        library = self.env.blocks[block_type]
        cpu_indices = indices.detach().cpu().to(torch.long)
        fp = library.fingerprints[cpu_indices].to(device, dtype=torch.float32)
        prop = library.properties[cpu_indices].to(device)
        type_index = self.env.block_type_to_index[block_type]
        block_types = torch.full(
            (len(indices),), type_index, dtype=torch.long, device=device
        )
        return self.block_embedding(fp, prop, block_types)

    def block_embedding(self, fp: Tensor, prop: Tensor, block_types: Tensor) -> Tensor:
        prop = prop / self.property_scale
        return self.mlp_block(
            torch.cat(
                [
                    self.lin_fp(fp),
                    self.lin_prop(prop),
                    self.emb_type(block_types),
                ],
                dim=-1,
            )
        )

    def forward_mdp(
        self, graph_emb: Tensor, action_name: str, logit_scale: Tensor
    ) -> Tensor:
        index = self.env.action_to_index[action_name]
        rxn_emb = self.emb_rxn.weight[index].expand(graph_emb.shape[0], -1)
        state_rxn_emb = torch.cat([graph_emb, rxn_emb], dim=-1)
        if action_name == "first_block":
            head = self.mlp_firstblock
        elif action_name in self.env.uni_reactions:
            head = self.mlp_unirxn
        else:
            head = self.mlp_birxn
        return head(state_rxn_emb) * logit_scale[:, None]

    def get_block_logits(
        self,
        graph_emb: Tensor,
        action_name: str,
        block_type: str,
        indices: Tensor,
        logit_scale: Tensor,
    ) -> Tensor:
        assert indices.ndim == 1 and graph_emb.shape[0] == 1
        state_emb = self.forward_mdp(graph_emb, action_name, logit_scale)
        block_emb = self.get_block_emb(block_type, indices, graph_emb.device)
        return F.normalize(block_emb, dim=-1) @ state_emb.squeeze(0)
