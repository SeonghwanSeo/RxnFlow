from itertools import chain

import torch
import torch.nn as nn
import torch_geometric.data as gd

from gflownet.config import Config
from gflownet.models.graph_transformer import GraphTransformer, mlp
from gflownet.envs.synthesis import ReactionActionType, SynthesisEnvContext

from .action_categorical import HierarchicalReactionActionCategorical


class HierarchicalGFN(nn.Module):
    def __init__(
        self,
        env_ctx: SynthesisEnvContext,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ) -> None:
        super().__init__()

        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )

        self.block_mlp = mlp(
            env_ctx.num_block_features,
            cfg.model.num_emb_building_block,
            cfg.model.num_emb_building_block,
            cfg.model.num_layers_building_block,
        )

        num_emb = cfg.model.num_emb
        num_emb_block = cfg.model.num_emb_building_block
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings

        self.num_bimolecular_rxns = env_ctx.num_bimolecular_rxns

        self._action_type_to_num_inputs_outputs = {
            ReactionActionType.Stop: (num_glob_final, 1),
            ReactionActionType.AddFirstReactant: (num_glob_final + num_emb_block, 1),
            ReactionActionType.ReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ReactionActionType.ReactBi: (num_glob_final, env_ctx.num_bimolecular_rxns * 2),
        }

        self.do_bck = do_bck
        mlps = {}
        for atype in chain(env_ctx.action_type_order, env_ctx.bck_action_type_order if do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.graph_transformer.num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.second_step_mlp = mlp(
            num_glob_final + env_ctx.num_bimolecular_rxns * 2 + num_emb_block,
            num_emb,
            1,
            cfg.model.graph_transformer.num_mlp_layers,
        )

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        return self._logZ(cond)

    def _make_cat(self, g, emb, fwd):
        return HierarchicalReactionActionCategorical(g, emb, model=self, fwd=fwd)

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        _, emb = self.transf(g, cond)
        graph_out = self.emb2graph_out(emb)
        fwd_cat = self._make_cat(g, emb, fwd=True)

        if self.do_bck:
            bck_cat = self._make_cat(g, emb, fwd=False)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out

    def hook_add_first_reactant(self, emb: torch.Tensor, block_emb: torch.Tensor):
        N_graph = emb.size(0)
        N_block = block_emb.size(0)
        mlp = self.mlps[ReactionActionType.AddFirstReactant.cname]

        logits = torch.empty((N_graph, N_block), device=emb.device)
        for i in range(N_graph):
            _emb = emb[i].unsqueeze(0).repeat(N_block, 1)
            expanded_input = torch.cat((_emb, block_emb), dim=-1)
            logits[i] = mlp(expanded_input).squeeze(-1)
        return logits

    def hook_stop(self, emb: torch.Tensor):
        return self.mlps[ReactionActionType.Stop.cname](emb)

    def hook_reactuni(self, emb: torch.Tensor, mask: torch.Tensor):
        logit = self.mlps[ReactionActionType.ReactUni.cname](emb)
        return logit.masked_fill(torch.logical_not(mask), -torch.inf)

    def hook_reactbi_primary(self, emb: torch.Tensor, mask: torch.Tensor):
        logit = self.mlps[ReactionActionType.ReactBi.cname](emb)
        return logit.masked_fill(torch.logical_not(mask).flatten(1), -torch.inf)

    def hook_reactbi_secondary(
        self, single_emb: torch.Tensor, rxn_id: int, block_is_first: bool, block_emb: torch.Tensor
    ):
        rxn_features = torch.zeros(self.num_bimolecular_rxns * 2, device=single_emb.device)
        rxn_features[rxn_id * 2 + int(block_is_first)] = 1

        num_blocks = block_emb.shape[0]
        _emb = torch.cat([single_emb, rxn_features], dim=-1)
        _emb = _emb.unsqueeze(0).repeat(num_blocks, 1)
        extended_emb = torch.cat([_emb, block_emb], dim=-1)
        return self.second_step_mlp(extended_emb).squeeze(-1)

    def single_hook_reactuni(self, emb: torch.Tensor, rxn_id: torch.Tensor):
        return self.mlps[ReactionActionType.ReactUni.cname](emb)[rxn_id]

    def single_hook_reactbi_primary(self, emb: torch.Tensor, rxn_id: int, block_is_first: bool):
        rxn_id = rxn_id * 2 + int(block_is_first)
        return self.mlps[ReactionActionType.ReactBi.cname](emb)[rxn_id]

    def single_hook_reactbi_secondary(
        self, emb: torch.Tensor, rxn_id: int, block_is_first: bool, block_emb: torch.Tensor
    ):
        rxn_features = torch.zeros(self.num_bimolecular_rxns * 2, device=emb.device)
        rxn_features[rxn_id * 2 + int(block_is_first)] = 1
        extended_emb = torch.cat([emb, rxn_features, block_emb], dim=-1)
        return self.second_step_mlp(extended_emb).view(-1)
