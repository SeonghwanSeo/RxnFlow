from itertools import chain

import torch
import torch.nn as nn
import torch_geometric.data as gd

from gflownet.config import Config
from gflownet.models.graph_transformer import GraphTransformer, mlp
from gflownet.envs.synthesis import ReactionActionType

from .env_context import RGFN_EnvContext
from .action_categorical import HierarchicalReactionActionCategorical


class RGFN(nn.Module):
    """GraphTransfomer class for a ASTB which outputs an ReactionActionCategorical.

    Outputs logits corresponding to each action (template).
    """

    def __init__(
        self,
        env_ctx: RGFN_EnvContext,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ) -> None:
        super().__init__()
        self.env_ctx: RGFN_EnvContext = env_ctx

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
            self.env_ctx.num_block_features,
            cfg.model.num_emb_building_block,
            cfg.model.num_emb_building_block,
            cfg.model.num_layers_building_block,
        )

        num_emb = cfg.model.num_emb
        num_emb_block = cfg.model.num_emb_building_block
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings

        self._action_type_to_num_inputs_outputs = {
            ReactionActionType.Stop: (num_glob_final, 1),
            ReactionActionType.AddFirstReactant: (num_glob_final, num_emb_block),
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
            num_glob_final + env_ctx.num_bimolecular_rxns * 2,
            num_emb,
            num_emb_block,
            cfg.model.graph_transformer.num_mlp_layers,
        )

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)
        self.gelu = nn.GELU()

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

    def hook_stop(self, emb: torch.Tensor):
        return self.mlps[ReactionActionType.Stop.cname](emb)

    def hook_add_first_reactant(self, emb: torch.Tensor, block_emb: torch.Tensor):
        emb = self.gelu(self.mlps[ReactionActionType.AddFirstReactant.cname](emb))  # N_graph, F
        return torch.matmul(emb, block_emb.T)

    def hook_reactbi_primary(self, emb: torch.Tensor, mask: torch.Tensor):
        logit = self.mlps[ReactionActionType.ReactBi.cname](emb)
        ngraph = mask.shape[0]
        return logit.masked_fill(torch.logical_not(mask).view(ngraph, -1), -torch.inf)

    def hook_reactbi_secondary(
        self, single_emb: torch.Tensor, rxn_id: int, block_is_first: bool, block_emb: torch.Tensor
    ):
        # Convert `rxn_id` to a one-hot vector
        rxn_features = torch.zeros(self.env_ctx.num_bimolecular_rxns * 2, device=single_emb.device)
        rxn_features[rxn_id * 2 + int(block_is_first)] = 1
        _emb = self.gelu(self.second_step_mlp(torch.cat([single_emb, rxn_features])))
        return torch.matmul(_emb, block_emb.T)

    def single_hook_reactbi_primary(self, emb: torch.Tensor, rxn_id: int, block_is_first: bool):
        rxn_id = rxn_id * 2 + int(block_is_first)
        return self.mlps[ReactionActionType.ReactBi.cname](emb)[rxn_id]

    def single_hook_reactbi_secondary(
        self, emb: torch.Tensor, rxn_id: int, block_is_first: bool, block_emb: torch.Tensor
    ):
        rxn_features = torch.zeros(self.env_ctx.num_bimolecular_rxns * 2, device=emb.device)
        rxn_features[rxn_id * 2 + int(block_is_first)] = 1
        _emb = self.gelu(self.second_step_mlp(torch.cat([emb, rxn_features])))
        return torch.matmul(_emb, block_emb).view(1)
