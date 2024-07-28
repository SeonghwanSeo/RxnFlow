from itertools import chain
from typing import Dict

import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch_scatter import scatter_mean

from gflownet.config import Config
from gflownet.envs.synthesis import ReactionActionType, SynthesisEnvContext
from gflownet.models.graph_transformer import GraphTransformer, mlp
from gflownet.models.synthesis_gfn import GFN_Synthesis

from gflownet.misc.tacogfn.gvp_embedding import GVP_embedding


class GFN_Synthesis_SBDD(GFN_Synthesis):
    def __init__(
        self,
        env_ctx: SynthesisEnvContext,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ):
        pocket_dim = cfg.task.tacogfn.pocket_dim
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim + pocket_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )

        self.block_mlp = mlp(
            cfg.model.fp_nbits_building_block,
            cfg.model.num_emb_building_block,
            cfg.model.num_emb_building_block,
            cfg.model.num_layers_building_block,
        )

        num_emb = cfg.model.num_emb
        num_emb_block = cfg.model.num_emb_building_block
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings

        self._action_type_to_num_inputs_outputs = {
            ReactionActionType.Stop: (num_glob_final, 1),
            ReactionActionType.AddFirstReactant: (num_glob_final + num_emb_block, 1),
            ReactionActionType.ReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ReactionActionType.ReactBi: (num_glob_final + env_ctx.num_bimolecular_rxns * 2 + num_emb_block, 1),
        }

        self.do_bck = do_bck
        mlps = {}
        for atype in chain(env_ctx.action_type_order, env_ctx.bck_action_type_order if do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.graph_transformer.num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim + pocket_dim, num_emb * 2, 1, 2)

        self.pocket_embedding: torch.Tensor

    def encode_pocket(self, pocket_data):
        p_node_feature = (
            pocket_data["protein"]["node_s"],
            pocket_data["protein"]["node_v"],
        )
        p_edge_index = pocket_data[("protein", "p2p", "protein")]["edge_index"]
        p_edge_feature = (
            pocket_data[("protein", "p2p", "protein")]["edge_s"],
            pocket_data[("protein", "p2p", "protein")]["edge_v"],
        )
        p_batch = pocket_data["protein"].batch
        p_embed = self.pocket_encoder(p_node_feature, p_edge_index, p_edge_feature, pocket_data.seq)

        pocket_cond = scatter_mean(p_embed, p_batch, dim=0)
        return pocket_cond

    def forward(self, mol_g: gd.Batch, cond: torch.Tensor):
        pocket_index = cond[:, -1].to(torch.long).tolist()
        pocket_g = gd.Batch.from_data_list([env_ctx.pocket_graphs[idx] for idx in pocket_index])
        self.pocket_embedding = self.encode_pocket(pocket_g.to(mol_g.x.device))
        cond_cat = torch.cat([cond[:, :-1], self.pocket_embedding], dim=-1)
        return self.graph_transformer(mol_g, cond_cat)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        cond_cat = torch.cat([cond[:, :-1], self.pocket_embedding], dim=-1)
        return self._logZ(cond_cat)
