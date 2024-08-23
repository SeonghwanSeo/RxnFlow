from itertools import chain

import torch
import torch.nn as nn
import torch_geometric.data as gd

from gflownet.config import Config
from gflownet.models.graph_transformer import GraphTransformer, mlp

from gflownet.envs.synthesis import ReactionActionType, SynthesisEnvContext, ReactionActionCategorical


class SynthesisGFN(nn.Module):
    """GraphTransfomer class which outputs an ReactionActionCategorical."""

    def __init__(
        self,
        env_ctx: SynthesisEnvContext,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ) -> None:
        super().__init__()
        self.num_bimolecular_rxns: int = env_ctx.num_bimolecular_rxns

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

        self._action_type_to_num_inputs_outputs = {
            ReactionActionType.Stop: (num_glob_final, 1),
            ReactionActionType.AddFirstReactant: (num_glob_final + num_emb_block, 1),
            ReactionActionType.ReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ReactionActionType.ReactBi: (num_glob_final + env_ctx.num_bimolecular_rxns * 2 + num_emb_block, 1),
        }

        assert do_bck is False
        self.do_bck = do_bck
        mlps = {}
        for atype in chain(env_ctx.action_type_order, env_ctx.bck_action_type_order if do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.graph_transformer.num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        return self._logZ(cond)

    def _make_cat(self, g, emb, fwd):
        return ReactionActionCategorical(g, emb, model=self, fwd=fwd)

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        """

        Parameters
        ----------
        g : gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond : torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        -------
        ReactionActionCategorical
        """
        _, emb = self.transf(g, cond)
        graph_out = self.emb2graph_out(emb)
        fwd_cat = self._make_cat(g, emb, fwd=True)

        if self.do_bck:
            bck_cat = self._make_cat(g, emb, fwd=False)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out

    def hook_stop(self, emb: torch.Tensor):
        """
        The hook function to be called for the Stop action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current state.

        Returns
        torch.Tensor
            The logits for Stop
        """

        return self.mlps[ReactionActionType.Stop.cname](emb)

    def hook_reactuni(self, emb: torch.Tensor, mask: torch.Tensor):
        """
        The hook function to be called for the ReactUni action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current state.

        Returns
        torch.Tensor
            The logits for ReactUni
        """

        logit = self.mlps[ReactionActionType.ReactUni.cname](emb)
        return logit.masked_fill(torch.logical_not(mask), -torch.inf)

    def hook_reactbi(
        self, emb: torch.Tensor, rxn_id: int, block_is_first: bool, block_emb: torch.Tensor, mask: torch.Tensor
    ):
        """
        The hook function to be called for the ReactBi action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current state.
        rxn_id : int
            The ID of the reaction selected by the sampler.
        block_is_first : bool
            The flag whether block is first reactant or second reactant of bimolecular reaction
        block_emb : torch.Tensor
            The embedding tensor for building blocks.

        Returns
        torch.Tensor
            The logits for (rxn_id, block_is_first).
        """
        N_graph = emb.size(0)
        N_block = block_emb.size(0)
        mlp = self.mlps[ReactionActionType.ReactBi.cname]

        # Convert `rxn_id` to a one-hot vector
        rxn_features = torch.zeros(N_block, self.num_bimolecular_rxns * 2, device=emb.device)
        rxn_features[:, rxn_id * 2 + int(block_is_first)] = 1

        logits = torch.full((N_graph, N_block), -torch.inf, device=emb.device)
        for i in range(N_graph):
            if mask[i]:
                _emb = emb[i].unsqueeze(0).repeat(N_block, 1)
                expanded_input = torch.cat((_emb, block_emb, rxn_features), dim=-1)
                logits[i] = mlp(expanded_input).squeeze(-1)
        return logits

    def hook_add_first_reactant(self, emb: torch.Tensor, block_emb: torch.Tensor):
        """
        The hook function to be called for the AddFirstReactant action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current state.
        block_emb : torch.Tensor
            The embedding tensor for building blocks.

        Returns
        torch.Tensor
            The logits or output of the MLP after being called with the expanded input.
        """
        N_graph = emb.size(0)
        N_block = block_emb.size(0)
        mlp = self.mlps[ReactionActionType.AddFirstReactant.cname]

        logits = torch.empty((N_graph, N_block), device=emb.device)
        for i in range(N_graph):
            _emb = emb[i].unsqueeze(0).repeat(N_block, 1)
            expanded_input = torch.cat((_emb, block_emb), dim=-1)
            logits[i] = mlp(expanded_input).squeeze(-1)
        return logits

    def single_hook_add_first_reactant(self, emb: torch.Tensor, block_emb: torch.Tensor):
        """
        The hook function to be called for the AddFirstReactant action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the a single current state.
        block_emb : torch.Tensor
            The embedding tensor for a single building blocks

        Returns
        torch.Tensor
            The logits or output of the MLP after being called with the expanded input.
        """
        expanded_input = torch.cat((emb, block_emb), dim=-1)
        return self.mlps[ReactionActionType.AddFirstReactant.cname](expanded_input).squeeze(-1)

    def single_hook_stop(self, emb: torch.Tensor):
        """
        The hook function to be called for the Stop action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current state.

        Returns
        torch.Tensor
            The logit for Stop
        """
        return self.mlps[ReactionActionType.Stop.cname](emb).view(-1)

    def single_hook_reactuni(self, emb: torch.Tensor, rxn_id: torch.Tensor):
        """
        The hook function to be called for the ReactUni action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current state.

        Returns
        torch.Tensor
            The logit for (rxn_id)
        """
        return self.mlps[ReactionActionType.ReactUni.cname](emb)[rxn_id]

    def single_hook_reactbi(self, emb: torch.Tensor, rxn_id: int, block_is_first: bool, block_emb: torch.Tensor):
        """
        The hook function to be called for the ReactBi action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current state.
        rxn_id : int
            The ID of the reaction selected by the sampler.
        block_is_first : bool
            The flag whether block is first reactant or second reactant of bimolecular reaction
        block_emb : torch.Tensor
            The embedding tensor for a single building block.

        Returns
        torch.Tensor
            The logit for (rxn_id, block_is_first, block).
        """
        rxn_features = torch.zeros(self.num_bimolecular_rxns * 2, device=emb.device)
        rxn_features[rxn_id * 2 + int(block_is_first)] = 1

        expanded_input = torch.cat((emb, block_emb, rxn_features), dim=-1)
        return self.mlps[ReactionActionType.ReactBi.cname](expanded_input).view(-1)
