from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.data as gd

from gflownet.config import Config
from gflownet.models.graph_transformer import GraphTransformer, mlp

from gflownet.envs.synthesis import ReactionActionType, SynthesisEnvContext, ReactionActionCategorical


class ASGFN_Synthesis(nn.Module):
    """GraphTransfomer class for a ASTB which outputs an ReactionActionCategorical.

    Outputs logits corresponding to each action (template).
    """

    # The GraphTransformer outputs per-graph embeddings
    _action_type_to_graph_part = {
        ReactionActionType.Stop: "graph",
        ReactionActionType.AddFirstReactant: "graph",
        ReactionActionType.ReactUni: "graph",
        ReactionActionType.ReactBi: "graph",
        ReactionActionType.BckRemoveFirstReactant: "graph",
        ReactionActionType.BckReactUni: "graph",
        ReactionActionType.BckReactBi: "graph",
    }

    def __init__(
        self,
        env_ctx: SynthesisEnvContext,
        cfg: Config,
        num_graph_out=1,
        do_bck=True,
    ) -> None:
        super().__init__()
        self.env_ctx: SynthesisEnvContext = env_ctx

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
            cfg.model.fp_nbits_building_block,
            cfg.model.num_emb_building_block,
            cfg.model.num_emb_building_block,
            cfg.model.num_layers_building_block,
        )

        num_emb = cfg.model.num_emb
        num_emb_block = cfg.model.num_emb_building_block
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings

        self.primary_action_type_order = (
            ReactionActionType.Stop,
            ReactionActionType.ReactUni,
            ReactionActionType.ReactBi,
        )
        self.secondary_action_type_order = (
            ReactionActionType.AddFirstReactant,
            ReactionActionType.AddReactant,
        )
        self.bck_primary_action_type_order = (
            ReactionActionType.BckRemoveFirstReactant,
            ReactionActionType.BckReactUni,
            ReactionActionType.BckReactBi,
        )
        self.bck_secondary_action_type_order = tuple()

        # 3 Action types: ADD_REACTANT, REACT_UNI, REACT_BI
        # Every action type gets its own MLP that is fed the output of the GraphTransformer.
        # Here we define the number of inputs and outputs of each of those (potential) MLPs.
        self._action_type_to_num_inputs_outputs = {
            ReactionActionType.Stop: (num_glob_final, 1),
            ReactionActionType.AddFirstReactant: (num_glob_final + num_emb_block, 1),
            ReactionActionType.AddReactant: (num_glob_final + env_ctx.num_bimolecular_rxns + num_emb_block, 1),
            ReactionActionType.ReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ReactionActionType.ReactBi: (num_glob_final, env_ctx.num_bimolecular_rxns),
            ReactionActionType.BckReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ReactionActionType.BckReactBi: (num_glob_final, env_ctx.num_bimolecular_rxns),
            ReactionActionType.BckRemoveFirstReactant: (num_glob_final, 1),
        }

        self.do_bck = do_bck
        mlps = {}
        for atype in chain(env_ctx.action_type_order, env_ctx.bck_action_type_order if do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.graph_transformer.num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def _action_type_to_mask(self, t, g) -> Optional[torch.Tensor]:
        # NOTE: ActionType.AddReactant gets masked in ActionCategorical class, not here
        # NOTE: For the first step, all logits get masked except those for AddFirstReactant
        assert t != ReactionActionType.AddReactant
        return getattr(g, t.mask_name, None)

    def _action_type_to_logit(self, t, emb, g) -> torch.Tensor:
        logits = self.mlps[t.cname](emb[self._action_type_to_graph_part[t]])
        return logits

    def _make_cat(self, g, emb, action_types, fwd):
        raw_logits = {typ: self._action_type_to_logit(typ, emb, g) for typ in action_types}
        masks = {typ: self._action_type_to_mask(typ, g) for typ in action_types}
        return ReactionActionCategorical(g, raw_logits, masks, emb, model=self, fwd=fwd)

    def add_first_reactant_hook(self, graph_embedding: torch.Tensor, block_emb: torch.Tensor):
        """
        The hook function to be called for the AddFirstReactant action.
        Parameters
        emb : torch.Tensor
            The embedding tensor for the current states.
        block_emb : torch.Tensor
            The embedding tensor for building blocks.
        g : Graph
            The current graph.

        Returns
        torch.Tensor
            The logits or output of the MLP after being called with the expanded input.
        """
        _emb = graph_embedding.unsqueeze(0).repeat(block_emb.size(0), 1)
        expanded_input = torch.cat((_emb, block_emb), dim=-1)  # [N_block, 2 * F]
        return self.mlps[ReactionActionType.AddFirstReactant.cname](expanded_input).squeeze(-1)

    def add_reactant_hook(self, rxn_id: int, graph_embedding: torch.Tensor, block_emb: torch.Tensor):
        """
        The hook function to be called for the AddReactant action.
        Parameters
        rxn_id : int
            The ID of the reaction selected by the sampler.
        emb : torch.Tensor
            The embedding tensor for the current state.
        block_emb : torch.Tensor
            The embedding tensor for building blocks.

        Returns
        torch.Tensor
            The logits or output of the MLP after being called with the expanded input.
        """

        # Convert `rxn_id` to a one-hot vector
        rxn_features = torch.zeros(self.env_ctx.num_bimolecular_rxns, device=graph_embedding.device)
        rxn_features[rxn_id] = 1

        expanded_input = torch.cat((graph_embedding, rxn_features), dim=-1)
        expanded_input = expanded_input.unsqueeze(0).repeat(block_emb.size(0), 1)
        expanded_input = torch.cat((expanded_input, block_emb), dim=-1)  # [N_block, 2 * F + Fcond]

        return self.mlps[ReactionActionType.AddReactant.cname](expanded_input).squeeze(-1)

    def _mask(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        assert m.dtype == torch.bool
        return x.masked_fill(torch.logical_not(m), -torch.inf)

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        """
        Forward pass of the ASTB.

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
        _, graph_embeddings = self.transf(g, cond)
        emb = {
            "graph": graph_embeddings,
        }
        graph_out = self.emb2graph_out(graph_embeddings)
        fwd_cat = self._make_cat(g, emb, self.env_ctx.primary_action_type_order, fwd=True)

        if self.do_bck:
            bck_cat = self._make_cat(g, emb, self.env_ctx.primary_bck_action_type_order, fwd=False)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out
