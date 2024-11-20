from itertools import chain

import torch
import torch.nn as nn
import torch_geometric.data as gd

from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer, mlp

from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext, RxnActionType
from rxnflow.policy.action_categorical import RxnActionCategorical


class RxnFlow(TrajectoryBalanceModel):
    """GraphTransfomer class which outputs an RxnActionCategorical."""

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
            RxnActionType.Stop: (num_glob_final, 1),
            RxnActionType.AddFirstReactant: (num_glob_final, num_emb_block),
            RxnActionType.ReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            RxnActionType.ReactBi: (num_glob_final + env_ctx.num_bimolecular_rxns * 2, num_emb_block),
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

    def _make_cat(self, g, emb, fwd):
        return RxnActionCategorical(g, emb, model=self, fwd=fwd)

    def forward(self, g: gd.Batch, cond: Tensor) -> tuple[RxnActionCategorical, Tensor]:
        """

        Parameters
        ----------
        g : gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond : Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        -------
        RxnActionCategorical
        """
        _, emb = self.transf(g, cond)
        graph_out = self.emb2graph_out(emb)
        fwd_cat = self._make_cat(g, emb, fwd=True)

        if self.do_bck:
            raise NotImplementedError
            bck_cat = self._make_cat(g, emb, fwd=False)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out

    def logZ(self, cond: Tensor) -> Tensor:
        return self._logZ(cond)

    def block_embedding(self, block: Tensor) -> Tensor:
        return self.block_mlp(block)

    def hook_stop(self, emb: Tensor):
        """
        The hook function to be called for the Stop action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.

        Returns
        Tensor
            The logits for Stop
        """

        return self.mlps[RxnActionType.Stop.cname](emb)

    def hook_reactuni(self, emb: Tensor, mask: Tensor):
        """
        The hook function to be called for the ReactUni action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.

        Returns
        Tensor
            The logits for ReactUni
        """

        logit = self.mlps[RxnActionType.ReactUni.cname](emb)
        return logit.masked_fill(torch.logical_not(mask), -torch.inf)

    def hook_reactbi(self, emb: Tensor, rxn_id: int, block_is_first: bool, block_emb: Tensor, mask: Tensor):
        """
        The hook function to be called for the ReactBi action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.
        rxn_id : int
            The ID of the reaction selected by the sampler.
        block_is_first : bool
            The flag whether block is first reactant or second reactant of bimolecular reaction
        block_emb : Tensor
            The embedding tensor for building blocks.

        Returns
        Tensor
            The logits for (rxn_id, block_is_first).
        """
        N_graph = emb.size(0)
        mlp = self.mlps[RxnActionType.ReactBi.cname]

        # NOTE: We improve the logit calculation
        rxn_features = torch.zeros(N_graph, self.num_bimolecular_rxns * 2, device=emb.device)
        rxn_features[:, rxn_id * 2 + int(block_is_first)] = 1
        _emb = mlp(torch.cat([emb, rxn_features], dim=-1))
        logits = torch.matmul(_emb, block_emb.T)  # [N_graph, N_block]
        return logits.masked_fill(torch.logical_not(mask), -torch.inf)

    def hook_add_first_reactant(self, emb: Tensor, block_emb: Tensor):
        """
        The hook function to be called for the AddFirstReactant action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.
        block_emb : Tensor
            The embedding tensor for building blocks.

        Returns
        Tensor
            The logits or output of the MLP after being called with the expanded input.
        """
        mlp = self.mlps[RxnActionType.AddFirstReactant.cname]
        _emb = mlp(emb)
        return torch.matmul(_emb, block_emb.T)

    def single_hook_add_first_reactant(self, emb: Tensor, block_emb: Tensor):
        """
        The hook function to be called for the AddFirstReactant action.
        Parameters
        emb : Tensor
            The embedding tensor for the a single current state.
        block_emb : Tensor
            The embedding tensor for a single building blocks

        Returns
        Tensor
            The logits or output of the MLP after being called with the expanded input.
        """
        mlp = self.mlps[RxnActionType.AddFirstReactant.cname]
        return torch.matmul(mlp(emb), block_emb.T)

    def single_hook_stop(self, emb: Tensor):
        """
        The hook function to be called for the Stop action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.

        Returns
        Tensor
            The logit for Stop
        """
        return self.mlps[RxnActionType.Stop.cname](emb).view(-1)

    def single_hook_reactuni(self, emb: Tensor, rxn_id: Tensor):
        """
        The hook function to be called for the ReactUni action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.

        Returns
        Tensor
            The logit for (rxn_id)
        """
        return self.mlps[RxnActionType.ReactUni.cname](emb)[rxn_id]

    def single_hook_reactbi(self, emb: Tensor, rxn_id: int, block_is_first: bool, block_emb: Tensor):
        """
        The hook function to be called for the ReactBi action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.
        rxn_id : int
            The ID of the reaction selected by the sampler.
        block_is_first : bool
            The flag whether block is first reactant or second reactant of bimolecular reaction
        block_emb : Tensor
            The embedding tensor for a single building block.

        Returns
        Tensor
            The logit for (rxn_id, block_is_first, block).
        """
        mlp = self.mlps[RxnActionType.ReactBi.cname]

        rxn_features = torch.zeros(self.num_bimolecular_rxns * 2, device=emb.device)
        rxn_features[rxn_id * 2 + int(block_is_first)] = 1
        _emb = mlp(torch.cat([emb, rxn_features], dim=-1))
        return torch.matmul(_emb, block_emb.T)
