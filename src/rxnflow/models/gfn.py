import torch
import torch.nn as nn
import torch_geometric.data as gd

from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer, mlp

from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext
from rxnflow.policy.action_categorical import RxnActionCategorical


class BiRxnMLP(nn.Module):
    def __init__(self, n_in: int, n_bi_rxn: int, n_hid: int, n_out: int, n_layer: int):
        super().__init__()
        self.lin_state = nn.Linear(n_in, n_hid)
        self.lin_rxn = nn.Embedding(n_bi_rxn, n_hid)
        self.mlp = mlp(n_hid, n_hid, n_out, n_layer)
        self.reakyrelu = nn.LeakyReLU()

    def forward(self, state: Tensor, bi_rxn_idx: int) -> Tensor:
        state_emb: Tensor = self.lin_state(state)
        dims = (1,) * (state_emb.dim() - 1) + (-1,)
        rxn_emb = self.lin_rxn.weight[bi_rxn_idx].view(dims)
        return self.mlp(self.reakyrelu(state_emb + rxn_emb))


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
        self.num_uni_rxns: int = env_ctx.num_uni_rxns
        self.num_bi_rxns: int = env_ctx.num_bi_rxns

        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.graph_transformer.num_layers,
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

        assert do_bck is False
        self.do_bck = do_bck

        num_mlp_layers: int = cfg.model.num_mlp_layers
        self.mlp_firstblock = mlp(num_glob_final, num_emb, num_emb_block, 0)
        self.mlp_stop = mlp(num_glob_final, num_emb, 1, num_mlp_layers)
        self.mlp_uni_rxn = mlp(num_glob_final, num_emb, self.num_uni_rxns, num_mlp_layers)
        self.mlp_bi_rxn = BiRxnMLP(num_glob_final, self.num_bi_rxns * 2, num_emb, num_emb_block, num_mlp_layers)

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def logZ(self, cond_info: Tensor) -> Tensor:
        return self._logZ(cond_info)

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

    def block_embedding(self, block: Tensor) -> Tensor:
        return self.block_mlp(block)

    def hook_firstblock(self, emb: Tensor, block_emb: Tensor):
        """
        The hook function to be called for the FirstBlock action.
        Parameters
        emb : Tensor
            The embedding tensor for the current states [Ngraph, F].
        block_emb : Tensor
            The embedding tensor for building blocks [Nblock, F].

        Returns
        Tensor
            The logit of the MLP.
        """
        return self.mlp_firstblock(emb) @ block_emb.T

    def single_hook_firstblock(self, emb: Tensor, block_emb: Tensor):
        """
        The hook function to be called for the FirstBlock action.
        Parameters
        emb : Tensor
            The embedding tensor for the a single current state. [F]
        block_emb : Tensor
            The embedding tensor for a single building block. [F]

        Returns
        Tensor
            The logit of the MLP.
        """
        return self.mlp_firstblock(emb) @ block_emb

    def hook_stop(self, emb: Tensor) -> Tensor:
        """
        The hook function to be called for the Stop action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.

        Returns
        Tensor
            The logits for Stop
        """

        return self.mlp_stop(emb)

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
        return self.mlp_stop(emb).view(-1)

    def hook_uni_rxn(self, emb: Tensor, mask: Tensor):
        """
        The hook function to be called for the UniRxn action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.

        Returns
        Tensor
            The logits for UniRxn
        """

        logit = self.mlp_uni_rxn(emb)
        return logit.masked_fill(torch.logical_not(mask), -torch.inf)

    def single_hook_uni_rxn(self, emb: Tensor, rxn_id: Tensor):
        """
        The hook function to be called for the UniRxn action.
        Parameters
        emb : Tensor
            The embedding tensor for the current state.

        Returns
        Tensor
            The logit for (rxn_id)
        """
        return self.mlp_uni_rxn(emb)[rxn_id]

    def hook_bi_rxn(self, emb: Tensor, rxn_id: int, block_is_first: bool, block_emb: Tensor, mask: Tensor) -> Tensor:
        """
        The hook function to be called for the BiRxn action.
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
        state_rxn_emb: Tensor = self.mlp_bi_rxn.forward(emb, rxn_id * 2 + int(block_is_first))
        logits: Tensor = state_rxn_emb @ block_emb.T
        return logits.masked_fill(torch.logical_not(mask.unsqueeze(-1)), -torch.inf)

    def single_hook_bi_rxn(self, emb: Tensor, rxn_id: int, block_is_first: bool, block_emb: Tensor) -> Tensor:
        """
        The hook function to be called for the BiRxn action.
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
        state_rxn_emb: Tensor = self.mlp_bi_rxn.forward(emb, rxn_id * 2 + int(block_is_first))
        return state_rxn_emb @ block_emb
