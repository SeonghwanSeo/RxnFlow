import torch
import torch.nn as nn
import torch_geometric.data as gd

from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer, mlp

from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext
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
        assert do_bck is False
        self.do_bck = do_bck

        num_emb = cfg.model.num_emb
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings
        num_mlp_layers = cfg.model.num_mlp_layers
        num_emb_block = cfg.model.num_emb_block
        num_mlp_layers_block = cfg.model.num_mlp_layers_block

        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.graph_transformer.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )
        # NOTE: Block embedding
        self.mlp_block = mlp(
            env_ctx.num_block_features,
            num_emb_block,
            num_emb_block,
            num_mlp_layers_block,
        )

        # NOTE: Markov Decision Process
        self.emb_unirxn = nn.ParameterDict(
            {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.unirxn_list}
        )
        self.emb_birxn = nn.ParameterDict(
            {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.birxn_list}
        )

        self.mlp_stop = mlp(num_glob_final, num_emb, 1, num_mlp_layers)
        self.mlp_firstblock = mlp(num_glob_final, num_emb, num_emb_block, num_mlp_layers)
        self.mlp_unirxn = mlp(num_glob_final, num_emb, 1, num_mlp_layers)
        self.mlp_birxn = mlp(num_glob_final, num_emb, num_emb_block, num_mlp_layers)
        self.act = nn.LeakyReLU()

        # NOTE: Etcs. (e.g., partition function)
        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def logZ(self, cond_info: Tensor) -> Tensor:
        return self._logZ(cond_info)

    def _make_cat(self, g: gd.Batch, emb: Tensor) -> RxnActionCategorical:
        action_masks = list(torch.unbind(g.protocol_mask, dim=1))  # [Ngraph, Nprotocol]
        return RxnActionCategorical(g, emb, action_masks=action_masks, model=self)

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
        fwd_cat = self._make_cat(g, emb)

        if self.do_bck:
            raise NotImplementedError
            bck_cat = self._make_cat(g, emb, fwd=False)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out

    def block_embedding(self, block: Tensor):
        return self.mlp_block(block)

    def hook_stop(self, emb: Tensor):
        """
        The hook function to be called for the Stop.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        return self.mlp_stop(emb)

    def hook_firstblock(self, emb: Tensor, block: Tensor):
        """
        The hook function to be called for the FirstBlock.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        block : Tensor
            The building block features.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        state_emb = self.mlp_firstblock(emb)
        block_emb = self.block_embedding(block)
        return state_emb @ block_emb.T

    def hook_unirxn(self, emb: Tensor, protocol: str):
        """
        The hook function to be called for the UniRxn.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        protocol: str
            The name of protocol.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        return self.mlp_unirxn(self.act(emb + self.emb_unirxn[protocol].view(1, -1)))

    def hook_birxn(
        self,
        emb: Tensor,
        block: Tensor,
        protocol: str,
    ):
        """
        The hook function to be called for the BiRxn.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        block : Tensor
            The building block features.
        protocol: str
            The name of protocol.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        state_emb = self.mlp_birxn(self.act(emb + self.emb_birxn[protocol].view(1, -1)))
        block_emb = self.block_embedding(block)
        return state_emb @ block_emb.T
