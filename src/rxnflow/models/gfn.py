import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer, mlp

from rxnflow.config import Config
from rxnflow.envs.env_context import SynthesisEnvContext
from rxnflow.policy.action_categorical import RxnActionCategorical


# NOTE: Hyperparameters
ACT_BLOCK = nn.LeakyReLU


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
        self.do_bck: bool = do_bck
        self.num_graph_out: int = num_graph_out

        num_emb = cfg.model.num_emb
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings
        num_mlp_layers: int = cfg.model.num_mlp_layers
        num_block_dim = env_ctx.num_block_features
        num_block_mlp_layers: int = cfg.model.num_mlp_layers_block

        # NOTE: State embedding
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim + env_ctx.num_graph_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.graph_transformer.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )
        # NOTE: Block embedding
        self.mlp_block = mlp(num_block_dim, num_emb, num_emb, n_layer=num_block_mlp_layers, act=ACT_BLOCK)

        mlps = {
            k.name: mlp(num_glob_final, num_emb, num_emb, num_mlp_layers)
            for k in env_ctx.firstblock_list + env_ctx.birxn_list
        }
        mlps.update(
            {k.name: mlp(num_glob_final, num_emb, 1, num_mlp_layers) for k in env_ctx.unirxn_list + env_ctx.stop_list}
        )
        self.mlp = nn.ModuleDict(mlps)

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.transf.graph2emb.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def logZ(self, cond_info: Tensor) -> Tensor:
        return self._logZ(cond_info)

    def _make_cat(self, g: gd.Batch, emb: Tensor) -> RxnActionCategorical:
        return RxnActionCategorical(emb, action_masks=g.protocol_mask, model=self)

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
        assert g.num_graphs == cond.shape[0]
        node_emb, graph_emb = self.transf(g, torch.cat([cond, g.graph_attr], dim=-1))
        # node_emb = node_emb[g.connect_atom] # NOTE: we need to modify here
        emb = graph_emb
        graph_out = self.emb2graph_out(emb)
        fwd_cat = self._make_cat(g, emb)

        if self.do_bck:
            raise NotImplementedError
        return fwd_cat, graph_out

    def hook_uni_action(self, emb: Tensor, protocol: str):
        """
        The hook function to be called for the UniRxn and Stop action.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate,]
        protocol: str
            The name of protocol.

        Returns
        Tensor
            The logits for protocol
            shape: [Nstate, 1]
        """
        return self.mlp[protocol](emb)

    def hook_adding_block(self, emb: Tensor, blocks: Tensor, protocol: str):
        """
        The hook function to be called for the FirstBlock and BiRxn action.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        blocks : Tensor
            The tensor for building blocks.
            shape: [Nblock, Fblock]
        protocol: str
            The name of protocol.

        Returns
        Tensor
            The logits for protocol
            shape: [Nstate, Nblock]
        """
        state_emb: Tensor = self.mlp[protocol](emb)
        block_emb = self.mlp_block(blocks)

        return state_emb @ block_emb.T
