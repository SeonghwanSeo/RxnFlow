import torch
import torch.nn as nn
import torch_geometric.data as gd

from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer, mlp

from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext
from rxnflow.policy.action_categorical import RxnActionCategorical
from rxnflow.models.utils import init_weight_linear

ACT_BLOCK = nn.SiLU
ACT_MDP = nn.SiLU
ACT_TB = nn.LeakyReLU


__all__ = ["RxnFlow", "BlockEmbedding"]


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

        # NOTE: State embedding
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
        self.emb_block = BlockEmbedding(
            env_ctx.block_fp_dim,
            env_ctx.block_prop_dim,
            num_emb_block,
            num_emb_block,
            num_mlp_layers_block,
        )

        # NOTE: Markov Decision Process
        self.mlp = nn.ModuleDict(
            {
                "stop": mlp(num_glob_final, num_emb, 1, num_mlp_layers, act=ACT_MDP),
                "firstblock": mlp(num_glob_final, num_emb, num_emb_block, num_mlp_layers, act=ACT_MDP),
                "unirxn": mlp(num_glob_final, num_emb, 1, num_mlp_layers, act=ACT_MDP),
                "birxn": mlp(num_glob_final, num_emb, num_emb_block, num_mlp_layers, act=ACT_MDP),
            }
        )
        # reaction embedding
        self.emb_unirxn = nn.ParameterDict(
            {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.unirxn_list}
        )
        self.emb_birxn = nn.ParameterDict(
            {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.birxn_list}
        )

        # NOTE: Etcs. (e.g., partition function)
        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, num_mlp_layers, act=ACT_TB)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2, act=ACT_TB)
        self._logit_scale: nn.Module = nn.Sequential(
            mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2, act=ACT_TB),
            nn.ELU(),  # to be (-1, inf)
        )
        self.reset_parameters()

    def logZ(self, cond_info: Tensor) -> Tensor:
        return self._logZ(cond_info)

    def logit_scale(self, cond_info: Tensor) -> Tensor:
        """return non-negative scale"""
        return self._logit_scale(cond_info).view(-1) + 1  # (-1, inf) -> (0, inf)

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
        protocol_masks = list(torch.unbind(g.protocol_mask, dim=1))  # [Ngraph, Nprotocol]
        logit_scale = self.logit_scale(cond)
        fwd_cat = RxnActionCategorical(g, emb, logit_scale, protocol_masks, model=self)
        if self.do_bck:
            raise NotImplementedError
        return fwd_cat, graph_out

    def hook_firstblock(
        self,
        emb: Tensor,
        block: tuple[Tensor, Tensor],
    ):
        """
        The hook function to be called for the FirstBlock.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        block : tuple[Tensor, Tensor]
            The building block features.
            shape:
                [Nblock, D_fp]
                [Nblock, D_prop]

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        state_emb = self.mlp["firstblock"](emb)
        block_emb = self.emb_block(block)
        return state_emb @ block_emb.T

    def hook_birxn(
        self,
        emb: Tensor,
        block: tuple[Tensor, Tensor],
        protocol: str,
    ):
        """
        The hook function to be called for the BiRxn.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        block : tuple[Tensor, Tensor]
            The building block features.
            shape:
                [Nblock, D_fp]
                [Nblock, D_prop]
        protocol: str
            The name of synthesis protocol.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        state_emb = self.mlp["birxn"](emb + self.emb_birxn[protocol].view(1, -1))
        block_emb = self.emb_block(block)
        return state_emb @ block_emb.T

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
            shape: [Nstate, 1]
        """
        return self.mlp["stop"](emb)

    def hook_unirxn(self, emb: Tensor, protocol: str):
        """
        The hook function to be called for the UniRxn.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        protocol: str
            The name of synthesis protocol.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, 1]
        """
        return self.mlp["unirxn"](emb + self.emb_unirxn[protocol].view(1, -1))

    def reset_parameters(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                init_weight_linear(m, ACT_MDP)

        for layer in [self.emb2graph_out, self._logZ, self._logit_scale]:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    init_weight_linear(m, ACT_TB)


class BlockEmbedding(nn.Module):
    def __init__(self, fp_dim: int, prop_dim: int, n_hid: int, n_out: int, n_layers: int):
        super().__init__()
        self.lin_fp = nn.Sequential(
            nn.Linear(fp_dim, n_hid),
            ACT_BLOCK(),
            nn.LayerNorm(n_hid),
        )
        self.lin_prop = nn.Sequential(
            nn.Linear(prop_dim, n_hid),
            ACT_BLOCK(),
            nn.LayerNorm(n_hid),
        )
        self.mlp = mlp(2 * n_hid, n_hid, n_out, n_layers, act=ACT_BLOCK)
        self.reset_parameters()

    def forward(self, block_data: tuple[Tensor, Tensor]):
        fp, prop = block_data
        x_fp = self.lin_fp(fp)
        x_prop = self.lin_prop(prop)
        x = torch.cat([x_fp, x_prop], dim=-1)
        return self.mlp(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weight_linear(m, ACT_BLOCK)
