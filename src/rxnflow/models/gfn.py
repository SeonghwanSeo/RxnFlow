import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext
from rxnflow.models.nn import init_weight_linear, mlp
from rxnflow.policy.action_categorical import RxnActionCategorical

ACT_BLOCK = nn.SiLU
ACT_MDP = nn.SiLU
ACT_TB = nn.LeakyReLU


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
        num_layers = cfg.model.num_mlp_layers
        num_emb_block = cfg.model.num_emb_block
        dropout = cfg.model.dropout

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
        # For regular GFN, normalization does not effect to model performance.
        # However, we add normalization to match the scale with reaction embedding
        self.norm_state = nn.LayerNorm(num_glob_final)

        # NOTE: Block embedding
        self.emb_block = BlockEmbedding(
            env_ctx.block_fp_dim,
            env_ctx.block_prop_dim,
            num_emb_block,
            num_emb_block,
            cfg.model.num_mlp_layers_block,
            ACT_BLOCK,
            dropout,
        )

        # NOTE: Markov Decision Process
        mlps = {
            "stop": mlp(num_glob_final, num_emb, 1, num_layers, ACT_MDP, dropout=dropout),
            "firstblock": mlp(num_glob_final, num_emb, num_emb_block, num_layers, ACT_MDP, dropout=dropout),
            "birxn": mlp(num_glob_final, num_emb, num_emb_block, num_layers, ACT_MDP, dropout=dropout),
            "unirxn": mlp(num_glob_final, num_emb, 1, num_layers, ACT_MDP, dropout=dropout),
        }
        self.mlp_mdp = nn.ModuleDict(mlps)

        # NOTE: reaction embedding
        self.emb_unirxn = nn.ParameterDict(
            {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.unirxn_list}
        )
        self.emb_birxn = nn.ParameterDict(
            {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.birxn_list}
        )
        self.act_mdp = ACT_MDP()

        # NOTE: Etcs. (e.g., partition function)
        self._emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, num_layers, ACT_TB, dropout=dropout)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2, ACT_TB, dropout=dropout)
        self._logit_scale = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2, ACT_TB, dropout=dropout)
        self.reset_parameters()

    def emb2graph_out(self, emb: Tensor) -> Tensor:
        return self._emb2graph_out(emb)

    def logZ(self, cond_info: Tensor) -> Tensor:
        """return log partition funciton"""
        return self._logZ(cond_info)

    def logit_scale(self, cond_info: Tensor) -> Tensor:
        """return non-negative scale"""
        return nn.functional.elu(self._logit_scale(cond_info).view(-1)) + 1  # (-1, inf) -> (0, inf)

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
        _, emb = self.transf(g, torch.cat([cond, g.graph_attr], axis=-1))
        emb = self.norm_state(emb)
        protocol_masks = list(torch.unbind(g.protocol_mask, dim=1))  # [Ngraph, Nprotocol]
        logit_scale = self.logit_scale(cond)
        fwd_cat = RxnActionCategorical(g, emb, logit_scale, protocol_masks, model=self)
        graph_out = self.emb2graph_out(emb)
        if self.do_bck:
            raise NotImplementedError
        return fwd_cat, graph_out

    def block_embedding(self, block: tuple[Tensor, Tensor]) -> Tensor:
        return self.emb_block(block)

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
        state_emb = self.mlp_mdp["firstblock"](emb)
        block_emb = self.block_embedding(block)
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
        emb = emb + self.emb_birxn[protocol].view(1, -1)
        state_emb = self.mlp_mdp["birxn"](self.act_mdp(emb))
        block_emb = self.block_embedding(block)
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
        return self.mlp_mdp["stop"](emb)

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
        emb = emb + self.emb_unirxn[protocol].view(1, -1)
        return self.mlp_mdp["unirxn"](self.act_mdp(emb))

    def reset_parameters(self):
        for m in self.mlp_mdp.modules():
            if isinstance(m, nn.Linear):
                init_weight_linear(m, ACT_MDP)

        for layer in [self._emb2graph_out, self._logZ, self._logit_scale]:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    init_weight_linear(m, ACT_TB)


class BlockEmbedding(nn.Module):
    def __init__(
        self,
        fp_dim: int,
        prop_dim: int,
        n_hid: int,
        n_out: int,
        n_layers: int,
        act: type[nn.Module],
        dropout: float,
    ):
        super().__init__()
        self.lin_fp = nn.Sequential(
            nn.Linear(fp_dim, n_hid),
            nn.LayerNorm(n_hid),
            act(),
            nn.Dropout(dropout),
        )
        self.lin_prop = nn.Sequential(
            nn.Linear(prop_dim, n_hid),
            nn.LayerNorm(n_hid),
            act(),
            nn.Dropout(dropout),
        )
        self.mlp = mlp(2 * n_hid, n_hid, n_out, n_layers, act=act, dropout=dropout)
        self.reset_parameters(act)

    def forward(self, block_data: tuple[Tensor, Tensor]):
        fp, prop = block_data
        x_fp = self.lin_fp(fp)
        x_prop = self.lin_prop(prop)
        x = torch.cat([x_fp, x_prop], dim=-1)
        return self.mlp(x)

    def reset_parameters(self, act):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weight_linear(m, act)
