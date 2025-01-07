import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.models.graph_transformer import GraphTransformer, mlp

from rxnflow.config import Config
from rxnflow.envs.env_context import SynthesisEnvContext
from rxnflow.policy.action_categorical import RxnActionCategorical


class BlockEmbedding(nn.Module):
    def __init__(
        self,
        fp_dim: int,
        desc_dim: int,
        n_type: int,
        n_hid: int,
        n_out: int,
        n_layers: int,
    ):
        super().__init__()
        self.emb_type = nn.Embedding(n_type, n_hid)
        self.lin_fp = nn.Linear(fp_dim, n_hid, bias=False)
        self.lin_desc = nn.Linear(desc_dim, n_hid, bias=True)
        self.mlp = mlp(n_hid, n_hid, n_out, n_layers)
        self.act = nn.LeakyReLU()

    def forward(self, block_data: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        typ, desc, fp = block_data
        x_typ = self.emb_type(typ)
        x_fp = self.lin_fp(fp)
        x_desc = self.lin_desc(desc)
        x = x_typ + x_fp + x_desc
        return self.mlp(self.act(x))


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

        num_emb_block = cfg.model.num_emb_block
        num_mlp_layers_block = cfg.model.num_mlp_layers_block

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
        self.emb_block = BlockEmbedding(
            env_ctx.block_fp_dim,
            env_ctx.block_desc_dim,
            env_ctx.num_block_types,
            num_emb_block,
            num_emb_block,
            num_mlp_layers_block,
        )

        embs = {p.name: nn.Parameter(torch.randn((num_glob_final,), requires_grad=True)) for p in env_ctx.protocols}
        self.emb_protocol = nn.ParameterDict(embs)
        self.mlp = mlp(num_glob_final, num_emb, num_emb_block, num_mlp_layers)
        self.act = nn.LeakyReLU()

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
        assert g.num_graphs == cond.shape[0]
        node_emb, graph_emb = self.transf(g, torch.cat([cond, g.graph_attr], dim=-1))
        # node_emb = node_emb[g.connect_atom] # NOTE: we need to modify here
        emb = graph_emb
        graph_out = self.emb2graph_out(emb)
        fwd_cat = self._make_cat(g, emb)
        return fwd_cat, graph_out

    def hook_adding_block(
        self,
        emb: Tensor,
        blocks: tuple[Tensor, Tensor, Tensor],
        protocol: str,
    ):
        """
        The hook function to be called for the FirstBlock, BiRxn.
        Parameters
        emb : Tensor
            The embedding tensor for the current states.
            shape: [Nstate, Fstate]
        blocks : tuple[Tensor, Tensor, Tensor]
            The building block features.
            shape:
                - LongTensor, [Nblock,]
                - FloatTensor, [Nblock, F_desc]
                - FloatTensor, [Nblock, F_fingerprint]
        protocol: str
            The name of protocol.

        Returns
        Tensor
            The logits of the MLP.
            shape: [Nstate, Nblock]
        """
        state_emb: Tensor = self.mlp(self.act(emb + self.emb_protocol[protocol].view(1, -1)))
        block_emb = self.emb_block(blocks)

        return state_emb @ block_emb.T
