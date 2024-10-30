import torch
import torch_geometric.data as gd
from torch import Tensor

from gflownet.utils.misc import get_worker_device
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext
from rxnflow.policy import RxnActionCategorical
from rxnflow.models.gfn import RxnFlow
from rxnflow.utils.misc import get_worker_env

from .utils import PocketDB
from .pocket.gvp import GVP_embedding


class RxnFlow_PocketConditional(RxnFlow):
    def __init__(self, env_ctx: SynthesisEnvContext, cfg: Config, num_graph_out=1, do_bck=False):
        pocket_dim = self.pocket_dim = cfg.task.pocket_conditional.pocket_dim
        org_num_cond_dim = env_ctx.num_cond_dim
        env_ctx.num_cond_dim = org_num_cond_dim + pocket_dim
        super().__init__(env_ctx, cfg, num_graph_out, do_bck)
        env_ctx.num_cond_dim = org_num_cond_dim

        self.pocket_encoder = GVP_embedding((6, 3), (pocket_dim, 16), (32, 1), (32, 1), seq_in=True, vocab_size=20)
        self.pocket_embed: torch.Tensor | None = None

    def get_pocket_embed(self, force: bool = False):
        if (self.pocket_embed is None) or force:
            dev = get_worker_device()
            task = get_worker_env("task")
            pocket_db: PocketDB = task.pocket_db
            _, pocket_embed = self.pocket_encoder.forward(pocket_db.batch_g.to(dev))
            self.pocket_embed = pocket_embed
        return self.pocket_embed


class RxnFlow_MP(RxnFlow_PocketConditional):
    """
    Model which can be trained on multiple pocket conditions,
    For Zero-shot sampling
    """

    def __init__(self, env_ctx: SynthesisEnvContext, cfg: Config, num_graph_out=1, do_bck=False):
        super().__init__(env_ctx, cfg, num_graph_out, do_bck)

    def forward(self, g: gd.Batch, cond: torch.Tensor, batch_idx: torch.Tensor) -> tuple[RxnActionCategorical, Tensor]:
        self.pocket_embed = self.get_pocket_embed()
        cond_cat = torch.cat([cond, self.pocket_embed[batch_idx]], dim=-1)
        return super().forward(g, cond_cat)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        self.pocket_embed = self.get_pocket_embed()
        cond_cat = torch.cat([cond, self.pocket_embed], dim=-1)
        return self._logZ(cond_cat)

    def clear_cache(self):
        self.pocket_embed = None


class RxnFlow_SP(RxnFlow_PocketConditional):
    """
    Model which can be trained on single pocket conditions
    For Inference or Few-shot training
    """

    def __init__(
        self,
        env_ctx: SynthesisEnvContext,
        cfg: Config,
        num_graph_out: int = 1,
        do_bck: bool = False,
        freeze_pocket_embedding: bool = True,
        freeze_action_embedding: bool = True,
    ):
        super().__init__(env_ctx, cfg, num_graph_out, do_bck)
        self.pocket_embed: torch.Tensor | None = None
        self.freeze_pocket_embedding: bool = freeze_pocket_embedding
        self.freeze_action_embedding: bool = freeze_action_embedding

        # NOTE: Freeze Pocket Encoder
        if freeze_pocket_embedding:
            for param in self.pocket_encoder.parameters():
                param.requires_grad = False

        if freeze_action_embedding:
            for param in self.block_mlp.parameters():
                param.requires_grad = False

    def forward(self, g: gd.Batch, cond: Tensor) -> tuple[RxnActionCategorical, Tensor]:
        if self.freeze_pocket_embedding:
            self.pocket_encoder.eval()
        if self.freeze_action_embedding:
            self.block_mlp.eval()

        self.pocket_embed = self.get_pocket_embed()
        pocket_embed = self.pocket_embed.view(1, -1).repeat(g.num_graphs, 1)
        cond_cat = torch.cat([cond, pocket_embed], dim=-1)
        return super().forward(g, cond_cat)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        self.pocket_embed = self.get_pocket_embed()
        pocket_embed = self.pocket_embed.view(1, -1).repeat(cond.shape[0], 1)
        cond_cat = torch.cat([cond, pocket_embed], dim=-1)
        return self._logZ(cond_cat)

    def get_pocket_embed(self, force: bool = False):
        if self.freeze_pocket_embedding:
            with torch.no_grad():
                return super().get_pocket_embed(force)
        else:
            return super().get_pocket_embed(force)

    def block_embedding(self, block: Tensor):
        if self.freeze_action_embedding:
            with torch.no_grad():
                return super().block_embedding(block)
        else:
            return super().block_embedding(block)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_pocket_embedding:
            self.pocket_encoder.eval()
        if self.freeze_action_embedding:
            self.block_mlp.eval()

        return self