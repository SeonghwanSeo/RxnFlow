import torch
import torch_geometric.data as gd
from torch import Tensor, nn

from gflownet.utils.misc import get_worker_device
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnvContext
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy import RxnActionCategorical
from rxnflow.utils.misc import get_worker_env

from .pocket.gvp import GVP_embedding
from .utils import PocketDB


class RxnFlow_PocketConditional(RxnFlow):
    def __init__(self, env_ctx: SynthesisEnvContext, cfg: Config, num_graph_out=1, do_bck=False):
        pocket_dim = self.pocket_dim = cfg.task.pocket_conditional.pocket_dim
        org_num_cond_dim = env_ctx.num_cond_dim
        env_ctx.num_cond_dim = org_num_cond_dim + pocket_dim
        super().__init__(env_ctx, cfg, num_graph_out, do_bck)
        env_ctx.num_cond_dim = org_num_cond_dim

        self.pocket_encoder = GVP_embedding((6, 3), (pocket_dim, 16), (32, 1), (32, 1), seq_in=True, vocab_size=20)
        self.pocket_norm = nn.LayerNorm(pocket_dim)
        self.pocket_embed: torch.Tensor | None = None

    def get_pocket_embed(self, force: bool = False):
        if (self.pocket_embed is None) or force:
            dev = get_worker_device()
            task = get_worker_env("task")
            pocket_db: PocketDB = task.pocket_db
            _, pocket_embed = self.pocket_encoder.forward(pocket_db.batch_g.to(dev))
            self.pocket_embed = self.pocket_norm(pocket_embed)
        return self.pocket_embed

    def clear_cache(self):
        self.pocket_embed = None


class RxnFlow_MultiPocket(RxnFlow_PocketConditional):
    """
    Model which can be trained on multiple pocket conditions,
    For Zero-shot sampling
    """

    def __init__(self, env_ctx: SynthesisEnvContext, cfg: Config, num_graph_out=1, do_bck=False):
        super().__init__(env_ctx, cfg, num_graph_out, do_bck)

    def forward(self, g: gd.Batch, cond: torch.Tensor) -> tuple[RxnActionCategorical, Tensor]:
        self.pocket_embed = self.get_pocket_embed(force=True)
        cond_cat = torch.cat([cond, self.pocket_embed[g.sample_idx]], dim=-1)
        return super().forward(g, cond_cat)

    def logZ(self, cond_info: torch.Tensor) -> torch.Tensor:
        self.pocket_embed = self.get_pocket_embed()
        cond_cat = torch.cat([cond_info, self.pocket_embed], dim=-1)
        return self._logZ(cond_cat)


class RxnFlow_SinglePocket(RxnFlow_PocketConditional):
    """
    Model which can be trained on single pocket conditions
    For Inference or Few-shot training
    """

    # TODO: check its validity - I do not check whether it works yet

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
            for param in self.emb_block.parameters():
                param.requires_grad = False

    def forward(self, g: gd.Batch, cond: Tensor) -> tuple[RxnActionCategorical, Tensor]:
        if self.freeze_pocket_embedding:
            self.pocket_encoder.eval()
        if self.freeze_action_embedding:
            self.emb_block.eval()

        self.pocket_embed = self.get_pocket_embed()
        pocket_embed = self.pocket_embed.view(1, -1).repeat(g.num_graphs, 1)
        cond_cat = torch.cat([cond, pocket_embed], dim=-1)
        return super().forward(g, cond_cat)

    def logZ(self, cond_info: torch.Tensor) -> torch.Tensor:
        self.pocket_embed = self.get_pocket_embed()
        pocket_embed = self.pocket_embed.view(1, -1).repeat(cond_info.shape[0], 1)
        cond_cat = torch.cat([cond_info, pocket_embed], dim=-1)
        return self._logZ(cond_cat)

    def get_pocket_embed(self, force: bool = False):
        if self.freeze_pocket_embedding:
            with torch.no_grad():
                return super().get_pocket_embed(force)
        else:
            return super().get_pocket_embed(force)

    def block_embedding(self, block: tuple[Tensor, Tensor]) -> Tensor:
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
            self.emb_block.eval()

        return self

    def clear_cache(self, force=False):
        if force or (not self.freeze_pocket_embedding):
            self.pocket_embed = None
