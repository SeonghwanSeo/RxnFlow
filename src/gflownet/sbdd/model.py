import torch
import torch_geometric.data as gd

from gflownet.config import Config
from gflownet.utils.misc import get_worker_env
from gflownet.envs.synthesis import SynthesisEnvContext
from gflownet.models.synthesis_gfn import SynthesisGFN
from gflownet.sbdd.pocket.gvp import GVP_embedding


class SynthesisGFN_SBDD(SynthesisGFN):
    def __init__(
        self,
        env_ctx: SynthesisEnvContext,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ):
        pocket_dim = self.pocket_dim = cfg.task.sbdd.pocket_dim
        org_num_cond_dim = env_ctx.num_cond_dim
        env_ctx.num_cond_dim = org_num_cond_dim + pocket_dim
        super().__init__(env_ctx, cfg, num_graph_out, do_bck)
        env_ctx.num_cond_dim = org_num_cond_dim

        self.pocket_encoder = GVP_embedding((6, 3), (pocket_dim, 16), (32, 1), (32, 1), seq_in=True, vocab_size=20)
        self.pocket_embed: torch.Tensor | None = None

    def forward(self, g: gd.Batch, cond: torch.Tensor, batch_idx: torch.Tensor):
        task = get_worker_env("task")
        pocket_db = task.pocket_db
        if self.pocket_embed is None:
            _, pocket_embed = self.pocket_encoder.forward(pocket_db.batch_g.to(cond.device))
            self.pocket_embed = pocket_embed

        cond_cat = torch.cat([cond, self.pocket_embed[batch_idx]], dim=-1)
        return super().forward(g, cond_cat)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        assert self.pocket_embed is not None
        cond_cat = torch.cat([cond, self.pocket_embed], dim=-1)
        self.pocket_embed = None
        return self._logZ(cond_cat)


class SynthesisGFN_SBDD_SingleOpt(SynthesisGFN_SBDD):
    training: bool = True

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        task = get_worker_env("task")
        pocket_db = task.pocket_db
        if self.pocket_embed is None:
            _, pocket_embed = self.pocket_encoder.forward(pocket_db.batch_g.to(cond.device))
            self.pocket_embed = pocket_embed
        pocket_embed = self.pocket_embed.view(1, -1).repeat(g.num_graphs, 1)
        cond_cat = torch.cat([cond, pocket_embed], dim=-1)
        return SynthesisGFN.forward(self, g, cond_cat)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        assert self.pocket_embed is not None
        pocket_embed = self.pocket_embed.view(1, -1).repeat(cond.shape[0], 1)
        if self.training:
            self.pocket_embed = None
        cond_cat = torch.cat([cond, pocket_embed], dim=-1)
        return self._logZ(cond_cat)
