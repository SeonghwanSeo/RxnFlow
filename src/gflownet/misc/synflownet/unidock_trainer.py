from gflownet.config import Config

from gflownet.tasks.unidock_moo_synthesis import UniDockMOOSynthesisTrainer, moo_config

from .gfn import SynFlowNet
from .env import ReactionTemplateEnv, ReactionTemplateEnvContext
from .trajectory_balance import SynFlowNet_TrajectoryBalance


class UniDockMOOSynFlowNetTrainer(UniDockMOOSynthesisTrainer):
    env: ReactionTemplateEnv
    ctx: ReactionTemplateEnvContext

    def set_default_hps(self, cfg: Config):
        # NOTE: SynFlowNet Setting
        super().set_default_hps(cfg)
        cfg.algo.max_len = 5
        cfg.algo.tb.do_parameterize_p_b = True

    def setup_env(self):
        self.env = ReactionTemplateEnv(self.cfg.env_dir)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        algo = SynFlowNet_TrajectoryBalance
        self.algo = algo(self.env, self.ctx, self.cfg)

    def setup_env_context(self):
        self.ctx = self.env.ctx
        self.ctx.num_cond_dim = self.task.num_cond_dim

    def setup_model(self):
        self.model = SynFlowNet(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )
