from gflownet.config import Config

from gflownet.tasks.unidock_moo_synthesis import UniDockMOOSynthesisTrainer, moo_config

from .gfn import RGFN
from .env_context import RGFN_EnvContext


class UniDockMOORGFNTrainer(UniDockMOOSynthesisTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.validate_every = 0
        cfg.algo.max_len = 5  # RGFN Setting

    def setup_env_context(self):
        self.ctx = RGFN_EnvContext(
            self.env,
            num_cond_dim=self.task.num_cond_dim,
            fp_radius_building_block=self.cfg.model.fp_radius_building_block,
            fp_nbits_building_block=self.cfg.model.fp_nbits_building_block,
        )

    def setup_model(self):
        self.model = RGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )
