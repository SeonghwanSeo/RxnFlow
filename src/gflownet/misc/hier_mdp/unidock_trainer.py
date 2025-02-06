from gflownet.config import Config

from gflownet.tasks.unidock_moo_synthesis import UniDockMOOSynthesisTrainer, moo_config

from .gfn import HierarchicalGFN


class UniDockMOOHierTrainer(UniDockMOOSynthesisTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.validate_every = 0
        cfg.algo.max_len = 4

    def setup_model(self):
        self.model = HierarchicalGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )
