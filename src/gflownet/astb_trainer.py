import os
import shutil

import numpy as np
from rdkit import RDLogger

from gflownet import trainer

from gflownet.models.astb_gfn import ASGFN_Synthesis
from gflownet.algo.astb_synthesis import ActionSamplingTrajectoryBalance
from gflownet.envs.synthesis.env import SynthesisEnv


class GFNTrainer(trainer.GFNTrainer):
    def setup_model(self):
        self.model = ASGFN_Synthesis(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        algo = ActionSamplingTrajectoryBalance
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)

    def setup(self):
        if os.path.exists(self.cfg.log_dir):
            if self.cfg.overwrite_existing_exp:
                shutil.rmtree(self.cfg.log_dir)
            else:
                raise ValueError(
                    f"Log dir {self.cfg.log_dir} already exists. Set overwrite_existing_exp=True to delete it."
                )
        os.makedirs(self.cfg.log_dir)

        RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(142857)
        self.env = SynthesisEnv(self.cfg.env_dir)
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()
