from pathlib import Path
import torch

from gflownet.base.base_trainer import SynthesisTrainer
from gflownet.config import Config, init_empty
from gflownet.sbdd.gfn import SynthesisGFN_SBDD
from gflownet.sbdd.task import SBDDTask
from gflownet.sbdd.trajectory_balance import SynthesisTrajectoryBalance_SBDD


def moo_config(env_dir: str | Path, pocket_db: str | Path, proxy: str) -> Config:
    config = init_empty(Config())
    config.desc = "Vina-QED optimization with proxy model"
    config.env_dir = str(env_dir)
    config.task.sbdd.pocket_db = str(pocket_db)
    config.task.sbdd.proxy = proxy
    config.checkpoint_every = 1_000
    config.store_all_checkpoints = True
    config.print_every = 10
    return config


class SynthesisTrainer_SBDD(SynthesisTrainer):
    task: SBDDTask

    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.validate_every = 0
        cfg.task.moo.objectives = ["docking", "qed"]
        cfg.task.sbdd.pocket_dim = 64
        cfg.num_training_steps = 50_000

    def setup_task(self):
        self.task = SBDDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        algo = SynthesisTrajectoryBalance_SBDD
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)

    def setup_model(self):
        self.model = SynthesisGFN_SBDD(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        super().log(info, index, key)

    def load_checkpoint(self, checkpoint_path: str | Path):
        state = torch.load(checkpoint_path, map_location="cpu")
        print(f"load pre-trained model from {checkpoint_path}")
        self.model.load_state_dict(state["models_state_dict"][0])
        if self.sampling_model is not self.model:
            self.sampling_model.load_state_dict(state["sampling_model_state_dict"][0])
        del state
