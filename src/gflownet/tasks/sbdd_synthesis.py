import torch
from pathlib import Path
from typing import Any

from gflownet.config import Config, init_empty
from gflownet.base.base_trainer import SynthesisTrainer
from gflownet.base.base_generator import SynthesisGFNSampler

from gflownet.sbdd.algo import SynthesisTrajectoryBalance_SBDD
from gflownet.sbdd.model import SynthesisGFN_SBDD, SynthesisGFN_SBDD_SingleOpt
from gflownet.sbdd.task import SBDDTask, SBDD_SingleOpt_Task


class SBDDTrainer(SynthesisTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.desc = "Proxy-QED optimization with proxy model"
        cfg.validate_every = 0
        cfg.task.moo.objectives = ["docking", "qed"]
        cfg.num_training_steps = 40_000

    def setup_task(self):
        self.task: SBDDTask = SBDDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

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
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.last_reward[obj].mean().item()
        super().log(info, index, key)


class SBDDSampler(SynthesisGFNSampler):
    def setup_model(self):
        self.model: SynthesisGFN_SBDD_SingleOpt = SynthesisGFN_SBDD_SingleOpt(
            self.ctx, self.cfg, do_bck=False, num_graph_out=self.cfg.algo.tb.do_predict_n + 1
        )

    def setup_task(self):
        self.task: SBDD_SingleOpt_Task = SBDD_SingleOpt_Task(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def calc_reward(self, samples, valid_idcs) -> list[Any]:
        samples = super().calc_reward(samples, valid_idcs)
        for idx, sample in enumerate(samples):
            for obj in self.task.objectives:
                sample["info"][f"reward_{obj}"] = self.task.last_reward[obj][idx]
        return samples

    @torch.no_grad()
    def sample_against_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float],
        n: int,
        calc_reward: bool = True,
    ) -> list[dict[str, Any]]:
        """
        samples = sampler.sample_against_pocket(<pocket_file>, <center>, <n>, calc_reward = False)
        samples[0] = {'smiles': <smiles>, 'traj': <traj>, 'info': <info>}
        samples[0]['traj'] = [
            (('StartingBlock',), smiles1),        # None    -> smiles1
            (('UniMolecularReaction', template), smiles2),  # smiles1 -> smiles2
            ...                                 # smiles2 -> ...
        ]
        samples[0]['info'] = {'beta': <beta>, ...}


        samples = sampler.sample_against_pocket(..., calc_reward = True)
        samples[0]['info'] = {
            'beta': <beta>,
            'reward': <reward>,
            'reward_qed': <qed>,
            'reward_docking': <proxy>,
        }
        """
        self.model.pocket_embed = None
        self.task.set_protein(str(protein_path), center)
        return self.sample(n, calc_reward)


def default_config(
    env_dir: str | Path,
    pocket_db: str | Path,
    proxy_model: str,
    proxy_docking: str,
    proxy_dataset: str,
) -> Config:
    config = init_empty(Config())
    config.desc = f"Proxy-QED optimization with proxy model: {proxy_docking} - ({proxy_model}, {proxy_dataset})"
    config.env_dir = str(env_dir)
    config.task.sbdd.pocket_db = str(pocket_db)
    config.task.sbdd.proxy = (proxy_model, proxy_docking, proxy_dataset)
    config.checkpoint_every = 1_000
    config.store_all_checkpoints = True
    config.print_every = 10
    return config
