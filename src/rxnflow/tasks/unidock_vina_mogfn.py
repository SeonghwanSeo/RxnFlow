import numpy as np
import torch
from rdkit.Chem import QED
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from rxnflow.config import Config, init_empty
from rxnflow.tasks.unidock_vina import VinaTask, VinaTrainer
from rxnflow.tasks.utils.chem_metrics import mol2qed, mol2sascore

aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class VinaMOGFNTask(VinaTask):
    is_moo = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.tensor([self.constraint(obj) for obj in objs], dtype=torch.bool)
        valid_objs = [obj for flag, obj in zip(is_valid_t, objs, strict=True) if flag]

        if len(valid_objs) > 0:
            flat_r: list[Tensor] = []
            self.avg_reward_info = []
            for prop in self.objectives:
                if prop == "vina":
                    docking_scores = self.mol2vina(objs)
                    fr = docking_scores * -0.1  # normalization
                else:
                    fr = aux_tasks[prop](objs)
                flat_r.append(fr)
                self.avg_reward_info.append((prop, fr.mean().item()))

            flat_rewards = torch.stack(flat_r, dim=1)
        else:
            flat_rewards = torch.zeros((0, self.num_objectives), dtype=torch.float32)
        assert flat_rewards.shape[0] == len(objs)
        self.oracle_idx += 1
        return ObjectProperties(flat_rewards), is_valid_t

    def update_storage(self, objs: list[RDMol], vina_scores: list[float]):
        """only consider QED > 0.5"""
        is_pass = [QED.qed(obj) > 0.5 for obj in objs]
        pass_objs = [mol for mol, flag in zip(objs, is_pass, strict=True) if flag]
        pass_vina_scores = [score for score, flag in zip(vina_scores, is_pass, strict=True) if flag]
        super().update_storage(pass_objs, pass_vina_scores)


class VinaMOGFNTrainer(VinaTrainer):
    task: VinaMOGFNTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]
        base.task.constraint.rule = None
        base.num_training_steps = 1000
        base.validate_every = 0

        base.algo.train_random_action_prob = 0.05
        base.algo.action_subsampling.sampling_ratio = 0.02

        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0.0, 64.0]
        base.replay.use = True
        base.replay.capacity = 64 * 100
        base.replay.warmup = 64 * 5

        # for training step = 1000
        base.opt.learning_rate = 1e-4
        base.opt.lr_decay = 500
        base.algo.tb.Z_learning_rate = 1e-2
        base.algo.tb.Z_lr_decay = 1000

    def setup_task(self):
        self.task = VinaMOGFNTask(self.cfg)

    def add_extra_info(self, info):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        if len(self.task.batch_vina) > 0:
            info["sampled_vina_avg"] = np.mean(self.task.batch_vina)
        best_vinas = list(self.task.topn_vina.values())
        for topn in [10, 100, 1000]:
            if len(best_vinas) > topn:
                info[f"top{topn}_vina"] = np.mean(best_vinas[:topn])


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-vina-mogfn/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    # config.task.docking.ref_ligand_path = "./data/examples/6oim_ligand.pdb"
    config.task.docking.center = (1.872, -8.260, -1.361)

    trial = VinaMOGFNTrainer(config)
    trial.run()
