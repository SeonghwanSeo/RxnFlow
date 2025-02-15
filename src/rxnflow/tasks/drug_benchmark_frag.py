import socket
from pathlib import Path
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import Mol as RDMol, QED

from gflownet import ObjectProperties
from gflownet.models import bengio2021flow
from gflownet.config import Config, init_empty
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.utils.misc import create_logger

from rxnflow.base import BaseTask
from rxnflow.tasks.utils.unidock import VinaReward
from rxnflow.tasks.utils.chem_metrics import mol2qed, mol2sascore

aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class BenchmarkTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.objectives: list[str] = cfg.task.moo.objectives
        assert set(self.objectives) <= {"vina", "qed", "sa"}
        assert cfg.task.docking.size == (22.5, 22.5, 22.5)
        self.oracle_idx = 0

        self.vina = VinaReward(
            cfg.task.docking.protein_path,
            cfg.task.docking.center,
            cfg.task.docking.ref_ligand_path,
            cfg.task.docking.size,
            search_mode="fast",
            num_workers=4,
        )

        self.topn_vina: OrderedDict[str, float] = OrderedDict()
        self.batch_vina: list[float] = []

        self.save_dir: Path = Path(cfg.log_dir) / "docking"
        self.save_dir.mkdir()

        self.log_path: Path = Path(cfg.log_dir) / "reward.csv"
        with open(self.log_path, "w") as w:
            w.write("oracle_idx,sample_idx,smiles," + ",".join(self.objectives) + "\n")

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)

        smiles_list = [Chem.MolToSmiles(obj) for obj in objs]
        reward_list: list[tuple[str, list[float]]] = [(smi, []) for smi in smiles_list]

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
            for i, v in enumerate(fr.tolist()):
                reward_list[i][1].append(v)

        # log reward
        with open(self.log_path, "a") as w:
            for i, (smi, rs) in enumerate(reward_list):
                log = f"{self.oracle_idx},{i},{smi}"
                for r in rs:
                    log += f",{r:.3f}"
                log += "\n"
                w.write(log)

        flat_rewards = torch.stack(flat_r, dim=1).prod(-1, keepdim=True).clip(min=0)
        assert flat_rewards.shape[0] == len(objs)
        self.oracle_idx += 1
        return ObjectProperties(flat_rewards), is_valid_t

    def mol2vina(self, mols: list[RDMol]) -> torch.Tensor:
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        vina_scores = self.vina.run_mols(mols, out_path)
        self.update_storage(mols, vina_scores)
        return torch.tensor(vina_scores, dtype=torch.float32)

    def update_storage(self, objs: list[RDMol], vina_scores: list[float]):
        """only consider QED > 0.5"""
        is_pass = [QED.qed(obj) > 0.5 for obj in objs]
        objs = [mol for mol, flag in zip(objs, is_pass, strict=True) if flag]
        vina_scores = [score for score, flag in zip(vina_scores, is_pass, strict=True) if flag]

        self.batch_vina = vina_scores
        smiles_list = [Chem.MolToSmiles(obj) for obj in objs]
        self.topn_vina.update(zip(smiles_list, vina_scores, strict=True))
        topn = sorted(list(self.topn_vina.items()), key=lambda v: v[1])[:1000]
        self.topn_vina = OrderedDict(topn)


class BenchmarkTrainer(StandardOnlineTrainer):
    task: BenchmarkTask

    def set_default_hps(self, base: Config):
        """GFlowNet hyperparameters for benchmark study
        To reduce the effects from hparam tuning, most settings are equal to `seh_frag`.
        I hope that future GFN studies will follow these settings as possible."""
        # Equal to seh_frag hparams
        base.hostname = socket.gethostname()
        base.opt.weight_decay = 1e-8
        base.opt.momentum = 0.9
        base.opt.clip_grad_type = "norm"
        base.opt.clip_grad_param = 10
        base.opt.adam_eps = 1e-8

        base.algo.method = "TB"
        base.algo.illegal_action_logreward = -75
        base.algo.tb.epsilon = None
        base.algo.tb.bootstrap_own_reward = False

        base.model.num_emb = 128  # <=128 is allowed
        base.model.num_layers = 4  # <=4 is allowed

        # Benchmark Setting
        ## online training w/o validation
        base.print_every = 1
        base.num_training_steps = 1000
        base.algo.num_from_policy = 64
        base.validate_every = 0
        base.algo.valid_num_from_policy = 0
        base.num_final_gen_steps = 0

        ## replay buffer
        ## it is allowed to disable replay buffer.
        base.replay.use = True
        base.replay.warmup = 640
        base.replay.capacity = 6_400  # previous 100 steps

        ## temperature-conditional GFlowNets (logit-gfn), beta ~ U(0,64)
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0.0, 64.0]

        ########################################
        # list of hparams that are allowed to be tuned (optional, {} is default (seh_frag))
        # RxnFlow used the default values for all hyperparameters.
        base.opt.learning_rate = 1e-4  # [1e-2, 1e-3, {1e-4}]
        base.opt.lr_decay = 20_000  # [1_000, 2_000, 10_000, {20_000}, 50_000]
        base.algo.tb.Z_learning_rate = 1e-3  # [1e-2, {1e-3}, 1e-4]
        base.algo.tb.Z_lr_decay = 50_000  # [1_000, 2_000, 10_000, 20_000, {50_000}]
        base.algo.sampling_tau = 0.90  # [0, {0.9}, 0.95, 0.99]
        ########################################

        # For FragGFN
        base.num_workers = 0
        base.algo.max_len = 128
        base.algo.max_nodes = 9

        # for non-synthesis-based models, you might want to add SA or aizynthfinder(not implemented in this work)
        base.task.moo.objectives = ["vina", "qed", "sa"]

    def setup_task(self):
        self.task = BenchmarkTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.seh.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

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
    config.log_dir = "./logs/debug-benchmark/"
    config.overwrite_existing_exp = True

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.ref_ligand_path = "./data/examples/6oim_ligand.pdb"

    # to prevent redundant logging due to unidock
    logger = create_logger()
    trial = BenchmarkTrainer(config)
    trial.run(logger)
