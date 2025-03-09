import torch
import torch_geometric.data as gd
from rdkit.Chem import QED, rdMolDescriptors
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from gflownet.models import bengio2021flow
from gflownet.utils import sascore
from gflownet.utils.misc import get_worker_device
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.config import Config, init_empty


def safe(f, x, default):
    try:
        return f(x)
    except Exception:
        return default


class SEHMOOTask(BaseTask):
    is_moo = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.seh_proxy = bengio2021flow.load_original_model()
        self.seh_proxy.to(get_worker_device())
        assert set(self.objectives) <= {"seh", "qed", "sa", "mw"} and len(self.objectives) == len(set(self.objectives))

    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        assert len(graphs) == len(mols)
        is_valid = [i is not None for i in graphs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        valid_graphs = [g for g in graphs if g is not None]
        valid_mols = [m for m, v in zip(mols, is_valid, strict=True) if v]
        if not any(is_valid):
            return ObjectProperties(torch.zeros((0, len(self.objectives)))), is_valid_t
        else:
            flat_r: list[Tensor] = []
            for obj in self.objectives:
                if obj == "seh":
                    flat_r.append(self.calc_seh_reward(valid_graphs))
                else:
                    flat_r.append(self.calc_mol_prop(valid_mols, obj))
            flat_rewards = torch.stack(flat_r, dim=1)  # [Nstate, Nprop]
            assert flat_rewards.shape == (len(mols), self.num_objectives)
            return ObjectProperties(flat_rewards), is_valid_t

    def calc_seh_reward(self, graphs: list[gd.Data]) -> Tensor:
        device = get_worker_device()
        batch = gd.Batch.from_data_list(graphs).to(device)
        preds = self.seh_proxy(batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))

    def calc_mol_prop(self, mols: list[RDMol], prop: str) -> Tensor:
        if prop == "mw":
            molwts = torch.tensor([safe(rdMolDescriptors.CalcExactMolWt, i, 1000) for i in mols])
            molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
            return molwts
        elif prop == "sa":
            sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i in mols])
            sas = (10 - sas) / 9  # Turn into a [0-1] reward
            return sas
        elif prop == "qed":
            return torch.tensor([safe(QED.qed, i, 0) for i in mols])
        else:
            raise ValueError(prop)


class SEHMOOTrainer(RxnFlowTrainer):
    def setup_task(self):
        self.task = SEHMOOTask(cfg=self.cfg)


if __name__ == "__main__":
    config = init_empty(Config())
    config.log_dir = "./logs/example/seh-moo"
    config.env_dir = "./data/envs/catalog"
    config.num_training_steps = 10000
    config.checkpoint_every = 1000
    config.store_all_checkpoints = True
    config.print_every = 1
    config.num_workers_retrosynthesis = 4

    config.algo.action_subsampling.sampling_ratio = 0.02

    config.algo.sampling_tau = 0.95
    config.task.moo.objectives = ["seh", "qed"]
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [60.0]
    config.cond.weighted_prefs.preference_type = "dirichlet"
    config.cond.focus_region.focus_type = None

    trial = SEHMOOTrainer(config)
    trial.run()
