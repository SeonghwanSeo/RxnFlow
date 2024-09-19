import torch

from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.base.base_generator import SynthesisGFNSampler
from gflownet.base.base_trainer import SynthesisTrainer
from gflownet.trainer import FlatRewards
from gflownet.base.base_task import BaseTask
from gflownet.misc.chem_metrics import mol2qed


class QEDTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock - TacoGFN Task."""

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        vina_scores = mol2qed(mols).reshape(-1, 1)
        is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
        return FlatRewards(vina_scores), is_valid_t


class QEDSynthesisTrainer(SynthesisTrainer):
    def setup_task(self):
        self.task: QEDTask = QEDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)


class QEDSynthesisSampler(SynthesisGFNSampler):
    def setup_task(self):
        self.task: QEDTask = QEDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)
