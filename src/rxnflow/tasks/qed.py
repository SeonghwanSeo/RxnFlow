import torch

from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from rxnflow.base import RxnFlowTrainer, RxnFlowSampler, BaseTask
from rxnflow.tasks.utils.chem_metrics import mol2qed


class QEDTask(BaseTask):
    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        fr = mol2qed(objs).reshape(-1, 1)
        is_valid_t = torch.ones((len(objs),), dtype=torch.bool)
        return ObjectProperties(fr), is_valid_t


class QEDTrainer(RxnFlowTrainer):  # For online training
    def setup_task(self):
        self.task: QEDTask = QEDTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


class QEDSampler(RxnFlowSampler):  # Sampling with pre-trained GFlowNet
    def setup_task(self):
        self.task: QEDTask = QEDTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)
