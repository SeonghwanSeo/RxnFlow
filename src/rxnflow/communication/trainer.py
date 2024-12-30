import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from rdkit import Chem

from collections.abc import Callable
from typing import Any

from gflownet import ObjectProperties
from rxnflow.config import Config
from rxnflow.base.task import BaseTask
from rxnflow.base.trainer import RxnFlowTrainer

__all__ = [
    "CommunicationTask",
    "MolCommunicationTask",
    "SeqCommunicationTask",
    "CommunicationTrainer",
    "MolCommunicationTrainer",
    "SeqCommunicationTrainer",
]


class CommunicationTask(BaseTask):
    """The rewards of objects are calculated by another process.
    The RewardModule requires minimal dependencies,
    I am not very good at English, so I request an appropriate name for this class.
    """

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        log_dir = Path(cfg.log_dir)
        self.tick: float = 0.1  # tick for wait

        self._lock_file = log_dir / "_wait.lock"
        self._save_sample_dir = log_dir / "_samples/"
        self._load_reward_dir = log_dir / "_rewards/"

        self._save_sample_dir.mkdir()
        self._load_reward_dir.mkdir()

    def object_to_str_repr(self, obj: Any) -> str:
        """Convert an Object to a string representation"""
        raise NotImplementedError

    def compute_obj_properties(self, objs: list[Any]) -> tuple[ObjectProperties, Tensor]:
        assert not self._lock_file.exists()
        assert self._post(objs)
        self._wait()
        rewards, is_valid = self._get()
        assert len(objs) == is_valid.shape[0]
        assert rewards.shape[0] == is_valid.sum()
        self.oracle_idx += 1
        return ObjectProperties(rewards), is_valid

    def _post(self, objs: list[Any]) -> bool:
        """Request to RewardModule

        Parameters
        ----------
        objs : list[Any]
            A list of n sampled objects

        Returns
        -------
        status: bool
        """
        sample_path = self._save_sample_dir / f"{self.oracle_idx}.csv"
        self._save_objects(objs, sample_path)
        self._lock(sample_path)
        return True

    def _wait(self):
        """Wait the response from RewardModule"""
        while self._lock_file.exists():
            time.sleep(self.tick)

    def _get(self) -> tuple[Tensor, Tensor]:
        """Response from RewardModule for n sampled objects

        Returns
        -------
        tuple[Tensor, Tensor]
            - rewards: FloatTensor [m, n_objectives]
                rewards of n<=m valid objects
            - is_valid: BoolTensor [n,]
                flags whether the objects are valid or not
        """
        load_path = self._load_reward_dir / f"{self.oracle_idx}.csv"
        return self._load_rewards(load_path)

    def _lock(self, obj_path: Path):
        with open(self._lock_file, "w") as w:
            w.write(str(self.oracle_idx))

    def _save_objects(self, objs: list[Any], save_path: Path):
        with open(save_path, "w") as w:
            w.write(",object\n")
            for i, obj in enumerate(objs):
                obj_repr = self.object_to_str_repr(obj)
                w.write(f"{i},{obj_repr}\n")

    def _load_rewards(self, reward_file: Path) -> tuple[Tensor, Tensor]:
        df = pd.read_csv(reward_file, index_col=0)
        fr = torch.from_numpy(df.iloc[:, 2:].to_numpy(np.float32))
        is_valid = torch.from_numpy(df.is_valid.to_numpy(np.bool_))
        fr = fr[is_valid]
        return fr, is_valid

    def compute_rewards(self, objs: list[Any]) -> Tensor:
        """Not to be reached"""
        raise ValueError("compute_rewards() is not used in CommunicateTask")


class MolCommunicationTask(CommunicationTask):
    def object_to_str_repr(self, obj: Chem.Mol) -> str:
        return Chem.MolToSmiles(obj)


class SeqCommunicationTask(CommunicationTask):
    def object_to_str_repr(self, obj: str) -> str:
        return obj


class CommunicationTrainer(RxnFlowTrainer):
    task: CommunicationTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        self.num_workers = 0

    def setup_task(self):
        self.task = CommunicationTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


class MolCommunicationTrainer(CommunicationTrainer):
    def setup_task(self):
        self.task = MolCommunicationTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


class SeqCommunicationTrainer(CommunicationTrainer):
    def setup_task(self):
        self.task = SeqCommunicationTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)
