import numpy as np
import torch
import torch.nn as nn

from typing import Callable, Dict, List, Tuple, Union
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward


class BaseTask(GFNTask):
    """Sets up a common structure of task"""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        self._wrap_model = wrap_model
        self.rng = rng
        self.cfg = cfg
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self):
        return {}

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        raise NotImplementedError

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return self._to_reward_scalar(cond_info, flat_reward)

    def _to_reward_scalar(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def flat_reward_transform(self, y: Union[float, Tensor, List[float]]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp
