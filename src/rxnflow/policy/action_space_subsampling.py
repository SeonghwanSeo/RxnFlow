import numpy as np
from numpy.typing import NDArray
from gflownet.utils.misc import get_worker_rng

from rxnflow.config import Config
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.action import RxnActionType


class BlockSpace:
    def __init__(
        self,
        allowable_block_indices: NDArray[np.int_],
        num_sampling: int,
    ):
        self.block_indices: NDArray[np.int_] = allowable_block_indices
        self.num_blocks: int = len(self.block_indices)
        self.num_sampling: int = num_sampling
        self.sampling_ratio: float = 1.0 if num_sampling == 0 else (self.num_sampling / self.num_blocks)

    def sampling(self) -> NDArray[np.int_]:
        if self.sampling_ratio < 1:
            rng: np.random.RandomState = get_worker_rng()
            block_indices = rng.choice(self.block_indices, self.num_sampling, replace=False)
            np.sort(block_indices)
        else:
            return self.block_indices.copy()
        return block_indices

    @classmethod
    def create1(cls, block_indices: NDArray[np.int_], num_sampling: int):
        num_blocks = len(block_indices)
        num_sampling = min(num_sampling, num_blocks)
        return cls(block_indices, num_sampling)

    @classmethod
    def create2(
        cls,
        block_indices: NDArray[np.int_],
        sampling_ratio: float,
        min_sampling: int,
    ):
        num_blocks = len(block_indices)
        min_sampling = min(min_sampling, num_blocks)
        if num_blocks != 0:
            num_sampling = max(int(num_blocks * sampling_ratio), min_sampling)
            return cls(block_indices, num_sampling)
        else:
            return cls(block_indices, 0)


class SubsamplingPolicy:
    def __init__(self, env: SynthesisEnv, cfg: Config):
        self.env: SynthesisEnv = env
        self.global_cfg = cfg
        self.cfg = cfg.algo.action_subsampling

        # NOTE: AddFirstReactant
        indices = np.arange(env.num_building_blocks)
        nsamp: int = self.cfg.num_sampling_first_block
        self._first_reactant_space = BlockSpace.create1(indices, nsamp)

        # NOTE: ReactBi
        sr = self.cfg.sampling_ratio_bi_rxn
        nmin = int(self.cfg.min_sampling_bi_rxn)
        self._bi_rxn_reactant_space = {}
        for i in range(env.num_bi_rxns):
            block_mask = env.building_block_mask[i]
            for block_is_first in (True, False):
                idx = 0 if block_is_first else 1
                indices = np.where(block_mask[idx])[0]
                self._bi_rxn_reactant_space[(i, block_is_first)] = BlockSpace.create2(indices, sr, nmin)

    def get_space(self, t: RxnActionType, rxn_type: tuple[int, bool] | None = None) -> BlockSpace:
        if t is RxnActionType.FirstBlock:
            assert rxn_type is None
            return self._first_reactant_space
        else:
            assert rxn_type is not None
            return self._bi_rxn_reactant_space[rxn_type]
