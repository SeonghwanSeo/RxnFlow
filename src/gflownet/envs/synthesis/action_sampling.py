import numpy as np
from numpy.typing import NDArray

from gflownet.config import Config
from gflownet.envs.synthesis.action import ReactionActionType
from gflownet.envs.synthesis.env import SynthesisEnv


def clip(x, minx, maxx):
    return min(maxx, max(x, minx))


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
            block_indices = np.random.choice(self.block_indices, self.num_sampling, replace=False)
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
        max_sampling: int,
    ):
        num_blocks = len(block_indices)
        max_sampling = min(max_sampling, num_blocks)
        min_sampling = min(min_sampling, num_blocks)
        if num_blocks != 0:
            num_sampling = int(num_blocks * sampling_ratio)
            num_sampling = clip(num_sampling, min_sampling, max_sampling)
            return cls(block_indices, num_sampling)
        else:
            return cls(block_indices, 0)


class ActionSamplingPolicy:
    def __init__(self, env: SynthesisEnv, cfg: Config):
        self.env: SynthesisEnv = env
        self.global_cfg = cfg
        self.cfg = cfg.algo.action_sampling
        self.num_mc_sampling: int = self.cfg.num_mc_sampling

        # NOTE: AddFirstReactant
        indices = np.arange(env.num_building_blocks)
        nsamp = self.cfg.num_sampling_add_first_reactant
        self._first_reactant_space = BlockSpace.create1(indices, nsamp)

        # NOTE: ReactBi
        sr = self.cfg.sampling_ratio_reactbi
        nmin = self.cfg.min_sampling_reactbi
        nmax = self.cfg.max_sampling_reactbi
        self._reactbi_reactant_space = {}
        for i in range(env.num_bimolecular_rxns):
            block_mask = env.building_block_mask[i]
            for block_is_first in (True, False):
                idx = 0 if block_is_first else 1
                indices = np.where(block_mask[idx])[0]
                self._reactbi_reactant_space[(i, block_is_first)] = BlockSpace.create2(
                    indices, sr, int(nmin), int(nmax)
                )

    def get_space(self, t: ReactionActionType, rxn_type: tuple[int, bool] | None = None) -> BlockSpace:
        if t is ReactionActionType.AddFirstReactant:
            assert rxn_type is None
            return self._first_reactant_space
        else:
            assert rxn_type is not None
            return self._reactbi_reactant_space[rxn_type]

    def get_space_addfirstreactant(self) -> BlockSpace:
        return self._first_reactant_space

    def get_space_reactbi(self, rxn_idx: int, block_is_first: bool) -> BlockSpace:
        return self._reactbi_reactant_space[(rxn_idx, block_is_first)]
