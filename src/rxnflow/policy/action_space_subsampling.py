import numpy as np
from gflownet.utils.misc import get_worker_rng

from rxnflow.config import Config
from rxnflow.envs.env import SynthesisEnv


class BlockSpace:
    def __init__(self, num_blocks: int, num_sampling: int):
        self.num_blocks: int = num_blocks
        self.num_sampling: int = num_sampling
        self.sampling_ratio: float = self.num_sampling / self.num_blocks

    def sampling(self) -> list[int]:
        # TODO: introduce importance subsampling instead of uniform subsampling
        if self.sampling_ratio < 1:
            rng: np.random.RandomState = get_worker_rng()
            block_indices = rng.choice(self.num_blocks, self.num_sampling, replace=False)
            np.sort(block_indices)
        else:
            return list(range(self.num_blocks))
        return block_indices.tolist()


class SubsamplingPolicy:
    def __init__(self, env: SynthesisEnv, cfg: Config):
        self.global_cfg = cfg
        self.cfg = cfg.algo.action_subsampling

        sr = self.cfg.sampling_ratio
        nmin = int(self.cfg.min_sampling)

        self.block_spaces: dict[str, BlockSpace] = {}
        self.num_blocks: dict[str, int] = {}
        for block_type, blocks in env.blocks.items():
            nblocks = len(blocks)
            nsamp = max(int(nblocks * sr), min(nblocks, nmin))
            self.num_blocks[block_type] = nsamp
            self.block_spaces[block_type] = BlockSpace(nblocks, nsamp)
        self.sampling_ratios = {t: space.sampling_ratio for t, space in self.block_spaces.items()}

    def sampling(self, block_type: str) -> list[int]:
        return self.block_spaces[block_type].sampling()

    def sampling_debug(self, block_type: str) -> list[int]:
        return list(range(self.block_spaces[block_type].num_sampling))
