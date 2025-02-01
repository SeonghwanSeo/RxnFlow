import math
import numpy as np
import torch

from gflownet.utils.misc import get_worker_rng
from rxnflow.config import Config
from rxnflow.envs.env import SynthesisEnv


class ActionSpace:
    def __init__(self, action_idcs: np.ndarray, sampling_ratio: float, min_sampling: int):
        assert sampling_ratio <= 1
        self.action_idcs = action_idcs
        num_actions = self.action_idcs.shape[0]
        min_sampling = min(num_actions, min_sampling)
        self.num_actions: int = num_actions
        self.num_sampling = max(int(num_actions * sampling_ratio), min_sampling)

        self.sampling_ratio: float = max(self.num_sampling, 1) / max(self.num_actions, 1)

    def sampling(self) -> torch.Tensor:
        # TODO: introduce importance subsampling instead of uniform subsampling
        if self.num_sampling == 0:
            return torch.tensor([], dtype=torch.long)
        if self.sampling_ratio < 1:
            rng: np.random.RandomState = get_worker_rng()
            indices = rng.choice(self.action_idcs, self.num_sampling, replace=False)
            np.sort(indices)
            return torch.from_numpy(indices).to(torch.long)
        else:
            return torch.from_numpy(self.action_idcs)


class SubsamplingPolicy:
    def __init__(self, env: SynthesisEnv, cfg: Config):
        self.global_cfg = cfg
        self.cfg = cfg.algo.action_subsampling

        sr = self.cfg.sampling_ratio
        nmin = int(self.cfg.min_sampling)

        self.block_spaces: dict[str, ActionSpace] = {}
        self.num_blocks: dict[str, int] = {}
        for protocol in env.firstblock_list:
            self.block_spaces[protocol.name] = ActionSpace(np.arange(env.num_blocks), sr, nmin)
        for protocol in env.birxn_list:
            self.block_spaces[protocol.name] = ActionSpace(env.birxn_block_indices[protocol.name], sr, nmin)
        self.sampling_ratios = {t: space.sampling_ratio for t, space in self.block_spaces.items()}
        self.weights = {t: math.log(1 / sr) for t, sr in self.sampling_ratios.items()}

    def sampling(self, protocol: str) -> tuple[torch.Tensor, float]:
        return self.block_spaces[protocol].sampling(), self.weights[protocol]
