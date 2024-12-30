import functools

import numpy as np
import torch
from omegaconf import OmegaConf

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from rdkit import Chem, RDLogger
from torch import Tensor

from gflownet.utils.misc import set_main_process_device, set_worker_rng_seed
from rxnflow.config import Config
from rxnflow.utils.misc import set_worker_env
from rxnflow.algo.trajectory_balance import SynthesisTB
from rxnflow.envs import SynthesisEnv, SynthesisEnvContext
from rxnflow.models.gfn import RxnFlow
from rxnflow.base.task import BaseTask

RDLogger.DisableLog("rdApp.*")

"""
config = init_empty(Config())
config.algo.num_from_policy = 100
sampler = RxnFlowSampler(config, <checkpoint-path>, 'cuda')

samples = sampler.sample(200, calc_reward = False)
samples[0] = {'smiles': <smiles>, 'traj': <traj>, 'info': <info>}
samples[0]['traj'] = [
    (('Start Block',), smiles1),        # None    -> smiles1
    (('UniRxn', template), smiles2),    # smiles1 -> smiles2
    ...                                 # smiles2 -> ...
]
samples[0]['info'] = {'beta': <beta> ...}
"""


class RxnFlowSampler:
    model: RxnFlow
    sampling_model: RxnFlow
    env: SynthesisEnv
    ctx: SynthesisEnvContext
    task: BaseTask
    algo: SynthesisTB

    def __init__(self, config: Config, checkpoint_path: str | Path, device: str):
        """Sampler for RxnFlow

        Parameters
        ---
        config: Config
            updating config (default: config in checkpoint)
        checkpoint_path: str (path)
            checkpoint path (.pt)
        device: str
            'cuda' | 'cpu'
        """
        state = torch.load(checkpoint_path, map_location=device)
        self.default_cfg: Config = state["cfg"]
        self.update_default_cfg(self.default_cfg)
        self.cfg: Config = OmegaConf.merge(self.default_cfg, config)
        self.cfg.device = device

        self.device = torch.device(device)
        self.setup()
        if "sampling_models_state_dict" in state:
            self.sampling_model.load_state_dict(state["sampling_models_state_dict"][0])
        else:
            self.sampling_model.load_state_dict(state["models_state_dict"][0])
        del state

    @torch.no_grad()
    def sample(self, n: int, calc_reward: bool = True) -> list[dict[str, Any]]:
        """
        samples = sampler.sample(200, calc_reward = False)
        samples[0] = {'smiles': <smiles>, 'traj': <traj>, 'info': <info>}
        samples[0]['traj'] = [
            (('Start Block',), smiles1),        # None    -> smiles1
            (('UniRxn', template), smiles2),  # smiles1 -> smiles2
            ...                                 # smiles2 -> ...
        ]
        samples[0]['info'] = {'beta': <beta>, ...}


        samples = sampler.sample(200, calc_reward = True)
        samples[0]['info'] = {'beta': <beta>, 'reward': <reward>, ...}
        """

        return list(self.iter(n, calc_reward))

    def update_default_cfg(self, config: Config):
        """Update default config which used in model training.
        config: checkpoint_state["cfg"]"""
        pass

    def setup_task(self):
        self.task = BaseTask(self.cfg, self._wrap_for_mp)
        pass
        # raise NotImplementedError()

    def setup_env(self):
        self.env = SynthesisEnv(self.cfg.env_dir)

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext(
            self.env,
            num_cond_dim=self.task.num_cond_dim,
        )

    def setup_model(self):
        self.model = RxnFlow(self.ctx, self.cfg, do_bck=False, num_graph_out=self.cfg.algo.tb.do_predict_n + 1)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB(self.env, self.ctx, self.cfg)

    def setup(self):
        self.rng = np.random.default_rng(142857)
        set_worker_rng_seed(self.cfg.seed)
        set_main_process_device(self.device)
        self.setup_env()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()
        set_worker_env("trainer", self)
        set_worker_env("env", self.env)
        set_worker_env("ctx", self.ctx)
        set_worker_env("algo", self.algo)
        set_worker_env("task", self.task)
        self.model = self.sampling_model = self.model.to(self.device)
        self.sampling_model.eval()

    @torch.no_grad()
    def iter(self, n: int, calc_reward: bool = True) -> Iterable[dict[str, Any]]:
        batch_size = min(n, self.cfg.algo.num_from_policy)
        idx = 0
        it = 0
        while True:
            samples = self.step(it, batch_size, calc_reward)
            for sample in samples:
                out = {
                    "smiles": Chem.MolToSmiles(self.ctx.graph_to_obj(sample["result"])),
                    "traj": self.ctx.read_traj(sample["traj"]),
                    "info": sample["info"],
                }
                yield out
                idx += 1
                if idx >= n:
                    return
            if idx >= n:
                return
            it += 1

    @torch.no_grad()
    def step(self, it: int = 0, batch_size: int = 64, calc_reward: bool = True):
        cond_info = self.task.sample_conditional_information(batch_size, it)
        samples = self.algo.graph_sampler.sample_inference(self.sampling_model, batch_size, cond_info["encoding"])
        for i, sample in enumerate(samples):
            sample["info"] = {k: self.to_item(v[i]) for k, v in cond_info.items() if k != "encoding"}

        valid_idcs = [i for i, sample in enumerate(samples) if sample["is_valid"]]
        samples = [samples[i] for i in valid_idcs]
        if calc_reward:
            samples = self.calc_reward(samples, valid_idcs)
        return samples

    def calc_reward(self, samples: list[Any], valid_idcs: list[int]) -> list[Any]:
        mols = [self.ctx.graph_to_obj(sample["result"]) for sample in samples]
        flat_r, m_is_valid = self.task.compute_obj_properties(mols)
        samples = [sample for sample, is_valid in zip(samples, m_is_valid, strict=True) if is_valid]
        for i, sample in enumerate(samples):
            sample["info"]["reward"] = self.to_item(flat_r[i])
        return samples

    def _wrap_for_mp(self, obj, send_to_device=False):
        if send_to_device:
            obj.to(self.device)
        return obj

    @staticmethod
    def to_item(t: Tensor) -> float | tuple[float, ...]:
        assert t.dim() <= 1
        if t.dim() == 0:
            return t.item()
        else:
            return tuple(t.tolist())


def moo_sampler(cls: type[RxnFlowSampler]):
    original_calc_reward = cls.calc_reward

    @functools.wraps(original_calc_reward)
    def new_calc_reward(self, samples: list[Any], valid_idcs: list[int]) -> list[Any]:
        samples = original_calc_reward(self, samples, valid_idcs)
        for sample in samples:
            for obj, r in zip(self.task.objectives, sample["info"]["reward"], strict=True):
                sample["info"][f"reward_{obj}"] = r
        return samples

    cls.calc_reward = new_calc_reward
    return cls
