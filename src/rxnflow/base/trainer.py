import socket
from pathlib import Path
from typing import Any

import torch
import torch_geometric.data as gd
from omegaconf import OmegaConf

import wandb
from gflownet.algo.config import Backward
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook
from rxnflow.algo.trajectory_balance import SynthesisTB
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnv, SynthesisEnvContext
from rxnflow.models.gfn import RxnFlow
from rxnflow.utils.misc import set_worker_env

from .gflownet.online_trainer import CustomStandardOnlineTrainer
from .task import BaseTask


class RxnFlowTrainer(CustomStandardOnlineTrainer):
    cfg: Config
    env: SynthesisEnv
    ctx: SynthesisEnvContext
    task: BaseTask
    algo: SynthesisTB
    model: RxnFlow
    sampling_model: RxnFlow

    def get_default_cfg(self):
        return Config()

    def set_default_hps(self, base: Config):
        # From SEHFragTrainer
        base.hostname = socket.gethostname()
        base.algo.illegal_action_logreward = -75

        base.opt.weight_decay = 1e-8
        base.opt.momentum = 0.9
        base.opt.adam_eps = 1e-8
        base.opt.lr_decay = 20_000
        base.opt.clip_grad_type = "norm"
        base.opt.clip_grad_param = 10

        base.algo.num_from_policy = 64
        base.algo.sampling_tau = 0.9
        base.algo.tb.epsilon = None
        base.algo.tb.bootstrap_own_reward = False
        base.algo.tb.Z_learning_rate = 1e-3
        base.algo.tb.Z_lr_decay = 50_000

        # RxnFlow model
        base.model.num_emb = 128
        base.model.graph_transformer.num_layers = 4
        base.model.num_mlp_layers = 1

        # RxnFlow max len
        base.algo.max_len = 3

        # From SEHFragMOOTrainer (No effect on single objective optimization)
        base.cond.weighted_prefs.preference_type = "dirichlet"
        base.cond.focus_region.focus_type = None

        # Required Settings for RxnFlow
        base.num_workers = 0
        base.algo.method = "TB"
        base.algo.tb.do_sample_p_b = False
        base.algo.tb.do_parameterize_p_b = False
        base.algo.tb.backward_policy = Backward.Free  # NOTE: custom fixed policy of rxnflow

        # Online Training Parameters
        base.validate_every = 0
        base.algo.num_from_policy = 64
        base.algo.valid_num_from_policy = 0
        base.algo.train_random_action_prob = 0.1  # suggest to set positive value

    def setup(self):
        self.cfg.cond.moo.num_objectives = len(self.cfg.task.moo.objectives)
        super().setup()

        # load checkpoint
        if self.cfg.pretrained_model_path is not None:
            self.load_checkpoint(self.cfg.pretrained_model_path)

        # setup multi-objective optimization
        self.is_moo: bool = self.task.is_moo
        if self.is_moo:
            if self.cfg.task.moo.online_pareto_front:
                hook = MultiObjectiveStatsHook(
                    256,
                    self.cfg.log_dir,
                    compute_igd=True,
                    compute_pc_entropy=True,
                    compute_focus_accuracy=True if self.cfg.cond.focus_region.focus_type is not None else False,
                    focus_cosim=self.cfg.cond.focus_region.focus_cosim,
                )
                self.sampling_hooks.append(hook)
                self.to_terminate.append(hook.terminate)

        set_worker_env("trainer", self)
        set_worker_env("env", self.env)
        set_worker_env("ctx", self.ctx)
        set_worker_env("algo", self.algo)
        set_worker_env("task", self.task)

    def setup_env(self):
        self.env = SynthesisEnv(self.cfg.env_dir, self.cfg.num_workers_retrosynthesis)

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext(self.env, num_cond_dim=self.task.num_cond_dim)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB(self.env, self.ctx, self.cfg)

    def setup_model(self):
        self.model = RxnFlow(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def load_checkpoint(self, checkpoint_path: str | Path, load_ema: bool = False):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"load pre-trained model from {checkpoint_path}")
        self.model.load_state_dict(state["models_state_dict"][0])
        if self.sampling_model is not self.model:
            if load_ema and "sampling_model_state_dict" in state:
                self.sampling_model.load_state_dict(state["sampling_model_state_dict"][0])
            else:
                self.sampling_model.load_state_dict(state["models_state_dict"][0])
        del state

    def run(self, logger=None):
        if wandb.run is not None:
            wandb.config.update({"config": OmegaConf.to_container(self.cfg)})
        super().run(logger)

    def terminate(self):
        super().terminate()
        self.env.retro_analyzer.pool.shutdown(wait=True, cancel_futures=True)
        if wandb.run is not None:
            wandb.finish()

    # MOO setting
    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> dict[str, Any]:
        if self.is_moo:
            if self.task.focus_cond is not None:
                self.task.focus_cond.step_focus_model(batch, train_it)
        return super().train_batch(batch, epoch_idx, batch_idx, train_it)

    def _save_state(self, it):
        if self.is_moo:
            if self.task.focus_cond is not None and self.task.focus_cond.focus_model is not None:
                self.task.focus_cond.focus_model.save(Path(self.cfg.log_dir))
        return super()._save_state(it)
