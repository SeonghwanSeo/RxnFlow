import functools
import socket
from pathlib import Path
import torch
import torch_geometric.data as gd
from typing import Any

from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook

from gflownet.utils.multiprocessing_proxy import mp_object_wrapper
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnv, SynthesisEnvContext
from rxnflow.algo.trajectory_balance import SynthesisTB
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy.action_categorical import RxnActionCategorical
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

    def set_default_hps(self, base: Config):
        # SEHFragTrainer
        base.hostname = socket.gethostname()
        base.num_workers = 0
        base.algo.illegal_action_logreward = -75
        base.algo.tb.Z_learning_rate = 1e-3

        # Important Parameters
        base.algo.train_random_action_prob = 0.05
        base.algo.tb.do_sample_p_b = False
        base.algo.tb.do_parameterize_p_b = False

        # Online Training Parameters
        base.algo.num_from_policy = 64
        base.algo.valid_num_from_policy = 0
        base.num_validation_gen_steps = 0
        base.validate_every = 0

        # Custom Parameters
        base.algo.sampling_tau = 0.0
        base.cond.temperature.sample_dist = "constant"
        base.cond.temperature.dist_params = [32.0]

    def load_checkpoint(self, checkpoint_path: str | Path):
        state = torch.load(checkpoint_path, map_location="cpu")
        print(f"load pre-trained model from {checkpoint_path}")
        self.model.load_state_dict(state["models_state_dict"][0])
        if self.sampling_model is not self.model:
            self.sampling_model.load_state_dict(state["sampling_model_state_dict"][0])
        del state

    def get_default_cfg(self):
        return Config()

    def setup(self):
        super().setup()
        set_worker_env("trainer", self)
        set_worker_env("env", self.env)
        set_worker_env("ctx", self.ctx)
        set_worker_env("algo", self.algo)
        set_worker_env("task", self.task)

    def setup_env(self):
        self.env = SynthesisEnv(self.cfg.env_dir)

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext(
            self.env,
            num_cond_dim=self.task.num_cond_dim,
        )

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

    def _wrap_for_mp(self, obj):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if self.cfg.num_workers > 0 and obj is not None:
            wrapper = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                cast_types=(gd.Batch, RxnActionCategorical),
                pickle_messages=self.cfg.pickle_mp_messages,
            )
            self.to_terminate.append(wrapper.terminate)
            return wrapper.placeholder
        else:
            return obj


def mogfn_trainer(cls: type[RxnFlowTrainer]):
    original_setup = cls.setup
    original_train_batch = cls.train_batch
    original_save_state = cls._save_state

    @functools.wraps(original_setup)
    def new_setup(self):
        self.cfg.cond.moo.num_objectives = len(self.cfg.task.moo.objectives)
        original_setup(self)
        if self.cfg.task.moo.online_pareto_front:
            self.sampling_hooks.append(
                MultiObjectiveStatsHook(
                    256,
                    self.cfg.log_dir,
                    compute_igd=True,
                    compute_pc_entropy=True,
                    compute_focus_accuracy=True if self.cfg.cond.focus_region.focus_type is not None else False,
                    focus_cosim=self.cfg.cond.focus_region.focus_cosim,
                )
            )
            self.to_terminate.append(self.sampling_hooks[-1].terminate)

    @functools.wraps(original_train_batch)
    def new_train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> dict[str, Any]:
        if self.task.focus_cond is not None:
            self.task.focus_cond.step_focus_model(batch, train_it)
        return original_train_batch(self, batch, epoch_idx, batch_idx, train_it)

    @functools.wraps(original_save_state)
    def new_save_state(self, it):
        if self.task.focus_cond is not None and self.task.focus_cond.focus_model is not None:
            self.task.focus_cond.focus_model.save(Path(self.cfg.log_dir))
        return original_save_state(self, it)

    cls.setup = new_setup
    cls.train_batch = new_train_batch
    cls._save_state = new_save_state
    return cls
