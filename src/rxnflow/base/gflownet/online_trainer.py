import copy
import os
import pathlib
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from omegaconf import OmegaConf
from rdkit import RDLogger
import git
import wandb

from collections.abc import Callable

from gflownet.config import Config
from gflownet.data.data_source import DataSource
from gflownet.trainer import Closable
from gflownet.online_trainer import StandardOnlineTrainer, AvgRewardHook
from gflownet.utils.misc import set_main_process_device, set_worker_rng_seed
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.data.replay_buffer import ReplayBuffer

from .sqlite_log import CustomSQLiteLogHook


class CustomStandardOnlineTrainer(StandardOnlineTrainer):
    def __init__(self, config: Config, print_config=True):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        config: Config
            The hyperparameters for the trainer.
        """
        self.print_config = print_config
        self.to_terminate: list[Closable] = []

        self.default_cfg: Config = self.get_default_cfg()
        self.set_default_hps(self.default_cfg)
        assert isinstance(self.default_cfg, Config) and isinstance(config, Config)
        self.cfg: Config = OmegaConf.merge(self.default_cfg, config)

        self.device = torch.device(self.cfg.device)
        set_main_process_device(self.device)
        self.print_every = self.cfg.print_every
        self.sampling_hooks: list[Callable] = []
        self.valid_sampling_hooks: list[Callable] = []
        self._validate_parameters = False

        self.setup()

    def get_default_cfg(self):
        return Config()

    def setup_env(self):
        return GraphBuildingEnv()

    def setup(self):
        if os.path.exists(self.cfg.log_dir):
            if self.cfg.overwrite_existing_exp:
                shutil.rmtree(self.cfg.log_dir)
            else:
                raise ValueError(
                    f"Log dir {self.cfg.log_dir} already exists. Set overwrite_existing_exp=True to delete it."
                )
        os.makedirs(self.cfg.log_dir)

        RDLogger.DisableLog("rdApp.*")
        torch.manual_seed(self.cfg.seed + 42)
        set_worker_rng_seed(self.cfg.seed)
        self.setup_env()
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()
        self.setup_online()

    def setup_online(self):
        self.offline_ratio = 0
        self.replay_buffer = ReplayBuffer(self.cfg) if self.cfg.replay.use else None
        self.sampling_hooks.append(AvgRewardHook())
        self.valid_sampling_hooks.append(AvgRewardHook())

        # Separate Z parameters from non-Z to allow for LR decay on the former
        if hasattr(self.model, "_logZ"):
            Z_params = list(self.model._logZ.parameters())
            non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        else:
            Z_params = []
            non_Z_params = list(self.model.parameters())
        self.opt = self._opt(non_Z_params)
        self.opt_Z = self._opt(Z_params, self.cfg.algo.tb.Z_learning_rate, 0.9)
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(
            self.opt_Z, lambda steps: 2 ** (-steps / self.cfg.algo.tb.Z_lr_decay)
        )

        self.sampling_tau = self.cfg.algo.sampling_tau
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

        self.clip_grad_callback = {
            "value": lambda params: torch.nn.utils.clip_grad_value_(params, self.cfg.opt.clip_grad_param),
            "norm": lambda params: [torch.nn.utils.clip_grad_norm_(p, self.cfg.opt.clip_grad_param) for p in params],
            "total_norm": lambda params: torch.nn.utils.clip_grad_norm_(params, self.cfg.opt.clip_grad_param),
            "none": lambda x: None,
        }[self.cfg.opt.clip_grad_type]

        # saving hyperparameters
        try:
            self.cfg.git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        except git.InvalidGitRepositoryError:
            self.cfg.git_hash = "unknown"  # May not have been installed through git

        yaml_cfg = OmegaConf.to_yaml(self.cfg)
        if self.print_config:
            print("\n\nHyperparameters:\n")
            print(yaml_cfg)
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        with open(pathlib.Path(self.cfg.log_dir) / "config.yaml", "w", encoding="utf8") as f:
            f.write(yaml_cfg)

    def log(self, info, index, key):
        # NOTE: wandb.run log (key_k -> key/k)
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)
        if wandb.run is not None:
            wandb.log({f"{key}/{k}": v for k, v in info.items()}, step=index)

    def build_training_data_loader(self) -> DataLoader:
        # NOTE: This is same function, but the SQLiteLogHook is our class.

        # Since the model may be used by a worker in a different process, we need to wrap it.
        # See implementation_notes.md for more details.
        model = self._wrap_for_mp(self.sampling_model)
        replay_buffer = self._wrap_for_mp(self.replay_buffer)

        if self.cfg.replay.use:
            # None is fine for either value, it will be replaced by num_from_policy, but 0 is not
            assert self.cfg.replay.num_from_replay != 0, "Replay is enabled but no samples are being drawn from it"
            assert self.cfg.replay.num_new_samples != 0, "Replay is enabled but no new samples are being added to it"

        n_drawn = self.cfg.algo.num_from_policy
        n_replayed = self.cfg.replay.num_from_replay or n_drawn if self.cfg.replay.use else 0
        n_new_replay_samples = self.cfg.replay.num_new_samples or n_drawn if self.cfg.replay.use else None
        n_from_dataset = self.cfg.algo.num_from_dataset

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, replay_buffer=replay_buffer)
        if n_from_dataset:
            src.do_sample_dataset(self.training_data, n_from_dataset, backwards_model=model)
        if n_drawn:
            src.do_sample_model(model, n_drawn, n_new_replay_samples)
        if n_replayed and replay_buffer is not None:
            src.do_sample_replay(n_replayed)
        if self.cfg.log_dir:
            src.add_sampling_hook(CustomSQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_validation_data_loader(self) -> DataLoader:
        # NOTE: This is same function, but the SQLiteLogHook is our class.
        model = self._wrap_for_mp(self.model)
        # TODO: we're changing the default, make sure anything that is using test data is adjusted
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        n_drawn = self.cfg.algo.valid_num_from_policy
        n_from_dataset = self.cfg.algo.valid_num_from_dataset

        if n_from_dataset:
            src.do_dataset_in_order(self.test_data, n_from_dataset, backwards_model=model)
        if n_drawn:
            assert self.cfg.num_validation_gen_steps is not None
            # TODO: might be better to change total steps to total trajectories drawn
            src.do_sample_model_n_times(model, n_drawn, num_total=self.cfg.num_validation_gen_steps * n_drawn)

        if self.cfg.log_dir:
            src.add_sampling_hook(CustomSQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "valid"), self.ctx))
        for hook in self.valid_sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_final_data_loader(self) -> DataLoader:
        # NOTE: This is same function, but the DataSource is our class.
        model = self._wrap_for_mp(self.model)

        n_drawn = self.cfg.algo.num_from_policy
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        assert self.cfg.num_final_gen_steps is not None
        # TODO: might be better to change total steps to total trajectories drawn
        src.do_sample_model_n_times(model, n_drawn, num_total=self.cfg.num_final_gen_steps * n_drawn)

        if self.cfg.log_dir:
            src.add_sampling_hook(CustomSQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "final"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)
