import os
import pathlib
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from rdkit import RDLogger
from torch.utils.data import DataLoader

from gflownet import trainer

from gflownet.models.astb_gfn import ASTB_GFN
from gflownet.algo.astb_synthesis import ActionSamplingTrajectoryBalance
from gflownet.data.astb_sampling_iterator import ASTBSamplingIterator
from gflownet.envs.synthesis.env import SynthesisEnv


class GFNTrainer(trainer.GFNTrainer):
    def setup_model(self):
        self.model = ASTB_GFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def setup_algo(self):
        assert self.cfg.algo.method == "ASTB"
        algo = ActionSamplingTrajectoryBalance
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)

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
        self.rng = np.random.default_rng(142857)
        self.env = SynthesisEnv(self.cfg.env_dir)
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()

    # NOTE: Difference:
    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        replay_buffer, _ = self._wrap_for_mp(self.replay_buffer, send_to_device=False)
        iterator = ASTBSamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            action_sampling_size=self.cfg.algo.astb.train_action_sampling_size,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=replay_buffer,
            ratio=self.cfg.algo.offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            det_after=self.cfg.algo.train_det_after,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.model, send_to_device=True)
        iterator = ASTBSamplingIterator(
            self.test_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            action_sampling_size=self.cfg.algo.astb.valid_action_sampling_size,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            ratio=self.cfg.algo.valid_offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "valid"),
            sample_cond_info=self.cfg.cond.valid_sample_cond_info,
            stream=False,
            random_action_prob=self.cfg.algo.valid_random_action_prob,
        )
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_final_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        iterator = ASTBSamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            action_sampling_size=self.cfg.algo.astb.final_action_sampling_size,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=None,
            ratio=0.0,
            log_dir=os.path.join(self.cfg.log_dir, "final"),
            random_action_prob=0.0,
            hindsight_ratio=0.0,
            init_train_iter=self.cfg.num_training_steps,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )
