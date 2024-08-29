import functools
from pathlib import Path
import torch
import socket

from gflownet.config import Config
from gflownet.online_trainer import StandardOnlineTrainer

# FragGFN
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow

# SynGFN
from gflownet.models.synthesis_gfn import SynthesisGFN
from gflownet.envs.synthesis import SynthesisEnv, SynthesisEnvContext
from gflownet.algo.trajectory_balance_synthesis import SynthesisTrajectoryBalance
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook


class BaseTrainer(StandardOnlineTrainer):
    def set_default_hps(self, cfg: Config):
        # SEHFragTrainer
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 0
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

        # Different Parameters
        cfg.cond.temperature.sample_dist = "uniform"
        cfg.cond.temperature.dist_params = [0, 64.0]
        cfg.algo.train_random_action_prob = 0.05

    def load_checkpoint(self, checkpoint_path: str | Path):
        state = torch.load(checkpoint_path, map_location="cpu")
        print(f"load pre-trained model from {checkpoint_path}")
        self.model.load_state_dict(state["models_state_dict"][0])
        if self.sampling_model is not self.model:
            self.sampling_model.load_state_dict(state["sampling_model_state_dict"][0])
        del state


def moo_trainer(cls: type[BaseTrainer]):
    original_setup = cls.setup

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

    cls.setup = new_setup
    return cls


class MOOTrainer(BaseTrainer):
    def setup(self):
        self.cfg.cond.moo.num_objectives = len(self.cfg.task.moo.objectives)
        super().setup()
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


class FragmentTrainer(BaseTrainer):
    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS,
        )


class SynthesisTrainer(BaseTrainer):
    env: SynthesisEnv
    ctx: SynthesisEnvContext

    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.algo.train_random_action_prob = 0.05
        cfg.algo.tb.do_sample_p_b = False

        # NOTE: For Synthesis-aware generation
        cfg.model.fp_nbits_building_block = 1024
        cfg.model.num_emb_building_block = 64
        cfg.model.num_layers_building_block = 0
        cfg.algo.min_len = 2
        cfg.algo.max_len = 4

    def setup_env(self):
        self.env = SynthesisEnv(self.cfg.env_dir)

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext(
            self.env,
            num_cond_dim=self.task.num_cond_dim,
            fp_radius_building_block=self.cfg.model.fp_radius_building_block,
            fp_nbits_building_block=self.cfg.model.fp_nbits_building_block,
        )

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        algo = SynthesisTrajectoryBalance
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)

    def setup_model(self):
        self.model = SynthesisGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )
