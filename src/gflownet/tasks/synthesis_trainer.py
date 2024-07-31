import socket

from gflownet.config import Config
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.models.synthesis_gfn import GFN_Synthesis
from gflownet.envs.synthesis.env import SynthesisEnv
from gflownet.envs.synthesis import SynthesisEnvContext
from gflownet.algo.trajectory_balance_synthesis import SynthesisTrajectoryBalance


class SynthesisTrainer(StandardOnlineTrainer):
    env: SynthesisEnv
    ctx: SynthesisEnvContext

    def set_default_hps(self, cfg: Config):
        # NOTE: Same to SEHFrag-MOO
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

        cfg.cond.temperature.sample_dist = "uniform"
        cfg.cond.temperature.dist_params = [0, 64.0]

        cfg.algo.method = "TB"
        cfg.algo.sampling_tau = 0.95  # 0.99 in SEHFrag, 0.95 in SEHFrag-MOO
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

        # NOTE: For Synthesis-aware generation
        cfg.model.fp_nbits_building_block = 1024
        cfg.model.num_emb_building_block = 64
        cfg.model.num_layers_building_block = 0
        cfg.algo.min_len = 2
        cfg.algo.max_len = 4
        cfg.validate_every = 0

    def setup_env(self):
        self.env = SynthesisEnv(self.cfg.env_dir)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        algo = SynthesisTrajectoryBalance
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)

    def setup_model(self):
        self.model = GFN_Synthesis(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext(
            self.env,
            num_cond_dim=self.task.num_cond_dim,
            fp_radius_building_block=self.cfg.model.fp_radius_building_block,
            fp_nbits_building_block=self.cfg.model.fp_nbits_building_block,
        )
