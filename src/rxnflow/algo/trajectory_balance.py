from gflownet.algo.config import TBVariant

from rxnflow.base.gflownet.trajectory_balance import CustomTB
from rxnflow.config import Config
from rxnflow.models.gfn import RxnFlow
from rxnflow.envs import SynthesisEnv, SynthesisEnvContext
from rxnflow.policy import SubsamplingPolicy
from rxnflow.algo.synthetic_path_sampling import SyntheticPathSampler
from rxnflow.utils.misc import set_worker_env

from torch import Tensor


class SynthesisTB(CustomTB):
    env: SynthesisEnv
    ctx: SynthesisEnvContext
    global_cfg: Config
    graph_sampler: SyntheticPathSampler

    def __init__(self, env: SynthesisEnv, ctx: SynthesisEnvContext, cfg: Config):
        assert cfg.algo.tb.variant == TBVariant.TB
        assert cfg.algo.tb.do_parameterize_p_b is False
        assert cfg.algo.tb.do_correct_idempotent is False

        self.action_subsampler: SubsamplingPolicy = SubsamplingPolicy(env, cfg)
        self.importance_temp = cfg.algo.action_subsampling.importance_temp
        set_worker_env("action_subsampler", self.action_subsampler)
        super().__init__(env, ctx, cfg)

    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler(
            self.ctx,
            self.env,
            self.action_subsampler,
            max_len=self.max_len,
            importance_temp=self.importance_temp,
            sample_temp=self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )

    def create_training_data_from_graphs(
        self,
        graphs,
        model: RxnFlow | None = None,
        cond_info: Tensor | None = None,
        random_action_prob: float | None = 0.0,
    ):
        # TODO: implement here
        assert len(graphs) == 0
        return []
