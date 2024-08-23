import numpy as np
import math
from gflownet.config import Config
from gflownet.algo.synthesis_sampling import SynthesisSampler
from gflownet.algo.trajectory_balance_synthesis import SynthesisTrajectoryBalance
from gflownet.envs.synthesis.retrosynthesis import RetroSynthesisTree
from gflownet.envs.synthesis import SynthesisEnv, SynthesisEnvContext


class ToyTrajectoryBalance(SynthesisTrajectoryBalance):
    def __init__(
        self,
        env: SynthesisEnv,
        ctx: SynthesisEnvContext,
        rng: np.random.RandomState,
        cfg: Config,
    ):
        super().__init__(env, ctx, rng, cfg)
        self.graph_sampler = ToySamper(
            ctx,
            env,
            cfg.algo.min_len,
            cfg.algo.max_len,
            rng,
            self.action_sampler,
            cfg.algo.action_sampling.onpolicy_temp,
            self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )


class ToySamper(SynthesisSampler):
    def cal_bck_logprob(self, curr_rt: RetroSynthesisTree, next_rt: RetroSynthesisTree):
        # If the max length is 2, we can know exact passing trajectories.
        if self.uniform_bck_logprob:
            # NOTE: PB is uniform
            return -math.log(len(next_rt))
        else:
            # NOTE: PB is proportional to the number of passing trajectories
            curr_rt_lens = curr_rt.length_distribution(self.max_len)
            next_rt_lens = next_rt.length_distribution(self.max_len)

            next_smi = next_rt.smi
            num_actions = 1
            for i, block in enumerate(self.env.building_blocks):
                if next_smi == block:
                    num_actions = (
                        self.env.precomputed_bb_masks[:, :, i].sum().item() + 1 + 1
                    )  # 1: Stop, 1: ReactUni(using single template)
                    break

            numerator = sum(
                curr_rt_lens[_t] * sum(num_actions**_i for _i in range(self.max_len - _t))
                for _t in range(0, self.max_len)  # T(s->s'), t=0~N-1, i=0~N-t-1
            )

            denominator = sum(
                next_rt_lens[_t] * sum(num_actions**_i for _i in range(self.max_len - _t + 1))
                for _t in range(1, self.max_len + 1)  # T(s'), t=1~N, i=0~N-t
            )
            return math.log(numerator) - math.log(denominator)
