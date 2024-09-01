import numpy as np
import torch
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.config import Config
from gflownet.algo.config import TBVariant
from gflownet.algo.trajectory_balance import TrajectoryBalance

from gflownet.models.synthesis_gfn import SynthesisGFN
from gflownet.envs.synthesis import SynthesisEnv, SynthesisEnvContext
from gflownet.envs.synthesis.action import ReactionActionIdx
from gflownet.algo.synthesis_sampling import SynthesisSampler
from gflownet.envs.synthesis.action_categorical import ReactionActionCategorical
from gflownet.envs.synthesis.action_sampling import ActionSamplingPolicy


class SynthesisTrajectoryBalance(TrajectoryBalance):
    env: SynthesisEnv
    ctx: SynthesisEnvContext
    graph_sampler: SynthesisSampler

    def __init__(self, env: SynthesisEnv, ctx: SynthesisEnvContext, rng: np.random.RandomState, cfg: Config):
        self.action_sampler: ActionSamplingPolicy = ActionSamplingPolicy(env, cfg)
        super().__init__(env, ctx, rng, cfg)

        # TODO: implement others
        assert self.cfg.variant == TBVariant.TB
        assert self.model_is_autoregressive is False
        assert self.cfg.do_parameterize_p_b is False
        assert self.cfg.do_correct_idempotent is False
        assert self.model_is_autoregressive is False

    def setup_graph_sampler(self):
        self.graph_sampler: SynthesisSampler = SynthesisSampler(
            self.ctx,
            self.env,
            self.global_cfg.algo.min_len,
            self.global_cfg.algo.max_len,
            self.rng,
            self.action_sampler,
            self.global_cfg.algo.action_sampling.onpolicy_temp,
            self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )

    def create_training_data_from_own_samples(
        self,
        model: SynthesisGFN,
        n: int,
        cond_info: Tensor,
        random_action_prob: float,
    ):
        dev = self.ctx.device
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, dev, random_action_prob)
        return data

    def create_training_data_from_graphs(
        self,
        graphs,
        model: SynthesisGFN | None,
        cond_info: Tensor | None = None,
        random_action_prob: float | None = None,
    ):
        return []

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """

        if self.model_is_autoregressive:
            raise NotImplementedError
        else:
            torch_graphs = [
                self.ctx.graph_to_Data(traj[0], tj_idx) for tj in trajs for tj_idx, traj in enumerate(tj["traj"])
            ]
            actions = [self.ctx.GraphAction_to_aidx(traj[1]) for tj in trajs for traj in tj["traj"]]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        if self.cfg.do_parameterize_p_b:
            batch.bck_actions = torch.tensor(
                [self.ctx.GraphAction_to_aidx(traj[1]) for tj in trajs for traj in tj["bck_a"]]
            )
            batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()

        if self.cfg.do_correct_idempotent:
            raise NotImplementedError()

        return batch

    def calculate_log_prob(
        self,
        action_categorical: ReactionActionCategorical,
        actions: list[ReactionActionIdx],
        **kwargs,
    ):
        # NOTE: To highlight the different btw fwd_cat.log_prob(...) & bck_cat.log_prob(...) in syn-tb
        return action_categorical.log_prob(actions, self.action_sampler, **kwargs)
