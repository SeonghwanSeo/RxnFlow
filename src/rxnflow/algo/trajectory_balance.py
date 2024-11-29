import torch
from torch import Tensor
import torch_geometric.data as gd

from gflownet.algo.config import TBVariant
from gflownet.utils.misc import get_worker_device

from rxnflow.base.gflownet.trajectory_balance import CustomTB
from rxnflow.config import Config
from rxnflow.models.gfn import RxnFlow
from rxnflow.envs import SynthesisEnv, SynthesisEnvContext
from rxnflow.policy import SubsamplingPolicy
from rxnflow.algo.synthetic_path_sampling import SyntheticPathSampler
from rxnflow.utils.misc import set_worker_env


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
        set_worker_env("action_subsampler", self.action_subsampler)
        super().__init__(env, ctx, cfg)

    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler(
            self.ctx,
            self.env,
            self.global_cfg.algo.min_len,
            self.global_cfg.algo.max_len,
            self.action_subsampler,
            self.global_cfg.algo.action_subsampling.onpolicy_temp,
            self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )

    def create_training_data_from_own_samples(
        self,
        model: RxnFlow,
        n: int,
        cond_info: Tensor | None = None,
        random_action_prob: float | None = 0.0,
    ):
        assert isinstance(model, RxnFlow)
        assert cond_info is not None
        random_action_prob = random_action_prob if random_action_prob else 0.0
        dev = get_worker_device()
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)
        return data

    def create_training_data_from_graphs(
        self,
        graphs,
        model: RxnFlow | None = None,
        cond_info: Tensor | None = None,
        random_action_prob: float | None = 0.0,
    ):
        # TODO: Do Implement!
        assert len(graphs) == 0
        return []

    def construct_batch(self, trajs: list[dict], cond_info: Tensor, log_rewards: Tensor) -> gd.Batch:
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
        batch.num_rxns = torch.tensor([len(i["traj"]) - 1 - i["is_sink"][-1] for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        if self.cfg.do_parameterize_p_b:
            raise NotImplementedError
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        if self.cfg.do_correct_idempotent:
            raise NotImplementedError()
        return batch

    def compute_batch_losses(
        self,
        model: RxnFlow,
        batch: gd.Batch,
        num_bootstrap: int = 0,  # type: ignore[override]
    ):
        loss, info = super().compute_batch_losses(model, batch, num_bootstrap)
        info["num_rxns"] = batch.num_rxns.float().mean()
        return loss, info
