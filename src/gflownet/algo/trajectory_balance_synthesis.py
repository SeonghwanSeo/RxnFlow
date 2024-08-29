import numpy as np
import torch
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.config import Config
from gflownet.algo.config import TBVariant
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.algo.synthesis_sampling import SynthesisSampler

from gflownet.models.synthesis_gfn import SynthesisGFN
from gflownet.envs.graph_building_env import Graph
from gflownet.envs.synthesis.action_sampling import ActionSamplingPolicy
from gflownet.envs.synthesis import SynthesisEnv, SynthesisEnvContext, ReactionAction


class SynthesisTrajectoryBalance(TrajectoryBalance):
    env: SynthesisEnv
    ctx: SynthesisEnvContext
    graph_sampler: SynthesisSampler

    def __init__(self, env: SynthesisEnv, ctx: SynthesisEnvContext, rng: np.random.RandomState, cfg: Config):
        self.action_sampler: ActionSamplingPolicy = ActionSamplingPolicy(env, cfg)
        super().__init__(env, ctx, rng, cfg)
        if self.cfg.variant == TBVariant.SubTB1:
            raise NotImplementedError

    def setup_graph_sampler(self):
        self.graph_sampler = SynthesisSampler(
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
        data = None
        while data is None:
            try:
                data = self.graph_sampler.sample_from_model(model, n, cond_info, dev, random_action_prob)
            except Exception as e:
                raise e
                print(f"ERROR - create_training_data_from_own_samples - {e}")
                data = None
        return data

    def create_training_data_from_graphs(
        self,
        graphs,
        model: SynthesisGFN | None,
        cond_info: Tensor | None = None,
        random_action_prob: float | None = None,
    ):
        return []

    def get_idempotent_actions(
        self, g: Graph, gd: gd.Data, gp: Graph, action: ReactionAction, return_aidx: bool = True
    ):
        raise NotImplementedError()

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
            actions = [self.ctx.ReactionAction_to_aidx(traj[1]) for tj in trajs for traj in tj["traj"]]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        if self.cfg.do_parameterize_p_b:
            batch.bck_actions = torch.tensor(
                [self.ctx.ReactionAction_to_aidx(traj[1]) for tj in trajs for traj in tj["bck_a"]]
            )
            batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()

        if self.cfg.do_correct_idempotent:
            raise NotImplementedError()

        return batch

    def compute_batch_losses(
        self,
        model: SynthesisGFN,
        batch: gd.Batch,
        num_bootstrap: int = 0,  # type: ignore[override]
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: SynthesisGFN
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        log_rewards = batch.log_rewards
        # Clip rewards
        assert log_rewards.ndim == 1
        clip_log_R = torch.maximum(
            log_rewards, torch.tensor(self.global_cfg.algo.illegal_action_logreward, device=dev)
        ).float()
        cond_info = batch.cond_info
        invalid_mask = 1 - batch.is_valid

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical representing the forward
        # policy P_F, optionally a backward policy P_B, and per-graph outputs (e.g. F(s) in SubTB).
        if self.cfg.do_parameterize_p_b:
            fwd_cat, bck_cat, per_graph_out = model(batch, cond_info[batch_idx])
        else:
            if self.model_is_autoregressive:
                fwd_cat, per_graph_out = model(batch, cond_info, batched=True)
            else:
                fwd_cat, per_graph_out = model(batch, cond_info[batch_idx])
        # Retreive the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        log_reward_preds = per_graph_out[final_graph_idx, 0]
        # Compute trajectory balance objective
        log_Z = model.logZ(cond_info)[:, 0]
        # Compute the log prob of each action in the trajectory
        if self.cfg.do_correct_idempotent:
            raise NotImplementedError()
        else:
            # Else just naively take the logprob of the actions we took
            log_p_F = fwd_cat.log_prob(batch.actions, self.action_sampler)
            if self.cfg.do_parameterize_p_b:
                log_p_B = bck_cat.log_prob(batch.bck_actions)

        if self.cfg.do_parameterize_p_b:
            raise NotImplementedError
            log_p_F[final_graph_idx] = 0
            if self.cfg.variant == TBVariant.SubTB1 or self.cfg.variant == TBVariant.DB:
                raise NotImplementedError
            log_p_B = torch.roll(log_p_B, -1, 0) * (1 - batch.is_sink)
        else:
            log_p_B = batch.log_p_B
        assert log_p_F.shape == log_p_B.shape

        # This is the log probability of each trajectory
        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        if self.cfg.variant == TBVariant.SubTB1:
            raise NotImplementedError()
        elif self.cfg.variant == TBVariant.DB:
            raise NotImplementedError()
        else:
            # Compute log numerator and denominator of the ASTB objective
            numerator = log_Z + traj_log_p_F
            denominator = clip_log_R + traj_log_p_B

            if self.mask_invalid_rewards:
                denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

            if self.cfg.epsilon is not None:
                epsilon = torch.tensor([self.cfg.epsilon], device=dev).float()
                numerator = torch.logaddexp(numerator, epsilon)
                denominator = torch.logaddexp(denominator, epsilon)
            if self.tb_loss_is_mae:
                traj_losses = abs(numerator - denominator)
            elif self.tb_loss_is_huber:
                pass  # TODO
            else:
                traj_losses = (numerator - denominator).pow(2)

        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens
        if self.reward_normalize_losses:
            # multiply each loss by how important it is, using R as the importance factor
            # factor = Rp.exp() / Rp.exp().sum()
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            # * num_trajs because we're doing a convex combination, and a .mean() later, which would
            # undercount (by 2N) the contribution of each loss
            traj_losses = factor * traj_losses * num_trajs

        if self.cfg.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(log_rewards)
            if self.reward_loss_is_mae:
                reward_losses = abs(log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap])
            else:
                reward_losses = (log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap]).pow(2)
            reward_loss = reward_losses.mean() * self.cfg.reward_loss_multiplier
        else:
            reward_loss = 0

        loss = traj_losses.mean() + reward_loss
        info = {
            "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "reward_loss": reward_loss,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_logprob": (invalid_mask * traj_log_p_F).sum() / (invalid_mask.sum() + 1e-4),
            "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            "logZ": log_Z.mean(),
            "loss": loss.item(),
        }
        return loss, info
