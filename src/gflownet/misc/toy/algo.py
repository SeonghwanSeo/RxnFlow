import copy
import math
import numpy as np
import torch
import torch_geometric.data as gd
from torch_scatter import scatter

from gflownet.config import Config
from gflownet.algo.synthesis_sampling import SynthesisSampler
from gflownet.algo.trajectory_balance_synthesis import SynthesisTrajectoryBalance
from gflownet.envs.synthesis.action_sampling import ActionSamplingPolicy
from gflownet.envs.synthesis.retrosynthesis import RetroSynthesisTree
from gflownet.envs.synthesis import SynthesisEnv, SynthesisEnvContext
from gflownet.models.synthesis_gfn import SynthesisGFN


@torch.no_grad()
def model_grad_norm(model):
    x = 0
    for i in model.parameters():
        if i.grad is not None:
            x += (i.grad * i.grad).sum()
    return torch.sqrt(x)


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
        cfg = copy.deepcopy(cfg)
        cfg.algo.action_sampling.num_sampling_add_first_reactant = 1_200_000
        cfg.algo.action_sampling.sampling_ratio_reactbi = 1.0
        cfg.algo.action_sampling.max_sampling_reactbi = 1_200_000
        self.action_sampler_all: ActionSamplingPolicy = ActionSamplingPolicy(env, cfg)

    def compute_batch_losses(
        self,
        model: SynthesisGFN,
        batch: gd.Batch,
        num_bootstrap: int = 0,  # type: ignore[override]
    ):
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

        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        fwd_cat, per_graph_out = model(batch, cond_info[batch_idx])
        log_reward_preds = per_graph_out[final_graph_idx, 0]
        log_Z = model.logZ(cond_info)[:, 0]

        log_p_B = batch.log_p_B
        log_p_F_true = fwd_cat.log_prob(batch.actions, self.action_sampler_all)
        log_p_F_est = fwd_cat.log_prob(batch.actions, self.action_sampler)
        assert log_p_F_true.shape == log_p_F_est.shape == log_p_B.shape

        traj_log_p_F_true = scatter(log_p_F_true, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_F_est = scatter(log_p_F_est, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        numerator_true = log_Z + traj_log_p_F_true
        numerator_est = log_Z + traj_log_p_F_est
        denominator = clip_log_R + traj_log_p_B

        if self.cfg.epsilon is not None:
            epsilon = torch.tensor([self.cfg.epsilon], device=dev).float()
            numerator_true = torch.logaddexp(numerator_true, epsilon)
            numerator_est = torch.logaddexp(numerator_est, epsilon)
            denominator = torch.logaddexp(denominator, epsilon)
        traj_losses_true = (numerator_true - denominator).pow(2)
        traj_losses_est = (numerator_est - denominator).pow(2)

        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses_true = traj_losses_true / batch.traj_lens
            traj_losses_est = traj_losses_est / batch.traj_lens
        if self.cfg.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(log_rewards)
            reward_losses = (log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap]).pow(2)
            reward_loss = reward_losses.mean() * self.cfg.reward_loss_multiplier
        else:
            reward_loss = 0

        loss_true = traj_losses_true.mean() + reward_loss
        loss_est = traj_losses_est.mean() + reward_loss

        loss_true.backward(retain_graph=True)
        grad_norm_true = model_grad_norm(model)
        model.zero_grad()

        info = {
            "offline_loss": traj_losses_est[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses_est[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "reward_loss": reward_loss,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_logprob": (invalid_mask * traj_log_p_F_est).sum() / (invalid_mask.sum() + 1e-4),
            "invalid_losses": (invalid_mask * traj_losses_est).sum() / (invalid_mask.sum() + 1e-4),
            "logZ": log_Z.mean(),
            "loss": loss_est.item(),
            "loss_true": loss_true.item(),
            "grad_norm_true": grad_norm_true.item(),
        }
        return loss_est, info


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
                        self.env.building_block_mask[:, :, i].sum().item() + 1 + 1
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
