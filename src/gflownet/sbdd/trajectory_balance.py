import torch
import torch_geometric.data as gd
from torch_scatter import scatter

from gflownet.algo.config import TBVariant

from gflownet.algo.trajectory_balance_synthesis import SynthesisTrajectoryBalance
from gflownet.sbdd.gfn import SynthesisGFN_SBDD
from gflownet.sbdd.synthesis_sampling import SynthesisSampler_SBDD


class SynthesisTrajectoryBalance_SBDD(SynthesisTrajectoryBalance):
    graph_sampler: SynthesisSampler_SBDD

    def setup_graph_sampler(self):
        self.graph_sampler = SynthesisSampler_SBDD(
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

    def compute_batch_losses(
        self,
        model: SynthesisGFN_SBDD,
        batch: gd.Batch,
        num_bootstrap: int = 0,  # type: ignore[override]
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
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

        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        if self.cfg.do_parameterize_p_b:
            raise NotImplementedError
            fwd_cat, bck_cat, per_graph_out = model(batch, cond_info[batch_idx])
        else:
            if self.model_is_autoregressive:
                raise NotImplementedError
                fwd_cat, per_graph_out = model(batch, cond_info, batched=True)
            else:
                fwd_cat, per_graph_out = model(batch, cond_info[batch_idx], batch_idx)
        log_reward_preds = per_graph_out[final_graph_idx, 0]
        log_Z = model.logZ(cond_info)[:, 0]
        if self.cfg.do_correct_idempotent:
            raise NotImplementedError
        else:
            log_p_F = fwd_cat.log_prob(batch.actions, self.action_sampler)
            if self.cfg.do_parameterize_p_b:
                raise NotImplementedError

        if self.cfg.do_parameterize_p_b:
            raise NotImplementedError
        else:
            log_p_B = batch.log_p_B
        assert log_p_F.shape == log_p_B.shape

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        if self.cfg.variant == TBVariant.SubTB1:
            raise NotImplementedError()
        elif self.cfg.variant == TBVariant.DB:
            raise NotImplementedError()
        else:
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
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
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
