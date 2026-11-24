"""Conditional trajectory-balance loss and reward scalarization."""

from typing import Literal

import numpy as np
import torch

from rxnflow.core.types import Trajectory
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.reward import INVALID_REWARD


def scalarize_rewards(
    scores: np.ndarray,
    preference: np.ndarray,
    method: Literal["sum", "mul"],
    valid: np.ndarray,
) -> np.ndarray:
    """Compute the arithmetic or geometric mean of finite component scores."""
    weights = preference / preference.sum(axis=-1, keepdims=True)
    if method == "sum":
        return (weights * scores).sum(axis=-1)
    assert method == "mul"
    # REINVENT4 floors each component before taking the weighted geometric mean.
    product = np.exp((weights * np.log(scores.clip(min=1e-8))).sum(axis=-1))
    return np.where(valid, product, INVALID_REWARD)


class TrajectoryBalance:
    """Evaluate TB on online and replay trajectories using the training policy."""

    def __init__(
        self,
        policy: RxnFlowPolicy,
        reward_floor: float,
        loss_fn: Literal["mse", "mae", "huber"] = "mse",
        *,
        moo_scalarization: Literal["sum", "mul"] = "sum",
    ) -> None:
        if loss_fn not in ("mse", "mae", "huber"):
            raise ValueError("loss_fn must be mse, mae, or huber")
        self.policy: RxnFlowPolicy = policy
        self.moo_scalarization = moo_scalarization
        self.eps: float = reward_floor
        self.loss_fn: Literal["mse", "mae", "huber"] = loss_fn

    def compute_batch_losses(
        self,
        trajectories: list[Trajectory],
        cond: torch.Tensor | None = None,
        *,
        num_online: int,
        num_replay: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute conditional TB loss for online trajectories followed by replay."""
        assert num_online >= 0 and num_replay >= 0, "batch counts must be non-negative"
        assert num_online + num_replay == len(trajectories), (
            "batch counts must match trajectories"
        )
        model = self.policy.model
        device = self.policy.device

        # 1. Compute the objective rewards.
        beta = np.array([t.beta for t in trajectories], dtype=np.float64)
        preference = np.array([t.preference for t in trajectories], dtype=np.float64)
        objective_rewards = np.array(
            [t.objective_rewards for t in trajectories], dtype=np.float64
        )
        valid = np.array([t.valid for t in trajectories], dtype=bool)
        task_reward = scalarize_rewards(
            objective_rewards, preference, self.moo_scalarization, valid
        )

        # 2. Compute the prior guidance
        prior_beta = np.array([t.prior_beta for t in trajectories], dtype=np.float64)
        log_p_prior = np.array(
            [0.0 if t.prior_log_prob is None else t.prior_log_prob for t in trajectories],
            dtype=np.float64,
        )

        # 3. Compute the property penalty
        property_penalty = np.array(
            [t.property_reward for t in trajectories], dtype=np.float64
        ).clip(min=self.eps)
        log_property_penalty = np.log(property_penalty)

        # 4. Compute the scaled log rewards for TB loss
        log_R_gfn = beta * task_reward + prior_beta * log_p_prior + log_property_penalty

        # Move to torch tensors
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa
        beta = to_tensor(beta)
        prior_beta = to_tensor(prior_beta)
        preference = to_tensor(preference)
        log_R_gfn = to_tensor(log_R_gfn)

        # Encode each trajectory's stored condition and predict logZ.
        if cond is not None:
            assert cond.shape == (len(trajectories), model.condition_dim)
            cond = cond.to(device)
        cond_emb = model.encode_cond(beta, preference, cond, prior_beta=prior_beta)
        log_Z = model.logZ(cond_emb).squeeze(-1)
        logit_scale = model.logit_scale(cond_emb).squeeze(-1)

        # 3. Score all observed transitions together, then sum by trajectory.
        transitions = []
        traj_indices: list[int] = []
        traj_log_p_B = torch.tensor(
            [
                sum(step.log_p_B for step in trajectory.steps)
                for trajectory in trajectories
            ],
            dtype=torch.float32,
            device=device,
        )
        for traj_idx, trajectory in enumerate(trajectories):
            for transition in trajectory.steps:
                transitions.append(transition)
                traj_indices.append(traj_idx)
        if transitions:
            batch_idx = torch.tensor(traj_indices, dtype=torch.long, device=device)
            log_p_F = self.policy.log_prob(
                [step.state for step in transitions],
                [step.action for step in transitions],
                beta[batch_idx],
                preference[batch_idx],
                cond=cond[batch_idx] if cond is not None else None,
                prior_beta=prior_beta[batch_idx],
            )
            traj_log_p_F = torch.zeros(len(trajectories), device=device)
            traj_log_p_F.index_add_(0, batch_idx, log_p_F)
        else:
            traj_log_p_F = torch.zeros(len(trajectories), device=device)

        # 4. Compute the trajectory-balance loss.
        tb_error = log_Z + traj_log_p_F - traj_log_p_B - log_R_gfn
        if self.loss_fn == "mse":
            tb_losses = tb_error.square()
        elif self.loss_fn == "mae":
            tb_losses = tb_error.abs()
        elif self.loss_fn == "huber":
            tb_losses = torch.nn.functional.huber_loss(
                tb_error, torch.zeros_like(tb_error), reduction="none", delta=1.0
            )
        else:
            raise ValueError(f"Unknown loss_fn: {self.loss_fn}")
        loss = tb_losses.mean()

        # 5. Derive diagnostics from the same pre-update values without gradients.
        # "batch_entropy" is trajectory surprisal on this online+replay batch,
        # not categorical entropy or an unbiased on-policy entropy estimate.
        with torch.no_grad():
            is_valid = torch.as_tensor(valid, device=device)
            info = {
                "loss": loss.detach(),
                "logZ": log_Z.mean(),
                "beta": beta.mean(),
                "logit_scale": logit_scale.mean(),
                "batch_entropy": -traj_log_p_F.mean(),
                "traj_log_p_F": traj_log_p_F.mean(),
                "traj_log_p_B": traj_log_p_B.mean(),
                "scaled_log_R": log_R_gfn.mean(),
                "tb_error": tb_error.mean(),
                "online_loss": tb_losses[:num_online].sum() / max(1, num_online),
                "replay_loss": tb_losses[num_online : num_online + num_replay].sum()
                / max(1, num_replay),
                "valid_losses": (tb_losses * is_valid).sum()
                / is_valid.sum().clamp_min(1),
                "invalid_losses": (tb_losses * ~is_valid).sum()
                / (~is_valid).sum().clamp_min(1),
                "invalid_logprob": (traj_log_p_F * ~is_valid).sum()
                / (~is_valid).sum().clamp_min(1),
                "invalid_trajectories": (~is_valid).float().mean(),
            }
        info["prior_beta"] = prior_beta.mean().detach()
        return loss, info
