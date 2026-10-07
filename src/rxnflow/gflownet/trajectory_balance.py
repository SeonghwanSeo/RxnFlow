"""Conditional trajectory-balance loss and reward scalarization."""

import math
from typing import Literal

import numpy as np
import torch

from rxnflow.core.types import Trajectory
from rxnflow.gflownet.policy import RxnFlowPolicy


def scalarize_log_rewards(
    values: np.ndarray,
    moo_scalarization: str,
    moo_preference: np.ndarray,
) -> np.ndarray:
    """Return scalarized log rewards from positive scores and normalized weights."""
    assert moo_scalarization in ("sum", "mul"), "moo_scalarization must be 'sum' or 'mul'"
    if moo_scalarization == "mul":
        log_rewards = (np.log(values) * moo_preference).sum(-1)
    else:
        log_rewards = np.log((values * moo_preference).sum(-1))
    return log_rewards


class TrajectoryBalance:
    """Evaluate TB on online and replay trajectories using the training policy."""

    def __init__(
        self,
        policy: RxnFlowPolicy,
        moo_scalarization: Literal["sum", "mul"],
        reward_floor: float,
        loss_fn: Literal["mse", "mae", "huber"] = "mse",
    ) -> None:
        if loss_fn not in ("mse", "mae", "huber"):
            raise ValueError("loss_fn must be mse, mae, or huber")
        self.policy = policy
        self.moo_scalarization = moo_scalarization
        self.reward_floor = reward_floor
        self.loss_fn = loss_fn

    def compute_batch_losses(
        self,
        trajectories: list[Trajectory],
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

        # 1. Compute rewards.
        beta = np.array([t.beta for t in trajectories], dtype=np.float64)
        preference = np.array([t.preferences for t in trajectories], dtype=np.float64)
        objective_rewards = np.array(
            [t.objective_rewards for t in trajectories], dtype=np.float64
        ).clip(min=self.reward_floor)
        property_rewards = np.array(
            [t.property_reward for t in trajectories], dtype=np.float64
        ).clip(min=self.reward_floor)

        # Scalarize the multi-objective rewards
        log_R = scalarize_log_rewards(
            objective_rewards, self.moo_scalarization, preference
        )
        # Apply the property penalty to the scalarized reward
        log_R += np.log(property_rewards)
        # Apply a floor
        log_R = np.clip(log_R, a_min=math.log(self.reward_floor), a_max=None)
        # Scale the log rewards by beta
        scaled_log_R = beta * log_R

        # Move to torch tensors
        beta = torch.from_numpy(beta.astype(np.float32)).to(device)
        preference = torch.from_numpy(preference.astype(np.float32)).to(device)
        scaled_log_R = torch.from_numpy(scaled_log_R.astype(np.float32)).to(device)

        # 2. Encode each trajectory's stored condition and predict logZ.
        cond_info = model.encode_cond(beta, preference)
        log_Z = model.logZ(cond_info).squeeze(-1)

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
            )
            traj_log_p_F = torch.zeros(len(trajectories), device=device)
            traj_log_p_F.index_add_(0, batch_idx, log_p_F)
        else:
            traj_log_p_F = torch.zeros(len(trajectories), device=device)

        # 4. Compute the trajectory-balance loss.
        tb_error = log_Z + traj_log_p_F - traj_log_p_B - scaled_log_R
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
            is_valid = torch.tensor(
                [value.valid for value in trajectories], device=device
            )
            info = {
                "loss": loss.detach(),
                "logZ": log_Z.mean(),
                "beta": beta.mean(),
                "logit_scale": model.logit_scale(cond_info).mean(),
                "batch_entropy": -traj_log_p_F.mean(),
                "traj_log_p_F": traj_log_p_F.mean(),
                "traj_log_p_B": traj_log_p_B.mean(),
                "scaled_log_R": scaled_log_R.mean(),
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
        return loss, info
