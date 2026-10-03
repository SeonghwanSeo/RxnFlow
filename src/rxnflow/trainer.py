"""Local trajectory-balance training with replay, EMA, and restart."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from rdkit import Chem

from rxnflow import __version__
from rxnflow.config import Config
from rxnflow.core.types import ActionType, Trajectory
from rxnflow.envs.env import SynthesisEnv
from rxnflow.gflownet.conditioning import ConditionSampler
from rxnflow.gflownet.policy import RxnFlowPolicy, resolve_device
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.gflownet.rewards import scalarize_log_rewards
from rxnflow.models import RxnFlowModel
from rxnflow.reward import RewardFunction


def sum_by_trajectory(
    values: torch.Tensor, indices: torch.Tensor, count: int
) -> torch.Tensor:
    """Aggregate transition values by trajectory with native tensor indexing."""

    result = torch.zeros(count, dtype=values.dtype, device=values.device)
    return result.index_add(0, indices, values)


class RxnFlowTrainer:
    def __init__(
        self,
        config: Config,
        reward: RewardFunction,
        restart: str | Path | None = None,
    ):
        # 1. Resolve reward conditions and independent random-number streams.
        config.validate()
        self.config = config
        self.reward = reward
        self.objectives = tuple(reward.objectives)
        if not self.objectives or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("reward.objectives must contain unique objective names")
        self.condition_sampler = ConditionSampler(
            config.reward.beta,
            config.reward.moo_preferences,
            len(self.objectives),
            config.reward.moo_scalarization,
        )
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)
        self.python_rng = random.Random(config.seed)
        self.rng = np.random.default_rng(config.seed)
        # 2. Load the environment and initialize training/EMA sampling models.
        self.env = SynthesisEnv(
            config.data.env_dir,
            config.data.max_atoms,
            config.generation.max_reactions,
            config.training.retrosynthesis_workers,
            config.property_penalty,
            min_synthons=config.generation.min_synthons,
            max_synthons=config.generation.max_synthons,
            min_reactions=config.generation.min_reactions,
        )
        self.model = RxnFlowModel(
            self.env,
            config.model,
            len(self.objectives),
            preference_conditioning=config.reward.moo_preferences[0] != "none",
        ).to(self.device)
        self.sampling_model = (
            RxnFlowModel(
                self.env,
                config.model,
                len(self.objectives),
                preference_conditioning=config.reward.moo_preferences[0] != "none",
            )
            .to(self.device)
            .eval()
        )
        self.sampling_model.load_state_dict(self.model.state_dict())
        # 3. Give the logZ head its own learning rate and exclude its parameters
        # from policy gradient clipping. The condition encoder stays in policy.
        self.policy_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("_logZ.")
        ]
        self.log_z_parameters = list(self.model._logZ.parameters())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy_parameters},
                {
                    "params": self.log_z_parameters,
                    "lr": config.training.learning_rate_logZ,
                },
            ],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 2 ** (-step / config.training.lr_decay_steps)
        )
        # 4. Initialize replay/output, then restore saved state for a restart.
        self.replay = ReplayBuffer(config.training.replay_capacity)
        self.step = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.sample_dir.mkdir(exist_ok=True)
        self.config.save(self.output_dir / "config.yaml")
        self.policy = RxnFlowPolicy(self.env, self.model, config, self.device, self.rng)
        self.sampling_policy = RxnFlowPolicy(
            self.env, self.sampling_model, config, self.device, self.rng
        )
        if restart is not None:
            self.load_checkpoint(restart)

    def _reward_class_name(self) -> str:
        return f"{type(self.reward).__module__}.{type(self.reward).__qualname__}"

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save model, optimizer, replay, and RNG state for an exact restart."""
        destination = (
            Path(path)
            if path is not None
            else self.checkpoint_dir / f"step_{self.step:06d}.ckpt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(
            {
                "rxnflow_version": __version__,
                "step": self.step,
                "config": self.config.to_dict(),
                "environment": self.env.signature,
                "model": self.model.state_dict(),
                "sampling_model": self.sampling_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "replay": self.replay.state_dict(),
                "numpy_rng": self.rng.bit_generator.state,
                # Library subsampling uses the NumPy generator above. Gumbel
                # sampling and dropout use the model-device global generator.
                # Beta/preference draws use the CPU global generator.
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda"
                    else None
                ),
                "python_random": self.python_rng.getstate(),
                "objectives": self.objectives,
                "reward_class": self._reward_class_name(),
            },
            temporary,
        )
        temporary.replace(destination)
        latest = self.checkpoint_dir / "latest.ckpt"
        latest_temporary = latest.with_suffix(".tmp")
        # Copy serialized bytes; do not deserialize the replay and optimizer
        # just to serialize the identical checkpoint again.
        shutil.copyfile(destination, latest_temporary)
        latest_temporary.replace(latest)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        # RNG states are CPU byte tensors even for a CUDA model. Parameter and
        # optimizer loaders move their own tensors to the model device.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("rxnflow_version") != __version__:
            raise ValueError(
                f"checkpoint was created by RxnFlow {checkpoint.get('rxnflow_version')!r}; "
                f"this installation is RxnFlow {__version__}"
            )
        if checkpoint.get("config") != self.config.to_dict():
            raise ValueError(
                "restart configuration differs from the resolved checkpoint configuration"
            )
        if checkpoint.get("environment") != self.env.signature:
            raise ValueError(
                "prepared environment differs from the checkpoint environment"
            )
        if checkpoint.get("reward_class") != self._reward_class_name():
            raise ValueError(
                "reward implementation differs from the checkpoint reward class"
            )
        if checkpoint["objectives"] != self.objectives:
            raise ValueError("reward objectives differ from the checkpoint")
        # Restore training state only after config, catalog, and reward match.
        self.model.load_state_dict(checkpoint["model"])
        self.sampling_model.load_state_dict(checkpoint["sampling_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.replay.load_state_dict(checkpoint["replay"])
        self.rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], self.device)
        self.python_rng.setstate(checkpoint["python_random"])
        self.step = int(checkpoint["step"])

    def _assign_rewards(self, trajectories: list[Trajectory]) -> None:
        """Attach raw objective values and preference-weighted scalar rewards."""
        values = self.reward.run(
            [
                Chem.MolFromSmiles(value.final_smiles) if value.valid else None
                for value in trajectories
            ],
        )
        # Objective values are independent of the condition; retain them for
        # logging and replay. The scalar reward has not been raised to beta.
        preferences = np.array([t.preferences for t in trajectories], dtype=np.float32)
        scalar_rewards = (
            scalarize_log_rewards(
                torch.from_numpy(values),
                torch.from_numpy(preferences),
                self.config.reward.moo_scalarization,
                self.config.training.reward_floor,
            )
            .exp()
            .tolist()
        )
        for trajectory, objectives, scalar in zip(
            trajectories, values.tolist(), scalar_rewards, strict=True
        ):
            trajectory.objective_rewards = objectives
            trajectory.reward = scalar if trajectory.valid else 0.0

    def compute_batch_losses(
        self, trajectories: list[Trajectory], num_online: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute conditional TB loss for online trajectories followed by replay."""
        # 1. Encode each trajectory's stored condition and predict logZ.
        beta = torch.tensor(
            [t.beta for t in trajectories], dtype=torch.float32, device=self.device
        )
        preferences = torch.tensor(
            [t.preferences for t in trajectories], dtype=torch.float32, device=self.device
        )
        cond_info = self.model.encode_cond(beta, preferences)
        log_Z = self.model.logZ(cond_info).squeeze(-1)
        # 2. Score all observed transitions together, then sum by trajectory.
        transitions = []
        traj_indices: list[int] = []
        traj_log_p_B = torch.tensor(
            [
                sum(step.log_p_B for step in trajectory.steps)
                for trajectory in trajectories
            ],
            dtype=torch.float32,
            device=self.device,
        )
        for traj_idx, trajectory in enumerate(trajectories):
            for transition in trajectory.steps:
                transitions.append(transition)
                traj_indices.append(traj_idx)
        if transitions:
            batch_idx = torch.tensor(traj_indices, dtype=torch.long, device=self.device)
            log_p_F = self.policy.log_prob(
                [step.state for step in transitions],
                [step.action for step in transitions],
                beta[batch_idx],
                preferences[batch_idx],
            )
            traj_log_p_F = sum_by_trajectory(log_p_F, batch_idx, len(trajectories))
        else:
            traj_log_p_F = torch.zeros(len(trajectories), device=self.device)
        # 3. Scalarize objectives, floor before log, and apply the reward exponent.
        objective_rewards = torch.tensor(
            [t.objective_rewards for t in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        scaled_log_R = (
            scalarize_log_rewards(
                objective_rewards,
                preferences,
                self.config.reward.moo_scalarization,
                self.config.training.reward_floor,
            )
            * beta
        )
        tb_residual = log_Z + traj_log_p_F - traj_log_p_B - scaled_log_R
        # Penalize the squared mismatch between forward and backward log flow.
        traj_losses = tb_residual.square()
        loss = traj_losses.mean()
        # 4. Derive diagnostics from the same pre-update values without gradients.
        # "batch_entropy" is trajectory surprisal on this online+replay batch,
        # not categorical entropy or an unbiased on-policy entropy estimate.
        with torch.no_grad():
            is_valid = torch.tensor(
                [value.valid for value in trajectories], device=self.device
            )
            info = {
                "loss": loss.detach(),
                "logZ": log_Z.mean(),
                "beta": beta.mean(),
                "logit_scale": self.model.logit_scale(cond_info).mean(),
                "batch_entropy": -traj_log_p_F.mean(),
                "traj_log_p_F": traj_log_p_F.mean(),
                "traj_log_p_B": traj_log_p_B.mean(),
                "scaled_log_R": scaled_log_R.mean(),
                "tb_residual": tb_residual.mean(),
                "online_loss": traj_losses[:num_online].mean(),
                # No replay on the first update (or when disabled).
                "replay_loss": traj_losses[num_online:].sum()
                / max(1, len(trajectories) - num_online),
                "valid_losses": (traj_losses * is_valid).sum()
                / is_valid.sum().clamp_min(1),
                "invalid_losses": (traj_losses * ~is_valid).sum()
                / (~is_valid).sum().clamp_min(1),
                "invalid_logprob": (traj_log_p_F * ~is_valid).sum()
                / (~is_valid).sum().clamp_min(1),
                "invalid_trajectories": (~is_valid).float().mean(),
            }
        return loss, info

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.config.training.ema_decay
        for target, source in zip(
            self.sampling_model.parameters(), self.model.parameters(), strict=True
        ):
            target.mul_(decay).add_(source, alpha=1 - decay)

    def _write_samples(self, trajectories: list[Trajectory]) -> None:
        """Write the compact reaction paths for one training update."""
        sample_path = self.sample_dir / f"step_{self.step:06d}.jsonl"
        with sample_path.open("w", encoding="utf-8") as handle:
            for index, value in enumerate(trajectories):
                traj = []
                for transition in value.steps:
                    action = transition.action
                    # The first reaction's state already contains the initial brick.
                    if action.action_type == ActionType.FIRST_SYNTHON:
                        continue
                    synthon_smiles = None
                    if action.library_name is not None:
                        assert action.synthon_index is not None
                        synthon_smiles = self.env.synthons[action.library_name].smiles[
                            action.synthon_index
                        ]
                    traj.append(
                        {
                            "state": transition.state.smiles,
                            "reaction": action.reaction,
                            "synthon_smiles": synthon_smiles,
                        }
                    )
                sample = {
                    "step": self.step,
                    "sample": index,
                    "final_smiles": value.final_smiles,
                    "reward": value.reward,
                    "objective_rewards": value.objective_rewards,
                    "beta": value.beta,
                    "preferences": value.preferences,
                    "valid": value.valid,
                    "invalid_reason": value.invalid_reason,
                    "traj": traj,
                }
                handle.write(json.dumps(sample) + "\n")

    def close(self) -> None:
        """Release environment workers; model and replay remain available."""
        self.env.close()

    def run(self, steps: int | None = None) -> Path:
        """Run additional optimization steps and return the final checkpoint."""
        try:
            final_step = self.step + (
                steps if steps is not None else self.config.training.steps
            )
            log_path = self.output_dir / "training.jsonl"
            last_saved_step = -1
            checkpoint = None
            while self.step < final_step:
                # 1. Generate online trajectories with the EMA model and score rewards.
                started = perf_counter()
                self.sampling_model.eval()
                count = self.config.training.num_online
                beta, preferences = self.condition_sampler.sample(count)
                online_trajs = self.sampling_policy.sample_from_model(
                    self.config.training.num_online,
                    random_action_prob=self.config.training.random_action_prob,
                    beta=beta,
                    preferences=preferences,
                )
                sample_time = perf_counter() - started
                self._assign_rewards(online_trajs)
                # 2. Sample replay before insertion so online trajectories cannot be
                # duplicated as replay entries in this same optimization batch.
                batch = online_trajs + self.replay.sample(
                    self.config.training.num_replay, self.python_rng
                )
                self.replay.add(online_trajs)
                # 3. Update the policy/logZ, learning rates, and EMA sampling weights.
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                loss, loss_info = self.compute_batch_losses(batch, len(online_trajs))
                loss.backward()
                # clip_grad_norm_ already returns the pre-clip norm. No second
                # traversal of policy gradients is needed for diagnostics.
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_parameters, 100.0
                )
                loss_info.update(
                    policy_grad_norm=policy_grad_norm,
                    # Total pre-clip norm includes the unclipped logZ head.
                    grad_norm=(
                        policy_grad_norm.square()
                        + sum(
                            p.grad.square().sum()
                            for p in self.log_z_parameters
                            if p.grad is not None
                        )
                    ).sqrt(),
                    policy_grad_clipped=(policy_grad_norm > 100.0).float(),
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self._update_ema()
                self.step += 1

                # 4. Collect scalar diagnostics, persist online attempts, and checkpoint.
                # Transfer scalar diagnostics together rather than synchronizing
                # CUDA separately for every metric. TB values are pre-update.
                loss_metrics = dict(
                    zip(
                        loss_info,
                        torch.stack(list(loss_info.values())).detach().cpu().tolist(),
                        strict=True,
                    )
                )
                record = {
                    "step": self.step,
                    **loss_metrics,
                    "num_online": len(online_trajs),
                    "num_replay": len(batch) - len(online_trajs),
                    "num_valid": sum(value.valid for value in batch),
                    "num_invalid": sum(not value.valid for value in batch),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "reward": sum(value.reward for value in online_trajs)
                    / len(online_trajs),
                    "objective_rewards": {
                        name: sum(t.objective_rewards[i] for t in online_trajs)
                        / len(online_trajs)
                        for i, name in enumerate(self.objectives)
                    },
                    "preferences": {
                        name: sum(t.preferences[i] for t in online_trajs)
                        / len(online_trajs)
                        for i, name in enumerate(self.objectives)
                    },
                    "online_valid_fraction": sum(value.valid for value in online_trajs)
                    / len(online_trajs),
                    # Unique fraction uses chemically valid online terminal molecules;
                    # reward filtering does not change their validity.
                    "online_unique_fraction": len(
                        {value.final_smiles for value in online_trajs if value.valid}
                    )
                    / max(1, sum(value.valid for value in online_trajs)),
                    # Selected actions per generated trajectory, including FirstSynthon
                    # and a failed selected action. Empty action spaces add no step.
                    "traj_lens": sum(len(value.steps) for value in online_trajs)
                    / len(online_trajs),
                    "sampling_time": sample_time,
                    "iteration_time": perf_counter() - started,
                }
                # Keep all online attempts, including invalid ones. This readable
                # path log omits replay-only state flags and backward probabilities;
                # checkpoints retain the complete training trajectories.
                log_started = perf_counter()
                self._write_samples(online_trajs)
                record["logging_time"] = perf_counter() - log_started
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if self.step % self.config.training.log_every == 0:
                    print(json.dumps(record, sort_keys=True), flush=True)
                if self.step % self.config.training.checkpoint_every == 0:
                    checkpoint = self.save_checkpoint()
                    last_saved_step = self.step
            if last_saved_step != self.step:
                checkpoint = self.save_checkpoint()
            assert checkpoint is not None
            return checkpoint
        finally:
            self.close()
