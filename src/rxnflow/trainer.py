"""Local trajectory-balance training with replay, EMA, and restart."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
from rdkit import Chem
from torch import Tensor

from rxnflow import __version__
from rxnflow.config import Config
from rxnflow.envs.env import SynthesisEnv
from rxnflow.gflownet.policy import SynthesisPolicy, resolve_device
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.gflownet.types import ActionKind, Trajectory
from rxnflow.models import RxnFlowModel
from rxnflow.reward import RewardFunction, SampleFilter, evaluate_rewards


def sum_by_trajectory(values: Tensor, indices: Tensor, count: int) -> Tensor:
    """Aggregate transition values by trajectory with native tensor indexing."""

    result = torch.zeros(count, dtype=values.dtype, device=values.device)
    return result.index_add(0, indices, values)


class RxnFlowTrainer:
    def __init__(
        self,
        config: Config,
        reward: RewardFunction,
        restart: str | Path | None = None,
        sample_filter: SampleFilter | None = None,
    ):
        config.validate()
        self.config = config
        self.reward = reward
        self.objectives = tuple(reward.objectives)
        if not self.objectives or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("reward.objectives must contain unique objective names")
        if config.reward.preferences is not None and len(
            config.reward.preferences
        ) != len(self.objectives):
            raise ValueError("reward.preferences must match reward.objectives")
        self.sample_filter = sample_filter
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)
        self.python_rng = random.Random(config.seed)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.env = SynthesisEnv(
            config.data.env_dir,
            config.data.max_atoms,
            config.generation.min_reactions,
            config.generation.max_reactions,
            config.training.retrosynthesis_workers,
            config.property_penalty,
        )
        self.model = RxnFlowModel(self.env, config.model, len(self.objectives)).to(
            self.device
        )
        self.sampling_model = (
            RxnFlowModel(self.env, config.model, len(self.objectives))
            .to(self.device)
            .eval()
        )
        self.sampling_model.load_state_dict(self.model.state_dict())
        # HSX main trains logZ at its own rate and excludes it from policy
        # gradient clipping. One optimizer with two groups keeps restart simple.
        self.policy_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("log_z.")
        ]
        self.log_z_parameters = list(self.model.log_z.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.policy_parameters},
                {
                    "params": self.log_z_parameters,
                    "lr": config.training.log_z_learning_rate,
                },
            ],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 2 ** (-step / config.training.lr_decay_steps)
        )
        self.replay = ReplayBuffer(config.training.replay_capacity)
        self.step = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.output_dir / "config.yaml")
        self.policy = SynthesisPolicy(
            self.env, self.model, config, self.device, self.generator
        )
        self.sampling_policy = SynthesisPolicy(
            self.env, self.sampling_model, config, self.device, self.generator
        )
        if restart is not None:
            self.load_checkpoint(restart)

    def _reward_class_name(self) -> str:
        return f"{type(self.reward).__module__}.{type(self.reward).__qualname__}"

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        destination = (
            Path(path)
            if path is not None
            else self.output_dir / f"checkpoint_{self.step:08d}.pt"
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
                "torch_generator": self.generator.get_state(),
                # Library subsampling uses the CPU generator above. Gumbel
                # sampling and dropout use the model-device global generator.
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
        latest = self.output_dir / "checkpoint_latest.pt"
        latest_temporary = latest.with_suffix(".tmp")
        # Copy serialized bytes; do not deserialize the replay and optimizer
        # just to serialize the identical checkpoint again.
        shutil.copyfile(destination, latest_temporary)
        latest_temporary.replace(latest)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        # RNG states are CPU ByteTensors even for a CUDA model. Parameter and
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
        self.model.load_state_dict(checkpoint["model"])
        self.sampling_model.load_state_dict(checkpoint["sampling_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.replay.load_state_dict(checkpoint["replay"])
        self.generator.set_state(checkpoint["torch_generator"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], self.device)
        self.python_rng.setstate(checkpoint["python_random"])
        self.step = int(checkpoint["step"])

    def _assign_rewards(self, trajectories: list[Trajectory]) -> dict[str, float]:
        values, metrics = evaluate_rewards(
            self.reward,
            [
                Chem.MolFromSmiles(value.final_smiles) if value.valid else None
                for value in trajectories
            ],
            self.sample_filter,
        )
        # Objective values are independent of the condition; retain them for
        # logging and replay. The scalar reward is the untempered weighted sum.
        preferences = values.new_tensor([t.preferences for t in trajectories])
        scalar_rewards = (values * preferences).sum(-1).cpu().tolist()
        for trajectory, objectives, scalar in zip(
            trajectories, values.cpu().tolist(), scalar_rewards, strict=True
        ):
            trajectory.objective_rewards = objectives
            trajectory.reward = scalar if trajectory.valid else 0.0
        return metrics

    def _loss(
        self, trajectories: list[Trajectory], num_fresh: int
    ) -> tuple[Tensor, dict[str, Tensor]]:
        beta = torch.tensor(
            [t.beta for t in trajectories], dtype=torch.float32, device=self.device
        )
        preferences = torch.tensor(
            [t.preferences for t in trajectories], dtype=torch.float32, device=self.device
        )
        condition = self.model.encode_condition(beta, preferences)
        log_z = self.model.log_z(condition).squeeze(-1)
        transitions = []
        trajectory_indices: list[int] = []
        backward_flows = torch.tensor(
            [
                sum(step.log_pb for step in trajectory.steps)
                for trajectory in trajectories
            ],
            dtype=torch.float32,
            device=self.device,
        )
        for trajectory_index, trajectory in enumerate(trajectories):
            for transition in trajectory.steps:
                transitions.append(transition)
                trajectory_indices.append(trajectory_index)
        if transitions:
            indices = torch.tensor(
                trajectory_indices, dtype=torch.long, device=self.device
            )
            values = self.policy.action_log_probabilities(
                [step.state for step in transitions],
                [step.action for step in transitions],
                beta[indices],
                preferences[indices],
            )
            forward_flow = sum_by_trajectory(values, indices, len(trajectories))
        else:
            forward_flow = torch.zeros(len(trajectories), device=self.device)
        objective_rewards = torch.tensor(
            [t.objective_rewards for t in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = (objective_rewards * preferences).sum(-1)
        log_reward = rewards.clamp_min(self.config.reward.floor).log() * beta
        residual = log_z + forward_flow - backward_flows - log_reward
        # Both HSX baselines default to the squared trajectory-balance residual.
        trajectory_losses = residual.square()
        loss = trajectory_losses.mean()
        # Reuse the TB terms already computed for optimization. Reference
        # "batch_entropy" is trajectory surprisal on this fresh+replay batch,
        # not categorical entropy or an unbiased on-policy entropy estimate.
        with torch.no_grad():
            valid = torch.tensor(
                [value.valid for value in trajectories], device=self.device
            )
            info = {
                "loss": loss.detach(),
                "log_z": log_z.mean(),
                "mean_beta": beta.mean(),
                "mean_logit_temperature": self.model.temperature(condition).mean(),
                "batch_entropy": -forward_flow.mean(),
                "mean_log_pf": forward_flow.mean(),
                "mean_log_pb": backward_flows.mean(),
                "mean_log_reward": log_reward.mean(),
                "mean_tb_residual": residual.mean(),
                "fresh_loss": trajectory_losses[:num_fresh].mean(),
                # No replay on the first update (or when disabled).
                "replay_loss": trajectory_losses[num_fresh:].sum()
                / max(1, len(trajectories) - num_fresh),
                "valid_loss": (trajectory_losses * valid).sum()
                / valid.sum().clamp_min(1),
                "invalid_loss": (trajectory_losses * ~valid).sum()
                / (~valid).sum().clamp_min(1),
                "invalid_logprob": (forward_flow * ~valid).sum()
                / (~valid).sum().clamp_min(1),
                "batch_invalid_fraction": (~valid).float().mean(),
            }
        return loss, info

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.config.training.ema_decay
        for target, source in zip(
            self.sampling_model.parameters(), self.model.parameters(), strict=True
        ):
            target.mul_(decay).add_(source, alpha=1 - decay)

    def run(self, steps: int | None = None) -> Path:
        final_step = self.step + (
            steps if steps is not None else self.config.training.steps
        )
        log_path = self.output_dir / "training.jsonl"
        last_saved_step = -1
        checkpoint = None
        while self.step < final_step:
            started = perf_counter()
            self.sampling_model.eval()
            count = self.config.training.batch_size
            exponent = self.config.reward.exponent
            if isinstance(exponent, list):
                beta = torch.empty(count).uniform_(*exponent, generator=self.generator)
            else:
                beta = torch.full((count,), float(exponent))
            fixed_preferences = self.config.reward.preferences
            if fixed_preferences is not None:
                preferences = torch.tensor(fixed_preferences, dtype=torch.float32).expand(
                    count, -1
                )
            elif len(self.objectives) == 1:
                preferences = torch.ones(count, 1)
            else:
                # Normalized independent Exp(1) samples give Dirichlet(1).
                preferences = (
                    -torch.rand(count, len(self.objectives), generator=self.generator)
                    .clamp_min(torch.finfo(torch.float32).tiny)
                    .log()
                )
                preferences = preferences / preferences.sum(-1, keepdim=True)
            fresh = self.sampling_policy.rollouts(
                self.config.training.batch_size,
                self.config.training.sampling_temperature,
                self.config.training.random_action_prob,
                beta=beta,
                preferences=preferences,
            )
            rollout_seconds = perf_counter() - started
            reward_metrics = self._assign_rewards(fresh)
            # HSX main samples replay before inserting the new trajectories, so
            # a new sample is not duplicated immediately into the same update.
            batch = fresh + self.replay.sample(
                self.config.training.replay_batch_size, self.python_rng
            )
            self.replay.add(fresh)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss, loss_info = self._loss(batch, len(fresh))
            loss.backward()
            # clip_grad_norm_ already returns the pre-clip norm. No second
            # traversal of policy gradients is needed for diagnostics.
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_parameters, 100.0
            )
            loss_info.update(
                policy_grad_norm=policy_grad_norm,
                # HSX main's grad_norm includes logZ, which is not clipped.
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

            # Transfer scalar diagnostics together rather than synchronizing
            # CUDA separately for every metric. TB values are pre-update.
            loss_metrics = dict(
                zip(
                    loss_info,
                    torch.stack(list(loss_info.values())).detach().cpu().tolist(),
                    strict=True,
                )
            )
            reactions = Counter(max(0, len(value.steps) - 1) for value in fresh)
            failures = Counter(value.invalid_reason for value in fresh if not value.valid)
            action_counts = Counter()
            for trajectory in fresh:
                for transition in trajectory.steps:
                    action = transition.action
                    if action.kind == ActionKind.FIRST_BLOCK:
                        role = "first_block"
                    elif action.kind == ActionKind.BI_REACTION:
                        role = (
                            "brick"
                            if self.env.blocks[action.block_type].is_brick
                            else "linker"
                        )
                    else:
                        terminal = (
                            self.env.uni_reactions[action.reaction].output_type is None
                        )
                        role = "uni_terminal" if terminal else "uni_continue"
                    action_counts[role] += 1
            record = {
                "step": self.step,
                **loss_metrics,
                "fresh_count": len(fresh),
                "replay_count": len(batch) - len(fresh),
                "batch_valid_count": sum(value.valid for value in batch),
                "batch_invalid_count": sum(not value.valid for value in batch),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "mean_reward": sum(value.reward for value in fresh) / len(fresh),
                "mean_objective_rewards": {
                    name: sum(t.objective_rewards[i] for t in fresh) / len(fresh)
                    for i, name in enumerate(self.objectives)
                },
                "mean_preferences": {
                    name: sum(t.preferences[i] for t in fresh) / len(fresh)
                    for i, name in enumerate(self.objectives)
                },
                "valid_fraction": sum(value.valid for value in fresh) / len(fresh),
                # These describe raw rollout attempts, before retry/filtering.
                # In particular, uniqueness is among valid terminal molecules.
                "unique_fraction": len(
                    {value.final_smiles for value in fresh if value.valid}
                )
                / max(1, sum(value.valid for value in fresh)),
                "mean_reactions": sum(max(0, len(value.steps) - 1) for value in fresh)
                / len(fresh),
                "reaction_counts": dict(reactions),
                "action_counts": dict(action_counts),
                "invalid_reasons": dict(failures),
                **reward_metrics,
                "rollout_seconds": rollout_seconds,
                "step_seconds": perf_counter() - started,
            }
            # Keep all fresh attempts, including invalid ones, independently
            # of replay eviction. Existing SMILES/action serialization suffices;
            # no graph tensors, molecule descriptors or fingerprints are added.
            log_started = perf_counter()
            with (self.output_dir / "samples.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.writelines(
                    json.dumps({"step": self.step, "sample": index, **value.to_dict()})
                    + "\n"
                    for index, value in enumerate(fresh)
                )
            record["sample_log_seconds"] = perf_counter() - log_started
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
