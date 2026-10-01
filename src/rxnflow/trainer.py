"""Local trajectory-balance training with replay, EMA, and restart."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from rxnflow._version import __version__
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnv
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.gflownet.runtime import PolicyRuntime, resolve_device, trajectory_sample
from rxnflow.models import RxnFlowModel
from rxnflow.reward import RewardFunction, SampleFilter, evaluate_rewards
from rxnflow.types import Trajectory


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
        self.model = RxnFlowModel(self.env, config.model).to(self.device)
        self.sampling_model = RxnFlowModel(self.env, config.model).to(self.device).eval()
        self.sampling_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.replay = ReplayBuffer(config.training.replay_capacity)
        self.step = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.output_dir / "config.yaml")
        self.runtime = PolicyRuntime(
            self.env, self.model, config, self.device, self.generator
        )
        self.sampling_runtime = PolicyRuntime(
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
                "replay": self.replay.state_dict(),
                "torch_generator": self.generator.get_state(),
                "python_random": self.python_rng.getstate(),
                "reward_class": self._reward_class_name(),
            },
            temporary,
        )
        temporary.replace(destination)
        latest = self.output_dir / "checkpoint_latest.pt"
        latest_temporary = latest.with_suffix(".tmp")
        torch.save(
            torch.load(destination, map_location="cpu", weights_only=False),
            latest_temporary,
        )
        latest_temporary.replace(latest)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
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
        self.model.load_state_dict(checkpoint["model"])
        self.sampling_model.load_state_dict(checkpoint["sampling_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.replay.load_state_dict(checkpoint["replay"])
        self.generator.set_state(checkpoint["torch_generator"])
        self.python_rng.setstate(checkpoint["python_random"])
        self.step = int(checkpoint["step"])

    def _assign_rewards(self, trajectories: list[Trajectory]) -> dict[str, float]:
        values, metrics = evaluate_rewards(
            self.reward,
            [trajectory_sample(value) for value in trajectories],
            self.sample_filter,
        )
        for trajectory, value in zip(trajectories, values, strict=True):
            trajectory.reward = value if trajectory.valid else 0.0
        return metrics

    def _loss(self, trajectories: list[Trajectory]) -> Tensor:
        log_probabilities: list[Tensor] = []
        trajectory_indices: list[int] = []
        backward_flows = torch.tensor(
            [
                sum(step.log_backward for step in trajectory.steps)
                for trajectory in trajectories
            ],
            dtype=torch.float32,
            device=self.device,
        )
        for trajectory_index, trajectory in enumerate(trajectories):
            for transition in trajectory.steps:
                log_probabilities.append(
                    self.runtime.action_log_probability(
                        transition.state, transition.action
                    )
                )
                trajectory_indices.append(trajectory_index)
        if log_probabilities:
            values = torch.stack(log_probabilities)
            indices = torch.tensor(
                trajectory_indices, dtype=torch.long, device=self.device
            )
            forward_flow = sum_by_trajectory(values, indices, len(trajectories))
        else:
            forward_flow = torch.zeros(len(trajectories), device=self.device)
        rewards = torch.tensor(
            [max(value.reward, self.config.reward.floor) for value in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        log_reward = rewards.log() * self.config.reward.exponent
        residual = self.model.log_z + forward_flow - backward_flows - log_reward
        return F.smooth_l1_loss(residual, torch.zeros_like(residual))

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
        while self.step < final_step:
            self.sampling_model.eval()
            fresh = self.sampling_runtime.rollouts(
                self.config.training.batch_size,
                self.config.training.sampling_temperature,
                self.config.training.random_action_prob,
            )
            reward_metrics = self._assign_rewards(fresh)
            self.replay.add(fresh)
            batch = fresh + self.replay.sample(
                self.config.training.replay_batch_size, self.python_rng
            )
            self.model.train()
            loss = self._loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            self._update_ema()
            self.step += 1

            record = {
                "step": self.step,
                "loss": float(loss.detach().cpu()),
                "mean_reward": sum(value.reward for value in fresh) / len(fresh),
                "valid_fraction": sum(value.valid for value in fresh) / len(fresh),
                **reward_metrics,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if self.step % self.config.training.log_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            if self.step % self.config.training.checkpoint_every == 0:
                self.save_checkpoint()
        return self.save_checkpoint()
