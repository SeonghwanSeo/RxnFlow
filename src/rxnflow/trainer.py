"""Local trajectory-balance training with replay, EMA, and restart."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from rxnflow import __version__
from rxnflow.config import Config
from rxnflow.core.types import ActionType, Trajectory
from rxnflow.envs.env import SynthesisEnv
from rxnflow.gflownet.conditioning import ConditionSampler
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.gflownet.property_penalty import compute_property_rewards
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.models import RxnFlowModel
from rxnflow.reward import RewardFunction

logger = logging.getLogger(__name__)


def scalarize_log_rewards(
    values: torch.Tensor,
    preferences: torch.Tensor,
    beta: torch.Tensor | float,
    method: str,
    floor: float,
) -> torch.Tensor:
    """Return beta-scaled log rewards from the weighted product or sum.

    Weights must sum to the number of objectives for ``mul`` or to one for
    ``sum``. Apply the floor to each objective for ``mul``, or to the sum.
    """
    if method == "mul":
        log_rewards = (values.clamp_min(floor).log() * preferences).sum(-1)
    else:
        log_rewards = (values * preferences).sum(-1).clamp_min(floor).log()
    return beta * log_rewards


def init_logger(log_path: Path) -> None:
    """Configure the logger to write to both console and file."""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class RxnFlowTrainer:
    def __init__(
        self,
        config: Config,
        reward: RewardFunction,
        *,
        output_dir: str | Path,
        device: str | torch.device = "cuda",
        seed: int = 1,
    ):
        # Validate the configuration
        config.validate()
        self.config: Config = config

        # Set up the output directory
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(f"output directory already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.config.save(self.output_dir / "config.yaml")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir()
        self.sample_dir = self.output_dir / "samples"
        self.sample_dir.mkdir()
        self.log_file = self.output_dir / "training.log"
        init_logger(self.log_file)

        self.device: torch.device = torch.device(device)
        self.seed: int = seed
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        logger.info("Initializing trainer: device=%s, seed=%d", device, seed)
        logger.info(
            "Config:\n%s",
            OmegaConf.to_yaml(OmegaConf.create(config.to_file_dict())).rstrip(),
        )
        logger.info("Output directory: %s", self.output_dir)

        # Reward function and objectives
        self.reward: RewardFunction = reward
        self.objectives: tuple[str, ...] = tuple(reward.objectives)
        self.num_objectives: int = len(self.objectives)
        if not self.objectives or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("reward.objectives must contain unique objective names")
        logger.info("Objectives: %s", ", ".join(self.objectives))

        self.setup()

    def setup(self) -> None:
        self._setup_env()
        self._setup_model()
        self._setup_optimizer()
        self._setup_replay()
        self._setup_training()

    def _setup_env(self) -> None:
        cfg = self.config
        # Initialize the environment
        logger.info("Loading environment: %s", cfg.env_dir)
        self.env = SynthesisEnv(
            cfg.env_dir,
            cfg.generation.max_atoms,
            cfg.generation.max_reactions,
            cfg.generation.min_synthons,
            cfg.generation.max_synthons,
            cfg.generation.min_reactions,
            cfg.property_penalty,
            cfg.training.retrosynthesis_workers,
        )
        logger.info(
            "Environment loaded: %s libraries, %s synthons.",
            format(len(self.env.synthons), ","),
            format(sum(len(lib) for lib in self.env.synthons.values()), ","),
        )

    def _setup_model(self) -> None:
        """Initialize the model and policy for online training"""
        cfg = self.config
        self.model = RxnFlowModel(self.env, cfg.model, self.num_objectives).to(
            self.device
        )
        self.sampling_model = (
            RxnFlowModel(self.env, cfg.model, self.num_objectives).to(self.device).eval()
        )
        self.sampling_model.load_state_dict(self.model.state_dict())
        self.policy = RxnFlowPolicy(self.env, self.model, cfg, self.device, self.rng)
        self.sampling_policy = RxnFlowPolicy(
            self.env, self.sampling_model, cfg, self.device, self.rng
        )
        logger.info(
            "Model initialized: %s parameters",
            format(sum(p.numel() for p in self.model.parameters()), ","),
        )

    def _setup_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler"""
        cfg = self.config
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
                    "lr": cfg.training.learning_rate_logZ,
                },
            ],
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 2 ** (-step / cfg.training.lr_decay_steps)
        )

    def _setup_replay(self) -> None:
        """Initialize the replay buffer."""
        cfg = self.config
        self.replay = ReplayBuffer(
            cfg.training.replay_capacity,
            cfg.training.num_replay_insert,
            cfg.training.replay_insert_priority,
        )

    def _setup_training(self) -> None:
        """Initialize the gfn training components"""
        cfg = self.config

        self.step = 0
        self.condition_sampler: ConditionSampler = ConditionSampler(
            cfg.reward.beta,
            cfg.reward.moo_preferences,
            self.num_objectives,
            cfg.reward.moo_scalarization,
        )

    def _reward_class_name(self) -> str:
        return f"{type(self.reward).__module__}.{type(self.reward).__qualname__}"

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save model, optimizer, replay, and RNG state for an exact restart."""
        destination = (
            Path(path)
            if path is not None
            else self.checkpoint_dir / f"step_{self.step:06d}.ckpt"
        )
        started = perf_counter()
        logger.info("Saving checkpoint at step %d: %s", self.step, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(
            {
                "rxnflow_version": __version__,
                "step": self.step,
                "config": self.config.to_dict(),
                "run": {
                    "output_dir": str(self.output_dir),
                    "device": str(self.device),
                    "seed": self.seed,
                },
                "environment": self.env.signature,
                "model": self.model.state_dict(),
                "sampling_model": self.sampling_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "replay": self.replay.state_dict(),
                "numpy_rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda"
                    else None
                ),
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
        logger.info("Checkpoint saved: %s (%.1fs)", destination, perf_counter() - started)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        # RNG states are CPU byte tensors even for a CUDA model. Parameter and
        # optimizer loaders move their own tensors to the model device.
        logger.info("Loading checkpoint: %s", path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("rxnflow_version") != __version__:
            raise ValueError(
                "checkpoint was created by RxnFlow "
                f"{checkpoint.get('rxnflow_version')!r}; "
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
        # CPU checkpoints have no CUDA RNG state; keep the initialized device RNG.
        if self.device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], self.device)
        self.seed = checkpoint["run"]["seed"]
        self.step = int(checkpoint["step"])
        logger.info(
            "Checkpoint restored: step=%d, replay=%s",
            self.step,
            format(len(self.replay), ","),
        )

    def _assign_rewards(self, trajectories: list[Trajectory]) -> None:
        """Retain user scores and apply terminal property penalties before beta."""
        mols = [
            Chem.MolFromSmiles(value.final_smiles) if value.valid else None
            for value in trajectories
        ]
        values = self.reward.run(mols)
        property_rewards, violations = compute_property_rewards(
            mols,
            self.env.property_limits,
            self.env.max_atoms,
            self.config.reward.property_penalty_ratio,
        )
        # Property penalties multiply the scalarized reward, independently of
        # MOO preferences. Keep zero rewards here; TB applies its floor below.
        preferences = np.array([t.preferences for t in trajectories], dtype=np.float32)
        log_rewards = scalarize_log_rewards(
            torch.from_numpy(values),
            torch.from_numpy(preferences),
            1.0,  # Logged and replay-priority rewards are independent of beta.
            self.config.reward.moo_scalarization,
            self.config.training.reward_floor,
        )
        scalar_rewards = (
            (log_rewards + torch.from_numpy(property_rewards).log()).exp().tolist()
        )
        for trajectory, objectives, property_reward, violation, scalar in zip(
            trajectories,
            values.tolist(),
            property_rewards.tolist(),
            violations.tolist(),
            scalar_rewards,
            strict=True,
        ):
            trajectory.objective_rewards = objectives
            trajectory.property_reward = property_reward
            trajectory.property_violation = violation
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
            traj_log_p_F = torch.zeros(len(trajectories), device=self.device)
            traj_log_p_F.index_add_(0, batch_idx, log_p_F)
        else:
            traj_log_p_F = torch.zeros(len(trajectories), device=self.device)
        # 3. Reconstruct log rewards from stored objectives and conditions.
        # beta * (log R + log property_reward) applies both before exponentiation.
        objective_rewards = torch.tensor(
            [t.objective_rewards for t in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        scaled_log_R = scalarize_log_rewards(
            objective_rewards,
            preferences,
            beta,
            self.config.reward.moo_scalarization,
            self.config.training.reward_floor,
        )
        log_property_rewards = torch.tensor(
            [t.property_reward for t in trajectories],
            dtype=torch.float64,
            device=self.device,
        ).log().to(dtype=torch.float32)
        scaled_log_R = (scaled_log_R + beta * log_property_rewards).clamp_min(
            beta * math.log(self.config.training.reward_floor)
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
                    "property_reward": value.property_reward,
                    "property_violation": value.property_violation,
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

    def run(
        self,
        num_steps: int,
        *,
        resume_from_checkpoint: str | Path | None = None,
    ) -> Path:
        """Run additional optimization steps and return the final checkpoint."""
        run_started = perf_counter()
        try:
            if resume_from_checkpoint is not None:
                self.load_checkpoint(resume_from_checkpoint)
            if num_steps < 0:
                raise ValueError("steps must be non-negative")
            final_step = self.step + num_steps
            log_path = self.output_dir / "training.jsonl"
            last_saved_step = -1
            checkpoint = None
            logger.info("Starting training; full log at %s", log_path)
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
                    self.config.training.num_replay, self.rng
                )
                self.replay.add(online_trajs, self.rng)
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
                # Generation statistics describe fresh online attempts, excluding replay.
                num_valid = sum(value.valid for value in online_trajs)
                num_unique = len(
                    {value.final_smiles for value in online_trajs if value.valid}
                )
                record = {
                    "step": self.step,
                    **loss_metrics,
                    "num_online": len(online_trajs),
                    "num_replay": len(batch) - len(online_trajs),
                    "num_valid": num_valid,
                    "num_invalid": len(online_trajs) - num_valid,
                    "num_unique": num_unique,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "reward": sum(value.reward for value in online_trajs)
                    / len(online_trajs),
                    "objective_rewards": {
                        f"r_{name}": sum(t.objective_rewards[i] for t in online_trajs)
                        / len(online_trajs)
                        for i, name in enumerate(self.objectives)
                    },
                    "property_reward": sum(t.property_reward for t in online_trajs)
                    / len(online_trajs),
                    "num_property_violations": sum(
                        t.property_violation for t in online_trajs
                    ),
                    "traj_lens": sum(len(value.steps) for value in online_trajs)
                    / len(online_trajs),
                    "sampling_time": sample_time,
                    "time": perf_counter() - started,
                }
                # Keep all online attempts, including invalid ones. This readable
                # path log omits replay-only state flags and backward probabilities;
                # checkpoints retain the complete training trajectories.
                log_started = perf_counter()
                self._write_samples(online_trajs)
                record["logging_time"] = perf_counter() - log_started
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if (
                    self.step == final_step
                    or self.step % self.config.training.log_every == 0
                ):
                    objectives = " ".join(
                        f"{name}={value:.4f}"
                        for name, value in record["objective_rewards"].items()
                    )
                    logger.info(
                        "Step %d/%d: loss=%.4f reward=%.4f valid=%.1f%% "
                        "unique=%.1f%% property_reward=%.4f violations=%d "
                        "replay=%d time=%.2fs | %s",
                        self.step,
                        final_step,
                        record["loss"],
                        record["reward"],
                        100 * record["num_valid"] / record["num_online"],
                        100 * record["num_unique"] / max(1, record["num_valid"]),
                        record["property_reward"],
                        record["num_property_violations"],
                        len(self.replay),
                        record["time"],
                        objectives,
                    )
                if self.step % self.config.training.checkpoint_every == 0:
                    checkpoint = self.save_checkpoint()
                    last_saved_step = self.step
            if last_saved_step != self.step:
                checkpoint = self.save_checkpoint()
            assert checkpoint is not None
            logger.info(
                "Training complete: step=%d, elapsed=%.1fs",
                self.step,
                perf_counter() - run_started,
            )
            return checkpoint
        except BaseException:
            logger.info(
                "Training interrupted at step %d; see traceback for details", self.step
            )
            raise
        finally:
            self.close()
