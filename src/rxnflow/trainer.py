"""Local trajectory-balance training with replay, EMA, and restart."""

from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from omegaconf import OmegaConf

from rxnflow import __version__
from rxnflow.__version__ import (
    check_library_compatibility,
    check_model_compatibility,
)
from rxnflow.conditioning import ConditioningModule
from rxnflow.config import Config
from rxnflow.core.types import Trajectory
from rxnflow.envs.env import SynthesisEnv
from rxnflow.gflownet.conditioning import ConditionSampler
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.gflownet.property_penalty import compute_property_penalties
from rxnflow.gflownet.replay import ReplayBuffer
from rxnflow.gflownet.trajectory_balance import TrajectoryBalance, scalarize_rewards
from rxnflow.models import RxnFlowModel
from rxnflow.prior_guide import PriorGuidance
from rxnflow.reward import RewardFunction

logger = logging.getLogger(__name__)


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
        reward: RewardFunction | list[RewardFunction],
        *,
        seed: int = 1,
        output_dir: str | Path,
        device: str | torch.device = "cuda",
        num_workers: int = 4,
        conditioning: ConditioningModule | None = None,
    ):
        # Set up the configuration
        config.validate()
        self.config: Config = config
        self.num_workers: int = num_workers
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        # Set up the reward functions and objectives
        self.rewards: list[RewardFunction] = (
            list(reward) if isinstance(reward, list) else [reward]
        )
        if not self.rewards:
            raise ValueError("supply at least one named reward")
        self.objectives: tuple[str, ...] = tuple(reward.name for reward in self.rewards)
        if not all(isinstance(name, str) and name.strip() for name in self.objectives):
            raise ValueError("each reward class must define a nonempty string name")
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("supplied reward names must be unique")
        self.num_objectives: int = len(self.objectives)

        # Set up the output directory
        self.output_dir: Path = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(f"output directory already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.config.save(self.output_dir / "config.yaml")
        self.checkpoint_dir: Path = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir()
        self.weights_dir = self.output_dir / "weights"
        self.weights_dir.mkdir()
        self.sample_path: Path = self.output_dir / "sample.jsonl"
        self.log_file: Path = self.output_dir / "training.log"
        init_logger(self.log_file)
        logger.info("Objectives: %s", ", ".join(self.objectives))

        self.device: torch.device = torch.device(device)
        self.seed: int = seed
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        logger.info(
            "Initializing trainer: device=%s, num_workers=%d, seed=%d",
            device,
            num_workers,
            seed,
        )
        if num_workers == 0:
            logger.warning(
                "num_workers=0: retrosynthesis runs in the main process "
                "and may slow training."
            )
        logger.info(
            "Config:\n%s",
            OmegaConf.to_yaml(OmegaConf.create(config.to_file_dict())).rstrip(),
        )
        logger.info("Output directory: %s", self.output_dir)

        self.conditioning: ConditioningModule | None = conditioning
        self.prior_guidance: PriorGuidance | None = (
            PriorGuidance(config.prior_guidance, self.device)
            if config.prior_guidance.model_name_or_path is not None
            else None
        )

        self.setup()

    def setup(self) -> None:
        try:
            self._setup_env()
            self._setup_model()
            self._setup_optimizer()
            self._setup_replay()
            self._setup_training()
        except Exception:
            self.close()
            raise

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
            self.num_workers,
        )
        logger.info(
            "Environment loaded: %s libraries, %s synthons.",
            format(len(self.env.synthons), ","),
            format(sum(len(lib) for lib in self.env.synthons.values()), ","),
        )

    def _setup_model(self) -> None:
        """Initialize the model and policy for online training."""
        cfg = self.config
        condition_dim = (
            self.conditioning.encoding_dim if self.conditioning is not None else None
        )
        use_cond_beta = cfg.gflownet.beta[0] != "fixed"
        use_cond_prior = (
            self.prior_guidance is not None and cfg.prior_guidance.strength[0] != "fixed"
        )
        use_cond_preference = (
            self.num_objectives > 1 and cfg.gflownet.moo_preference[0] == "dirichlet"
        )

        if cfg.pretrained_model is not None:
            logger.info("Loading pretrained model: %s", cfg.pretrained_model)
            ckpt = torch.load(
                cfg.pretrained_model, map_location="cpu", weights_only=False
            )
            check_model_compatibility(ckpt["rxnflow_version"])
            check_library_compatibility(ckpt["environment"]["rxnflow_version"])
            if ckpt["environment"]["synthesis"] != self.env.signature["synthesis"]:
                raise ValueError("pretrained synthesis definitions differ")
            if ckpt["config"]["model"] != asdict(cfg.model):
                raise ValueError("pretrained model architecture differs")
            if ckpt["condition_dim"] != condition_dim:
                raise ValueError("pretrained external condition dimension differs")
            if (
                ckpt["use_cond_preference"]
                and tuple(ckpt["objectives"]) != self.objectives
            ):
                raise ValueError(
                    "pretrained preference conditioning requires matching objectives"
                )
            use_cond_beta = use_cond_beta or ckpt["use_cond_beta"]
            use_cond_prior = use_cond_prior or ckpt["use_cond_prior"]
            use_cond_preference = use_cond_preference or ckpt["use_cond_preference"]

        self.model = RxnFlowModel(
            self.env,
            cfg.model,
            self.num_objectives,
            condition_dim,
            use_cond_beta=use_cond_beta,
            use_cond_prior=use_cond_prior,
            use_cond_preference=use_cond_preference,
        )
        if cfg.pretrained_model is not None:
            weights = dict(ckpt["sampling_model"])
            # Reset _logZ for the downstream reward scale instead of copying it.
            for key, value in self.model._logZ.state_dict().items():
                weights[f"_logZ.{key}"] = value
            # Preserve the initial weights of newly enabled branches.
            encoder = self.model.gfn_condition_encoder
            for name, layer, learned in (
                ("emb_beta", encoder.emb_beta, ckpt["use_cond_beta"]),
                ("emb_prior", encoder.emb_prior, ckpt["use_cond_prior"]),
                ("emb_pref", encoder.emb_pref, ckpt["use_cond_preference"]),
            ):
                if layer is not None and not learned:
                    for key, value in layer.state_dict().items():
                        weights[f"gfn_condition_encoder.{name}.{key}"] = value
            self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.sampling_model = deepcopy(self.model).eval()

        self.policy = RxnFlowPolicy(self.env, self.model, cfg, self.device, self.rng)
        self.sampling_policy = RxnFlowPolicy(
            self.env, self.sampling_model, cfg, self.device, self.rng
        )
        self.prior_model: RxnFlowModel | None = None
        self.prior_policy: RxnFlowPolicy | None = None
        if cfg.exploration.source == "prior":
            assert cfg.pretrained_model is not None
            self._setup_prior_policy(ckpt)
        if cfg.pretrained_model is not None:
            del ckpt
        logger.info(
            "Model initialized: %s parameters",
            format(sum(p.numel() for p in self.model.parameters()), ","),
        )

    def _setup_prior_policy(self, ckpt: dict[str, object]) -> None:
        """Build the frozen pretrained prior policy used for prior-based exploration."""
        cfg = self.config
        exp_cfg = cfg.exploration
        saved_cfg = Config.from_dict(ckpt["config"])
        trained_prior_strength = saved_cfg.prior_guidance.strength
        if exp_cfg.strength is not None:
            Config.validate_sampling_beta(
                trained_prior_strength, ("fixed", [exp_cfg.strength])
            )
        else:
            Config.validate_sampling_beta(
                trained_prior_strength, cfg.prior_guidance.strength
            )
        if ckpt["use_cond_beta"]:
            Config.validate_sampling_beta(saved_cfg.gflownet.beta, ("fixed", [0.0]))
        self.prior_model = (
            RxnFlowModel(
                self.env,
                saved_cfg.model,
                len(ckpt["objectives"]),
                ckpt["condition_dim"],
                use_cond_beta=ckpt["use_cond_beta"],
                use_cond_prior=ckpt["use_cond_prior"],
                use_cond_preference=ckpt["use_cond_preference"],
            )
            .to(self.device)
            .eval()
        )
        self.prior_model.load_state_dict(ckpt["sampling_model"])
        for parameter in self.prior_model.parameters():
            parameter.requires_grad_(False)
        self.prior_policy = RxnFlowPolicy(
            self.env, self.prior_model, cfg, self.device, self.rng
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
            cfg.training.replay_sample_priority,
        )

    def _setup_training(self) -> None:
        """Initialize the gfn training components"""
        cfg = self.config

        self.tb = TrajectoryBalance(
            self.policy,
            cfg.training.reward_floor,
            loss_fn=cfg.training.loss_fn,
            moo_scalarization=cfg.gflownet.moo_scalarization,
        )
        self.step = 0
        self.condition_sampler: ConditionSampler = ConditionSampler(
            cfg.gflownet.beta,
            cfg.gflownet.moo_preference,
            cfg.prior_guidance.strength,
            num_objectives=self.num_objectives,
        )

    def _reward_class_names(self) -> list[str]:
        return [f"{type(r).__module__}.{type(r).__qualname__}" for r in self.rewards]

    def save_checkpoint(self, name: str) -> Path:
        """Save model, optimizer, replay, and RNG state for an exact restart."""
        ckpt_path = self.checkpoint_dir / f"{name}.ckpt"
        model_path = self.weights_dir / f"{name}_model.pth"
        ema_path = self.weights_dir / f"{name}_ema.pth"

        logger.info("Saving checkpoint at step %d: %s", self.step, ckpt_path)
        started = perf_counter()
        metadata = {
            "rxnflow_version": __version__,
            "step": self.step,
            "config": self.config.to_dict(),
            "environment": self.env.signature,
            "condition_dim": self.model.condition_dim,
            "use_cond_beta": self.model.use_cond_beta,
            "use_cond_prior": self.model.use_cond_prior,
            "use_cond_preference": self.model.use_cond_preference,
            "objectives": self.objectives,
        }

        model_state_dict = self.model.state_dict()
        sampling_model_state_dict = self.sampling_model.state_dict()

        ckpt_tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                **metadata,
                "run": {
                    "output_dir": str(self.output_dir),
                    "device": str(self.device),
                    "seed": self.seed,
                },
                "model": model_state_dict,
                "sampling_model": sampling_model_state_dict,
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
                "reward_classes": self._reward_class_names(),
            },
            ckpt_tmp,
        )
        model_tmp = model_path.with_suffix(".tmp")
        torch.save(
            {
                **metadata,
                "sampling_model": model_state_dict,
            },
            model_tmp,
        )
        ema_tmp = ema_path.with_suffix(".tmp")
        torch.save(
            {
                **metadata,
                "sampling_model": sampling_model_state_dict,
            },
            ema_tmp,
        )

        latest_ckpt = self.checkpoint_dir / "latest.ckpt"
        latest_model = self.weights_dir / "latest_model.pth"
        latest_ema = self.weights_dir / "latest_ema.pth"
        shutil.copyfile(ckpt_tmp, latest_ckpt.with_suffix(".tmp"))
        shutil.copyfile(model_tmp, latest_model.with_suffix(".tmp"))
        shutil.copyfile(ema_tmp, latest_ema.with_suffix(".tmp"))

        ckpt_tmp.rename(ckpt_path)
        model_tmp.rename(model_path)
        ema_tmp.rename(ema_path)
        latest_ckpt.with_suffix(".tmp").rename(latest_ckpt)
        latest_model.with_suffix(".tmp").rename(latest_model)
        latest_ema.with_suffix(".tmp").rename(latest_ema)
        logger.info("Checkpoint saved: %s (%.1fs)", ckpt_path, perf_counter() - started)
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        logger.info("Loading checkpoint: %s", path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        check_model_compatibility(checkpoint["rxnflow_version"])
        saved_environment = checkpoint["environment"]
        check_library_compatibility(saved_environment["rxnflow_version"])
        if checkpoint["config"] != self.config.to_dict():
            raise ValueError(
                "restart configuration differs from the resolved checkpoint configuration"
            )
        # Resume restores catalog-indexed actions in replay; sampling can replace
        # the catalog, but resume requires identical content and synthesis rules.
        if (
            saved_environment["content_sha256"] != self.env.signature["content_sha256"]
            or saved_environment["synthesis"] != self.env.signature["synthesis"]
        ):
            raise ValueError(
                "prepared environment differs from the checkpoint environment"
            )
        if checkpoint["reward_classes"] != self._reward_class_names():
            raise ValueError(
                "reward implementation differs from the checkpoint reward class"
            )
        if checkpoint["objectives"] != self.objectives:
            raise ValueError("reward objectives differ from the checkpoint")
        if checkpoint["condition_dim"] != self.model.condition_dim:
            raise ValueError("external condition dimension differs from the checkpoint")
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
        """Store reward factors and untempered diagnostics."""
        smiles_list = [
            value.final_smiles if value.valid else None for value in trajectories
        ]
        prior_values = (
            self.prior_guidance.run(smiles_list)[:, 0]
            if self.prior_guidance is not None
            else [None] * len(smiles_list)
        )
        for trajectory, prior in zip(trajectories, prior_values, strict=True):
            trajectory.prior_log_prob = float(prior) if prior is not None else None
            trajectory.prior_score = (
                float(np.clip(1.0 + float(prior) / 100.0, 0.0, 1.0))
                if prior is not None
                else None
            )

        cond_list = (
            [t.cond for t in trajectories] if self.conditioning is not None else None
        )
        scores = np.stack(
            [reward.run(smiles_list, cond_list) for reward in self.rewards], axis=1
        ).astype(np.float64)
        reward_floor = self.config.training.reward_floor
        property_penalties, violations = compute_property_penalties(
            smiles_list,
            self.env.property_limits,
            self.env.max_atoms,
            self.config.gflownet.property_penalty_ratio,
        )
        preference = np.array([t.preference for t in trajectories], dtype=np.float64)
        rewards = scalarize_rewards(
            scores,
            preference,
            self.config.gflownet.moo_scalarization,
            valid=np.array([t.valid for t in trajectories], dtype=bool),
        )
        rewards += np.log(property_penalties.astype(np.float64).clip(min=reward_floor))
        # Replay priority uses the untempered task and property terms.
        for trajectory, objectives, property_penalty, violation, scalar in zip(
            trajectories,
            scores.tolist(),
            property_penalties.tolist(),
            violations.tolist(),
            rewards.tolist(),
            strict=True,
        ):
            trajectory.objective_rewards = objectives
            trajectory.property_penalty = property_penalty
            trajectory.property_violation = violation
            trajectory.reward = scalar

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.config.training.ema_decay
        for target, source in zip(
            self.sampling_model.parameters(), self.model.parameters(), strict=True
        ):
            target.mul_(decay).add_(source, alpha=1 - decay)

    def _clip_policy_gradients(self) -> torch.Tensor:
        """Clip policy gradients and return their pre-clip norm."""
        grad_clip = self.config.training.grad_clip
        norm = torch.nn.utils.clip_grad_norm_(self.policy_parameters, grad_clip)
        return norm

    def _serialize_samples(self, trajectories: list[Trajectory]) -> list[str]:
        """Serialize online molecules, rewards and complete synthesis routes."""
        lines: list[str] = []
        for index, value in enumerate(trajectories):
            traj = []
            for transition in value.steps:
                traj.append(
                    {
                        "state": transition.state.smiles,
                        **self.env.action_to_dict(transition.action),
                        "product_smiles": transition.product_smiles,
                    }
                )
            sample = {
                "step": self.step,
                "sample": index,
                "final_smiles": value.final_smiles,
                "reward": value.reward,
                "objective_rewards": value.objective_rewards,
                "property_penalty": value.property_penalty,
                "property_violation": value.property_violation,
                "beta": value.beta,
                "prior_strength": value.prior_strength,
                "prior_score": value.prior_score,
                "prior_log_prob": value.prior_log_prob,
                "preference": value.preference,
                **{f"cond_{key}": item for key, item in (value.cond or {}).items()},
                "valid": value.valid,
                "invalid_reason": value.invalid_reason,
                "traj": traj,
            }
            lines.append(json.dumps(sample) + "\n")
        return lines

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
            if num_steps < 0:
                raise ValueError("steps must be non-negative")
            if resume_from_checkpoint is not None:
                self.load_checkpoint(resume_from_checkpoint)
            final_step = self.step + num_steps
            log_path = self.output_dir / "training.jsonl"
            sample_lines: list[str] = []
            log_lines: list[str] = []
            logger.info("Starting training; full log at %s", log_path)
            while self.step < final_step:
                # 1. Generate online trajectories with the EMA model and score rewards.
                started = perf_counter()
                self.sampling_model.eval()
                if self.prior_model is not None:
                    self.prior_model.eval()
                count = self.config.training.num_online
                beta, preference, prior_strength = self.condition_sampler.sample(
                    count, self.rng
                )
                cond_list, cond = None, None
                if self.conditioning is not None:
                    cond_list = self.conditioning.sample(count, self.rng)
                    cond = torch.from_numpy(self.conditioning.encode(cond_list)).to(
                        self.device
                    )
                online_trajs = self.sampling_policy.sample_from_model(
                    count,
                    beta=beta,
                    prior_strength=prior_strength,
                    preference=preference,
                    cond=cond,
                    cond_list=cond_list,
                    random_action_prob=self.config.training.random_action_prob,
                    exploration_policy=self.prior_policy,
                )
                sample_time = perf_counter() - started
                self._assign_rewards(online_trajs)
                # 2. Sample replay before insertion so online trajectories cannot be
                # duplicated as replay entries in this same optimization batch.
                replay_trajs = self.replay.sample(
                    self.config.training.num_replay, self.rng
                )
                batch = online_trajs + replay_trajs
                self.replay.add(online_trajs, self.rng)
                # 3. Update the policy/logZ, learning rates, and EMA sampling weights.
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                # Replay retains its original conditions; never sample new labels here.
                cond = None
                if self.conditioning is not None:
                    cond = torch.from_numpy(
                        self.conditioning.encode([t.cond for t in batch])
                    ).to(self.device)
                loss, loss_info = self.tb.compute_batch_losses(
                    batch,
                    cond=cond,
                    num_online=len(online_trajs),
                    num_replay=len(replay_trajs),
                )
                loss.backward()
                # clip_grad_norm_ already returns the pre-clip norm. No second
                # traversal of policy gradients is needed for diagnostics.
                policy_grad_norm = self._clip_policy_gradients()
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
                    "num_replay": len(replay_trajs),
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
                    "property_penalty": sum(t.property_penalty for t in online_trajs)
                    / len(online_trajs),
                    "prior_score": (
                        sum(t.prior_score for t in online_trajs) / len(online_trajs)
                        if self.prior_guidance is not None
                        else None
                    ),
                    "prior_log_prob": (
                        sum(t.prior_log_prob for t in online_trajs) / len(online_trajs)
                        if self.prior_guidance is not None
                        else None
                    ),
                    "num_property_violations": sum(
                        t.property_violation for t in online_trajs
                    ),
                    "traj_lens": sum(len(value.steps) for value in online_trajs)
                    / len(online_trajs),
                    "sampling_time": sample_time,
                    "time": perf_counter() - started,
                }

                sample_lines.extend(self._serialize_samples(online_trajs))
                log_lines.append(json.dumps(record, sort_keys=True) + "\n")
                report_step = (
                    self.step == final_step
                    or self.step % self.config.training.log_every == 0
                )
                checkpoint_step = self.step % self.config.training.checkpoint_every == 0
                # Save every step's records while amortizing file opens and flushes.
                if report_step or checkpoint_step:
                    with self.sample_path.open("a") as handle:
                        handle.write("".join(sample_lines))
                    sample_lines.clear()
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write("".join(log_lines))
                    log_lines.clear()
                if report_step:
                    objectives = " ".join(
                        f"{name}={value:.4f}"
                        for name, value in record["objective_rewards"].items()
                    )
                    logger.info(
                        "Step %d/%d: loss=%.4f reward=%.4f valid=%.2f "
                        "unique=%.2f violation=%.2f "
                        "replay=%d time=%.2fs | %s",
                        self.step,
                        final_step,
                        record["loss"],
                        record["reward"],
                        record["num_valid"] / record["num_online"],
                        record["num_unique"] / max(1, record["num_valid"]),
                        record["num_property_violations"] / record["num_online"],
                        len(self.replay),
                        record["time"],
                        objectives,
                    )
                if checkpoint_step and self.step < final_step:
                    self.save_checkpoint(f"step_{self.step:06d}")
            checkpoint = self.save_checkpoint(f"step_{self.step:06d}")
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
